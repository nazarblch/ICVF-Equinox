from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, default_init, ensemblize

import flax.linen as nn
import jax
import jax.numpy as jnp

import equinox as eqx
import equinox.nn as eqxnn
import functools

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class ICVFWithEncoder(nn.Module):
    encoder: nn.Module
    vf: nn.Module

    def get_encoder_latent(self, observations: jnp.ndarray) -> jnp.ndarray:     
        return get_latent(self.encoder, observations)
    
    def get_phi(self, observations: jnp.ndarray) -> jnp.ndarray:
        latent = get_latent(self.encoder, observations)
        return self.vf.get_phi(latent)

    def __call__(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf(latent_s, latent_g, latent_z)
    
    def get_info(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf.get_info(latent_s, latent_g, latent_z)
    
def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):    
    if isinstance(icvf_cls_or_name, str):
        icvf_cls = icvfs[icvf_cls_or_name]
    else:
        icvf_cls = icvf_cls_or_name

    if ensemble:
        vf = ensemblize(icvf_cls, 2, methods=['__call__', 'get_info', 'get_phi'])(**kwargs)
    else:
        vf = icvf_cls(**kwargs)
    
    if encoder is None:
        return vf

    return ICVFWithEncoder(encoder, vf)



##
#
# Actual ICVF definitions below
##

class ICVFTemplate(nn.Module):

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Returns useful metrics
        raise NotImplementedError
    
    def get_phi(self, observations):
        # Returns phi(s) for downstream use
        raise NotImplementedError
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Returns V(s, g, z)
        raise NotImplementedError

class MonolithicVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.net = network_cls((*self.hidden_dims, 1), activate_final=False)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return {
            'v': jnp.squeeze(v, -1),
            'psi': outcomes,
            'z': z,
            'phi': observations,
        }
    
    def get_phi(self, observations):
        print('Warning: StandardVF does not define a state representation phi(s). Returning phi(s) = s')
        return observations
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return jnp.squeeze(v, -1)

class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        self.matrix_a = nn.Dense(self.hidden_dims[-1], name='matrix_a')
        self.matrix_b = nn.Dense(self.hidden_dims[-1], name='matrix_b')
        
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> jnp.ndarray:
        return self.get_info(observations, outcomes, intents)['v']
        
    def get_phi(self, observations):
        return self.phi_net(observations)
                        #(1, 29)
    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents) # psi and z are same for V(s, g, g)
        Tz = self.T_net(z)

        # T(z) should be a dxd matrix, but having a network output d^2 parameters is inefficient
        # So we'll make a low-rank approximation to T(z) = (diag(Tz) * A * B * diag(Tz))
        # where A and B are (fixed) dxd matrices and Tz is a d-dimensional parameter dependent on z

        phi_z = self.matrix_a(Tz * phi) # (1, 256)
        psi_z = self.matrix_b(Tz * psi) # (1, 256)
        v = (phi_z * psi_z).sum(axis=-1) # (1, 256) -> (1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
        }

class MultilinearVF_EQX(eqx.Module):
    phi_net: eqx.Module
    psi_net: eqx.Module
    T_net: eqx.Module
    matrix_a: eqx.Module
    matrix_b: eqx.Module
    
    def __init__(self, key, state_dim, hidden_dims):
        key, phi_key, psi_key, t_key, matrix_a_key, matrix_b_key = jax.random.split(key, 6)
        network_cls = functools.partial(eqxnn.MLP, in_size=state_dim, out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu)
        self.phi_net = network_cls(key=phi_key)
        self.psi_net = network_cls(key=psi_key)
        self.T_net = eqxnn.MLP(in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu, key=t_key)
        self.matrix_a = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_a_key)
        self.matrix_b = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_b_key)
        
    def __call__(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi) # (1, 256)
        psi_z = self.matrix_b(Tz * psi) # (1, 256)
        v = (phi_z * psi_z).sum(axis=-1)
        return v
         
icvfs = {
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}