from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import optax
from jaxrl_m.common import TrainTargetStateEQX

import ml_collections
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
import equinox as eqx
import equinox.nn as nn
from src.icvf_networks import MultilinearVF_EQX
import dataclasses
import functools

def expectile_loss(adv, diff, expectile=0.9):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2


def icvf_loss(value_fn, target_value_fn, batch, config):

    assert all([k in config for k in ['no_intent', 'min_q', 'expectile', 'discount']]), 'Missing ICVF config keys'

    if config['no_intent']:
        batch['desired_goals'] = jax.tree_map(jnp.ones_like, batch['desired_goals'])

    ###
    # Compute TD error for outcome s_+
    # 1(s == s_+) + V(s', s_+, z) - V(s, s_+, z)
    ###

    (next_v1_gz, next_v2_gz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['goals'], batch['desired_goals'])
    q1_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v1_gz
    q2_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v2_gz
    #q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = eval_ensemble(value_fn, batch['observations'], batch['goals'], batch['desired_goals'])

    ###
    # Compute the advantage of s -> s' under z
    # r(s, z) + V(s', z, z) - V(s, z, z)
    ###

    (next_v1_zz, next_v2_zz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['desired_goals'], batch['desired_goals'])
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2
    
    q_zz = batch['desired_rewards'] + config['discount'] * batch['desired_masks'] * next_v_zz

    (v1_zz, v2_zz) = eval_ensemble(target_value_fn, batch['observations'], batch['desired_goals'], batch['desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2
    adv = q_zz - v_zz

    if config['no_intent']:
        adv = jnp.zeros_like(adv)

    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    def masked_mean(x, mask):
        return (x * mask).sum() / (1e-5 + mask.sum())

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v_gz max': v1_gz.max(),
        'v_gz min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
        'reward mean': batch['rewards'].mean(),
        'mask mean': batch['masks'].mean(),
        'q_gz max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
        'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
    }

class ICVF_EQX_Agent(eqx.Module):
    value_learner: TrainTargetStateEQX
    config: dict

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None), out_axes=0)
def eval_ensemble(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble)(s, g, z)

@eqx.filter_jit
def update(agent, batch):
    (val, value_aux), v_grads = eqx.filter_value_and_grad(icvf_loss, has_aux=True)(agent.value_learner.model, agent.value_learner.target_model, batch, agent.config)
    updated_v_learner = agent.value_learner.apply_updates(v_grads).soft_update()
    return dataclasses.replace(agent, value_learner=updated_v_learner), value_aux
    
def create_eqx_learner(seed: int,
                       observations: jnp.array,
                       hidden_dims: list,
                       load_pretrained_icvf: bool=False,
                       optim_kwargs: dict = {
                            'learning_rate': 0.00005,
                            'eps': 0.0003125
                        },
                        discount: float = 0.99,
                        target_update_rate: float = 0.005,
                        expectile: float = 0.9,
                        no_intent: bool = False,
                        min_q: bool = True,
                        periodic_target_update: bool = False,
                        **kwargs):
        print('Extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        
        if load_pretrained_icvf:
            network_cls_phi = functools.partial(nn.MLP, in_size=observations.shape[-1], out_size=hidden_dims[-1], final_activation=jax.nn.relu,
                                        width_size=hidden_dims[0], depth=len(hidden_dims))
            network_cls_psi = functools.partial(nn.MLP, in_size=observations.shape[-1], out_size=hidden_dims[-1], final_activation=jax.nn.relu,
                                        width_size=hidden_dims[0], depth=len(hidden_dims))
            network_cls_T = functools.partial(nn.MLP, in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], final_activation=jax.nn.relu,
                                              depth=len(hidden_dims))
            loaded_matrix_a = functools.partial(nn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
            loaded_matrix_b = functools.partial(nn.Linear, in_features=hidden_dims[-1], out_features=hidden_dims[-1])
            
            phi_net = network_cls_phi(key=rng)
            psi_net = network_cls_psi(key=rng)
            T_net = network_cls_T(key=rng)
            matrix_a = loaded_matrix_a(key=rng)
            matrix_b = loaded_matrix_b(key=rng)
            loaded_phi_net = eqx.tree_deserialise_leaves("/home/m_bobrin/icvf_release/icvf_model_phi.eqx", phi_net)
            loaded_psi_net = eqx.tree_deserialise_leaves("/home/m_bobrin/icvf_release/icvf_model_psi.eqx", psi_net)
            loaded_T_net = eqx.tree_deserialise_leaves("/home/m_bobrin/icvf_release/icvf_model_T.eqx", T_net)
            loaded_matrix_a = eqx.tree_deserialise_leaves("/home/m_bobrin/icvf_release/icvf_model_a.eqx", matrix_a)
            loaded_matrix_b = eqx.tree_deserialise_leaves("/home/m_bobrin/icvf_release/icvf_model_b.eqx", matrix_b)
        else:
            loaded_phi_net = None
            loaded_psi_net = None
            loaded_T_net = None
            loaded_matrix_a = None
            loaded_matrix_b = None
            
        @eqx.filter_vmap
        def ensemblize(keys):
            return MultilinearVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims,
                                     pretrained_phi=loaded_phi_net, pretrained_psi=loaded_psi_net, pretrained_T=loaded_T_net,
                                     pretrained_a=loaded_matrix_a, pretrained_b=loaded_matrix_b)
            
        value_learner = TrainTargetStateEQX.create(
            model=ensemblize(jax.random.split(rng, 2)),
            target_model=ensemblize(jax.random.split(rng, 2)),
            optim=optax.adam(**optim_kwargs)
        )
        config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
        return ICVF_EQX_Agent(value_learner=value_learner, config=config)
    
def get_default_config():
    config = ml_collections.ConfigDict({
        'optim_kwargs': {
            'learning_rate': 3e-4,
            'eps': 0.0003125
        }, # LR for vision here. For FC, use standard 1e-3
        'discount': 0.99,
        'expectile': 0.9,  # The actual tau for expectiles.
        'target_update_rate': 0.005,  # For soft target updates.
        'no_intent': False,
        'min_q': True,
        'periodic_target_update': False,
    })

    return config
