import torch
from tqdm import tqdm
from jaxrl_m.dataset import Dataset
import dataclasses
import numpy as np
import jax
import ml_collections
import equinox as eqx
import numpy as np
from functools import partial
from jax import random
from ott.tools.k_means import k_means
from ott.geometry import pointcloud
from jax import numpy as jnp

@eqx.filter_jit
@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None))
def eval_ensemble_psi(ensemble, s):
    return eqx.filter_vmap(ensemble.psi_net)(s)

@eqx.filter_jit
def kmeans_jax(x, K):
    geom = pointcloud.PointCloud(x, x)
    return k_means(geom, K, n_init=5).assignment

@eqx.filter_jit
def knn_jax(batch_z, all_z, K):
    distance_matrix = jnp.sum((batch_z[:, None, :] - all_z[None, :, :]) ** 2, axis=-1)
    return jax.lax.top_k(-distance_matrix, K)[1]


def batched_knn(X, K, batch_size=2000):
    n = X.shape[0]
    if n > 10_000:
        nb_ids = [jax.device_get(knn_jax(X[i:i+batch_size], X, K)) for i in range(0, n, batch_size)]
        nb_ids = np.concatenate(nb_ids)
    else:
        nb_ids = jax.device_get(knn_jax(X, X, K))
    return nb_ids


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    max_distance: int = None
    curr_goal_shift: int = 0
    neighbours_count: int = 3000

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'max_distance': ml_collections.config_dict.placeholder(int),
            'curr_goal_shift': 0,
            'neighbours_count': 3000
        })

    def __post_init__(self):
        (self.terminal_locs)  = np.nonzero(self.dataset[self.terminal_key] > 0)[0]
        self.terminal_locs = np.concatenate(
            [self.terminal_locs, [self.dataset['observations'].shape[0] - 1]], axis=0
        )
        self.intents = None
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def update_intents(self, icvf_model):
        obs = self.dataset['observations']
        N = obs.shape[0]
        z = []
        for i in range(0, N, 50_000):
            zi = eval_ensemble_psi(icvf_model.value_learner.model, obs[i:i+50_000]).mean(axis=0)
            z.append(jax.device_get(zi))
        z = np.concatenate(z, 0)
        # NC = 50
        # assignment = jax.device_get(kmeans_jax(z, NC))

        self.intents = z
        # self.assignment = assignment
        # pos_ids = np.arange(N)
        print("intents updated")
        # K = np.min([(assignment == c).astype(np.int32).sum() for c in range(NC)] + [3000]) - 1
        K=self.neighbours_count
        self.neighbours = np.empty((N, K), dtype=np.int32)

        for c in tqdm(range(0, N, 50_000)):
            to = c+50_000
            if to + K >= N:
                to = to + K
            zc = self.intents[c:to]
            self.neighbours[c:to] = batched_knn(zc, K) + c

            if to >= N:
                break

        print("neighbours found")

    def sample_from_neighbours(self, indices: np.ndarray):
        goal_indx = np.random.randint(self.neighbours.shape[1], size=indices.shape[0])
        return self.neighbours[indices, goal_indx]

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size-self.curr_goal_shift, size=batch_size)

        if self.intents is not None:
            goal_indx = self.sample_from_neighbours(indx)
            # norm = jnp.linalg.norm(self.intents[indx] - self.intents[goal_indx]) 
            # print(norm)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        if self.max_distance is not None:
            final_state_indx = np.clip(final_state_indx, 0, indx + self.max_distance)
            
        distance = np.random.rand(batch_size)
        middle_goal_indx = np.round(((indx) * distance + final_state_indx * (1- distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx+self.curr_goal_shift], self.dataset['observations'])

        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    p_samegoal: float = 0.5
    intent_sametraj: bool = False

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'p_samegoal': 0.5,
            'intent_sametraj': False,
            'max_distance': ml_collections.config_dict.placeholder(int),
            'curr_goal_shift': 0,
            'neighbours_count': 3000
        })

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        if self.intent_sametraj:
            desired_goal_indx = self.sample_goals(indx, p_randomgoal=0.0, p_trajgoal=1.0 - self.p_currgoal, p_currgoal=self.p_currgoal)
        else:
            desired_goal_indx = self.sample_goals(indx)
        
        goal_indx = self.sample_goals(indx)
        goal_indx = np.where(np.random.rand(batch_size) < self.p_samegoal, desired_goal_indx, goal_indx)

        success = (indx == goal_indx)
        desired_success = (indx == desired_goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['desired_rewards'] = desired_success.astype(float) * self.reward_scale + self.reward_shift
        
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
            batch['desired_masks'] = (1.0 - desired_success.astype(float))
        
        else:
            batch['masks'] = np.ones(batch_size)
            batch['desired_masks'] = np.ones(batch_size)
        
        goal_indx = np.clip(goal_indx + self.curr_goal_shift, 0, self.dataset.size-1)
        desired_goal_indx = np.clip(desired_goal_indx + self.curr_goal_shift, 0, self.dataset.size-1)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        batch['desired_goals'] = jax.tree_map(lambda arr: arr[desired_goal_indx], self.dataset['observations'])

        return batch
