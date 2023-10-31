import d4rl
import gym
import numpy as np

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor

from typing import *

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def compute_mean_std(states: np.ndarray, eps: float):
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize(states, mean, std):
    return (states - mean) / std

def get_dataset(env: gym.Env,
                gcrl: bool = True,
                dataset=None,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                obs_dtype=np.float32,
                normalize_states=False,
                normalize_rewards=True
                ):
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        if 'antmaze' in env.spec.id.lower() and gcrl:
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.

            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
        else:
            dones_float = dataset['terminals'].copy()

        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        if normalize_states:
            state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
            observations = normalize(dataset["observations"], state_mean, state_std)
            next_observations = normalize(dataset["next_observations"], state_mean, state_std)
        
        if normalize_rewards:
            dataset = modify_reward(dataset, env_name=env.spec.id)
            print(f"Rewards mean: {dataset['rewards'].mean()}")
            
        return Dataset.create(
            observations=observations,
            next_observations=next_observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
        )

def modify_reward(
    dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000
):
    if any(s in env_name.lower() for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = get_normalization(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return dataset

def get_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    if 'antmaze' in env_name:
         return  dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset