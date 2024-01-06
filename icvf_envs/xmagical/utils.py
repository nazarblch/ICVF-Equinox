import gym
import xmagical
xmagical.register_envs()
from PIL import Image
import numpy as np
from jaxrl_m.evaluation import EpisodeMonitor
from jaxrl_m.dataset import Dataset
import jax

def make_env(modality, randomize=False):
    modality = modality.capitalize()
    assert modality in ['Gripper', 'Shortstick', 'Mediumstick', 'Longstick']
    if randomize:
        env = gym.make(f'SweepToTop-{modality}-Pixels-Allo-TestLayout-v0')
    else:
        env = gym.make(f'SweepToTop-{modality}-Pixels-Allo-Demo-v0')
    transform_obs = lambda obs: np.asarray(Image.fromarray(obs).resize((64, 64)))
    env = gym.wrappers.TransformObservation(env, transform_obs)
    env = EpisodeMonitor(env)
    return env

def get_dataset(modality, dir_name='', keys=None):
    """ If keys is None, return all keys, else return specified keys"""
    modality = modality.lower()
    assert modality in ['gripper', 'shortstick', 'mediumstick', 'longstick']
    fname = f'{dir_name}/{modality}_train.npz'
    buffer = np.load(fname)
    if keys is None: keys = buffer.keys()
    dataset_dict = {k: buffer[k] for k in keys}
    dataset_dict['images'] = dataset_dict['observations']
    dataset_dict['observations'] = dataset_dict['states'].reshape(dataset_dict['states'].shape[0], 3, -1)[:, 0, :]
    dataset_dict['next_observations'] = dataset_dict['next_states'].reshape(dataset_dict['next_states'].shape[0], 3, -1)[:, 0, :]
    return Dataset(dataset_dict)

def get_all_datasets(dir_name=''):
    return {modality: get_dataset(modality, dir_name) for modality in ['gripper', 'shortstick', 'mediumstick', 'longstick']}

def crossembodiment_dataset(not_modality, dir_name):
    datasets = []
    keys = ['states', 'observations', 'next_states', 'rewards', 'masks', 'dones_float']
    for i, modality in enumerate(['longstick', 'shortstick', 'mediumstick']): # 'gripper', 'shortstick', 'mediumstick', 
        if modality == not_modality: continue
        dataset = get_dataset(modality, dir_name, keys=keys)
        dataset = dataset.copy({'embodiment': np.full(dataset.size, i)})
        datasets.append(dataset._dict)
    full_dataset = Dataset(jax.tree_map(lambda *arrs: np.concatenate(arrs, axis=0), *datasets))
    return full_dataset