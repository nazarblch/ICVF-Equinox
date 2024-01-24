import os
import warnings

warnings.filterwarnings("ignore")
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import pickle
from ml_collections import config_flags
import wandb

from jaxrl_m.wandb import setup_wandb, default_wandb_config
from src.gc_dataset import GCSDataset
from src.icvf_learner import update, eval_ensemble
from src import icvf_learner as learner
from jaxrl_m.vision import encoders
from src.icvf_networks import icvfs
from icvf_envs import xmagical
import os
from absl import app, flags
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from src import viz_utils as viz

import equinox as eqx
import equinox.nn as nn

FLAGS = flags.FLAGS
flags.DEFINE_enum('modality', 'gripper', [
                  'gripper', 'shortstick', 'mediumstick', 'longstick'], 'Modality name')
flags.DEFINE_enum('video_type', 'same', [
                  'same', 'cross', 'all'], 'Type of video data to train on (only modality, all but modality, or all)')
flags.DEFINE_string('dataset', f'/home/m_bobrin/icvf_release/experiments/xmagical/xmagical_replay',
                    'Directory containing datasets')

flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 200, 'Metric logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 25000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_enum('icvf_type', 'multilinear',
                  list(icvfs), 'Which model to use.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden sizes.')
flags.DEFINE_bool('view_mode', True, 'whether to use pixel based or state based.')

def update_dict(d, additional):
    d.update(additional)
    return d


wandb_config = update_dict(
    default_wandb_config(),
    {
        'project': 'ICVF_Baseline',
        'group': 'icvf_baseline',
        'name': '{icvf_type}_{modality}_{video_type}',
    }
)

config = learner.get_default_config()
gcdataset_config = GCSDataset.get_default_config()

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

def main(_):
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    setup_wandb(params_dict, **FLAGS.wandb)
    if FLAGS.view_mode:
        keys = ['states', 'next_states', 'rewards', 'masks', 'dones_float']
    if FLAGS.video_type == 'same':
        video_dataset = xmagical.get_dataset(FLAGS.modality, FLAGS.dataset, keys=keys)
    elif FLAGS.video_type == 'cross':
        video_dataset = xmagical.crossembodiment_dataset(
            FLAGS.modality, FLAGS.dataset)
    elif FLAGS.video_type == 'all':
        video_dataset = xmagical.crossembodiment_dataset(None, FLAGS.dataset)
    else:
        raise ValueError(f'Invalid video type {FLAGS.video_type}')

    gc_dataset = GCSDataset(video_dataset, **FLAGS.gcdataset.to_dict())
    example_batch = gc_dataset.sample(1)

    hidden_dims = tuple([int(h) for h in FLAGS.hidden_dims])
    agent = learner.create_eqx_learner(FLAGS.seed,
                                   example_batch['observations'],
                                   hidden_dims=hidden_dims,
                                   load_pretrained_icvf=False,
                                   **FLAGS.config)

    visualizer = DebugPlotGenerator(video_dataset)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        agent, update_info = update(agent, batch)

        if i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, batch)
            train_metrics = {f'training/{k}': v for k,
                             v in update_info.items()}
            train_metrics.update(
                {f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            visualizations = visualizer.generate_debug_plots(agent)
            eval_metrics = {f'visualizations/{k}': v for k,
                            v in visualizations.items()}
            wandb.log(eval_metrics, step=i)
            
        if i % FLAGS.save_interval == 0:
            base_path = "/home/nazar/projects/AILOT/pretrained_icvf/halfcheetah-medium-decomposed/"
            print("save to", base_path)
            unensemble_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, agent.value_learner.model)
            with open(base_path + "icvf_model_phi.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.phi_net)
            with open(base_path + "icvf_model_psi.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.psi_net)
            with open(base_path + "icvf_model_T.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.T_net)
            with open(base_path + "icvf_model_a.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.matrix_a)
            with open(base_path + "icvf_model_b.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.matrix_b)
###################################################################################################
#
# Creates wandb plots
#
###################################################################################################


class DebugPlotGenerator:
    def __init__(self, video_dataset):
        first_done = (video_dataset['masks'] == 0).argmax()
        self.traj_batch = video_dataset.get_subset(np.arange(first_done+1))

    def generate_debug_plots(self, unrep_agent):
        plot_items = [
            partial(viz.visualize_metric, metric_name=k) if isinstance(k, str)
            else partial(viz.visualize_metrics, metric_names=k)
            for k in [
                'dist_from_beginning',
                'dist_from_end',
                'dist_from_middle',
                'density_from_beginning',
                'density_from_end',
                'density_from_middle',
            ]
        ]

        metrics = get_distances(unrep_agent, self.traj_batch)
        img = viz.make_visual(
            self.traj_batch['images'], metrics, plot_items) #observations
        return {'image': wandb.Image(img)}

###################################################################################################
#
# Helper functions for visualization
#
###################################################################################################


@eqx.filter_jit
def get_debug_statistics(agent, batch):
    def get_info(s, g, z):
        if agent.config['no_intent']:
            return agent.value(s, g, jnp.ones_like(z), method='get_info')
        else:
            return eval_ensemble(agent.value_learner.model, s, g, z)

    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals']

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz.mean(),
        'v_szz': info_szz.mean(),
        'v_sgz': info_sgz.mean(),
        'v_sgg': info_sgg.mean(),
        'v_szg': info_szg.mean(),
        'diff_szz_szg': (info_szz - info_szg).mean(),
        'diff_sgg_sgz': (info_sgg - info_sgz).mean(),
    })
    return stats


@eqx.filter_jit
def get_distances(agent, batch):
    def get_v(o, g):
        v = eval_ensemble(agent.value_learner.model, o[None], g[None], g[None]).mean(0)
        return v
        #v1, v2 = agent.value(o, g, g)
        #return v1

    distances = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(
        0, None))(batch['observations'], batch['observations'])

    def get_density(start, o):
        v = eval_ensemble(agent.value_learner.model, start[None], o[None], batch['observations'][-1][None]).mean(0)
        return v
        #v1, v2 = agent.value(start, o, batch['observations'][-1])
        #return v1
    densities = jax.vmap(jax.vmap(get_density, in_axes=(None, 0)), in_axes=(
        0, None))(batch['observations'], batch['observations'])

    return {
        'dist_from_beginning': distances[:, 0],
        'dist_from_end': distances[:, -1],
        'dist_from_middle': distances[:, distances.shape[1]//2],
        'density_from_beginning': densities[0],
        'density_from_end': densities[-1],
        'density_from_middle': densities[densities.shape[0]//2],
    }


####################


if __name__ == '__main__':
    app.run(main)
