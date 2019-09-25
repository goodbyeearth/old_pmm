import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from subproc_vec_env import *
import pommerman
from pommerman.agents import BaseAgent,SimpleAgent,PlayerAgent,SimpleNoBombAgent

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def make_envs():
    def _thunk():
        env = pommerman.make('PommeFFACompetition-v0',[SimpleAgent(),SimpleAgent(),SimpleAgent(),SimpleAgent()])
        env._agents[0].is_alive = False
        env._agents[2].is_alive = False

        env._agents[3].restart = True
        env._agents[1].restart = True
        env._agents[2].restart = False
        env._agents[0].restart = False
        return env
    return _thunk

def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    print('-----------------------------------')
    learn = get_learn_function(args.alg,submodule=args.submodule)
    # alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    # alg_kwargs.update(extra_args)

    # env = build_env(args)
    # if args.save_video_interval != 0:
    #     env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs() for _ in range(num_envs)]
    env = SubprocVecEnv(envs)
    # if args.network:
    #     alg_kwargs['network'] = args.network
    # else:
    #     if alg_kwargs.get('network') is None:
    #         alg_kwargs['network'] = get_default_network(env_type)

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    # with tf.Session(config=config):
    model = learn(
        env=env,
        seed=seed,
        save_path=args.save_path,
        total_timesteps=total_timesteps,
        nsteps=args.nsteps,
        network=args.network,
        **extra_args
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    print(submodule)
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg, submodule):
    return get_alg_module(alg,submodule).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if args.extra_import is not None:
        import_module(args.extra_import)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        if args.log_path == None:
            logger.configure()
        else:
            if any(args.log_path):
                os.makedirs(args.log_path, exist_ok=True)
            logger.configure(args.log_path)
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)

if __name__ == '__main__':
    # main(sys.argv)

    #a2c_kfac_distillation
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_kfac_distillation',
    #       '--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/420000',
    #       '--nstep=1','--save_path=parameter/parameter75/', '--num_env=2','--log_path=temp/实验75/'])

    # a2c_accuracy
    main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_multihead',
          '--env=PongNoFrameskip-v4','--network=cnn','--load_path=parameter/parameter79/updates21',
          '--nstep=1','--save_path=parameter/parameter87/', '--num_env=2','--log_path=temp/实验87/'])

    #a2c_multihead
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_multihead',
    #       '--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/26000_kfac',
    #       '--nstep=1','--save_path=parameter/parameter85/', '--num_env=2','--log_path=temp/实验85/'])

    #a2c_multihead
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_multihead',
    #       '--env=PongNoFrameskip-v4','--network=cnn','--load_path=parameter/parameter78/updates1',
    #       '--nstep=1','--save_path=parameter/parameter81/', '--num_env=2','--log_path=temp/实验81/'])

    # #a2c_kfac_fm
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_kfac_fm', '--env=PongNoFrameskip-v4','--network=cnn','--load_path=parameter/parameter64/33',
    #       '--nstep=1','--save_path=fish','--num_timesteps=4000', '--num_env=1'])

    ##a2c_distillaiton
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_distillation',
    #       '--env=PongNoFrameskip-v4','--network=cnn',
    #       '--nstep=1','--save_path=parameter/static_agent/', '--num_env=1','--load_path=initial_parameter/26000'])

    ## a2c_kfac_distillation_KLpenality
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_kfac_distillation_KLpenalty',
    #       '--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/26000_kfac',
    #       '--nstep=1','--save_path=parameter/static_simple/', '--num_env=2'])

    # ##a2c_collection
    # main(['collect_data_distillation','--alg=a2c','--submodule=a2c_collect','--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/480000'
    #       ,'--nstep=32','--save_path=distillation_data/3simple_random.npz','--num_timesteps=320000','--num_env=10'])

    ##a2c
    # main(['collect_data_distillation','--alg=a2c','--submodule=a2c', '--env=PongNoFrameskip-v4',
    #       '--network=cnn','--save_path=parameter/parameter57/','--num_timesteps=1e9'])

    ##parameter_fusion
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=parameter_fusion',
    #       '--env=PongNoFrameskip-v4','--network=cnn','--load_path=initial_parameter/420000',
    #       '--nstep=1','--save_path=parameter/parameter999/', '--num_env=2'])

    #a2c_compute_FM
    # main(['collect_data_distillation', '--alg=a2c', '--submodule=a2c_compute_FM', '--env=PongNoFrameskip-v4',
    #       '--network=cnn','--load_path=parameter/parameter69/22',
    #        '--nstep=1','--save_path=fisher_matrix_ewc/22simple__agent_random_1000','--num_timesteps=1000', '--num_env=1'])