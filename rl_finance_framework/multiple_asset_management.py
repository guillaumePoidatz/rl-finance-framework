import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import torch
import numpy as np
import argparse
import json

from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.logger import *
from stable_baselines3.common.monitor import *

from importlib import import_module

# BE CAREFUL ! THIS SCRIPT IS DEPRECATED
# not always necessary
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

#  if you try to access a value that doesn't exist, it will create a default key asscoiated with it
_game_envs = defaultdict(set)
# check for all env inside gym and add the id to our local dict
for env in gym.envs.registry.values():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
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



def build_testenv(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    seed = args.seed

    env_type, env_id = get_env_type('RLTestStock-v0')

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)

    return env
    

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type == 'atari':
        if alg == 'acer':
            env = make_vec_env(env_id, env_type, nenv, seed)
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            # TODO check if the second seeding is necessary, and eventually remove
            env.seed(seed)
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_vec_env(env_id, nenv, seed), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000,
                                        use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)

    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        env = make_vec_env(env_id, args.num_env or 1, seed)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env

def get_alg_module(alg,submodule=None):
    submodule = submodule or alg
    alg_module = import_module('.'.join(['stable_baselines3', alg, submodule]))

    return alg_module


def get_RL_algo(alg):
    if alg == 'ppo':
        from stable_baselines3 import PPO
        return PPO
    elif alg == 'dqn':
        from stable_baselines3 import DQN
        return DQN
    elif alg == 'ddpg':
        from stable_baselines3 import DDPG
        return DDPG
    elif alg == 'a2c':
        from stable_baselines3 import A2C
        return A2C
    elif alg == 'sac':
        from stable_baselines3 import SAC
        return SAC
    elif alg == 'td3':
        from stable_baselines3 import TD3
        return TD3
    
def get_env_type(env_id):
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


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    RL_algo = get_RL_algo(args.alg)

    env = build_env(args)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id,args.network, args.num_timesteps))

    model = RL_algo(args.network, env, seed=seed, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    return model, env

def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval

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

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='RLStock-v0')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str, default='rlstock')
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ddpg',choices = ['a2c','ddpg','dqn','sac','td3','ppo'])
    parser.add_argument('--num_timesteps', type=float, default=1e4),
    parser.add_argument('--network', help='network type' , default='mlp',choices = ['MlpPolicy','CnnPolicy','MultiInputPolicy'])
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser

# args we need env, alg:
def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)
    
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure()
    else:
        # disable logger with empty string
        configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    env.close()


    # if args.save_path is not None and rank == 0:
    #     save_path = osp.expanduser(args.save_path)
    #     model.save(save_path)

    if args.play:
        # just a message to tell the user that it is runnung
        Logger.log("Running trained model")
        env = build_testenv(args)
        obs = env.reset()
        # done = False

        #hardcode the data length
        for i in range(686):

            actions, _, _, _ = model.step(obs)
            obs, _, done, _ = env.step(actions)
            # env.render()
            # done = done.any() if isinstance(done, np.ndarray) else done

            # if done:
            #     obs = env.reset()

        env.close()
        
    model.save(args.log_path)

if __name__ == '__main__':
    main()
    
