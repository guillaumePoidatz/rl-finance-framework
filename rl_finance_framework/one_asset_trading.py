import os
import argparse

import ray
from ray import tune

from rl_finance_framework.baseline_config import (
    ppo_baseline_config,
)
from rl_finance_framework.config import (
    ppo_config,
)  # for One-way strategy
# from config_long import ppo_config # for Long only strategy


def main():
    parser = argparse.ArgumentParser(
        description="RLlib training ressources configuration"
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=1,
        help="Number of environment runners (parallel rollout workers)",
    )
    parser.add_argument(
        "--num_envs_per_env_runner",
        type=int,
        default=1,
        help="Number of environments per environment runner",
    )
    parser.add_argument(
        "--num-cpus-per-learner",
        type=int,
        default=1,
        help="Number of CPUs allocated per learner process",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=1,
        help="Number of learner processes (for multi-GPU or distributed setup)",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=int,
        default=0,
        help="Number of GPUs to assign per learner process",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=2000, help="Number of training iterations"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="file://" + os.path.abspath("./results"),
        help="Storage path for results and checkpoints",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=168,
        help="Length of the decision sequence",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=1000,
        help="Size of the minibatch",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=ppo_baseline_config,
        help="configuration template to use: ppo_baseline_config or ppo_config",
    )
    args = parser.parse_args()

    ppo_config["num_env_runners"] = args.num_env_runners
    ppo_config["num_envs_per_env_runner"] = args.num_envs_per_env_runner
    ppo_config["num_gpus_per_learner"] = args.num_gpus_per_learner
    ppo_config["num_cpus_per_learner"] = args.num_cpus_per_learner
    ppo_config["num_learners"] = args.num_learners
    ppo_config["rollout_fragment_length"] = args.rollout_fragment_length
    ppo_config["train_batch_size_per_learner"] = (
        args.num_env_runners
        * args.num_envs_per_env_runner
        * ppo_config["rollout_fragment_length"]
    )
    ppo_config["minibatch_size"] = min(
        args.minibatch_size, ppo_config["train_batch_size_per_learner"]
    )
    ray.shutdown()
    ray.init()

    tune.run(
        "PPO",
        stop={"training_iteration": 2000},
        base_config=(
            ppo_baseline_config
            if args.configuration == "ppo_baseline_config"
            else ppo_config
        ),
        storage_path=args.storage_path,  # default folder "~ray_results"
        checkpoint_config={
            "checkpoint_frequency": 12,
            "checkpoint_at_end": False,
            "num_to_keep": None,
            # keep all the checkpoints (put a number x to keep the x last checkpoints only)
        },
        checkpoint_at_end=False,
        keep_checkpoints_num=None,
        verbose=2,
        reuse_actors=False,
        log_to_file=True,
    )

    # kind of algorithm that can be used : PPO DQN A3C DDPG SAC TD3 APPO IMPALA
    # verbose : 0 = silent, 1 = default, 2 = verbose


if __name__ == "__main__":
    main()
