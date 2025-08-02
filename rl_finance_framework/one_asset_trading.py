import os
import argparse
from utils.training_ressources_config import TrainingRessourcesConfig

import ray
from ray import tune

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
        "--num-envs-per-runner",
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
    args = parser.parse_args()

    ppo_config["num_rollout_workers"] = args.num_env_runners
    ppo_config["num_envs_per_worker"] = args.num_envs_per_runner
    ppo_config["num_gpus_per_learner"] = args.num_gpus_per_learner
    ppo_config["num_cpus_per_learner"] = args.num_cpus_per_learner
    ppo_config["num_learners"] = args.num_learners

    ray.shutdown()
    ray.init()

    tune.run(
        "PPO",
        stop={"training_iteration": 2000},
        config=ppo_config,
        storage_path="file://"
        + os.path.abspath("./results"),  # default folder "~ray_results"
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
