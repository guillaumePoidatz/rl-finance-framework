import ray
from ray import tune
from rl_finance_framework.config import (
    ppo_config,
)  # for One-way strategy
import os
# from config_long import ppo_config # for Long only strategy

directory = os.getcwd()

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
