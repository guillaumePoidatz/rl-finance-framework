import os

from ray.tune import registry
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

from rl_finance_framework.envs.one_asset_env import (
    LearningCryptoEnv,
)
from rl_finance_framework.models.one_asset_trade_transformer import (
    oneAssetTradeTransformer,
)

registry.register_env(
    name="CryptoEnv", env_creator=lambda env_config: LearningCryptoEnv(**env_config)
)

num_env_runners = 1
num_envs_per_env_runner = 1
num_obs_in_history = 168
num_cpus_per_learner = 7
num_learners = 1
original_lr = 5e-5

# create RL agent
ppo_config = (
    PPOConfig()
    .environment(
        env="CryptoEnv",
        env_config={
            "dataset_name": "dataset",  # .npy files should be in ./data/dataset/
            "leverage": 2,  # leverage for perpetual futures
            "episode_max_len": num_obs_in_history * 2,  # train episode length, 2 weeks
            "lookback_window_len": num_obs_in_history,
            "train_start": [2000, 7000, 12000, 17000, 22000],
            "train_end": [6000, 11000, 16000, 21000, 26000],
            "test_start": [6000, 11000, 16000, 21000, 26000],
            "test_end": [7000, 12000, 17000, 22000, 29377 - 1],
            "order_size": 50,  # dollars
            "initial_capital": 1000,  # dollars
            "open_fee": 0.12e-2,  # taker_fee
            "close_fee": 0.12e-2,  # taker_fee
            "maintenance_margin_percentage": 0.012,  # 1.2 percent
            "initial_random_allocated": 0,  # opened initial random long/short position up to initial_random_allocated $
            "regime": "training",
            "record_stats": False,  # True for backtesting
            "logdir": os.getcwd(),
            # "cold_start_steps": 0, # do nothing at the beginning of the episode
        },
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=oneAssetTradeTransformer,
            model_config={
                "d_history_flat": num_obs_in_history * 38,
                "num_obs_in_history": num_obs_in_history,
                "d_obs": 38,
                "d_time": 2,  # hour, day
                "d_account": 2,  # unrealized_pnl, available_balance
                "d_candlesticks_btc": 34,
                "d_obs_enc": 256,
                "num_attn_blocks": 3,
                "num_heads": 4,
                "dropout_rate": 0.1,
                "num_outputs": 4,
            },
        )
    )
    .training(
        lr=[[0, original_lr], [100, original_lr * (num_learners**0.5)]],
        gamma=0.995,  # 1 recent reward are more important (discount factor).
        grad_clip=30.0,  # max value of the gradient will be 30
        entropy_coeff=0.03,  # for exploration of action space
        kl_coeff=0.05,  # is set in order to slow down or speed up the training depending on kl_target
        kl_target=0.01,  # target of the divergence between two policies
        num_epochs=10,
        use_gae=True,  # Generalized Advantage Estimation
        use_critic=True,  # use critic (value function) to compute the advantage
        lambda_=0.95,
        clip_param=0.3,  # limit the difference between two successive policies
        vf_clip_param=10,  #  limit the difference between two successive value functions
        train_batch_size_per_learner=num_env_runners
        * num_envs_per_env_runner
        * num_obs_in_history,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
        evaluation_config={"explore": False},
        evaluation_num_env_runners=1,
    )
    .env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        rollout_fragment_length=num_obs_in_history,
        batch_mode="complete_episodes",
        preprocessor_pref=None,
        gym_env_vectorize_mode=("ASYNC"),
    )
    .learners(
        num_learners=1,
        num_cpus_per_learner=num_cpus_per_learner,
        num_gpus_per_learner=0,
    )
    .debugging(
        log_level="WARN"  # DEBUG INFO WARN ERROR CRITICAL
    )
    .api_stack(
        enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
    )
)
