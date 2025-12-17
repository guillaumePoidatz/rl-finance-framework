from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

num_env_runners = 1
num_envs_per_env_runner = 1
num_obs_in_history = 168
num_cpus_per_learner = 4
num_learners = 1
num_gpus_per_learner = 1
original_lr = 5e-5

# create RL agent
ppo_baseline_config = (
    PPOConfig()
    .environment(
        env="CartPole",
    )
    .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[32, 32]))
    .training(
        lr=[[0, original_lr], [100, original_lr * (num_learners**0.5)]],
        gamma=0.995,  # 1 recent reward are more important (discount factor).
        grad_clip=30.0,  # max value of the gradient will be 30
        entropy_coeff=0.03,  # for exploration of action space
        kl_coeff=0.05,  # is set in order to slow down or speed up the training depending on kl_target
        kl_target=0.01,  # target of the divergence between two policies
        use_gae=True,  # Generalized Advantage Estimation
        use_critic=True,  # use critic (value function) to compute the advantage
        lambda_=0.95,
        clip_param=0.3,  # limit the difference between two successive policies
        vf_clip_param=10,  #  limit the difference between two successive value functions
        train_batch_size_per_learner=30000,
        minibatch_size=10000,
        num_epochs=10,
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
        gym_env_vectorize_mode=("async"),
    )
    .learners(
        num_learners=1,
        num_cpus_per_learner=num_cpus_per_learner,
        num_gpus_per_learner=num_gpus_per_learner,
    )
    .debugging(
        log_level="WARN"  # DEBUG INFO WARN ERROR CRITICAL
    )
    .api_stack(
        enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
    )
)
