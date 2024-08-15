import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym  # Use Gymnasium instead of Gym

# Initialize Ray
ray.init()

# Define the training configuration using PPOConfig
config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .framework("torch")  # Use "torch" for PyTorch
    .rollouts(num_rollout_workers=2)  # Number of parallel workers
    .training(
        lr=1e-3,
        model={"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"},
    )
)

# Run the training using Tune with Ray AIR
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 200, "training_iteration": 10},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1,
            checkpoint_at_end=True,
        ),
    ),
)
results = tuner.fit()

# Save the trained model
checkpoint_path = results.get_best_result().checkpoint
print(f"Best checkpoint saved at {checkpoint_path}")

# Shutdown Ray
ray.shutdown()
