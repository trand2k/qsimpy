import argparse

import ray
from ray import tune, air, train
from ray.tune.registry import register_env
from env_creator import qsimpy_env_creator
from ray.rllib.algorithms.impala import IMPALAConfig  # Changed to IMPALA
from ray.rllib.utils.framework import try_import_tf
import os

tf1, tf, tfv = try_import_tf()
parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()
    register_env("QSimPyEnv", qsimpy_env_creator)

    config = (
        IMPALAConfig()
        .framework(framework=args.framework)
        .resources(num_gpus=0)
        .environment(
            env="QSimPyEnv",
            env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_36.csv",
            },
        )
        .training(
            lr=tune.grid_search([0.01]),
            train_batch_size=50,  # Adjusted to match expected tensor size
            vtrace=True,
        )
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
    )


    stop_config = {
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }
    
    # Get the absolute path of the current directory
    current_directory = os.getcwd()

    # Append the "result" folder to the current directory path
    result_directory = os.path.join(current_directory, "results")

    # Create the storage_path with the "file://" scheme
    storage_path = f"file://{result_directory}"

    dict_ = config.to_dict()
    dict_["num_env_runners"] = 2
    results = tune.Tuner(
        "IMPALA",  # Changed to IMPALA
        run_config=air.RunConfig(
            stop=stop_config,
            checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
            storage_path=storage_path, 
            name="IMPALA_QCE_1000"  # Changed the name accordingly
        ),
        param_space=dict_,
    ).fit()

    ray.shutdown()
