import sys, os
import metaworld
import gymnasium as gym
import metaworld, mujoco
import numpy as np
import wandb
import argparse
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
sys.path.append("..")
from rlkit.envs.wrappers import NormalizedBoxEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--train_num', type=int, default=8)
    parser.add_argument('--eval_num', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=16000)
    parser.add_argument('--max_episode_steps', type=int, default=500)
    parser.add_argument('--train_max_steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--project_name', type=str, default="metaworld")
    parser.add_argument('--exp_name', type=str, default="oracle")
    parser.add_argument('--reward_path', type=str, default=None)

    args = parser.parse_args()

    if args.reward_path is not None:
        with open(args.reward_path, "r") as f:
            reward_code_str = f.read()
        namespace = {**globals()}
        exec(reward_code_str, namespace)
        new_function = namespace['compute_dense_reward']
        ContinuousTaskWrapper.compute_dense_reward = new_function


