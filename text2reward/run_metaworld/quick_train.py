#!/usr/bin/env python3
"""
Quick training script for MetaWorld environments
Trains a SAC model for a short duration for testing purposes.
"""

import sys
import os
import gymnasium as gym
import numpy as np
import argparse
from pathlib import Path

# Add MetaWorld to path
sys.path.insert(0, '/home/yingyue/scratch/LLMR/Metaworld')
import metaworld, mujoco
from metaworld.env_dict import ALL_V3_ENVIRONMENTS

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback


class ContinuousTaskWrapper(gym.Wrapper):
    """Wrapper for continuous task execution with episode length control."""
    
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        if hasattr(super(), 'reset'):
            self.pre_obs = super().reset(**kwargs)
        else:
            self.pre_obs = super().reset()
        return self.pre_obs
    
    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


def make_env(env_id, max_episode_steps: int = None):
    """Create environment factory function."""
    def _init() -> gym.Env:
        try:
            # Try gymnasium API first (for newer MetaWorld versions)
            env = gym.make(f"Meta-World/MT1", env_name=env_id)
        except:
            # Fall back to direct MetaWorld API
            env_cls = ALL_V3_ENVIRONMENTS[env_id]
            env = env_cls()
            env._freeze_rand_vec = False
            env._set_task_called = True

        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Quick training for MetaWorld environments')
    parser.add_argument('--env_id', type=str, default='drawer-open-v2',
                       help='MetaWorld environment ID')
    parser.add_argument('--train_steps', type=int, default=50000,
                       help='Number of training steps')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./quick_models',
                       help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Validate environment
    if args.env_id not in ALL_V3_ENVIRONMENTS.keys():
        print(f"Error: Environment '{args.env_id}' not found!")
        print("Available environments:")
        for env_name in sorted(ALL_V3_ENVIRONMENTS.keys()):
            print(f"  - {env_name}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_random_seed(args.seed)
    
    print(f"Training SAC on {args.env_id} for {args.train_steps} steps")
    
    # Create training environment
    env = SubprocVecEnv([
        make_env(args.env_id, args.max_episode_steps) 
        for _ in range(args.num_envs)
    ])
    env = VecMonitor(env)
    env.seed(args.seed)
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([
        make_env(args.env_id, args.max_episode_steps) 
        for _ in range(2)
    ])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.seed)
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=args.output_dir,
        log_path=args.output_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Create SAC model
    model = SAC(
        "MlpPolicy", 
        env, 
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, 
        batch_size=256, 
        gamma=0.99, 
        learning_rate=0.0003, 
        tau=0.005, 
        learning_starts=1000,
        tensorboard_log=args.output_dir
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=args.train_steps,
        callback=eval_callback
    )
    
    # Save the final model
    model_path = os.path.join(args.output_dir, f"sac_{args.env_id}_final")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Quick evaluation
    print("Running evaluation...")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    print("Training completed!")
    print(f"You can now render the environment using:")
    print(f"python render_metaworld.py --env_id {args.env_id} --model_path {model_path}.zip")


if __name__ == "__main__":
    main()
