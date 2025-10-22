#!/usr/bin/env python3
"""
MetaWorld Environment Renderer with Trained Policy
Renders MetaWorld environments using trained SAC policies with various display options.
"""

import sys
import os
import gymnasium as gym
import numpy as np
import argparse
import time
import cv2
from pathlib import Path
from typing import Optional, Dict, Any

# Add MetaWorld to path
sys.path.insert(0, '/home/yingyue/scratch/LLMR/Metaworld')
import metaworld, mujoco
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
    ML10_V3,
    ML45_V3,
    MT10_V3,
    MT50_V3,
    EnvDict,
    TrainTestEnvDict,
)

# Import SAC from stable-baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy


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


class SuccessInfoWrapper(gym.Wrapper):
    """Wrapper that adds success information."""
    
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info.get("success", False)
        if info["is_success"]:
            done = True
        return ob, rew, done, info


class MetaWorldRenderer:
    """Main class for rendering MetaWorld environments with trained policies."""
    
    def __init__(self, env_id: str, model_path: Optional[str] = None, 
                 max_episode_steps: int = 500, render_mode: str = "human"):
        self.env_id = env_id
        self.model_path = model_path
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.model = None
        self.env = None
        
        # Video recording settings
        self.video_writer = None
        self.video_frames = []
        self.record_video = False
        self.video_path = None
        
    def setup_environment(self):
        """Setup the MetaWorld environment."""
        print(f"Setting up environment: {self.env_id}")
        
        # Validate environment ID
        if self.env_id not in ALL_V3_ENVIRONMENTS.keys():
            print(f"Error: Environment '{self.env_id}' not found!")
            print("Available environments:")
            for env_name in sorted(ALL_V3_ENVIRONMENTS.keys()):
                print(f"  - {env_name}")
            raise ValueError(f"Invalid environment ID: {self.env_id}")
        
        try:
            # Try gymnasium API first (for newer MetaWorld versions)
            env = gym.make(f"Meta-World/MT1", env_name=self.env_id, render_mode=self.render_mode)
            print("Using gymnasium API")
        except Exception as e:
            print(f"Gymnasium API failed: {e}")
            print("Falling back to direct MetaWorld API")
            # Fall back to direct MetaWorld API
            env_cls = ALL_V3_ENVIRONMENTS[self.env_id]
            env = env_cls()
            env._freeze_rand_vec = False
            env._set_task_called = True
            
            # Enable rendering if supported
            if hasattr(env, 'render_mode'):
                env.render_mode = self.render_mode
        
        # Apply wrappers
        env = ContinuousTaskWrapper(env, self.max_episode_steps)
        env = SuccessInfoWrapper(env)
        
        self.env = env
        print(f"Environment setup complete: {env}")
        return env
    
    def load_model(self, model_path: str):
        """Load a trained SAC model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Load the model
            self.model = SAC.load(model_path, verbose=1)
            print("Model loaded successfully!")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_video_recording(self, video_path: str, fps: int = 30):
        """Setup video recording."""
        self.video_path = video_path
        self.record_video = True
        self.video_frames = []
        print(f"Video recording enabled: {video_path} (FPS: {fps})")
    
    def render_episode(self, deterministic: bool = True, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Render a single episode with the trained policy."""
        if self.env is None:
            self.setup_environment()
        
        if self.model is None and self.model_path:
            self.load_model(self.model_path)
        
        # Reset environment
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gymnasium API
        
        episode_reward = 0.0
        episode_length = 0
        success = False
        done = False
        
        max_steps = max_steps or self.max_episode_steps
        
        print(f"Starting episode (max steps: {max_steps})")
        
        while not done and episode_length < max_steps:
            # Get action from policy
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=deterministic)
            else:
                # Random action if no model
                action = self.env.action_space.sample()
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle gymnasium API
            
            episode_reward += reward
            episode_length += 1
            
            # Check for success
            if info.get("is_success", False):
                success = True
                print(f"Success achieved at step {episode_length}!")
            
            # Render frame
            if self.render_mode == "human":
                self.env.render()
                time.sleep(0.01)  # Small delay for visualization
            elif self.render_mode == "rgb_array":
                frame = self.env.render()
                if frame is not None:
                    if self.record_video:
                        self.video_frames.append(frame)
                    else:
                        # Display frame using OpenCV
                        cv2.imshow('MetaWorld', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Quit key pressed!")
                            done = True
            
            # Print progress
            if episode_length % 50 == 0:
                print(f"Step {episode_length}, Reward: {episode_reward:.2f}")
        
        # Save video if recording
        if self.record_video and self.video_frames:
            self._save_video()
        
        episode_stats = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": success,
            "final_info": info
        }
        
        print(f"Episode completed!")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Success: {success}")
        
        return episode_stats
    
    def _save_video(self):
        """Save recorded video frames to file."""
        if not self.video_frames:
            return
        
        print(f"Saving video to: {self.video_path}")
        
        # Get video dimensions from first frame
        height, width, channels = self.video_frames[0].shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (width, height))
        
        # Write frames
        for frame in self.video_frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved successfully: {self.video_path}")
    
    def render_multiple_episodes(self, num_episodes: int = 5, deterministic: bool = True) -> Dict[str, Any]:
        """Render multiple episodes and collect statistics."""
        print(f"Rendering {num_episodes} episodes...")
        
        all_rewards = []
        all_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment for new episode
            self.env.reset()
            
            episode_stats = self.render_episode(deterministic=deterministic)
            
            all_rewards.append(episode_stats["episode_reward"])
            all_lengths.append(episode_stats["episode_length"])
            if episode_stats["success"]:
                success_count += 1
        
        # Calculate statistics
        stats = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_length": np.mean(all_lengths),
            "std_length": np.std(all_lengths),
            "success_rate": success_count / num_episodes,
            "success_count": success_count,
            "all_rewards": all_rewards,
            "all_lengths": all_lengths
        }
        
        print(f"\n--- Episode Statistics ---")
        print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
        print(f"Success rate: {stats['success_rate']:.2%} ({success_count}/{num_episodes})")
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()


def main():
    """Main function for the renderer."""
    parser = argparse.ArgumentParser(description='Render MetaWorld environments with trained policies')
    
    # Environment arguments
    parser.add_argument('--env_id', type=str, required=True,
                       help='MetaWorld environment ID (e.g., drawer-open-v2)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file (.zip)')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                       help='Maximum steps per episode')
    
    # Rendering arguments
    parser.add_argument('--render_mode', type=str, default='human', 
                       choices=['human', 'rgb_array'],
                       help='Rendering mode')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='Number of episodes to render')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    parser.add_argument('--random', action='store_true',
                       help='Use random actions instead of trained policy')
    
    # Video recording arguments
    parser.add_argument('--record_video', action='store_true',
                       help='Record video of episodes')
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to save video file')
    parser.add_argument('--video_fps', type=int, default=30,
                       help='Video frame rate')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create renderer
    renderer = MetaWorldRenderer(
        env_id=args.env_id,
        model_path=args.model_path,
        max_episode_steps=args.max_episode_steps,
        render_mode=args.render_mode
    )
    
    # Setup video recording if requested
    if args.record_video:
        video_path = args.video_path or f"{args.env_id}_episode.mp4"
        renderer.setup_video_recording(video_path, args.video_fps)
    
    try:
        # Render episodes
        if args.num_episodes == 1:
            renderer.render_episode(deterministic=args.deterministic)
        else:
            renderer.render_multiple_episodes(
                num_episodes=args.num_episodes,
                deterministic=args.deterministic
            )
    
    except KeyboardInterrupt:
        print("\nRendering interrupted by user")
    except Exception as e:
        print(f"Error during rendering: {e}")
        raise
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
