#!/usr/bin/env python3
"""
Example usage of the MetaWorld renderer
Demonstrates different ways to use the renderer with and without trained models.
"""

import os
import sys
from render_metaworld import MetaWorldRenderer


def example_without_model():
    """Example: Render environment with random actions (no trained model)."""
    print("=== Example 1: Random Actions (No Model) ===")
    
    # Create renderer without model
    renderer = MetaWorldRenderer(
        env_id="drawer-open-v2",
        model_path=None,  # No model
        max_episode_steps=200,
        render_mode="human"
    )
    
    # Render a single episode with random actions
    stats = renderer.render_episode(deterministic=False)
    print(f"Random episode stats: {stats}")
    
    renderer.close()


def example_with_model():
    """Example: Render environment with trained model."""
    print("=== Example 2: With Trained Model ===")
    
    # Check if model exists
    model_path = "./quick_models/sac_drawer-open-v2_final.zip"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please run quick_train.py first to train a model")
        return
    
    # Create renderer with model
    renderer = MetaWorldRenderer(
        env_id="drawer-open-v2",
        model_path=model_path,
        max_episode_steps=500,
        render_mode="human"
    )
    
    # Load the model
    renderer.load_model(model_path)
    
    # Render multiple episodes
    stats = renderer.render_multiple_episodes(num_episodes=3, deterministic=True)
    print(f"Trained model stats: {stats}")
    
    renderer.close()


def example_video_recording():
    """Example: Record video of episodes."""
    print("=== Example 3: Video Recording ===")
    
    # Create renderer
    renderer = MetaWorldRenderer(
        env_id="drawer-open-v2",
        model_path=None,  # Random actions
        max_episode_steps=200,
        render_mode="rgb_array"  # Required for video recording
    )
    
    # Setup video recording
    video_path = "drawer_open_random_episode.mp4"
    renderer.setup_video_recording(video_path, fps=30)
    
    # Render episode (will be recorded)
    stats = renderer.render_episode(deterministic=False)
    print(f"Video recorded: {video_path}")
    print(f"Episode stats: {stats}")
    
    renderer.close()


def example_different_environments():
    """Example: Try different MetaWorld environments."""
    print("=== Example 4: Different Environments ===")
    
    # List of environments to try
    environments = [
        "drawer-open-v2",
        "drawer-close-v2", 
        "button-press-v2",
        "door-open-v2",
        "reach-v2"
    ]
    
    for env_id in environments:
        print(f"\nTrying environment: {env_id}")
        
        try:
            renderer = MetaWorldRenderer(
                env_id=env_id,
                model_path=None,  # Random actions
                max_episode_steps=100,
                render_mode="human"
            )
            
            # Quick episode
            stats = renderer.render_episode(deterministic=False)
            print(f"  {env_id}: Reward={stats['episode_reward']:.2f}, Length={stats['episode_length']}")
            
            renderer.close()
            
        except Exception as e:
            print(f"  {env_id}: Error - {e}")


def main():
    """Run all examples."""
    print("MetaWorld Renderer Examples")
    print("=" * 50)
    
    # Example 1: Random actions
    try:
        example_without_model()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 2: With trained model (if available)
    try:
        example_with_model()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 3: Video recording
    try:
        example_video_recording()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 4: Different environments
    try:
        example_different_environments()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
