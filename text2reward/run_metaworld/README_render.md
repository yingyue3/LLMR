# MetaWorld Environment Renderer

This directory contains tools for rendering MetaWorld environments with trained policies.

## Files

- `render_metaworld.py` - Main rendering script with comprehensive functionality
- `quick_train.py` - Quick training script for testing purposes
- `example_usage.py` - Example usage demonstrations
- `sac.py` - Original SAC training script
- `llmr_sac.py` - LLMR-enhanced SAC training script
- `llmr_interactive.py` - Interactive LLMR system

## Quick Start

### 1. Train a Quick Model (Optional)

If you don't have a trained model, you can quickly train one:

```bash
python quick_train.py --env_id drawer-open-v2 --train_steps 50000
```

This will create a model in `./quick_models/sac_drawer-open-v2_final.zip`

### 2. Render with Trained Model

```bash
# Render with trained model
python render_metaworld.py --env_id drawer-open-v2 --model_path ./quick_models/sac_drawer-open-v2_final.zip

# Render with random actions (no model)
python render_metaworld.py --env_id drawer-open-v2

# Render multiple episodes
python render_metaworld.py --env_id drawer-open-v2 --model_path ./quick_models/sac_drawer-open-v2_final.zip --num_episodes 5

# Record video
python render_metaworld.py --env_id drawer-open-v2 --model_path ./quick_models/sac_drawer-open-v2_final.zip --record_video --video_path my_episode.mp4
```

### 3. Run Examples

```bash
python example_usage.py
```

## Available Environments

The renderer supports all MetaWorld v3 environments. Some popular ones include:

- `drawer-open-v2` - Open a drawer by its handle
- `drawer-close-v2` - Close a drawer by its handle  
- `button-press-v2` - Press a button
- `door-open-v2` - Open a door
- `door-close-v2` - Close a door
- `reach-v2` - Reach a target position
- `push-v2` - Push an object to target
- `pick-place-v2` - Pick up and place an object
- `sweep-v2` - Sweep an object to target
- `throw-v2` - Throw an object to target

## Command Line Options

### Basic Options
- `--env_id` - MetaWorld environment ID (required)
- `--model_path` - Path to trained model file (.zip)
- `--max_episode_steps` - Maximum steps per episode (default: 500)
- `--num_episodes` - Number of episodes to render (default: 1)

### Rendering Options
- `--render_mode` - Rendering mode: 'human' or 'rgb_array' (default: 'human')
- `--deterministic` - Use deterministic policy (default: True)
- `--random` - Use random actions instead of trained policy

### Video Recording Options
- `--record_video` - Enable video recording
- `--video_path` - Path to save video file
- `--video_fps` - Video frame rate (default: 30)

### Other Options
- `--seed` - Random seed (default: 42)

## Examples

### Basic Rendering
```bash
# Render with random actions
python render_metaworld.py --env_id drawer-open-v2

# Render with trained model
python render_metaworld.py --env_id drawer-open-v2 --model_path my_model.zip
```

### Video Recording
```bash
# Record video of trained policy
python render_metaworld.py --env_id drawer-open-v2 --model_path my_model.zip --record_video --video_path trained_policy.mp4

# Record video of random actions
python render_metaworld.py --env_id drawer-open-v2 --record_video --video_path random_actions.mp4
```

### Multiple Episodes
```bash
# Render 5 episodes and show statistics
python render_metaworld.py --env_id drawer-open-v2 --model_path my_model.zip --num_episodes 5
```

### Different Environments
```bash
# Try different environments
python render_metaworld.py --env_id button-press-v2
python render_metaworld.py --env_id door-open-v2
python render_metaworld.py --env_id reach-v2
```

## Programmatic Usage

You can also use the renderer programmatically:

```python
from render_metaworld import MetaWorldRenderer

# Create renderer
renderer = MetaWorldRenderer(
    env_id="drawer-open-v2",
    model_path="my_model.zip",
    render_mode="human"
)

# Load model
renderer.load_model("my_model.zip")

# Render episode
stats = renderer.render_episode()

# Render multiple episodes
stats = renderer.render_multiple_episodes(num_episodes=5)

# Setup video recording
renderer.setup_video_recording("output.mp4", fps=30)
stats = renderer.render_episode()  # This will be recorded

# Clean up
renderer.close()
```

## Troubleshooting

### Common Issues

1. **Environment not found**: Make sure the environment ID is valid. Check available environments with:
   ```python
   from metaworld.env_dict import ALL_V3_ENVIRONMENTS
   print(list(ALL_V3_ENVIRONMENTS.keys()))
   ```

2. **Model loading error**: Ensure the model file exists and is a valid SAC model:
   ```bash
   ls -la your_model.zip
   ```

3. **Rendering issues**: Try different render modes:
   - `--render_mode human` for real-time display
   - `--render_mode rgb_array` for video recording

4. **Import errors**: Make sure MetaWorld is properly installed and the path is correct in the scripts.

### Dependencies

- gymnasium
- metaworld
- stable-baselines3
- opencv-python (for video recording)
- numpy
- mujoco

## Notes

- The renderer supports both gymnasium API (newer MetaWorld versions) and direct MetaWorld API
- Video recording requires `render_mode="rgb_array"`
- For headless rendering, use `render_mode="rgb_array"` without display
- The renderer automatically handles environment wrappers and episode length control
