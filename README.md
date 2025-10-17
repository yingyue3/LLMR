# Load required modules
module load StdEnv/2023
module load python/3.10
module load mujoco/3.0.1

# Check Version Requirements:
module spider mujoco/3.0.1
module avail mujoco

# For GPU support (if needed):
module load cuda/11.8 

# Create and activate virtual environment
virtualenv $your_env_name
source $your_env_name/bin/activate


IMPORTANT: Use the modern 'mujoco' package, not 'mujoco_py'
The old mujoco_py package has compatibility issues with Python 3.11+ and MuJoCo 3.0.1

Correct usage:
import mujoco  # ✅ This works with MuJoCo 3.0.1

Avoid:
import mujoco_py  # ❌ This causes compilation errors


# install wandb
pip install --no-index wandb
wandb offline