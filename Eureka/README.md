# Load required modules
module load StdEnv/2023
module load python/3.11
module load mujoco/3.0.1

module load opencv/4.10.0
module load gcc

# Isaac Sim Install
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

virtualenv $your_env_name
source $your_env_name/bin/activate