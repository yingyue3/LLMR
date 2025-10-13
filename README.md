module load StdEnv/2023
module load python/3.10
module load mujoco/3.0.1


Check Version Requirement:
module spider mujoco/3.0.1
module avail mujoco

module load cuda/11.8 

virtualenv $your_env_name

source $your_env_name/bin/activate