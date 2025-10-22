#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=aip-mtaylor3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=yingyue3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=metaworld_exp

module load StdEnv/2023
module load python/3.10
module load mujoco/3.0.1

source /home/yingyue/scratch/metavenv/bin/activate


python ./run_metaworld/sac.py --env_id button-press-v3  \
        --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 500 \
        --train_max_steps 1_000_000 --seed 12345 --exp_name zero-shot \
        --reward_path ./reward_code/button-press-v3/specific.py
