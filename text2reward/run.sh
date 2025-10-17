# $ENV_ID = button-press-v2 
python ./run_metaworld/sac.py --env_id button-press-v3  \
        --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 500 \
        --train_max_steps 1_000_000 --seed 12345 --exp_name zero-shot \
        --reward_path ./reward_code/button-press-v3/specific.py
