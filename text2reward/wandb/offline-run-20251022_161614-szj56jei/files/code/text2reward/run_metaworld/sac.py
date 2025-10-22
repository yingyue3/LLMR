# import gym, sys, os
import sys, os
import gymnasium as gym
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
# from rlkit.envs.wrappers import NormalizedBoxEnv


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs
    
    def compute_dense_reward(self, action, obs):
        assert (0)

    def step(self, action):
        ob, rew, done, info = super().step(action)
        if args.reward_path is not None:
            rew = self.compute_dense_reward(action, ob)  # TODO: uncomment this line
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        if info["success"]:
            done = True
        return ob, rew, done, info


def make_env(env_id, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        try:
            # Try gymnasium API first (for newer MetaWorld versions)
            env = gym.make(f"Meta-World/MT1", env_name=env_id)
        except:
            env_cls = ALL_V3_ENVIRONMENTS[env_id]
            env = env_cls()
            env._freeze_rand_vec = False
            env._set_task_called = True

            # env = NormalizedBoxEnv(env)

            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = ContinuousTaskWrapper(env, env.max_path_length)
                env = SuccessInfoWrapper(env)
        return env

    return _init


if __name__ == '__main__':
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--train_num', type=int, default=8)
    parser.add_argument('--eval_num', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=16_000)
    parser.add_argument('--max_episode_steps', type=int, default=500)
    parser.add_argument('--train_max_steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=12345)
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

    if args.env_id not in ALL_V3_ENVIRONMENTS.keys():
        print("Please specify a valid environment!")
        assert (0)

    # initialize wandb
    run = wandb.init(project=args.project_name, entity="code4reward",
                     config={"env": "{}".format(args.env_id)},
                     name=args.env_id[:-2] + args.exp_name, sync_tensorboard=True, save_code=True)
    
    # create a dir on wandb to store the codes, copy these to wandb
    if args.reward_path is not None:
        os.makedirs(f"{wandb.run.dir}/codes/{run.id}", exist_ok=True)
        os.system(f"cp -r {args.reward_path[:-11]} {wandb.run.dir}/codes/{run.id}")


    # set up eval environment
    eval_env = SubprocVecEnv([make_env(args.env_id, record_dir="logs/videos") for i in range(args.eval_num)])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.seed)
    eval_env.reset()

    # set up training environment
    env = SubprocVecEnv([make_env(args.env_id, max_episode_steps=args.max_episode_steps) for i in range(args.train_num)])
    env = VecMonitor(env)
    env.seed(args.seed)
    obs = env.reset()

    # set up callback
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", log_path="./logs/",
                                 eval_freq=args.eval_freq // args.train_num, deterministic=True, render=False,
                                 n_eval_episodes=10)
    set_random_seed(args.seed)

    print("check point 4")
    # set up sac algorithm
    policy_kwargs = dict(net_arch=[256, 256, 256])
    print("check point 1")
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, batch_size=512, gamma=0.99, target_update_interval=2,
                learning_rate=0.0003, tau=0.005, learning_starts=4000, ent_coef='auto_0.1', tensorboard_log="./logs")
    print("check point 2")
    model.learn(args.train_max_steps, callback=[eval_callback, WandbCallback(verbose=2)])
    print("check point 3")
    model.save("./logs/latest_model_" + args.env_id[:-2] + args.exp_name)
    print("check point 4")

    # print("training done, rendering...")
    # env.render_mode = "human"
    # env.render()
    print("training done")
    eval_env.close()
    env.close()
    wandb.finish()