#!/usr/bin/env python3.10

import sys
import os
sys.path.insert(0, '/home/yingyue/scratch/LLMR/Metaworld')

import gymnasium as gym
import metaworld

seed = 42 # for reproducibility

env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed) # MT1 with the reach environment

obs, info = env.reset()

a = env.action_space.sample() # randomly sample an action
obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action