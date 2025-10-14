import gym, sys, os
import gymnasium as gym_gymnasium
import metaworld, mujoco
import metaworld.envs.mujoco.env_dict as _env_dict
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
from rlkit.envs.wrappers import NormalizedBoxEnv

# LLMR imports
sys.path.append("../../LLMR")
from generation import ZeroShotGenerator
from MetaworldPrompt import METAWORLD_PROMPT
from post_process import RewardFunctionConverter


class LLMRContinuousTaskWrapper(gym.Wrapper):
    """Enhanced wrapper for LLMR-based reward functions with dynamic code generation."""
    
    def __init__(self, env, max_episode_steps: int, task_instruction: str = None, 
                 llm_model: str = "gpt-4", use_llmr: bool = True) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps
        self.use_llmr = use_llmr
        self.task_instruction = task_instruction
        self.llm_model = llm_model
        
        # LLMR components
        if self.use_llmr and self.task_instruction:
            self._setup_llmr_reward_function()
    
    def _setup_llmr_reward_function(self):
        """Setup LLMR-based reward function generation."""
        try:
            # Mapping dictionary for converting general terms to specific observations
            self.mapping_dicts = {
                "self.robot.ee_position": "obs[:3]",
                "self.robot.gripper_openness": "obs[3]", 
                "self.obj1.position": "obs[4:7]",
                "self.obj1.quaternion": "obs[7:11]",
                "self.obj2.position": "obs[11:14]",
                "self.obj2.quaternion": "obs[14:18]",
                "self.goal_position": "self.env._get_pos_goal()",
            }
            
            # Generate reward function using LLMR
            self.code_generator = ZeroShotGenerator(METAWORLD_PROMPT, self.llm_model)
            self.general_code, self.specific_code = self.code_generator.generate_code(
                self.task_instruction, self.mapping_dicts
            )
            
            # Create namespace for executing the generated reward function
            self.reward_namespace = {
                'np': np,
                'self': self,
                'obs': None,
                'action': None
            }
            
            # Execute the specific code to define the reward function
            exec(self.specific_code, self.reward_namespace)
            self.compute_dense_reward = self.reward_namespace['compute_dense_reward']
            
            print(f"LLMR reward function generated successfully for task: {self.task_instruction}")
            
        except Exception as e:
            print(f"Failed to setup LLMR reward function: {e}")
            print("Falling back to default reward function")
            self.use_llmr = False

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs
    
    def compute_dense_reward(self, action, obs):
        """Default reward function - can be overridden by LLMR generation."""
        # Simple distance-based reward as fallback
        if hasattr(self.env, '_get_pos_goal'):
            goal_pos = self.env._get_pos_goal()
            dist_to_goal = np.linalg.norm(obs[:3] - goal_pos)
            return -dist_to_goal
        return 0.0

    def step(self, action):
        ob, rew, done, info = super().step(action)
        
        # Use LLMR-generated reward function if available
        if self.use_llmr and hasattr(self, 'compute_dense_reward'):
            try:
                # Update namespace with current observation and action
                self.reward_namespace['obs'] = ob
                self.reward_namespace['action'] = action
                self.reward_namespace['self'] = self
                
                # Compute dense reward using LLMR-generated function
                rew = self.compute_dense_reward(action, ob)
            except Exception as e:
                print(f"Error in LLMR reward computation: {e}")
                # Fall back to original reward
                pass
        
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


class LLMRSuccessInfoWrapper(gym.Wrapper):
    """Wrapper that adds success information for LLMR tasks."""
    
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info.get("success", False)
        if info["is_success"]:
            done = True
        return ob, rew, done, info


def make_llmr_env(env_id, max_episode_steps: int = None, record_dir: str = None, 
                  task_instruction: str = None, llm_model: str = "gpt-4", 
                  use_llmr: bool = True):
    """Create LLMR-enhanced MetaWorld environment."""
    
    def _init() -> gym.Env:
        # Create environment using gymnasium API for newer MetaWorld versions
        try:
            # Try gymnasium API first (for newer MetaWorld versions)
            env = gym_gymnasium.make(f"Meta-World/MT1", env_name=env_id)
        except:
            # Fall back to old MetaWorld API
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_id]
            env = env_cls()
            env._freeze_rand_vec = False
            env._set_task_called = True

        # Apply normalization wrapper
        env = NormalizedBoxEnv(env)

        # Apply LLMR-enhanced task wrapper
        if max_episode_steps is not None:
            env = LLMRContinuousTaskWrapper(
                env, max_episode_steps, 
                task_instruction=task_instruction,
                llm_model=llm_model,
                use_llmr=use_llmr
            )
        
        # Apply success info wrapper for evaluation
        if record_dir is not None:
            env = LLMRSuccessInfoWrapper(env)
            
        return env

    return _init


# Task instruction mapping for LLMR
LLMR_TASK_INSTRUCTIONS = {
    "window-open-v2": "Push and open a sliding window by its handle.",
    "window-close-v2": "Push and close a sliding window by its handle.",
    "door-close-v2": "Close a door with a revolving joint by pushing door's handle.",
    "drawer-open-v2": "Open a drawer by its handle.",
    "drawer-close-v2": "Close a drawer by its handle.",
    "door-unlock-v2": "Unlock the door by rotating the lock counter-clockwise.",
    "sweep-into-v2": "Sweep a puck from the initial position into a hole.",
    "button-press-v2": "Press a button in y coordination.",
    "handle-press-v2": "Press a handle down.",
    "handle-press-side-v2": "Press a handle down sideways.",
    "pick-place-v2": "Pick up an object and place it at a target location.",
    "push-v2": "Push an object to a target location.",
    "reach-v2": "Reach a target position with the robot's end-effector.",
    "soccer-v2": "Kick a ball into a goal.",
    "sweep-v2": "Sweep an object to a target location.",
    "throw-v2": "Throw an object to a target location.",
    "window-open-v2": "Open a window by its handle.",
    "window-close-v2": "Close a window by its handle.",
}


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='LLMR-Enhanced MetaWorld SAC Training')

    parser.add_argument('--env_id', type=str, default=None, 
                       help='MetaWorld environment ID')
    parser.add_argument('--train_num', type=int, default=8,
                       help='Number of training environments')
    parser.add_argument('--eval_num', type=int, default=5,
                       help='Number of evaluation environments')
    parser.add_argument('--eval_freq', type=int, default=16_000,
                       help='Evaluation frequency')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                       help='Maximum episode steps')
    parser.add_argument('--train_max_steps', type=int, default=1_000_000,
                       help='Maximum training steps')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed')
    parser.add_argument('--project_name', type=str, default="llmr-metaworld",
                       help='Wandb project name')
    parser.add_argument('--exp_name', type=str, default="llmr-sac",
                       help='Experiment name')
    parser.add_argument('--llm_model', type=str, default="gpt-4",
                       help='LLM model for reward generation')
    parser.add_argument('--use_llmr', action='store_true', default=True,
                       help='Use LLMR for reward generation')
    parser.add_argument('--custom_instruction', type=str, default=None,
                       help='Custom task instruction for LLMR')

    args = parser.parse_args()

    # Validate environment ID
    if args.env_id not in LLMR_TASK_INSTRUCTIONS.keys():
        print("Please specify a valid environment!")
        print("Available environments:", list(LLMR_TASK_INSTRUCTIONS.keys()))
        assert False

    # Get task instruction
    task_instruction = args.custom_instruction or LLMR_TASK_INSTRUCTIONS[args.env_id]
    
    # Initialize wandb
    run = wandb.init(
        project=args.project_name, 
        entity="llmr-metaworld",
        config={
            "env": args.env_id,
            "task_instruction": task_instruction,
            "llm_model": args.llm_model,
            "use_llmr": args.use_llmr
        },
        name=f"{args.env_id}-{args.exp_name}",
        sync_tensorboard=True, 
        save_code=True
    )
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([
        make_llmr_env(
            args.env_id, 
            record_dir="logs/videos",
            task_instruction=task_instruction,
            llm_model=args.llm_model,
            use_llmr=args.use_llmr
        ) for i in range(args.eval_num)
    ])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.seed)
    eval_env.reset()

    # Create training environment
    env = SubprocVecEnv([
        make_llmr_env(
            args.env_id, 
            max_episode_steps=args.max_episode_steps,
            task_instruction=task_instruction,
            llm_model=args.llm_model,
            use_llmr=args.use_llmr
        ) for i in range(args.train_num)
    ])
    env = VecMonitor(env)
    env.seed(args.seed)
    obs = env.reset()

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./logs/", 
        log_path="./logs/",
        eval_freq=args.eval_freq // args.train_num, 
        deterministic=True, 
        render=False,
        n_eval_episodes=10
    )
    set_random_seed(args.seed)

    # Setup SAC algorithm
    policy_kwargs = dict(net_arch=[256, 256, 256])
    model = SAC(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        batch_size=512, 
        gamma=0.99, 
        target_update_interval=2,
        learning_rate=0.0003, 
        tau=0.005, 
        learning_starts=4000, 
        ent_coef='auto_0.1', 
        tensorboard_log="./logs"
    )
    
    # Train the model
    print(f"Starting LLMR-enhanced SAC training for {args.env_id}")
    print(f"Task instruction: {task_instruction}")
    print(f"Using LLMR: {args.use_llmr}")
    print(f"LLM Model: {args.llm_model}")
    
    model.learn(
        args.train_max_steps, 
        callback=[eval_callback, WandbCallback(verbose=2)]
    )
    
    # Save the model
    model_path = f"./logs/llmr_model_{args.env_id}_{args.exp_name}"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Final evaluation
    print("Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.finish()
