"""
Interactive LLMR/MetaWorld System
Provides real-time interaction with MetaWorld environments using natural language commands.
"""

import gym
import gymnasium as gym_gymnasium
import metaworld
import numpy as np
import argparse
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Import LLMR components
sys.path.append("../../LLMR")
from llmr_generation import EnhancedLLMRGenerator
from generation import ZeroShotGenerator
from MetaworldPrompt import METAWORLD_PROMPT


class LLMRInteractiveEnvironment:
    """Interactive environment that responds to natural language commands."""
    
    def __init__(self, env_id: str, model_name: str = "gpt-4", render: bool = True):
        self.env_id = env_id
        self.model_name = model_name
        self.render = render
        
        # Initialize environment
        self._setup_environment()
        
        # Initialize LLMR generator
        self.llmr_generator = EnhancedLLMRGenerator(model_name=model_name)
        
        # Current state
        self.current_obs = None
        self.current_reward_function = None
        self.task_history = []
        self.performance_metrics = {}
        
        # Command queue for real-time interaction
        self.command_queue = queue.Queue()
        self.is_running = False
        
    def _setup_environment(self):
        """Setup the MetaWorld environment."""
        try:
            # Try gymnasium API first (for newer MetaWorld versions)
            self.env = gym_gymnasium.make(f"Meta-World/MT1", env_name=self.env_id)
            self.use_gymnasium = True
        except:
            # Fall back to old MetaWorld API
            import metaworld.envs.mujoco.env_dict as _env_dict
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[self.env_id]
            self.env = env_cls()
            self.env._freeze_rand_vec = False
            self.env._set_task_called = True
            self.use_gymnasium = False
        
        # Apply normalization wrapper
        from rlkit.envs.wrappers import NormalizedBoxEnv
        self.env = NormalizedBoxEnv(self.env)
        
        print(f"Environment {self.env_id} initialized successfully")
    
    def reset_environment(self, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.env.seed(seed)
        
        if self.use_gymnasium:
            self.current_obs, _ = self.env.reset()
        else:
            self.current_obs = self.env.reset()
        
        print("Environment reset to initial state")
        return self.current_obs
    
    def process_natural_language_command(self, command: str) -> Dict:
        """Process a natural language command and return response."""
        
        # Parse command type
        command_type = self._classify_command(command)
        
        response = {
            "command_type": command_type,
            "original_command": command,
            "status": "processing",
            "result": None,
            "message": ""
        }
        
        try:
            if command_type == "task_instruction":
                result = self._handle_task_instruction(command)
                response["result"] = result
                response["message"] = f"Task instruction processed: {command}"
                
            elif command_type == "reward_modification":
                result = self._handle_reward_modification(command)
                response["result"] = result
                response["message"] = f"Reward function modified: {command}"
                
            elif command_type == "environment_query":
                result = self._handle_environment_query(command)
                response["result"] = result
                response["message"] = f"Environment query answered: {command}"
                
            elif command_type == "action_command":
                result = self._handle_action_command(command)
                response["result"] = result
                response["message"] = f"Action executed: {command}"
                
            elif command_type == "performance_query":
                result = self._handle_performance_query(command)
                response["result"] = result
                response["message"] = f"Performance metrics retrieved: {command}"
                
            else:
                response["status"] = "error"
                response["message"] = f"Unknown command type: {command_type}"
                
        except Exception as e:
            response["status"] = "error"
            response["message"] = f"Error processing command: {str(e)}"
        
        response["status"] = "completed"
        return response
    
    def _classify_command(self, command: str) -> str:
        """Classify the type of natural language command."""
        command_lower = command.lower()
        
        # Task instruction keywords
        if any(word in command_lower for word in ["do", "perform", "complete", "achieve", "accomplish", "task"]):
            return "task_instruction"
        
        # Reward modification keywords
        elif any(word in command_lower for word in ["reward", "penalty", "bonus", "incentive", "modify", "change"]):
            return "reward_modification"
        
        # Environment query keywords
        elif any(word in command_lower for word in ["what", "where", "how", "show", "describe", "explain"]):
            return "environment_query"
        
        # Action command keywords
        elif any(word in command_lower for word in ["move", "grasp", "push", "pull", "press", "rotate"]):
            return "action_command"
        
        # Performance query keywords
        elif any(word in command_lower for word in ["performance", "success", "progress", "metrics", "stats"]):
            return "performance_query"
        
        else:
            return "unknown"
    
    def _handle_task_instruction(self, command: str) -> Dict:
        """Handle task instruction commands."""
        # Generate new reward function based on instruction
        general_code, specific_code = self.llmr_generator.generate_interactive_reward_function(
            self.env_id, command
        )
        
        # Update current reward function
        self.current_reward_function = specific_code
        
        # Add to task history
        self.task_history.append({
            "timestamp": time.time(),
            "type": "task_instruction",
            "command": command,
            "reward_function": specific_code
        })
        
        return {
            "reward_function_generated": True,
            "general_code": general_code,
            "specific_code": specific_code
        }
    
    def _handle_reward_modification(self, command: str) -> Dict:
        """Handle reward modification commands."""
        if self.current_reward_function is None:
            return {"error": "No current reward function to modify"}
        
        # Generate modified reward function
        general_code, specific_code = self.llmr_generator.generate_interactive_reward_function(
            self.env_id, f"Modify the current reward function: {command}"
        )
        
        # Update current reward function
        self.current_reward_function = specific_code
        
        # Add to task history
        self.task_history.append({
            "timestamp": time.time(),
            "type": "reward_modification",
            "command": command,
            "reward_function": specific_code
        })
        
        return {
            "reward_function_modified": True,
            "new_code": specific_code
        }
    
    def _handle_environment_query(self, command: str) -> Dict:
        """Handle environment query commands."""
        if self.current_obs is None:
            return {"error": "Environment not initialized"}
        
        # Analyze current observation
        obs_analysis = self._analyze_observation(self.current_obs)
        
        # Generate response based on query
        if "position" in command.lower():
            return {
                "robot_position": obs_analysis["robot_position"],
                "object_positions": obs_analysis["object_positions"],
                "goal_position": obs_analysis["goal_position"]
            }
        elif "state" in command.lower():
            return {
                "observation_analysis": obs_analysis,
                "environment_info": self._get_environment_info()
            }
        else:
            return {
                "observation_analysis": obs_analysis,
                "environment_info": self._get_environment_info()
            }
    
    def _handle_action_command(self, command: str) -> Dict:
        """Handle direct action commands."""
        # Parse action from command
        action = self._parse_action_from_command(command)
        
        if action is None:
            return {"error": "Could not parse action from command"}
        
        # Execute action
        if self.use_gymnasium:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, info = self.env.step(action)
        
        # Update current observation
        self.current_obs = obs
        
        # Compute custom reward if available
        if self.current_reward_function:
            try:
                custom_reward = self._compute_custom_reward(action, obs)
                reward = custom_reward
            except Exception as e:
                print(f"Error computing custom reward: {e}")
        
        # Update performance metrics
        self._update_performance_metrics(reward, done, info)
        
        return {
            "action_executed": True,
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    
    def _handle_performance_query(self, command: str) -> Dict:
        """Handle performance query commands."""
        return {
            "performance_metrics": self.performance_metrics,
            "task_history": self.task_history[-10:],  # Last 10 tasks
            "current_reward_function": self.current_reward_function is not None
        }
    
    def _analyze_observation(self, obs: np.ndarray) -> Dict:
        """Analyze current observation and extract meaningful information."""
        return {
            "robot_position": obs[:3].tolist(),
            "gripper_openness": float(obs[3]),
            "object_positions": {
                "obj1": obs[4:7].tolist() if len(obs) > 7 else None,
                "obj2": obs[11:14].tolist() if len(obs) > 14 else None
            },
            "goal_position": self._get_goal_position(),
            "observation_shape": obs.shape,
            "observation_range": [float(obs.min()), float(obs.max())]
        }
    
    def _get_goal_position(self) -> List[float]:
        """Get current goal position."""
        try:
            if hasattr(self.env, '_get_pos_goal'):
                return self.env._get_pos_goal().tolist()
            else:
                return [0.0, 0.0, 0.0]
        except:
            return [0.0, 0.0, 0.0]
    
    def _get_environment_info(self) -> Dict:
        """Get general environment information."""
        return {
            "env_id": self.env_id,
            "action_space": str(self.env.action_space),
            "observation_space": str(self.env.observation_space),
            "use_gymnasium": self.use_gymnasium
        }
    
    def _parse_action_from_command(self, command: str) -> Optional[np.ndarray]:
        """Parse action from natural language command."""
        # This is a simplified parser - in practice, you'd use more sophisticated NLP
        command_lower = command.lower()
        
        # Default action (no movement)
        action = np.zeros(4)
        
        # Parse movement commands
        if "move up" in command_lower or "lift" in command_lower:
            action[2] = 1.0  # Move up in z
        elif "move down" in command_lower or "lower" in command_lower:
            action[2] = -1.0  # Move down in z
        elif "move left" in command_lower:
            action[0] = -1.0  # Move left in x
        elif "move right" in command_lower:
            action[0] = 1.0  # Move right in x
        elif "move forward" in command_lower:
            action[1] = 1.0  # Move forward in y
        elif "move backward" in command_lower:
            action[1] = -1.0  # Move backward in y
        
        # Parse gripper commands
        if "grasp" in command_lower or "close" in command_lower:
            action[3] = -1.0  # Close gripper
        elif "release" in command_lower or "open" in command_lower:
            action[3] = 1.0  # Open gripper
        
        return action
    
    def _compute_custom_reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        """Compute custom reward using current reward function."""
        if self.current_reward_function is None:
            return 0.0
        
        try:
            # Create namespace for reward computation
            namespace = {
                'np': np,
                'obs': obs,
                'action': action,
                'self': self
            }
            
            # Execute reward function
            exec(self.current_reward_function, namespace)
            return namespace['compute_dense_reward'](action, obs)
        except Exception as e:
            print(f"Error computing custom reward: {e}")
            return 0.0
    
    def _update_performance_metrics(self, reward: float, done: bool, info: Dict):
        """Update performance metrics."""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                "total_reward": 0.0,
                "episode_count": 0,
                "success_count": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0
            }
        
        self.performance_metrics["total_reward"] += reward
        self.performance_metrics["avg_reward"] = (
            self.performance_metrics["total_reward"] / 
            max(1, self.performance_metrics["episode_count"])
        )
        
        if done:
            self.performance_metrics["episode_count"] += 1
            if info.get("success", False):
                self.performance_metrics["success_count"] += 1
            self.performance_metrics["success_rate"] = (
                self.performance_metrics["success_count"] / 
                self.performance_metrics["episode_count"]
            )
    
    def start_interactive_session(self):
        """Start an interactive session."""
        print(f"Starting interactive LLMR session for {self.env_id}")
        print("Type 'help' for available commands, 'quit' to exit")
        
        # Reset environment
        self.reset_environment()
        
        while True:
            try:
                command = input("\nLLMR> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'help':
                    self._print_help()
                    continue
                elif command.lower() == 'reset':
                    self.reset_environment()
                    continue
                elif command.lower() == 'status':
                    self._print_status()
                    continue
                
                # Process command
                response = self.process_natural_language_command(command)
                
                # Print response
                print(f"\nResponse: {response['message']}")
                if response['result']:
                    print(f"Result: {json.dumps(response['result'], indent=2)}")
                
            except KeyboardInterrupt:
                print("\nExiting interactive session...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Interactive session ended.")
    
    def _print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
- Task Instructions: "Open the drawer", "Press the button", "Sweep the puck into the hole"
- Reward Modifications: "Increase the reward for getting closer to the goal", "Add penalty for dropping objects"
- Environment Queries: "What is the current position?", "Show me the object locations"
- Action Commands: "Move up", "Grasp the object", "Release the gripper"
- Performance Queries: "How am I performing?", "Show success rate"
- System Commands: "reset", "status", "help", "quit"
        """
        print(help_text)
    
    def _print_status(self):
        """Print current status."""
        if self.current_obs is not None:
            obs_analysis = self._analyze_observation(self.current_obs)
            print(f"Robot Position: {obs_analysis['robot_position']}")
            print(f"Gripper Openness: {obs_analysis['gripper_openness']}")
            print(f"Goal Position: {obs_analysis['goal_position']}")
        
        print(f"Performance Metrics: {self.performance_metrics}")
        print(f"Task History: {len(self.task_history)} tasks recorded")


def main():
    """Main function for interactive LLMR system."""
    parser = argparse.ArgumentParser(description='Interactive LLMR/MetaWorld System')
    parser.add_argument('--env_id', type=str, default='drawer-open-v2',
                       help='MetaWorld environment ID')
    parser.add_argument('--model_name', type=str, default='gpt-4',
                       help='LLM model name')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Enable rendering')
    
    args = parser.parse_args()
    
    # Create interactive environment
    interactive_env = LLMRInteractiveEnvironment(
        env_id=args.env_id,
        model_name=args.model_name,
        render=args.render
    )
    
    # Start interactive session
    interactive_env.start_interactive_session()


if __name__ == "__main__":
    main()
