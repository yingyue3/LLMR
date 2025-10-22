import numpy as np # Importing numpy for array operations

def compute_dense_reward(self, action, obs) -> float:
    """
    Computes a dense reward for pressing a button in y coordination.

    Args:
        self: The environment instance (BaseEnv), providing access to robot, objects, and goal.
        action: The action taken by the agent.
        obs: The observation from the environment (not explicitly used in this reward, but available).

    Returns:
        float: The calculated dense reward.
    """
    reward = 0.0

    # --- Reward Constants ---
    # Coefficients for exponential rewards. Higher values mean steeper rewards/penalties.
    K_REACH_EE_OBJ = 10.0  # For end-effector approaching the button
    K_PRESS_OBJ_Y = 50.0   # For button's y-position approaching the goal y-position (primary task)
    K_ACTION_REG = 0.005 # For penalizing large actions (regularization)

    # Thresholds for activating specific reward phases or giving bonuses
    EE_CLOSE_ENOUGH_FOR_PRESS = 0.05 # [meters] End-effector must be this close to obj1 to fully activate pressing reward
    GOAL_TOLERANCE_Y = 0.01          # [meters] Tolerance for obj1.position[1] to be considered 'at goal_position[1]'

    # --- Retrieve current states ---
    ee_pos = self.robot.ee_position
    obj_pos = self.obj1.position
    goal_y = self.goal_position[1] # We only care about the y-coordinate for the goal state

    # --- 1. Reward for End-Effector reaching the button (Guiding the robot to the interaction point) ---
    # Calculate the Euclidean distance between the end-effector and the button's center.
    dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
    
    # Use an exponential reward: gets larger as distance decreases.
    # np.exp(-K * dist) provides a smooth, non-linear reward.
    reward_reach_ee_obj = np.exp(-K_REACH_EE_OBJ * dist_ee_obj)
    reward += reward_reach_ee_obj

    # --- 2. Reward for the button's Y position approaching the goal Y position (Primary task reward) ---
    # Calculate the absolute difference in y-coordinates between the button and its goal.
    dist_obj_goal_y = abs(obj_pos[1] - goal_y)

    # Calculate the base pressing reward using an exponential function.
    reward_press_obj_y = np.exp(-K_PRESS_OBJ_Y * dist_obj_goal_y)

    # Condition the pressing reward: It's more heavily weighted only when the end-effector is close enough to interact.
    # This prevents the agent from getting rewarded for the button moving without actual interaction.
    if dist_ee_obj < EE_CLOSE_ENOUGH_FOR_PRESS:
        # When EE is close, the pressing reward is a significant component.
        reward += reward_press_obj_y * 2.0 # Give it more weight
        
        # Add a substantial bonus if the button's Y position is within the goal tolerance.
        if dist_obj_goal_y < GOAL_TOLERANCE_Y:
            reward += 1000.0 # Large positive bonus for successful task completion
    else:
        # If EE is not close, provide a very small, attenuated pressing reward.
        # This can still guide initial exploration but doesn't dominate the 'reach' reward.
        reward += reward_press_obj_y * 0.1 # Small exploratory reward for pressing progress


    # --- 3. Action Regularization (Optional, but good for smooth and stable control) ---
    # Penalize the magnitude of the action to encourage smaller, smoother movements and prevent jerky behavior.
    # np.sum(np.square(action)) calculates the squared L2 norm of the action vector.
    reward_action_reg = -K_ACTION_REG * np.sum(np.square(action))
    reward += reward_action_reg

    return reward
