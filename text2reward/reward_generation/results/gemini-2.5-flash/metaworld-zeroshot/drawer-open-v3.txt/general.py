import numpy as np

def compute_dense_reward(self, action, obs) -> float:
    # Initialize the total reward for the current step
    reward = 0.0

    # --- 1. Reward for Reaching the Handle (self.obj2) ---
    # Encourage the robot's end-effector to get close to the handle.
    # We use a negative exponential of the distance, so reward is higher when closer.
    ee_to_handle_dist = np.linalg.norm(self.robot.ee_position - self.obj2.position)

    # Scale and exponential factor for the reach reward.
    # A higher scale makes this component more impactful.
    # A higher exponent makes the reward drop off faster with distance.
    reach_reward_scale = 100.0
    reach_exponential_factor = 10.0 # Controls how quickly reward decays with distance
    reach_reward = reach_reward_scale * np.exp(-reach_exponential_factor * ee_to_handle_dist)
    reward += reach_reward

    # --- 2. Reward for Gripper Engagement (Grasping) ---
    # Once the end-effector is close enough to the handle,
    # encourage the gripper to close to a desired state for grasping.
    # Assuming `gripper_openness` ranges from -1 (fully closed) to 1 (fully open).
    # For grasping a handle, we typically want a somewhat closed state, e.g., around -0.5.
    
    gripper_engagement_threshold = 0.05 # Distance threshold for EE to be near handle
    desired_gripper_openness = -0.5     # Target gripper state for grasping

    if ee_to_handle_dist < gripper_engagement_threshold:
        # If the EE is close enough, reward for having the desired gripper openness.
        # We use a negative exponential of the absolute difference to reward closeness to the desired value.
        gripper_reward_scale = 50.0
        gripper_exponential_factor = 20.0 # Controls how quickly reward decays from desired openness
        gripper_reward = gripper_reward_scale * np.exp(-gripper_exponential_factor * np.abs(self.robot.gripper_openness - desired_gripper_openness))
        reward += gripper_reward
    else:
        # Optionally, penalize being closed too early if far from handle
        # This prevents the robot from closing its gripper unnecessarily.
        # However, for dense reward, it's often better to just not reward if not in range.
        pass # No gripper reward if not close to handle

    # --- 3. Reward for Drawer Opening Progress (self.obj1 moving towards self.goal_position) ---
    # This is the primary objective: moving the drawer to its open state.
    # Reward is higher as the drawer (obj1) gets closer to its `goal_position`.
    drawer_to_goal_dist = np.linalg.norm(self.obj1.position - self.goal_position)

    # Scale and exponential factor for the drawer progress reward.
    # This should typically be the highest scaled reward as it's the main task goal.
    drawer_progress_reward_scale = 500.0
    drawer_exponential_factor = 5.0 # Controls how quickly reward decays as drawer moves away from goal
    drawer_progress_reward = drawer_progress_reward_scale * np.exp(-drawer_exponential_factor * drawer_to_goal_dist)
    reward += drawer_progress_reward

    # --- 4. Action Regularization ---
    # Penalize large, jerky actions to encourage smoother movements and energy efficiency.
    # The action space is Box(-1, 1, (4,)), so squaring and summing is appropriate.
    action_regularization_scale = -0.01 # Negative scale to apply penalty
    action_regularization_penalty = action_regularization_scale * np.sum(np.square(action))
    reward += action_regularization_penalty

    return reward