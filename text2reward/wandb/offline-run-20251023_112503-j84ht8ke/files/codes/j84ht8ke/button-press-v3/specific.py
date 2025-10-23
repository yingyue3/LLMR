import numpy as np # For np.linalg.norm and np.square

def compute_dense_reward(self, action, obs) -> float:
    """
    Computes a dense reward for the "Press a button in y coordination" task.

    Reward components:
    1.  **Alignment (X-Z)**: Encourages the end-effector (EE) to align its X and Z
        coordinates with the target goal position's X and Z.
    2.  **Pressing (Y)**: Encourages the EE to move towards the target goal position's
        Y coordinate, emphasizing the primary pressing motion along the Y-axis.
    3.  **Gripper State**: Rewards the gripper being in a closed state,
        which is suitable for pressing rather than grasping.
    4.  **Action Regularization**: Penalizes the magnitude of the robot's action
        to encourage smoother and more energy-efficient movements.
    """
    reward = 0.0

    # Define weights for different reward components.
    # These weights can be tuned based on task difficulty and desired behavior.
    # Higher weights mean a stronger influence on the total reward.
    W_ALIGN_XZ = 10.0      # Weight for alignment in the XZ plane
    W_PRESS_Y = 20.0       # Weight for moving along the Y-axis (pressing)
    W_GRIPPER_CLOSED = 1.0 # Weight for maintaining a closed gripper state
    W_ACTION_REG = 0.01    # Weight for action regularization (penalty)

    # Extract relevant positions from the environment state
    ee_pos = obs[:3]
    goal_pos = self.env._get_pos_goal() # This is the target 3D position for the EE to press the button

    # --- 1. Reward for alignment in X-Z plane ---
    # Calculate the 2D Euclidean distance in the XZ plane between the EE and the goal.
    # This encourages the robot to position its end-effector horizontally above/in front of the button.
    dist_xz_ee_goal = np.linalg.norm(ee_pos[[0, 2]] - goal_pos[[0, 2]])
    reward_xz_alignment = -W_ALIGN_XZ * dist_xz_ee_goal
    reward += reward_xz_alignment

    # --- 2. Reward for pressing along Y-axis ---
    # Calculate the absolute difference in the Y coordinate between the EE and the goal.
    # This directly rewards the "pressing" motion along the specified Y-axis.
    dist_y_ee_goal = abs(ee_pos[1] - goal_pos[1])
    reward_y_press = -W_PRESS_Y * dist_y_ee_goal
    reward += reward_y_press

    # --- 3. Reward for gripper state ---
    # For pressing a button, the gripper should typically be closed to make effective contact.
    # `obs[3]` ranges from -1 (fully closed) to 1 (fully open).
    # We penalize deviations from the fully closed state (-1.0).
    gripper_target_openness = -1.0
    reward_gripper_closed = -W_GRIPPER_CLOSED * abs(obs[3] - gripper_target_openness)
    reward += reward_gripper_closed

    # --- 4. Action regularization ---
    # Penalize the magnitude of the action to encourage smooth, less jerky movements.
    # `action` is a Box(-1, 1, (4,), float32) representing EE movements and gripper control.
    reward_action_reg = -W_ACTION_REG * np.sum(np.square(action))
    reward += reward_action_reg

    return reward