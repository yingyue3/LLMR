def compute_dense_reward(self, action, obs) -> float:
    # PART 1: Distance between the end of the robot arm (gripper) and the button (goal)
    gripper_to_goal = np.linalg.norm(obs[:3] - self.env._get_pos_goal())

    # Lower the distance, higher should be the reward
    reward_dist = -gripper_to_goal

    # PART 2: Regularization of the robot's action
    # We want the gripper to close when it's at the button to press it.
    # We give larger penalty as the gripper's openness deviates from fully closed position
    # assuming that the gripper needs to be fully closed to press the button properly
    reward_grip = -np.abs(obs[3] - (-1))

    # PART 3: Difference between current state of object and its goal state
    # As the task is to press a button, there will not be any object manipulation required.
    # The button itself can be the object of interest, in this case we might want to include its current state and expected state.
    # However, in some cases, the button can be just a fixed location without consideration of its state, hence this part could be omitted here.

    # Total Reward Calculation
    # Coefficients for these rewards can be tuned as per the importance of the each reward part. Here we keep them same.
    reward = reward_dist + reward_grip

    return reward