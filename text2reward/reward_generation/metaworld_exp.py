import os, argparse

from reward_generation import ZeroShotGenerator
from MetaworldPrompt import METAWORLD_PROMPT

instruction_mapping = {
    "window-open-v3": "Push and open a sliding window by its handle.",
    "window-close-v3": "Push and close a sliding window by its handle.",
    "door-close-v3": "Close a door with a revolving joint by pushing door's handle.",
    "drawer-open-v3": "Open a drawer by its handle.",
    "drawer-close-v3": "Close a drawer by its handle.",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise.",
    "sweep-into-v3": " Sweep a puck from the initial position into a hole.",
    "button-press-v3": "Press a button in y coordination.",
    "handle-press-v3": "Press a handle down.",
    "handle-press-side-v3": "Press a handle down sideways.",
}

mapping_dicts = {
    "self.robot.ee_position": "obs[:3]",
    "self.robot.gripper_openness": "obs[3]",
    "self.obj1.position": "obs[4:7]",
    "self.obj1.quaternion": "obs[7:11]",
    "self.obj2.position": "obs[11:14]",
    "self.obj2.quaternion": "obs[14:18]",
    "self.goal_position": "self.env._get_pos_goal()",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--TASK', type=str, default="drawer-open-v3", \
                        help="choose one task from: drawer-open-v3, drawer-close-v3, window-open-v3, window-close-v3, button-press-v3, sweep-into-v3, door-unlock-v3, door-close-v3, handle-press-v3, handle-press-side-v3")
    parser.add_argument('--FILE_PATH', type=str, default=None)
    parser.add_argument('--MODEL_NAME', type=str, default="gpt-5-mini")

    args = parser.parse_args()

    # File path to save result
    if args.FILE_PATH == None:
        args.FILE_PATH = "results/{}/metaworld-zeroshot/{}.txt".format(args.MODEL_NAME, args.TASK)

    os.makedirs(args.FILE_PATH, exist_ok=True)

    code_generator = ZeroShotGenerator(METAWORLD_PROMPT, args.MODEL_NAME)
    print("check point")
    general_code, specific_code = code_generator.generate_code(instruction_mapping[args.TASK], mapping_dicts)

    with open(os.path.join(args.FILE_PATH, "general.py"), "w") as f:
        f.write(general_code)

    with open(os.path.join(args.FILE_PATH, "specific.py"), "w") as f:
        f.write(specific_code)