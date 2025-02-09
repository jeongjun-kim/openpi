# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image

# from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# JJ : from robomimic lib
try:
    import furniture_bench
except:
    # Skip if furniture_bench is not installed.
    pass
from collections import OrderedDict
from typing import Optional, Union

FURNITURE_ENV_NUM = 2


def create_env():
    env_name = "FurnitureBench-v0"

    import gym

    env = gym.make(
        env_name,
        furniture="cabinet",
        randomness="high",
        joint_impedance=True,
        record_dir_path=os.getcwd(),
        manual_reset=False,
        num_cameras=2,
        no_apriltag=True,
    )
    env.name = env_name
    return env


@dataclasses.dataclass
class Args:
    # Hardware parameters
    # CAM_WRIST_SERIAL = os.getenv("CAM_WRIST_SERIAL", "327122077530")
    # CAM_FRONT_SERIAL = os.getenv("CAM_FRONT_SERIAL", "042222071115")
    # CAM_LEFT_SERIAL = os.getenv("CAM_REAR_SERIAL", "236522071022")
    # CAM_RIGHT_SERIAL = os.getenv("CAM_REAR_SERIAL", "943222070986")

    if FURNITURE_ENV_NUM == 2:
        left_camera_id: str = "042222071115"  # TODO: camera organization is needed for fb
        right_camera_id: str = "<your_camera_id>"
        wrist_camera_id: str = "327122077530"
    else:
        raise ValueError("Invalid furniture environment number")

    # Policy parameters
    external_camera: Optional[str] = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    # assert (
    #     args.external_camera is not None and args.external_camera in ["left", "right"]
    # ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    env = create_env()
    # env_name = env.name
    print("Created the FurnitureBench-robomimic env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        env_obs = env.reset()

        for t_step in bar:
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env_obs,
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"front_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs["front_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 32] of
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                # import pdb

                # pdb.set_trace()
                env_obs = env.step_joint(action)[0]  # First 8 dimensions only.
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        # success: str | float | None = None
        success: Union[str, float, None] = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    front_image = obs_dict["color_image2"]
    wrist_image = obs_dict["color_image1"]

    # Drop the alpha dimension (not sure)
    front_image = front_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB (not sure)
    front_image = front_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    ee_pos = np.array(obs_dict["robot_state"]["ee_pos"])
    ee_quat = np.array(obs_dict["robot_state"]["ee_quat"])
    joint_position = np.array(obs_dict["robot_state"]["joint_positions"])
    gripper_position = np.array(obs_dict["robot_state"]["gripper_width"])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([front_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "front_image": front_image,
        "wrist_image": wrist_image,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
