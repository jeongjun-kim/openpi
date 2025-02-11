"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import pickle


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    cameras = [
            "wrist_image",
            "agentview_image",
        ]
    features = {
        "action": {
            "dtype": "float32",
            "shape": (8,),  # Action length is 8 based on provided structure
            "names": [
                "pos1", "pos2", "pos3", "quat1",
                "quat2", "quat3", "quat4", "gripper_width",
            ],
        },

        "observation.state": {
            "dtype": "float32",
            "shape": (28,),  # Flattened state
            "names": [
                "ee_pos_x", "ee_pos_y", "ee_pos_z",
                "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
                "ee_pos_vel_x", "ee_pos_vel_y", "ee_pos_vel_z",
                "ee_ori_vel_x", "ee_ori_vel_y", "ee_ori_vel_z",
                "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5", "joint_pos_6",
                "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5", "joint_vel_6",
                "gripper_width",
            ],
        },
        # "parts_poses": {
        #     "dtype": "float32",
        #     "shape": (28,),
        #     "names": ["pose_" + str(i) for i in range(28)],
        # },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

# def get_cameras(hdf5_files: list[Path]) -> list[str]:
#     with h5py.File(hdf5_files[0], "r") as ep:
#         # ignore depth channel, not currently handled
#         return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


# def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
#     imgs_per_cam = {}
#     for camera in cameras:
#         uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

#         if uncompressed:
#             # load all images in RAM
#             imgs_array = ep[f"/observations/images/{camera}"][:]
#         else:
#             import cv2

#             # load one compressed image after the other in RAM and uncompress
#             imgs_array = []
#             for data in ep[f"/observations/images/{camera}"]:
#                 imgs_array.append(cv2.imdecode(data, 1))
#             imgs_array = np.array(imgs_array)

#         imgs_per_cam[camera] = imgs_array
#     return imgs_per_cam


# def load_raw_episode_data(
#     ep_path: Path,
# ) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
#     with h5py.File(ep_path, "r") as ep:
#         state = torch.from_numpy(ep["/observations/qpos"][:])
#         action = torch.from_numpy(ep["/action"][:])

#         velocity = None
#         if "/observations/qvel" in ep:
#             velocity = torch.from_numpy(ep["/observations/qvel"][:])

#         effort = None
#         if "/observations/effort" in ep:
#             effort = torch.from_numpy(ep["/observations/effort"][:])

#         imgs_per_cam = load_raw_images_per_camera(
#             ep,
#             [
#                 "cam_high",
#                 "cam_low",
#                 "cam_left_wrist",
#                 "cam_right_wrist",
#             ],
#         )

#     return imgs_per_cam, state, action, velocity, effort

def load_raw_episode_data(pkl_path: Path):
    """ Load raw data from a PKL file with multiple timesteps. """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    obs = data['observations']  # List of timestep dicts
    demo_id = data['demo_id']
    furniturebench_id = data['furniturebench_id']

    # Initialize lists for each type of data
    images = {  # Store lists of images for each camera
        "wrist_image": [],
        "agentview_image": [],
    }
    robot_states = []  # Robot state over timesteps
    # parts_poses = []  # Parts poses over timesteps

    # Iterate through each timestep and extract the data
    for timestep in obs:
        # Store images
        images["wrist_image"].append(timestep["color_image1"])
        images["agentview_image"].append(timestep["color_image2"])

        # Store robot state
        robot_state_dict = timestep["robot_state"]
        robot_states.append(np.concatenate([
            robot_state_dict["ee_pos"],  # (3,)
            robot_state_dict["ee_quat"],  # (4,)
            robot_state_dict["ee_pos_vel"],  # (3,)
            robot_state_dict["ee_ori_vel"],  # (3,)
            robot_state_dict["joint_positions"],  # (7,)
            robot_state_dict["joint_velocities"],  # (7,)
            robot_state_dict["gripper_width"],  # (1,)
        ]))  # Shape: (28,)

        # Store parts poses and actions
        # parts_poses.append(timestep["parts_poses"])  # (28,)
    for key, value in images.items():
        images[key] = np.stack(value, axis=0)
    # Convert lists to numpy arrays (adding batch dimension)
    # images = value, axis=0) for key, value in images.items()}  # (T, 224, 224, 3)
    robot_states = torch.from_numpy(np.stack(robot_states, axis=0))  # (T, 28)
    # parts_poses = np.stack(parts_poses, axis=0)  # (T, 28)
    action = torch.from_numpy(np.array(data['actions']))  # (T, 8)

    return images, robot_states, action, demo_id, furniturebench_id

def populate_dataset(
    dataset: LeRobotDataset,
    pkl_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(pkl_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = pkl_files[ep_idx]

        imgs_per_cam, state, action, demo_id, furniturebench_id = load_raw_episode_data(ep_path)

        num_frames = state.shape[0]

        # print(state[0])
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                # "parts_poses": parts_poses,
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            dataset.add_frame(frame)
        dataset.save_episode(task=task)

    return dataset

def port_furniture_bench(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    is_packed: bool = True,
):
    print("LEROBOT_HOME:", LEROBOT_HOME)
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    # Load all PKL files from the directory
    pkl_files = sorted(raw_dir.glob("*.pkl"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="furniture_bench",
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        pkl_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_furniture_bench)
