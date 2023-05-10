import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from utils.sensor_data import SensorData


def create_intrinsics(fov: float, height: int, width: int) -> np.ndarray:
    """a function that create 3 by 3 matrix representing the simplified camera projection matrix

    Args:
        fov (float): the field of view, as indicated in the dataset
        height (int): height of the depth map in pixels
        width (int): width of the depth map in pixels

    Returns:
        np.ndarray: a 3 by 3 matrix (camera intrinsics)
    """
    real_focal_length = height // 2 / np.tan(fov / 2)
    intrinsics = np.asarray(
        [
            [real_focal_length, 0, height / 2],
            [
                0,
                real_focal_length,
                width / 2,
            ],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return intrinsics


def data_gen_from_scannet(
    path_to_scans: Path,
    total_wanted_scans: int,
    available_data: int,
    output_path,
    n: int = 0,
) -> tuple[int, float, list[float], list[float]]:
    """this function handles the data flow from scannet, it access the scene folders and generate correspandent depth maps and intrinsics parameters, it automaticallt samples data based on the user's inputs

    Args:
        path_to_scans : path to the native dataset (as presented by the creators)
        total_wanted_scans : total number of depth maps wanted by the user
        available_data : the total number of depth maps present in the input dataset
        output_path : path to where the user would like to save the outputs
        n : an index used for naming the output, it's needed when creating the dataset from multiple sources or processes


    Returns:
        tuple: the process return an integer to make sure next process won't overwrite the previously prepared data, also returns a list of focals and scales that can be used to inspect statistics of the resulting dataset
    """
    (output_path / "depths").mkdir(parents=True, exist_ok=True)
    (output_path / "intrinsics").mkdir(parents=True, exist_ok=True)
    scenes = list(path_to_scans.glob("*"))
    k = n
    focals = []
    scales = []
    for scene in tqdm(scenes):
        sens_file = SensorData(scene / f"{scene.name}.sens")
        wanted_frames = int(sens_file.num_frames * total_wanted_scans / available_data)
        sens_file = SensorData(
            scene / f"{scene.name}.sens", wanted_frames=wanted_frames
        )

        for i in tqdm(range(len(sens_file.frames))):
            depth = sens_file.read_depth_from_sens(i) / 1000
            intrinsics = sens_file.intrinsic_depth
            depth = np.where((depth > 0) & (depth < 50), depth, 0)

            np.savez_compressed(output_path / "depths" / f"{k}", depth)
            np.save(output_path / "intrinsics" / f"{k}", intrinsics)
            focals.append((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
            scales.append(depth.max())
            k += 1
    return k, focals, scales


def data_gen_from_taskonomy(
    path_to_data: Path,
    total_wanted_scans: int,
    available_data: int,
    output_path,
    n: int = 0,
) -> tuple[int, float, list[float], list[float]]:
    """similar to dataGenFromScannet, working with the Taskonomy dataset

    Args:
        path_to_data (Path): path to the native dataset (as presented by the creators)
        total_wanted_scans (int): total number of depth maps wanted by the user
        available_data (int): the total number of depth maps present in the input dataset
        output_path (_type_): path to where the user would like to save the outputs
        n (int, optional): an index used for naming the output, it's needed when creating the dataset from multiple sources or processes

    Returns:
        tuple: the process return an integer to make sure next process won't overwrite the previously prepared data, also returns a list of focals and scales that can be used to inspect statistics of the resulting dataset
    """
    (output_path / "depths").mkdir(parents=True, exist_ok=True)
    (output_path / "intrinsics").mkdir(parents=True, exist_ok=True)
    path_to_depths = path_to_data / "depth_zbuffer/taskonomy"
    path_to_camera_infos = path_to_data / "point_info/taskonomy"
    scene_names = [path.name for path in list(path_to_depths.glob("*"))]
    k = n
    focals = []
    scales = []
    for scene in tqdm(scene_names):
        path_to_scene_depths = path_to_depths / scene
        depths = list(path_to_scene_depths.glob("*"))
        intrinsics_path = path_to_camera_infos / scene
        depths = random.sample(
            depths, int(len(depths) * total_wanted_scans / available_data)
        )

        for depth_path in tqdm(depths):
            depth = cv2.imread(f"{depth_path}", cv2.IMREAD_UNCHANGED)
            depth = depth / 1000

            h, w = depth.shape
            depth = np.where((depth > 0) & (depth < 50), depth, 0)

            f = open(intrinsics_path / f"{depth_path.name[:-17]}point_info.json")
            fov = float(json.load(f)["field_of_view_rads"])
            f.close()

            intrinsics = create_intrinsics(fov, h, w)

            np.savez_compressed(output_path / "depths" / f"{k}", depth)
            np.save(output_path / "intrinsics" / f"{k}", intrinsics)
            focals.append((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
            scales.append(depth.max())
            k += 1
    return k, focals, scales


def data_gen_from_burns(
    path_to_data: Path,
    total_wanted_scans: int,
    available_data: int,
    output_path,
    n: int = 0,
):
    """similar to dataGenFromScannet, working with the Ken Burns dataset

    Args:
        path_to_data (Path): path to the native dataset (as presented by the creators)
        total_wanted_scans (int): total number of depth maps wanted by the user
        available_data (int): the total number of depth maps present in the input dataset
        output_path (_type_): path to where the user would like to save the outputs
        n (int, optional): an index used for naming the output, it's needed when creating the dataset from multiple sources or processes

    Returns:
        tuple: the process return an integer to make sure next process won't overwrite the previously prepared data, also returns a list of focals and scales that can be used to inspect statistics of the resulting dataset
    """
    (output_path / "depths").mkdir(parents=True, exist_ok=True)
    (output_path / "intrinsics").mkdir(parents=True, exist_ok=True)

    path_to_depths = path_to_data / "depths"
    path_to_rgb = path_to_data / "rgb"
    scenes = list(path_to_rgb.glob("*"))
    k = n
    focals = []
    scales = []
    for scene in tqdm(scenes):
        path_to_scene_depths = path_to_depths / f"{scene.name}-depth"
        depths = list(path_to_scene_depths.glob("*"))
        depths = random.sample(
            depths, int(len(depths) * total_wanted_scans / available_data)
        )

        for depth_path in tqdm(depths):
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth / 1000

            h, w = depth.shape
            depth = np.where((depth > 0) & (depth < 60), depth, 0)
            f = open(path_to_rgb / scene / f"{depth_path.name[:-12]}meta.json")
            fov = float(json.load(f)["fltFov"]) * np.pi / 180
            f.close()

            intrinsics = create_intrinsics(fov, h, w)

            np.savez_compressed(output_path / "depths" / f"{k}", depth)
            np.save(output_path / "intrinsics" / f"{k}", intrinsics)
            focals.append((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
            scales.append(depth.max())
            k += 1
    return k, focals, scales
