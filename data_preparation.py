import argparse
import os
from pathlib import Path

import numpy as np
import utils.points_cloud_structuring as utils
from utils.sensor_data import SensorData

# an environment variable needed by opencv for .exr files support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="config for data generation")

    parser.add_argument(
        "-ss",
        "--src-scannet",
        required=False,
        help="path to raw Scannet dataset",
        type=Path,
    )
    parser.add_argument(
        "-sk",
        "--src-kenberns",
        required=False,
        help="path to raw KenBurns dataset",
        type=Path,
    )
    parser.add_argument(
        "-st",
        "--src-takonomy",
        required=False,
        help="path to raw taskonomy dataset",
        type=Path,
    )

    parser.add_argument(
        "--out", required=False, help="path to save the output files dataset", type=Path
    )
    parser.add_argument(
        "-ns",
        "--nbr-samples",
        help="a list of the data quantities from each dataset - \
        have to be inferior than the total number of samles in each dataset",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    total_num_smaples_s = np.asarray(
        [
            SensorData(scene / f"{scene.name}.sens").num_frames
            for scene in (args.src_scannet).glob("*")
        ]
    )
    total_num_smaples_k = np.asarray(
        [
            len(list((scene).glob("*.exr")))
            for scene in (args.src_kenberns / "depths").glob("*")
        ]
    )
    total_num_smaples_t = np.asarray(
        [
            len(list((scene).glob("*.png")))
            for scene in (args.src_takonomy / "depth_zbuffer/taskonomy").glob("*")
        ]
    )

    print(
        "total number of samples in scannet",
        sum(total_num_smaples_s),
        ", KenBurns",
        sum(total_num_smaples_k),
        ", taskonomy",
        sum(total_num_smaples_t),
    )
    total_wanted_samples = [int(item) for item in args.nbr_samples.split(",")]

    if total_wanted_samples < [
        sum(total_num_smaples_s),
        sum(total_num_smaples_k),
        sum(total_num_smaples_t),
    ]:
        print(total_wanted_samples)
        focals = []
        scales = []
        k, f, s = utils.data_gen_from_scannet(
            args.src_scannet,
            total_wanted_samples[0],
            sum(total_num_smaples_s),
            args.out,
        )
        focals += f
        scales += s
        k, f, s = utils.data_gen_from_burns(
            args.src_kenberns,
            total_wanted_samples[1],
            sum(total_num_smaples_k),
            args.out,
            k,
        )
        focals += f
        scales += s
        k, f, s = utils.data_gen_from_taskonomy(
            args.src_takonomy,
            total_wanted_samples[2],
            sum(total_num_smaples_t),
            args.out,
            k,
        )
        focals += f
        scales += s
        np.save("focals", focals)
        np.save("scales", scales)
        print("total generated data samples:", k)
    else:
        print("you want more data than you got")
        print("total generated data samples:", 0)
