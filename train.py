import argparse
from pathlib import Path

import torch
from utils.models import SPVCNN_CLASSIFICATION
from utils.training_scripts import (
    init_tenssorboard,
    prepare_train_and_val_datasets,
    train_n_epochs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Configs for 3D training")
    parser.add_argument(
        "--data", required=True, help="path to training dataset", type=Path
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="name to use to save the model and for tensorboard logs",
    )
    parser.add_argument(
        "--input_chanels",
        required=True,
        choices=[3, 5],
        help="specify the features dimentions, 3 if your are using only x, y and z, 5 for x,y,z and u,v",
        type=int,
    )
    parser.add_argument(
        "--targets",
        required=True,
        choices=[0, 1, 2],
        help="a variable needed by the data loader to specify what targets to train, 0 for the shift, 1 for the scale and 2 for focal_length",
        type=int,
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=0.24, type=float)
    parser.add_argument(
        "--distort_s",
        action="store_true",
        help="indicate to the data loader whether to apply distortion to the cloudpoint through a applying a shift to the depth values, default is False",
    )
    parser.add_argument(
        "--distort_f",
        action="store_true",
        help="indicate to the data loader whether to apply distortion to the cloudpoint through an incorrect focal, default is False",
    )
    parser.add_argument("--voxel_size", default=0.005, type=float)
    parser.add_argument("--init_scale", default=1, type=float)
    parser.add_argument("--init_fov", default=60, type=float)
    parser.add_argument(
        "--train_split",
        default=0.9,
        help="percentage / 100 of data used for training, by default 1 - train_split represent the portion of data used for validation",
        type=float,
    )
    parser.add_argument(
        "--learning_rate_decay",
        default=0.1,
        help="used for a linear learning rate decay",
        type=float,
    )
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument(
        "--logs",
        action="store_true",
        help="indicate if you would like to ge tensorboard logs and helper prints",
    )
    parser.add_argument("--port", default=6006, help="default is 6006")
    parser.add_argument(
        "--print_losses_evry", default=1000, type=int, help="frequency of the logs"
    )
    parser.add_argument(
        "--num_w", default=10, type=int, help="multithreading for the dataloader"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SPVCNN_CLASSIFICATION(
        input_channel=args.input_chanels, num_classes=1, cr=1.0, pres=0.01, vres=0.01
    ).to(device)

    (Path("best") / args.model_name).mkdir(parents=True, exist_ok=True)

    tensorboard_writer = init_tenssorboard(Path("runs"), args.model_name)

    train, val = prepare_train_and_val_datasets(
        args.data,
        args.voxel_size,
        args.input_chanels,
        args.targets,
        args.init_fov,
        args.init_scale,
        args.distort_f,
        args.distort_s,
        args.train_split,
        args.batch_size,
        args.num_w,
    )

    train_n_epochs(
        device,
        model,
        args.model_name,
        train,
        val,
        args.learning_rate,
        args.learning_rate_decay,
        tensorboard_writer,
        args.epochs,
        args.print_losses_evry,
        args.logs,
    )
