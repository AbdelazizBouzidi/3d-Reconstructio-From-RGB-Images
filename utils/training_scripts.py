from pathlib import Path

import torch
import torch.cuda
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsparse.utils.collate import sparse_collate_fn
from tqdm import tqdm
from utils.datasets import PointCloudDataset
from utils.models import SPVCNN_CLASSIFICATION


def prepare_train_and_val_datasets(
    dataset_path: Path,
    voxel_size: float,
    num_channels: int,
    label: int,
    initial_fov: float,
    initial_scale: float,
    distort_f: bool,
    distort_s: bool,
    train_split: float,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader[dict], DataLoader[dict]]:
    use_uv = True if num_channels == 5 else False
    dataset = PointCloudDataset(
        dataset_path,
        voxel_size,
        label,
        initial_fov,
        initial_scale,
        use_uv,
        distort_s,
        distort_f,
    )
    train, valid = torch.utils.data.random_split(
        dataset, [train_split, 1 - train_split]
    )

    train_dataflow = DataLoader(
        train,
        batch_size=batch_size,
        collate_fn=sparse_collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    print("number of batches in the training set", len(train_dataflow))

    valid_dataflow = DataLoader(
        valid,
        batch_size=batch_size,
        collate_fn=sparse_collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )
    print("number of batches in the validation set", len(valid_dataflow))

    return train_dataflow, valid_dataflow


def init_tenssorboard(path_to_logs: Path, model_name: str):
    writer = SummaryWriter(path_to_logs / model_name)
    print(
        "logs for this training session will be written at",
        (path_to_logs / model_name).absolute(),
    )

    return writer


def train_n_epochs(
    device,
    model: SPVCNN_CLASSIFICATION,
    model_name: str,
    train_split: DataLoader,
    validation_split: DataLoader,
    lr: float,
    lr_decay: float,
    tensorboard_writer: SummaryWriter,
    epochs: int,
    print_evry: int,
    logs: bool,
):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lmbda = lambda step: lr_decay * step
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    scaler = amp.GradScaler(enabled=True)

    for i in range(epochs):
        train_one_epoch(
            device,
            train_split,
            validation_split,
            model,
            model_name,
            criterion,
            optimizer,
            scaler,
            tensorboard_writer,
            print_evry,
            logs,
            i * len(train_split),
        )
        scheduler.step()


def train_one_epoch(
    device,
    train_dataflow: DataLoader,
    valid_dataflow: DataLoader,
    model: SPVCNN_CLASSIFICATION,
    model_name: str,
    criterion,
    optimizer,
    scaler,
    tensorboard_writer: SummaryWriter,
    print_evry: int,
    logs: bool,
    train_step: int = 0,
):
    train_loss_sum = 0
    best_val_loss = 1000
    for k, feed_dict in enumerate(tqdm(train_dataflow)):
        inputs = feed_dict["input"].to(device=device)
        labels = feed_dict["label"].to(device=device)

        with amp.autocast(enabled=False):
            outputs = model(inputs)
            train_loss = criterion(outputs, labels[:, None])
            train_loss_sum += train_loss.item()
            train_step += 1

        optimizer.zero_grad()
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (train_step % print_evry == 0) & logs:
            tensorboard_writer.add_scalar(
                "training loss", train_loss_sum / print_evry, train_step
            )
            valid_loss_sum = validate_once(
                device=device,
                valid_dataflow=valid_dataflow,
                model=model,
                criterion=criterion,
                tensorboard_writer=tensorboard_writer,
                train_step=train_step,
                logs=logs,
            )
            print(
                "train_loss:",
                train_loss_sum / print_evry,
                "----------- validation_loss:",
                valid_loss_sum,
            )
            train_loss_sum = 0
            if valid_loss_sum < best_val_loss:
                torch.save(
                    model.state_dict(),
                    "best/" + model_name + "/" + f"{train_step // len(train_dataflow)}",
                )
                best_val_loss = valid_loss_sum / len(valid_dataflow)


def validate_once(
    device,
    valid_dataflow: DataLoader,
    model: SPVCNN_CLASSIFICATION,
    criterion,
    tensorboard_writer: SummaryWriter,
    train_step: int,
    logs: bool,
) -> float:
    valid_loss_sum = 0
    model.eval()
    with torch.no_grad():
        for k, feed_dict in enumerate(tqdm(valid_dataflow)):
            inputs = feed_dict["input"].to(device=device)
            labels = feed_dict["label"].to(device=device)

            with amp.autocast(enabled=False):
                outputs = model(inputs)
                valid_loss = criterion(outputs, labels[:, None])
                valid_loss_sum += valid_loss.item()
        if logs:
            tensorboard_writer.add_scalar(
                "validation loss", valid_loss_sum / len(valid_dataflow), train_step
            )
    return valid_loss_sum / len(valid_dataflow)
