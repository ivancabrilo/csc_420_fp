import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from torchvision import datasets, transforms, models

from utils import accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument(
        "--global-batch",
        type=int,
        default=256,
        help="Total batch size across all GPUs.",
    )
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument(
        "--out",
        type=str,
        default="../results/logs/res152_2gpu_ddp.txt",
    )
    return p.parse_args()


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback: single-process (useful for quick debugging)
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def get_cifar10_ddp(batch_size, num_workers, rank, world_size):

    cifar_root = os.environ.get("CIFAR10_ROOT", "../data")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=False,
        transform=transform_train,
    )

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, train_sampler


def main():
    args = parse_args()

    rank, world_size, local_rank = init_distributed()

    if args.global_batch % world_size != 0:
        raise ValueError(
            f"global-batch ({args.global_batch}) must be divisible by world_size ({world_size})"
        )

    local_batch = args.global_batch // world_size

    model = models.resnet152(num_classes=10)
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    train_loader, train_sampler = get_cifar10_ddp(
        batch_size=local_batch,
        num_workers=4,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        f = open(args.out, "w")
        f.write(
            f"# DDP ResNet152, world_size={world_size}, "
            f"global_batch={args.global_batch}, local_batch={local_batch}\n"
        )
        f.write("epoch,train_loss,train_acc,throughput,epoch_time\n")
    else:
        f = None

    for epoch in range(1, args.epochs + 1):
        # important for DistributedSampler shuffling
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            mb = x.size(0)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            raw_loss = criterion(out, y)
            loss = raw_loss / 1.0  
            loss.backward()
            optimizer.step()

            total_loss += raw_loss.item() * mb
            total_acc += accuracy(out, y) * mb
            total_samples += mb

        t1 = time.time()
        epoch_time = t1 - t0

        # Convert to tensors and all_reduce
        loss_tensor = torch.tensor(total_loss, device=device)
        acc_tensor = torch.tensor(total_acc, device=device)
        samples_tensor = torch.tensor(
            total_samples, dtype=torch.float64, device=device
        )

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

        total_loss = loss_tensor.item()
        total_acc = acc_tensor.item()
        total_samples = samples_tensor.item()

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        throughput = total_samples / epoch_time

        if rank == 0:
            line = (
                f"{epoch},{avg_loss:.4f},{avg_acc:.4f},"
                f"{throughput:.2f},{epoch_time:.2f}\n"
            )
            print(line.strip())
            f.write(line)
            f.flush()

    if rank == 0 and f is not None:
        f.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
