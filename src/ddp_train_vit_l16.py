import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import datasets, transforms, models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--global-batch", type=int, default=256,
                   help="Total batch size across all GPUs")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", type=str,
                   default="../results/logs/vith14_2gpu_ddp.txt")
    return p.parse_args()


def build_dataloaders(global_batch, world_size, rank):
    """
    CIFAR-10 → ViT-H/14:
      - resize to 224x224
      - ImageNet normalization
    """

    batch_per_rank = global_batch // world_size
    if global_batch % world_size != 0:
        raise ValueError(
            f"global-batch ({global_batch}) must be divisible "
            f"by world_size ({world_size})"
        )

    # ViT ImageNet-style transforms
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    cifar_root = os.environ.get("CIFAR10_ROOT", "../data")

    if rank == 0:
        datasets.CIFAR10(root=cifar_root,
                         train=True,
                         download=True)
        datasets.CIFAR10(root=cifar_root,
                         train=False,
                         download=True)
    dist.barrier()

    train_set = datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=False,
        transform=train_tf,
    )
    test_set = datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=False,
        transform=test_tf,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_per_rank,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # keeps batch size consistent
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, train_sampler


def build_model(device):
    vit = models.vit_l_16(weights=None)
    in_features = vit.heads.head.in_features
    vit.heads.head = nn.Linear(in_features, 10)
    vit.to(device)
    return vit


def accuracy(logits, targets):
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().sum()


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"[DDP] world_size={world_size}, device={device}")
        print(f"Using global_batch={args.global_batch}, epochs={args.epochs}")

    train_loader, test_loader, train_sampler = build_dataloaders(
        args.global_batch, world_size, rank
    )

    model = build_model(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Only rank 0 writes log file
    if rank == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        f = open(args.out, "w")
        f.write(f"# ViT-H/14 DDP, world_size={world_size}, "
                f"global_batch={args.global_batch}\n")
        f.write("epoch,train_loss,train_acc,throughput,epoch_time\n")
        f.flush()
    else:
        f = None

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss_local = 0.0
        correct_local = 0.0
        total_local = 0.0

        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            epoch_loss_local += loss.item() * bs
            correct_local += accuracy(out, y).item()
            total_local += bs

        t1 = time.time()
        epoch_time_local = t1 - t0

        # Reduce across all ranks
        stats = torch.tensor(
            [epoch_loss_local, correct_local, total_local, epoch_time_local],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_loss = stats[0].item()
        total_correct = stats[1].item()
        total_samples = stats[2].item()
        epoch_time_sum = stats[3].item()

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        epoch_time_avg = epoch_time_sum / world_size
        throughput = total_samples / epoch_time_avg

        if rank == 0:
            line = (f"{epoch},{avg_loss:.4f},{avg_acc:.4f},"
                    f"{throughput:.2f},{epoch_time_avg:.2f}\n")
            print(line.strip())
            f.write(line)
            f.flush()

    if rank == 0 and f is not None:
        f.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
