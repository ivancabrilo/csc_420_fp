import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_cifar10
from models import ResNet18Stage1, ResNet18Stage2
from utils import accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument(
        "--global-batch",
        type=int,
        default=256,
        help="Total batch size per step (split into micro-batches).",
    )
    p.add_argument(
        "--m",
        type=int,
        default=4,
        help="Number of micro-batches (must divide global-batch).",
    )
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--out", type=str, default="../results/logs/res18_2gpu.txt")
    return p.parse_args()

def main():
    args = parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 GPUs for this pipeline script.")

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    if args.global_batch % args.m != 0:
        raise ValueError("global-batch must be divisible by m (micro-batches).")

    micro_batch_size = args.global_batch // args.m
    print(f"Using global batch {args.global_batch}, m={args.m}, micro-batch={micro_batch_size}")

    # Build 2-stage model
    stage1 = ResNet18Stage1().to(device0)
    stage2 = ResNet18Stage2().to(device1)

    train_loader, _ = get_cifar10(batch_size=args.global_batch)

    criterion = nn.CrossEntropyLoss()
    params = list(stage1.parameters()) + list(stage2.parameters())
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    with open(args.out, "a") as f:
        f.write(f"\n# m={args.m}, global_batch={args.global_batch}\n")
        f.write("epoch,train_loss,train_acc,throughput,epoch_time\n")

        for epoch in range(1, args.epochs + 1):
            stage1.train()
            stage2.train()

            epoch_samples = 0
            total_loss = 0.0
            total_acc = 0.0

            start_epoch = time.time()
            for x, y in train_loader:
                bsz = x.size(0)
                assert bsz == args.global_batch, (
                    f"Expected loader batch {args.global_batch}, got {bsz}"
                )

                x_chunks = x.chunk(args.m, dim=0)
                y_chunks = y.chunk(args.m, dim=0)

                optimizer.zero_grad(set_to_none=True)

                for xb, yb in zip(x_chunks, y_chunks):
                    # move micro-batch to GPU 0
                    xb0 = xb.to(device0, non_blocking=True)

                    # 1tage 1 on GPU 0
                    h = stage1(xb0)

                    # transfer activations + labels to GPU 1
                    h1 = h.to(device1, non_blocking=True)
                    y1 = yb.to(device1, non_blocking=True)

                    # stage 2 on GPU 1
                    out = stage2(h1)

    
                    raw_loss = criterion(out, y1)

                    # scale by 1/m for gradient accumulation
                    loss = raw_loss / args.m
                    loss.backward()

                    mb_size = xb.size(0)
                    total_loss += raw_loss.item() * mb_size
                    total_acc += accuracy(out, y1) * mb_size
                    epoch_samples += mb_size

                optimizer.step()

            end_epoch = time.time()
            epoch_time = end_epoch - start_epoch
            throughput = epoch_samples / epoch_time if epoch_time > 0 else 0.0

            avg_loss = total_loss / epoch_samples
            avg_acc = total_acc / epoch_samples

            line = (
                f"{epoch},{avg_loss:.4f},{avg_acc:.4f},"
                f"{throughput:.2f},{epoch_time:.2f}\n"
            )
            print(line.strip())
            f.write(line)

if __name__ == "__main__":
    main()
