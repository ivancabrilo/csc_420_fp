import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_cifar10
from models import (
    ResNet50Stage1_4,
    ResNet50Stage2_4,
    ResNet50Stage3_4,
    ResNet50Stage4_4,
)
from utils import accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--m", type=int, default=4,
                   help="number of micro-batches; must divide global-batch")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--out", type=str,
                   default="../results/logs/res50_4gpu.txt")
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.device_count() < 4:
        raise RuntimeError("Need at least 4 GPUs for this script.")

    if args.global_batch % args.m != 0:
        raise ValueError("global-batch must be divisible by m.")

    micro_batch_size = args.global_batch // args.m
    print(f"ResNet-50 4-GPU pipeline: global_batch={args.global_batch}, "
          f"m={args.m}, micro_batch={micro_batch_size}")

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    dev2 = torch.device("cuda:2")
    dev3 = torch.device("cuda:3")

    # Build 4-stage model
    s1 = ResNet50Stage1_4().to(dev0)
    s2 = ResNet50Stage2_4().to(dev1)
    s3 = ResNet50Stage3_4().to(dev2)
    s4 = ResNet50Stage4_4().to(dev3)

    train_loader, _ = get_cifar10(batch_size=args.global_batch)

    criterion = nn.CrossEntropyLoss()
    params = list(s1.parameters()) + list(s2.parameters()) \
             + list(s3.parameters()) + list(s4.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    with open(args.out, "a") as f:
        f.write(f"\n# m={args.m}, global_batch={args.global_batch}\n")
        f.write("epoch,train_loss,train_acc,throughput,epoch_time\n")

        for epoch in range(1, args.epochs + 1):
            s1.train(); s2.train(); s3.train(); s4.train()
            total_loss = 0.0
            total_acc = 0.0
            total_samples = 0

            t0 = time.time()
            for x, y in train_loader:
                bsz = x.size(0)
                assert bsz == args.global_batch

                x_chunks = x.chunk(args.m, dim=0)
                y_chunks = y.chunk(args.m, dim=0)

                optimizer.zero_grad(set_to_none=True)

                for xb, yb in zip(x_chunks, y_chunks):
                    xb0 = xb.to(dev0, non_blocking=True)
                    yb3 = yb.to(dev3, non_blocking=True)

                    h1 = s1(xb0)
                    h2 = s2(h1.to(dev1))
                    h3 = s3(h2.to(dev2))
                    out = s4(h3.to(dev3))

                    raw_loss = criterion(out, yb3)
                    loss = raw_loss / args.m
                    loss.backward()

                    mb = xb.size(0)
                    total_loss += raw_loss.item() * mb
                    total_acc += accuracy(out, yb3) * mb
                    total_samples += mb

                optimizer.step()

            t1 = time.time()
            epoch_time = t1 - t0
            throughput = total_samples / epoch_time
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples

            line = (f"{epoch},{avg_loss:.4f},{avg_acc:.4f},"
                    f"{throughput:.2f},{epoch_time:.2f}\n")
            print(line.strip())
            f.write(line)


if __name__ == "__main__":
    main()
