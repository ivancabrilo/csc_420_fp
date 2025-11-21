import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from models import make_vit_h14_4stage_pipeline, get_cifar10_for_vit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--m", type=int, default=16, help="number of micro-batches")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.global_batch % args.m == 0, "global-batch must be divisible by m"
    micro_batch = args.global_batch // args.m

    print(
        f"[PPL] ViT-H/14 4-GPU pipeline, global_batch={args.global_batch}, "
        f"m={args.m}, micro_batch={micro_batch}"
    )

    # ---------- Data ----------
    train_loader = get_cifar10_for_vit(batch_size=args.global_batch)

    # ---------- Model / Pipeline stages ----------
    s1, s2, s3, s4 = make_vit_h14_4stage_pipeline()
    stages = [s1, s2, s3, s4]

    # Collect params from all stages
    params = list(s1.parameters()) + list(s2.parameters()) \
             + list(s3.parameters()) + list(s4.parameters())

    optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss().to("cuda:3")  # loss on last stage device

    # for logging
    if args.out is not None:
        fout = open(args.out, "w")
        fout.write("epoch,train_loss,train_acc,throughput,epoch_time\n")
    else:
        fout = None

    device_last = torch.device("cuda:3")

    for epoch in range(1, args.epochs + 1):
        s1.train(); s2.train(); s3.train(); s4.train()

        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            bs = xb.size(0)
            assert bs == args.global_batch, "DataLoader batch size must equal global-batch"

            optimizer.zero_grad(set_to_none=True)

            # split into micro-batches
            xs = xb.split(micro_batch, dim=0)
            ys = yb.split(micro_batch, dim=0)

         
            # keep activations for backward
            h1_list = [None] * args.m
            h2_list = [None] * args.m
            h3_list = [None] * args.m
            out_list = [None] * args.m
            loss_list = [None] * args.m

            # FP with micro-batches
            for i in range(args.m):
                x_mb = xs[i]
                y_mb = ys[i].to(device_last, non_blocking=True)
                with autocast(dtype=torch.float16):
                    h1 = s1(x_mb)
                    h2 = s2(h1)
                    h3 = s3(h2)
                    out = s4(h3)
                    loss = criterion(out, y_mb)

                h1_list[i] = h1
                h2_list[i] = h2
                h3_list[i] = h3
                out_list[i] = out
                loss_list[i] = loss

            # Backward
            for i in reversed(range(args.m)):
                scaler.scale(loss_list[i] / args.m).backward()

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logits = out_list[-1]
                y_last = ys[-1].to(device_last)
                _, preds = logits.max(1)
                correct += preds.eq(y_last).sum().item()
                total += y_last.size(0)
                running_loss += loss_list[-1].item()

        epoch_time = time.time() - epoch_start
        throughput = (len(train_loader) * args.global_batch) / epoch_time

        avg_loss = running_loss / len(train_loader)
        acc = correct / total if total > 0 else 0.0

        msg = f"{epoch},{avg_loss:.4f},{acc:.4f},{throughput:.2f},{epoch_time:.2f}"
        print(msg)
        if fout is not None:
            fout.write(msg + "\n")
            fout.flush()

    if fout is not None:
        fout.close()


if __name__ == "__main__":
    main()
