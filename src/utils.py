import time
import torch

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Returns: avg_loss, avg_acc, epoch_time, throughput (samples/s)
    """
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    start = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bsz = x.size(0)
        total_samples += bsz
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz

    end = time.time()
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    epoch_time = end - start
    throughput = total_samples / epoch_time if epoch_time > 0 else 0.0

    return avg_loss, avg_acc, epoch_time, throughput

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bsz = x.size(0)
        total_samples += bsz
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc
