import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet152
import torchvision.models as tv_models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ---------- Base helpers for CIFAR-10 (18 / 34 / 50) ----------

def resnet18_cifar10():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def resnet34_cifar10():
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def resnet50_cifar10():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# +------------------------------+
#  2-GPU PIPELINE STAGES
# +------------------------------+

# ---------- ResNet-18: 2-stage split ----------

class ResNet18Stage1(nn.Module):
    """
    Stage 1 for 2-GPU pipeline:
    conv1 → bn1 → relu → maxpool → layer1 → layer2
    """
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResNet18Stage2(nn.Module):
    """
    Stage 2 for 2-GPU pipeline:
    layer3 → layer4 → avgpool → flatten → fc
    """
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.layer3 = full.layer3
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- ResNet-34: 2-stage split ----------

class ResNet34Stage1(nn.Module):
    """
    Stage 1 for 2-GPU pipeline:
    conv1 → bn1 → relu → maxpool → layer1 → layer2
    """
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResNet34Stage2(nn.Module):
    """
    Stage 2 for 2-GPU pipeline:
    layer3 → layer4 → avgpool → flatten → fc
    """
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.layer3 = full.layer3
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- ResNet-50: 2-stage split ----------

class ResNet50Stage1(nn.Module):
    """
    Stage 1 for 2-GPU pipeline:
    conv1 → bn1 → relu → maxpool → layer1 → layer2
    """
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResNet50Stage2(nn.Module):
    """
    Stage 2 for 2-GPU pipeline:
    layer3 → layer4 → avgpool → flatten → fc
    """
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.layer3 = full.layer3
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# +------------------------------+
#  4-GPU PIPELINE STAGES
# +------------------------------+

# ---------- ResNet-18: 4-stage split ----------

class ResNet18Stage1_4(nn.Module):
    """Stage 1: conv1..layer1"""
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class ResNet18Stage2_4(nn.Module):
    """Stage 2: layer2"""
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.layer2(x)
        return x


class ResNet18Stage3_4(nn.Module):
    """Stage 3: layer3"""
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.layer3 = full.layer3

    def forward(self, x):
        x = self.layer3(x)
        return x


class ResNet18Stage4_4(nn.Module):
    """Stage 4: layer4..fc"""
    def __init__(self):
        super().__init__()
        full = resnet18_cifar10()
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# +---------- ResNet-34: 4-stage split ----------+

class ResNet34Stage1_4(nn.Module):
    """Stage 1: conv1..layer1"""
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class ResNet34Stage2_4(nn.Module):
    """Stage 2: layer2"""
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.layer2(x)
        return x


class ResNet34Stage3_4(nn.Module):
    """Stage 3: layer3"""
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.layer3 = full.layer3

    def forward(self, x):
        x = self.layer3(x)
        return x


class ResNet34Stage4_4(nn.Module):
    """Stage 4: layer4..fc"""
    def __init__(self):
        super().__init__()
        full = resnet34_cifar10()
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- ResNet-50: 4-stage split ----------

class ResNet50Stage1_4(nn.Module):
    """Stage 1: conv1..layer1"""
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.conv1 = full.conv1
        self.bn1 = full.bn1
        self.relu = full.relu
        self.maxpool = full.maxpool
        self.layer1 = full.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class ResNet50Stage2_4(nn.Module):
    """Stage 2: layer2"""
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.layer2 = full.layer2

    def forward(self, x):
        x = self.layer2(x)
        return x


class ResNet50Stage3_4(nn.Module):
    """Stage 3: layer3"""
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.layer3 = full.layer3

    def forward(self, x):
        x = self.layer3(x)
        return x


class ResNet50Stage4_4(nn.Module):
    """Stage 4: layer4..fc"""
    def __init__(self):
        super().__init__()
        full = resnet50_cifar10()
        self.layer4 = full.layer4
        self.avgpool = full.avgpool
        self.fc = full.fc

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class ResNet152Stage1_2(nn.Module):
    """
    Stage 1 for 2-GPU pipeline:
    conv1 + bn1 + relu + maxpool + layer1 + layer2
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)

        self.stem_and_early = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
            m.layer2,
        )

    def forward(self, x):
        return self.stem_and_early(x)


class ResNet152Stage2_2(nn.Module):
    """
    Stage 2 for 2-GPU pipeline:
    layer3 + layer4 + avgpool + fc
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)

        self.blocks = nn.Sequential(
            m.layer3,
            m.layer4,
            m.avgpool,
        )
        self.fc = m.fc

    def forward(self, x):
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class ResNet152Stage1_4(nn.Module):
    """
    Stage 1 (GPU 0): conv1 + bn1 + relu + maxpool + layer1
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)

        self.block = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
        )

    def forward(self, x):
        return self.block(x)


class ResNet152Stage2_4(nn.Module):
    """
    Stage 2 (GPU 1): layer2
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)
        self.block = m.layer2

    def forward(self, x):
        return self.block(x)


class ResNet152Stage3_4(nn.Module):
    """
    Stage 3 (GPU 2): layer3  (the big block)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)
        self.block = m.layer3

    def forward(self, x):
        return self.block(x)


class ResNet152Stage4_4(nn.Module):
    """
    Stage 4 (GPU 3): layer4 + avgpool + fc
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet152(weights=None, num_classes=num_classes)

        self.block = nn.Sequential(
            m.layer4,
            m.avgpool,
        )
        self.fc = m.fc

    def forward(self, x):
        x = self.block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




def get_cifar10_for_vit(batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),          # ViT expects 224×224 patches
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


class ViTL16Stage1_4(nn.Module):
    """
    Stage 1: patch embedding + cls token + pos embedding + encoder layers 0–5
    """
    def __init__(self, full_vit):
        super().__init__()
        enc = full_vit.encoder

        self.conv_proj = full_vit.conv_proj
        self.cls_token = full_vit.class_token
        self.pos_embedding = enc.pos_embedding
        self.pos_dropout = enc.dropout

        # first 6 encoder layers
        self.encoder_layers = nn.ModuleList(enc.layers[:6])

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.conv_proj(x)                    # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)         # [B, N, C]

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+N, C]

        # ViT-L/16 pos_embedding is [1, 1+N, C]
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x


class ViTL16Stage2_4(nn.Module):
    """
    Stage 2: encoder layers 6–11
    """
    def __init__(self, full_vit):
        super().__init__()
        enc = full_vit.encoder
        self.encoder_layers = nn.ModuleList(enc.layers[6:12])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ViTL16Stage3_4(nn.Module):
    """
    Stage 3: encoder layers 12–17
    """
    def __init__(self, full_vit):
        super().__init__()
        enc = full_vit.encoder
        self.encoder_layers = nn.ModuleList(enc.layers[12:18])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ViTL16Stage4_4(nn.Module):
    """
    Stage 4: encoder layers 18–23 + final layernorm + classifier head
    """
    def __init__(self, full_vit, num_classes=10):
        super().__init__()
        enc = full_vit.encoder
        self.encoder_layers = nn.ModuleList(enc.layers[18:])
        self.ln = enc.ln
        self.heads = full_vit.heads   # linear head

        # adjust classifier to CIFAR-10
        if full_vit.heads.head.out_features != num_classes:
            in_features = full_vit.heads.head.in_features
            self.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.ln(x)
        # take CLS token
        cls = x[:, 0]          # [B, C]
        out = self.heads(cls)  # [B, num_classes]
        return out


def make_vit_l16_4stage_pipeline():
    """
    Build ViT-L/16 and split into 4 stages.
    """
    full = tv_models.vit_l_16(weights=None)  # ~304M params
    s1 = ViTL16Stage1_4(full)
    s2 = ViTL16Stage2_4(full)
    s3 = ViTL16Stage3_4(full)
    s4 = ViTL16Stage4_4(full, num_classes=10)
    return s1, s2, s3, s4
######
# +------------ custom CIFAR-10 loader for ViT --------------------+

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_for_vit(batch_size: int):
    """
    CIFAR-10 loader with 224x224 resize for ViT models.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


# +------------------------  ViT-H/14 4-stage pipeline  ------------------------+

class ViTH14Stage1(nn.Module):
    def __init__(self, conv_proj, class_token, pos_embedding, dropout, layers, device):
        super().__init__()
        self.device = device
        # patch embedding / CLS / pos / dropout
        self.conv_proj = conv_proj.to(device)
        # NOTE: wrap as Parameters so they are trainable
        self.class_token = nn.Parameter(class_token.to(device))
        self.pos_embedding = nn.Parameter(pos_embedding.to(device))
        self.dropout = dropout.to(device)
        self.layers = nn.ModuleList(layers).to(device)

    def forward(self, x):
        x = x.to(self.device, non_blocking=True)
        x = self.conv_proj(x)                      # [B, C, H/14, W/14]
        x = x.flatten(2).transpose(1, 2)          # [B, N, C]

        B = x.size(0)
        cls = self.class_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([cls, x], dim=1)            # [B, 1+N, C]
        x = x + self.pos_embedding
        x = self.dropout(x)

        for blk in self.layers:
            x = blk(x)
        return x


class ViTH14EncoderStage(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(layers).to(device)

    def forward(self, x):
        x = x.to(self.device, non_blocking=True)
        for blk in self.layers:
            x = blk(x)
        return x


class ViTH14FinalStage(nn.Module):
    def __init__(self, layers, ln, head, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList(layers).to(device)
        self.ln = ln.to(device)
        self.head = head.to(device)

    def forward(self, x):
        x = x.to(self.device, non_blocking=True)
        for blk in self.layers:
            x = blk(x)
        x = self.ln(x)
        x = x[:, 0]          # CLS token
        x = self.head(x)
        return x


def make_vit_h14_4stage_pipeline():
    """
    Build ViT-H/14 split across 4 GPUs: cuda:0..3.
    """
    full = tv_models.vit_h_14(weights=None)

    # ----- adjust classifier for CIFAR-10 -----
    if isinstance(full.heads, nn.Linear):
        full.heads = nn.Linear(full.heads.in_features, 10)
    elif isinstance(full.heads, nn.Sequential):
        last = list(full.heads.children())[-1]
        assert isinstance(last, nn.Linear), f"Unexpected head type: {type(last)}"
        in_features = last.in_features
        full.heads[-1] = nn.Linear(in_features, 10)
    else:
        raise TypeError(f"Unexpected type for full.heads: {type(full.heads)}")

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    dev2 = torch.device("cuda:2")
    dev3 = torch.device("cuda:3")

    conv_proj = full.conv_proj
    class_token = full.class_token
    pos_embedding = full.encoder.pos_embedding
    dropout = full.encoder.dropout
    layers = list(full.encoder.layers)
    ln = full.encoder.ln
    head = full.heads  # this is now adjusted to 10 classes

    n = len(layers)
    s1_end = n // 4
    s2_end = n // 2
    s3_end = (3 * n) // 4
    s4_end = n

    s1_layers = layers[:s1_end]
    s2_layers = layers[s1_end:s2_end]
    s3_layers = layers[s2_end:s3_end]
    s4_layers = layers[s3_end:s4_end]

    s1 = ViTH14Stage1(conv_proj, class_token, pos_embedding, dropout, s1_layers, dev0)
    s2 = ViTH14EncoderStage(s2_layers, dev1)
    s3 = ViTH14EncoderStage(s3_layers, dev2)
    s4 = ViTH14FinalStage(s4_layers, ln, head, dev3)

    return s1, s2, s3, s4
