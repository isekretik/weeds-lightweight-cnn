"""Общие функции: данные DeepWeeds, разбиение, модели, метрики.

Данные ожидаются в одном из двух видов (путь передаётся через --data_dir):
  1) папка с изображениями *.jpg и файлом labels.csv (как в репозитории DeepWeeds:
     столбцы Filename, Label, Species);
  2) папка с подпапками по классам (формат ImageFolder).
"""
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Названия классов DeepWeeds: метки 0-7 - виды сорняков, 8 - "не сорняк"
CLASS_NAMES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
    "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative",
]
MODEL_NAMES = ["resnet50", "resnet18", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_samples(data_dir):
    """Возвращает список (путь, метка) и список имён классов."""
    import csv
    csv_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(csv_path):
        samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames}
            fn, lb = cols["filename"], cols["label"]
            for row in reader:
                path = os.path.join(data_dir, row[fn])
                if not os.path.exists(path):
                    alt = os.path.join(data_dir, "images", row[fn])
                    path = alt if os.path.exists(alt) else path
                samples.append((path, int(row[lb])))
        return samples, CLASS_NAMES
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    samples = []
    for i, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(data_dir, c))):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(data_dir, c, fn), i))
    return samples, classes


def stratified_split(samples, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Разбиение 60/20/20 с сохранением доли классов. Фиксировано seed=0, чтобы
    все модели и все запуски сравнивались на одном и том же тесте."""
    rng = random.Random(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, val, test = [], [], []
    for _, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        a = int(n * fractions[0])
        b = int(n * (fractions[0] + fractions[1]))
        train += items[:a]
        val += items[a:b]
        test += items[b:]
    return train, val, test


def get_transforms(img_size=224, train=True):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])


class WeedDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(name, num_classes=9, pretrained=True):
    """Создаёт модель из torchvision и меняет последний слой на num_classes выходов."""
    w = "DEFAULT" if pretrained else None
    if name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model {name}")
    return m


def head_parameters(model, name):
    """Параметры последнего слоя (для обучения с замороженной основой)."""
    if name.startswith("resnet"):
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        ps.append(out.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y_true, y_pred, num_classes=9):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def load_checkpoint(model_name, path, device="cpu"):
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def model_name_from_run(run_dir):
    """'results/mobilenet_v3_small_s0' -> 'mobilenet_v3_small'."""
    import re
    base = os.path.basename(run_dir.rstrip("/"))
    return re.match(r"(.+?)_s\d+", base).group(1)
