"""Grad-CAM: карты внимания для нескольких тестовых картинок (правильные и ошибочные ответы)."""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from common import (model_name_from_run, find_samples, stratified_split, get_transforms, load_checkpoint, CLASS_NAMES,
                    IMAGENET_MEAN, IMAGENET_STD)


def target_layer(model, name):
    """Последний свёрточный блок перед классификатором."""
    if name.startswith("resnet"):
        return model.layer4
    return model.features


def grad_cam(model, layer, x, cls=None):
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    pred = int(out.argmax(1))
    cls = pred if cls is None else cls
    out[0, cls].backward()
    h1.remove(); h2.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    name = model_name_from_run(args.run_dir)
    model = load_checkpoint(name, os.path.join(args.run_dir, "best.pt"), "cpu")
    for p in model.parameters():
        p.requires_grad = True
    layer = target_layer(model, name)
    samples, classes = find_samples(args.data_dir)
    _, _, test_s = stratified_split(samples, seed=0)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(test_s), size=min(args.n, len(test_s)), replace=False)
    tf = get_transforms(args.img_size, False)
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        path, label = test_s[i]
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
        cam, pred = grad_cam(model, layer, x)
        img = np.clip((x[0].numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        names = classes if len(classes) == 9 else CLASS_NAMES
        ax.set_title(f"факт: {names[label]}\nответ: {names[pred]}", fontsize=8, color="green" if pred == label else "red")
    plt.tight_layout()
    out = os.path.join(args.run_dir, "gradcam.png")
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
