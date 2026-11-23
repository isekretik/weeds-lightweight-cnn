"""Проверка устойчивости: macro-F1 на тесте при размытии, шуме и изменении яркости."""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from common import (model_name_from_run, find_samples, stratified_split, get_transforms, WeedDataset, load_checkpoint,
                    predict, compute_metrics, save_json, set_seed)


class Corrupt:
    """Искажение, применяемое к уже нормализованному тензору-картинке не нужно:
    поэтому применяем к PIL-изображению до ToTensor."""
    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def __call__(self, img):
        if self.kind == "blur":
            return TF.gaussian_blur(img, kernel_size=2 * int(self.level) * 2 + 1, sigma=float(self.level))
        if self.kind == "brightness":
            return TF.adjust_brightness(img, self.level)
        if self.kind == "noise":
            t = TF.to_tensor(img)
            t = (t + torch.randn_like(t) * self.level).clamp(0, 1)
            return TF.to_pil_image(t)
        return img


CORRUPTIONS = {"clean": [0], "blur": [1, 2, 3], "noise": [0.05, 0.1, 0.2], "brightness": [0.5, 0.7, 1.4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", required=True, help="папка запуска, например results/mobilenet_v3_small_s0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_test", type=int, default=0)
    args = ap.parse_args()
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_name_from_run(args.run_dir)
    model = load_checkpoint(name, os.path.join(args.run_dir, "best.pt"), device)
    samples, _ = find_samples(args.data_dir)
    _, _, test_s = stratified_split(samples, seed=0)
    if args.max_test:
        test_s = test_s[:args.max_test]
    base = get_transforms(args.img_size, False).transforms  # Resize, ToTensor, Normalize
    res = {}
    for kind, levels in CORRUPTIONS.items():
        for lv in levels:
            pre = [] if kind == "clean" else [Corrupt(kind, lv)]
            tf = transforms.Compose([base[0]] + pre + base[1:])
            dl = DataLoader(WeedDataset(test_s, tf), batch_size=args.batch_size, num_workers=args.workers)
            y, p = predict(model, dl, device)
            m = compute_metrics(y, p)
            res[f"{kind}_{lv}"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            print(kind, lv, res[f"{kind}_{lv}"], flush=True)
    save_json(res, os.path.join(args.run_dir, "robustness.json"))


if __name__ == "__main__":
    main()
