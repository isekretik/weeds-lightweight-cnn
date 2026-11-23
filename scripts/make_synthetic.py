"""Создаёт ИГРУШЕЧНЫЙ набор в формате DeepWeeds (картинки + labels.csv), чтобы проверить, что код работает.
Это не настоящие данные: цвет и пятна зависят от класса. Для результатов в записку используйте настоящий DeepWeeds."""
import argparse
import csv
import os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="toy_data")
ap.add_argument("--n", type=int, default=270)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.RandomState(0)
probs = np.array([.06] * 8 + [.52]); probs /= probs.sum()
rows = []
for i in range(args.n):
    lab = int(rng.choice(9, p=probs))
    base = np.array([30 + lab * 20, 120 - lab * 8, 60 + lab * 10])
    img = np.clip(rng.normal(base, 25, (256, 256, 3)), 0, 255).astype(np.uint8)
    fn = f"toy_{i:05d}.jpg"
    Image.fromarray(img).save(os.path.join(args.out, fn))
    rows.append((fn, lab, f"class{lab}"))
with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Filename", "Label", "Species"]); w.writerows(rows)
print("готово:", args.out, len(rows))
