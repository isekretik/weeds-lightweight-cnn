"""Блок «большие данные»: скорость чтения и подготовки изображений (картинок/с) при разном числе процессов-загрузчиков,
и сводная таблица метаданных набора (pandas)."""
import argparse
import time
import pandas as pd
from torch.utils.data import DataLoader
from common import find_samples, get_transforms, WeedDataset, CLASS_NAMES, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_images", type=int, default=2000)
    ap.add_argument("--workers_list", default="0,2,4,8")
    ap.add_argument("--out", default="results/throughput.json")
    args = ap.parse_args()
    samples, classes = find_samples(args.data_dir)
    df = pd.DataFrame(samples, columns=["path", "label"])
    names = classes if len(classes) == 9 else CLASS_NAMES
    df["class"] = df["label"].map(lambda i: names[i])
    counts = df["class"].value_counts().rename_axis("class").reset_index(name="n")
    counts["share_%"] = (counts["n"] / len(df) * 100).round(1)
    print(counts.to_string(index=False))
    counts.to_csv("results/class_distribution.csv", index=False)
    df.to_parquet("results/metadata.parquet") if _has_parquet() else df.to_csv("results/metadata.csv", index=False)
    sub = samples[:args.n_images]
    res = []
    for w in [int(v) for v in args.workers_list.split(",")]:
        dl = DataLoader(WeedDataset(sub, get_transforms(224, True)), batch_size=64, num_workers=w)
        t = time.perf_counter()
        n = sum(len(y) for _, y in dl)
        dt = time.perf_counter() - t
        res.append({"workers": w, "images": n, "seconds": dt, "images_per_s": n / dt})
        print(res[-1], flush=True)
    save_json({"n_total": len(df), "rows": res}, args.out)


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
