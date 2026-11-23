"""Сводные таблицы и графики для записки: качество (mean ± std по seed), скорость, устойчивость."""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import MODEL_NAMES, CLASS_NAMES

R = "results"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for p in glob.glob(f"{R}/*/metrics.json"):
        m = load(p)
        if m["loss"] != "ce":
            continue
        rows.append({k: m[k] for k in ["model", "seed", "accuracy", "macro_f1", "macro_precision", "macro_recall"]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("нет результатов"); return
    g = df.groupby("model")
    q = g.agg(n_seeds=("seed", "count"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
              f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).reindex(
        [m for m in MODEL_NAMES if m in g.groups]).fillna(0)
    bench = {r["model"]: r for r in load(f"{R}/benchmark.json")["rows"]} if os.path.exists(f"{R}/benchmark.json") else {}
    out = []
    for name, r in q.iterrows():
        b = bench.get(name, {})
        out.append({"Модель": name, "Seeds": int(r.n_seeds),
                    "Accuracy, %": f"{r.acc_mean * 100:.2f} ± {r.acc_std * 100:.2f}",
                    "Macro-F1, %": f"{r.f1_mean * 100:.2f} ± {r.f1_std * 100:.2f}",
                    "Параметры, млн": round(b.get("params_M", float("nan")), 2),
                    "Размер, МБ": round(b.get("size_MB", float("nan")), 1),
                    "GFLOPs": round(b.get("GFLOPs", float("nan")), 2),
                    "CPU, мс/кадр": round(b.get("cpu_ms", float("nan")), 1),
                    "GPU, мс/кадр": round(b.get("gpu_ms", float("nan")), 1) if "gpu_ms" in b else ""})
    table = pd.DataFrame(out)
    table.to_csv(f"{R}/table_main.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))

    # график: macro-F1 против задержки на CPU (размер точки - число параметров)
    if bench:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for name, r in q.iterrows():
            b = bench[name]
            ax.scatter(b["cpu_ms"], r.f1_mean * 100, s=40 + b["params_M"] * 6)
            ax.annotate(name, (b["cpu_ms"], r.f1_mean * 100), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Задержка на CPU, мс/кадр"); ax.set_ylabel("Macro-F1, %"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{R}/fig_f1_vs_latency.png", dpi=200); plt.close()

    # устойчивость: macro-F1 при искажениях (seed 0)
    rob = {}
    for name in MODEL_NAMES:
        p = f"{R}/{name}_s0/robustness.json"
        if os.path.exists(p):
            rob[name] = {k: v["macro_f1"] * 100 for k, v in load(p).items()}
    if rob:
        pd.DataFrame(rob).T.round(1).to_csv(f"{R}/table_robustness.csv", encoding="utf-8-sig")

    # матрица ошибок лучшей модели (seed 0)
    best = q["f1_mean"].idxmax()
    p = f"{R}/{best}_s0/metrics.json"
    if os.path.exists(p):
        cm = np.array(load(p)["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cmn, cmap="Blues")
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
        ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > .5 else "black")
        ax.set_xlabel("Предсказано"); ax.set_ylabel("Факт"); ax.set_title(f"{best}, доля по строкам")
        plt.tight_layout(); plt.savefig(f"{R}/fig_confusion_{best}.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
