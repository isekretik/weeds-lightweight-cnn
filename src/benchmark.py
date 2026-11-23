"""Размер модели, число параметров, FLOPs и задержка на один кадр (ms/frame).
Модели берутся без обучения (скорость от весов не зависит), поэтому скрипт можно запускать сразу."""
import argparse
import time
import statistics
import torch
from torch.utils.flop_counter import FlopCounterMode
from common import build_model, count_params, model_size_mb, save_json, MODEL_NAMES


def flops_g(model, size):
    x = torch.randn(1, 3, size, size)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9  # FLOPs (умножение и сложение считаются отдельно)


@torch.no_grad()
def latency_ms(model, device, size, warmup=20, runs=100, threads=None):
    if threads:
        torch.set_num_threads(threads)
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), statistics.pstdev(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=4, help="потоки CPU (имитация слабого бортового процессора)")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    rows = []
    for name in MODEL_NAMES:
        m = build_model(name, pretrained=False).eval()
        row = {"model": name, "params_M": count_params(m) / 1e6, "size_MB": model_size_mb(m),
               "GFLOPs": flops_g(m, args.img_size)}
        row["cpu_ms"], row["cpu_ms_std"] = latency_ms(m, "cpu", args.img_size, threads=args.threads)
        if torch.cuda.is_available():
            m = m.cuda()
            row["gpu_ms"], row["gpu_ms_std"] = latency_ms(m, "cuda", args.img_size)
        rows.append(row)
        print(row, flush=True)
    info = {"cpu_threads": args.threads, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "img_size": args.img_size}
    save_json({"info": info, "rows": rows}, args.out)


if __name__ == "__main__":
    main()
