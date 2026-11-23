"""(Необязательный блок) Экспорт в ONNX и статическое квантование INT8 через ONNX Runtime.
Сравнивает размер, задержку на CPU и качество на тесте до и после квантования."""
import argparse
import os
import time
import statistics
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from torch.utils.data import DataLoader
from common import (model_name_from_run, find_samples, stratified_split, get_transforms, WeedDataset, load_checkpoint,
                    compute_metrics, save_json)


class Reader(CalibrationDataReader):
    def __init__(self, loader, input_name, n_batches=8):
        self.it = iter([x.numpy() for i, (x, _) in zip(range(n_batches), loader)])
        self.name = input_name

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {self.name: b}


def evaluate(sess, loader):
    name = sess.get_inputs()[0].name
    ys, ps = [], []
    for x, y in loader:
        out = sess.run(None, {name: x.numpy()})[0]
        ps.append(out.argmax(1)); ys.append(y.numpy())
    return compute_metrics(np.concatenate(ys), np.concatenate(ps))


def latency(sess, size, runs=60):
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(10):
        sess.run(None, {name: x})
    t = []
    for _ in range(runs):
        s = time.perf_counter(); sess.run(None, {name: x}); t.append((time.perf_counter() - s) * 1000)
    return statistics.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    name = model_name_from_run(args.run_dir)
    model = load_checkpoint(name, os.path.join(args.run_dir, "best.pt"), "cpu")
    fp32 = os.path.join(args.run_dir, "model_fp32.onnx")
    int8 = os.path.join(args.run_dir, "model_int8.onnx")
    torch.onnx.export(model, torch.randn(1, 3, args.img_size, args.img_size), fp32, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    samples, _ = find_samples(args.data_dir)
    train_s, _, test_s = stratified_split(samples, seed=0)
    if args.max_test:
        test_s = test_s[:args.max_test]
    tf = get_transforms(args.img_size, False)
    calib = DataLoader(WeedDataset(train_s[:256], tf), batch_size=16)
    test = DataLoader(WeedDataset(test_s, tf), batch_size=32, num_workers=args.workers)
    quantize_static(fp32, int8, Reader(calib, "input"), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    res = {}
    for tag, path in [("fp32", fp32), ("int8", int8)]:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        m = evaluate(sess, test)
        res[tag] = {"size_MB": os.path.getsize(path) / 1e6, "cpu_ms": latency(sess, args.img_size),
                    "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        print(tag, res[tag], flush=True)
    save_json(res, os.path.join(args.run_dir, "quantization.json"))


if __name__ == "__main__":
    main()
