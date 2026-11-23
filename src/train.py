"""Обучение одной модели с одним seed. Результат: results/<model>_s<seed>/ (метрики, история, веса)."""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common import (set_seed, find_samples, stratified_split, get_transforms, WeedDataset,
                    build_model, head_parameters, predict, compute_metrics, save_json, MODEL_NAMES)


def focal_loss_fn(gamma=2.0, weight=None):
    ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def f(logits, y):
        l = ce(logits, y)
        pt = torch.exp(-l)
        return ((1 - pt) ** gamma * l).mean()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze_epochs", type=int, default=2, help="эпох с замороженной основой")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="lr головы; основа дообучается с lr/10")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--loss", default="ce", choices=["ce", "weighted", "focal"])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--no_pretrained", action="store_true", help="только для проверки кода без интернета")
    ap.add_argument("--max_train", type=int, default=0, help="для быстрых проверок: ограничить число обучающих картинок")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"{args.model}_s{args.seed}" + ("" if args.loss == "ce" else f"_{args.loss}")
    out = os.path.join(args.out_dir, run)
    os.makedirs(out, exist_ok=True)

    samples, classes = find_samples(args.data_dir)
    train_s, val_s, test_s = stratified_split(samples, seed=0)  # разбиение одно для всех запусков
    if args.max_train:
        train_s = train_s[:args.max_train]
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=(device == "cuda"))
    g = torch.Generator()
    g.manual_seed(args.seed)
    tr = DataLoader(WeedDataset(train_s, get_transforms(args.img_size, True)), shuffle=True, generator=g, **kw)
    va = DataLoader(WeedDataset(val_s, get_transforms(args.img_size, False)), **kw)
    te = DataLoader(WeedDataset(test_s, get_transforms(args.img_size, False)), **kw)

    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    weight = None
    if args.loss in ("weighted", "focal"):
        counts = torch.bincount(torch.tensor([s[1] for s in train_s]), minlength=len(classes)).float()
        weight = (counts.sum() / (len(classes) * counts.clamp(min=1))).to(device) if args.loss == "weighted" else None
    crit = focal_loss_fn(2.0) if args.loss == "focal" else nn.CrossEntropyLoss(weight=weight)

    head_ids = {id(p) for p in head_parameters(model, args.model)}
    head = [p for p in model.parameters() if id(p) in head_ids]
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": body, "lr": args.lr / 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_f1, bad, best_epoch = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for p in body:
            p.requires_grad = epoch >= args.freeze_epochs
        model.train()
        total, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv = predict(model, va, device)
        m = compute_metrics(yv, pv, len(classes))
        history.append({"epoch": epoch + 1, "train_loss": total / n, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"[{run}] epoch {epoch + 1}/{args.epochs} loss={total / n:.4f} val_acc={m['accuracy']:.4f} val_f1={m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best_f1:
            best_f1, bad, best_epoch = m["macro_f1"], 0, epoch + 1
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    yt, pt = predict(model, te, device)
    res = compute_metrics(yt, pt, len(classes))
    res.update({"model": args.model, "seed": args.seed, "loss": args.loss, "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1, "train_time_s": train_time, "history": history,
                "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s), "device": device})
    save_json(res, os.path.join(out, "metrics.json"))
    print(f"[{run}] TEST acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
