"""
EEGNet benchmark for 100-trial DSO matdata variants.

Runs selected subject-level folds. Within each held-out subject, evaluation is
leave-one-block-out over blocks 2/4/6/8/10, matching older EEGNet baselines.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


BLOCKS = [2, 4, 6, 8, 10]
LR = 1e-3
BS = 64


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_chan, n_time, n_class=5, dropout_p=0.5, f1=8, d=2, c1=64):
        super().__init__()
        f2 = d * f1
        block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, c1), padding=(0, c1 // 2), bias=False),
            nn.BatchNorm2d(f1),
            Conv2dWithConstraint(f1, f1 * d, (n_chan, 1), bias=False, max_norm=1, groups=f1),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout_p),
        )
        block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 22), padding=(0, 11), bias=False, groups=f1 * d),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout_p),
        )
        self.features = nn.Sequential(block1, block2)
        with torch.no_grad():
            feat_size = self.features(torch.zeros(1, 1, n_chan, n_time)).shape[-1]
        self.classifier = nn.Conv2d(f2, n_class, (1, feat_size))

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.features(x)).squeeze(-1).squeeze(-1)


def cosine_lr(base, final, epochs, steps_per_ep, warmup=10):
    warmup_vals = np.linspace(0, base, warmup * steps_per_ep) if warmup > 0 else np.array([])
    iters = np.arange(epochs * steps_per_ep - len(warmup_vals))
    vals = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_vals, vals))


def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_block(train_x, train_y, test_x, test_y, device, epochs, num_workers):
    model = EEGNet(n_chan=train_x.shape[1], n_time=train_x.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=BS,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).long()),
        batch_size=BS,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    sched_vals = cosine_lr(LR, 1e-5, epochs, len(train_loader), warmup=min(10, epochs))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: sched_vals[min(step, len(sched_vals) - 1)] / LR
    )

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))


def load_subject(data_dir, subject, scale_factor):
    eeg = loadmat(str(data_dir / f"{subject}_eeg_trials.mat"))
    labels = loadmat(str(data_dir / f"{subject}_trial_labels.mat"))
    x = eeg["eeg_data"].astype(np.float32) * np.float32(scale_factor)
    y = labels["word_label"].ravel().astype(np.int64)
    if sorted(np.unique(y).tolist()) == [1, 2, 3, 4, 5]:
        y = y - 1
    block = labels["block"].ravel().astype(np.int64)
    return x, y, block


def parse_folds(value):
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scale-factor", type=float, default=1e6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tag", default="ica_artifact_t09_no_baseline_scale1e6_eegnet_63ch_100ep")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    subjects = sorted(path.name.split("_")[0] for path in data_dir.glob("S*_eeg_trials.mat"))
    folds = set(parse_folds(args.folds))
    print(f"Device: {device}")
    print(f"Subjects: {len(subjects)} | folds: {sorted(folds)} | epochs: {args.epochs}")
    print(f"Data: {data_dir}")

    splitter = KFold(n_splits=10, shuffle=False)
    for fold, (_, test_idx) in enumerate(splitter.split(subjects)):
        if fold not in folds:
            continue
        out_path = out_dir / f"fold{fold}_{args.tag}.npz"
        if out_path.exists():
            print(f"SKIP fold{fold}: {out_path} exists", flush=True)
            continue

        fold_subjects = [subjects[i] for i in test_idx]
        print(f"\nFold {fold}: {fold_subjects}", flush=True)
        results = {}
        for si, subject in enumerate(fold_subjects, start=1):
            t0 = time.time()
            seed_all(42)
            x, y, block = load_subject(data_dir, subject, args.scale_factor)
            block_scores = []
            for test_block in BLOCKS:
                mask = block == test_block
                score = run_one_block(
                    x[~mask], y[~mask], x[mask], y[mask], device, args.epochs, args.num_workers
                )
                block_scores.append(score)
            results[subject] = np.array(block_scores, dtype=np.float32)
            elapsed = time.time() - t0
            print(
                f"[{si:2d}/{len(fold_subjects)}] {subject}: "
                f"{' '.join(f'{v:.3f}' for v in block_scores)} | "
                f"mean={np.mean(block_scores):.4f} | {elapsed:.0f}s",
                flush=True,
            )

        means = np.array([np.mean(v) for v in results.values()], dtype=np.float32)
        payload = {subject: scores for subject, scores in results.items()}
        payload["subjects"] = np.array(list(results.keys()), dtype=object)
        payload["subject_mean"] = means
        payload["grand_mean"] = np.float32(np.mean(means))
        payload["grand_std"] = np.float32(np.std(means))
        payload["scale_factor"] = np.float64(args.scale_factor)
        payload["data_dir"] = str(data_dir)
        np.savez(out_path, **payload)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
