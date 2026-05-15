#!/usr/bin/env python3
"""Linear SVM block-CV benchmark for 100-trial MAT folders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


BLOCKS = [2, 4, 6, 8, 10]


def parse_folds(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def load_subject(data_dir: Path, subject: str, scale_factor: float, tmin: float, tmax: float):
    eeg = loadmat(str(data_dir / f"{subject}_eeg_trials.mat"))
    labels = loadmat(str(data_dir / f"{subject}_trial_labels.mat"))
    x = eeg["eeg_data"].astype(np.float64) * float(scale_factor)
    y = labels["word_label"].ravel().astype(np.int64)
    if sorted(np.unique(y).tolist()) == [1, 2, 3, 4, 5]:
        y = y - 1
    block = labels["block"].ravel().astype(np.int64)
    time_vec = eeg["time_vec"].ravel().astype(float)
    keep = (time_vec >= tmin) & (time_vec <= tmax)
    if not np.any(keep):
        raise RuntimeError(f"{subject}: no samples in window {tmin}..{tmax}")
    return x[:, :, keep], y, block


def run_subject(x: np.ndarray, y: np.ndarray, block: np.ndarray, max_iter: int):
    x_flat = x.reshape(x.shape[0], -1)
    all_preds = np.zeros_like(y)
    scores = []
    for test_block in BLOCKS:
        test_mask = block == test_block
        train_mask = ~test_mask
        x_train, x_test = x_flat[train_mask], x_flat[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        mu = x_train.mean(axis=0)
        sigma = x_train.std(axis=0)
        sigma[sigma == 0] = 1.0
        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma

        clf = OneVsRestClassifier(LinearSVC(C=1.0, max_iter=max_iter, dual="auto"))
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        all_preds[test_mask] = preds
        scores.append(balanced_accuracy_score(y_test, preds))
    return np.array(scores, dtype=np.float32), balanced_accuracy_score(y, all_preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=1.5)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tag", default="fast_benchmark_58_svm_0to1p5s")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(path.name.split("_")[0] for path in data_dir.glob("S*_eeg_trials.mat"))
    folds = set(parse_folds(args.folds))
    print(f"SVM subjects={len(subjects)} window={args.tmin}..{args.tmax}s data={data_dir}")

    for fold, (_, test_idx) in enumerate(KFold(n_splits=10, shuffle=False).split(subjects)):
        if fold not in folds:
            continue
        out_path = out_dir / f"fold{fold}_{args.tag}.npz"
        if out_path.exists():
            print(f"SKIP fold{fold}: {out_path} exists", flush=True)
            continue
        results = {}
        overall = {}
        fold_subjects = [subjects[i] for i in test_idx]
        print(f"\nFold {fold}: {fold_subjects}", flush=True)
        for si, subject in enumerate(fold_subjects, start=1):
            t0 = time.time()
            x, y, block = load_subject(data_dir, subject, args.scale_factor, args.tmin, args.tmax)
            scores, all_score = run_subject(x, y, block, args.max_iter)
            results[subject] = scores
            overall[subject] = np.float32(all_score)
            print(
                f"[{si:2d}/{len(fold_subjects)}] {subject}: "
                f"{' '.join(f'{v:.3f}' for v in scores)} | "
                f"mean={float(np.mean(scores)):.4f} overall={all_score:.4f} | "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        means = np.array([np.mean(v) for v in results.values()], dtype=np.float32)
        payload = {subject: scores for subject, scores in results.items()}
        payload["subjects"] = np.array(list(results.keys()), dtype=object)
        payload["subject_mean"] = means
        payload["subject_overall"] = np.array([overall[s] for s in results], dtype=np.float32)
        payload["grand_mean"] = np.float32(np.mean(means))
        payload["grand_std"] = np.float32(np.std(means))
        payload["window_sec"] = np.array([args.tmin, args.tmax], dtype=np.float32)
        payload["data_dir"] = str(data_dir)
        np.savez(out_path, **payload)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
