#!/usr/bin/env python3
"""Append leftover FAST subject outputs omitted by hard-coded 57-subject aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def subject_scores(epoacc: np.ndarray) -> np.ndarray:
    if epoacc.ndim == 3:
        return epoacc[:, -1, :].mean(axis=1)
    if epoacc.ndim == 2:
        return epoacc[:, -1]
    raise ValueError(f"Unsupported epoacc shape: {epoacc.shape}")


def aggregate_one(run_dir: Path, kind: str, n_subjects: int) -> dict:
    base_npz = run_dir / f"{kind}.npz"
    if not base_npz.exists():
        raise FileNotFoundError(base_npz)

    z = np.load(base_npz, allow_pickle=True)
    epoacc = z["epoacc"]
    logits = z["logits"]
    start = epoacc.shape[0]

    extra_epo = []
    extra_logits = []
    for idx in range(start, n_subjects):
        csv_path = run_dir / f"{kind}-{idx:02d}.csv"
        npy_path = run_dir / f"{kind}-{idx:02d}.npy"
        if not csv_path.exists() or not npy_path.exists():
            raise FileNotFoundError(f"Missing {csv_path} or {npy_path}")
        extra_epo.append(np.loadtxt(csv_path, delimiter=","))
        extra_logits.append(np.load(npy_path))

    if extra_epo:
        epoacc = np.concatenate([epoacc, np.stack(extra_epo, axis=0)], axis=0)
        logits = np.concatenate([logits, np.concatenate(extra_logits, axis=0)], axis=0)

    out_npz = run_dir / f"{kind}_{n_subjects}.npz"
    np.savez(out_npz, epoacc=epoacc, logits=logits)

    scores = subject_scores(epoacc)
    return {
        "kind": kind,
        "output": str(out_npz),
        "epoacc_shape": list(epoacc.shape),
        "logits_shape": list(logits.shape),
        "n": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "sd_pop": float(scores.std(ddof=0)),
        "sd_sample": float(scores.std(ddof=1)) if scores.shape[0] > 1 else None,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "appended_subject_indices": list(range(start, n_subjects)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n-subjects", required=True, type=int)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()

    results = [aggregate_one(args.run_dir, kind, args.n_subjects) for kind in ("pre", "tune")]
    summary = {"run_dir": str(args.run_dir), "n_subjects": args.n_subjects, "results": results}
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary:
        args.summary.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
