#!/usr/bin/env python3
"""FAST-compatible whole-trial EEG preprocessing for COSMO benchmark data.

This is the public benchmark preprocessing entrypoint. It intentionally does
not hard-code participant lists or demographic
metadata. Subjects are discovered from BrainVision file names, or supplied in a
plain text list outside the source tree.

For simultaneous EEG-fMRI, this script expects BrainVision files after the
scanner-specific AAS and BCG correction stage. All subsequent steps match the
FAST-compatible benchmark settings unless command-line options override them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mne
import numpy as np
from scipy.io import savemat


mne.set_log_level("WARNING")

DEFAULT_COVERT_BLOCKS = [2, 4, 6, 8, 10]
SUBJECT_RE = re.compile(r"S\d{4}")


@dataclass(frozen=True)
class SubjectSummary:
    subject_id: str
    dataset: str
    raw_file: str
    x_shape: tuple[int, int, int]
    y_counts: list[int]
    block_counts: dict[int, int]
    eog_idx: list[int]
    muscle_idx: list[int]
    exclude_idx: list[int]
    clip_limits: tuple[float, float]
    filter_l_freq: float
    filter_h_freq: float
    out_npz: str
    out_mat: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["st", "si"], required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--montage", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--subject-list", default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--expected-trials", type=int, default=None)
    parser.add_argument("--min-trials", type=int, default=1)
    parser.add_argument("--strict-balanced", action="store_true")
    parser.add_argument("--no-combine", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resample", type=int, default=200)
    parser.add_argument("--filter-l-freq", type=float, default=1.0)
    parser.add_argument("--filter-h-freq", type=float, default=50.0)
    parser.add_argument("--marker-sfreq", type=float, default=1000.0)
    parser.add_argument("--eog-threshold", type=float, default=3.5)
    parser.add_argument("--muscle-topk", type=int, default=10)
    parser.add_argument("--ica-components", type=int, default=40)
    parser.add_argument("--covert-blocks", nargs="+", type=int, default=DEFAULT_COVERT_BLOCKS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-components", action="store_true")
    return parser.parse_args()


def load_subject_list(args: argparse.Namespace) -> list[str] | None:
    if args.subjects:
        return sorted(args.subjects)
    if not args.subject_list:
        return None
    rows = []
    for line in Path(args.subject_list).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split(",")[0].strip())
    return sorted(rows)


def discover_subjects(raw_root: Path) -> list[str]:
    subjects = set()
    for vhdr in raw_root.rglob("*.vhdr"):
        match = SUBJECT_RE.search(str(vhdr))
        if match:
            subjects.add(match.group(0))
    return sorted(subjects)


def selected_subjects(args: argparse.Namespace) -> list[str]:
    subjects = load_subject_list(args) or discover_subjects(Path(args.raw_root))
    if args.subject_limit is not None:
        subjects = subjects[: args.subject_limit]
    if args.expected_subjects is not None and args.subject_limit is None:
        if len(subjects) != args.expected_subjects:
            raise RuntimeError(f"Expected {args.expected_subjects} subjects, found {len(subjects)}")
    if not subjects:
        raise RuntimeError(f"No subjects found under {args.raw_root}")
    return subjects


def find_vhdr(raw_root: Path, subject_id: str, dataset: str) -> Path:
    exact = []
    if dataset == "st":
        exact.append(raw_root / f"{subject_id}_Filters.vhdr")
    else:
        exact.extend(
            [
                raw_root / subject_id / f"{subject_id}-fmri_BCG_Correction.vhdr",
                raw_root / f"{subject_id}-fmri_BCG_Correction.vhdr",
            ]
        )
    for path in exact:
        if path.exists() and path.with_suffix(".vmrk").exists():
            return path

    candidates = [p for p in raw_root.rglob(f"*{subject_id}*.vhdr") if p.with_suffix(".vmrk").exists()]
    if not candidates:
        raise FileNotFoundError(f"Missing BrainVision vhdr/vmrk pair for {subject_id} under {raw_root}")

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        value = 0
        if dataset == "st" and "filters" in name:
            value -= 10
        if dataset == "si" and "bcg_correction" in name:
            value -= 10
        if dataset == "si" and "fmri" in name:
            value -= 5
        return value, len(str(path)), str(path)

    return sorted(candidates, key=score)[0]


def read_vmrk_word_markers(vmrk_path: Path, marker_sfreq: float) -> list[dict]:
    rows = []
    for line in vmrk_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("Mk") or "=" not in line:
            continue
        parts = line.split("=", 1)[1].split(",")
        if len(parts) < 3:
            continue
        match = re.fullmatch(r"S\s*([1-5])", parts[1].strip())
        if not match:
            continue
        pos = int(parts[2])
        marker_num = int(match.group(1))
        rows.append({"marker_num": marker_num, "pos": pos, "onset": pos / marker_sfreq})

    if not rows:
        raise RuntimeError(f"No S1-S5 stimulus markers found in {vmrk_path}")

    positions = np.array([row["pos"] for row in rows], dtype=float)
    blocks = np.ones(len(rows), dtype=np.int32)
    breaks = np.where(np.diff(positions) / marker_sfreq > 20.0)[0]
    for block_number, break_idx in enumerate(breaks, start=2):
        blocks[break_idx + 1 :] = block_number
    for row, block in zip(rows, blocks):
        row["block"] = int(block)
    return rows


def select_trial_markers(args: argparse.Namespace, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept = []
    prev_pos = None
    dedup_samples = 5.0 * args.marker_sfreq
    for row in rows:
        if prev_pos is not None and row["pos"] - prev_pos < dedup_samples:
            prev_pos = row["pos"]
            continue
        kept.append(row)
        prev_pos = row["pos"]

    covert = [row for row in kept if row["block"] in set(args.covert_blocks)]
    if args.expected_trials is not None and len(covert) != args.expected_trials:
        raise RuntimeError(f"Expected {args.expected_trials} covert trials, got {len(covert)}")
    if len(covert) < args.min_trials:
        raise RuntimeError(f"Expected at least {args.min_trials} covert trials, got {len(covert)}")

    onsets = np.array([row["onset"] for row in covert], dtype=np.float64)
    labels = np.array([row["marker_num"] - 1 for row in covert], dtype=np.int64)
    blocks = np.array([row["block"] for row in covert], dtype=np.int64)

    if args.strict_balanced:
        counts = np.bincount(labels, minlength=5)
        expected_per_class = len(covert) // 5
        if len(covert) % 5 != 0 or not np.all(counts == expected_per_class):
            raise RuntimeError(f"Unbalanced word counts: {counts.tolist()}")
        block_counts = {int(b): int(np.sum(blocks == b)) for b in np.unique(blocks)}
        expected_blocks = {b: expected_per_class for b in args.covert_blocks}
        if block_counts != expected_blocks:
            raise RuntimeError(f"Unexpected covert block counts: {block_counts}")

    return onsets, labels, blocks


def set_montage(info_or_raw, montage_path: str | None) -> None:
    if not montage_path:
        return
    montage_file = Path(montage_path)
    if not montage_file.exists():
        raise FileNotFoundError(f"Montage file not found: {montage_file}")
    montage = mne.channels.read_custom_montage(str(montage_file))
    try:
        info_or_raw.set_montage(montage, verbose=False)
    except Exception:
        info_or_raw.set_montage(montage, on_missing="ignore", verbose=False)


def process_subject(args: argparse.Namespace, subject_id: str) -> SubjectSummary:
    out_root = Path(args.out_root)
    subject_dir = out_root / "subjects"
    mat_dir = out_root / "mat"
    fig_dir = out_root / "ica_fig"
    subject_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_npz = subject_dir / f"{subject_id}.npz"
    out_eeg = mat_dir / f"{subject_id}_eeg_trials.mat"
    out_lbl = mat_dir / f"{subject_id}_trial_labels.mat"
    vhdr = find_vhdr(Path(args.raw_root), subject_id, args.dataset)
    vmrk = vhdr.with_suffix(".vmrk")

    if out_npz.exists() and out_eeg.exists() and out_lbl.exists() and not args.force:
        with np.load(out_npz, allow_pickle=True) as data:
            has_filter_metadata = "filter_l_freq" in data.files and "filter_h_freq" in data.files
            if has_filter_metadata:
                stored_l_freq = float(data["filter_l_freq"])
                stored_h_freq = float(data["filter_h_freq"])
                if not (
                    np.isclose(stored_l_freq, args.filter_l_freq)
                    and np.isclose(stored_h_freq, args.filter_h_freq)
                ):
                    raise RuntimeError(
                        f"{subject_id} existing output filter "
                        f"{stored_l_freq:g}-{stored_h_freq:g} Hz does not match requested "
                        f"{args.filter_l_freq:g}-{args.filter_h_freq:g} Hz"
                    )
            elif not (np.isclose(args.filter_l_freq, 1.0) and np.isclose(args.filter_h_freq, 50.0)):
                raise RuntimeError(
                    f"{subject_id} existing output lacks filter metadata; use --force or a new out-root"
                )
            labels = data["Y"].astype(np.int64)
            blocks = data["block"].astype(np.int64)
            return SubjectSummary(
                subject_id=subject_id,
                dataset=args.dataset,
                raw_file=str(vhdr),
                x_shape=tuple(int(v) for v in data["X"].shape),
                y_counts=np.bincount(labels, minlength=5).astype(int).tolist(),
                block_counts={int(b): int(np.sum(blocks == b)) for b in np.unique(blocks)},
                eog_idx=data["eog_idx"].astype(int).tolist(),
                muscle_idx=data["muscle_idx"].astype(int).tolist(),
                exclude_idx=data["exclude_idx"].astype(int).tolist(),
                clip_limits=tuple(float(v) for v in data["clip_limits"]),
                filter_l_freq=args.filter_l_freq,
                filter_h_freq=args.filter_h_freq,
                out_npz=str(out_npz),
                out_mat=str(out_eeg),
            )

    marker_rows = read_vmrk_word_markers(vmrk, args.marker_sfreq)
    onsets, labels, blocks = select_trial_markers(args, marker_rows)

    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    raw.resample(sfreq=args.resample, verbose=False)
    set_montage(raw, args.montage)
    if "ECG" in raw.ch_names:
        raw.drop_channels(["ECG"])
    sfreq = int(round(raw.info["sfreq"]))

    events = np.column_stack(
        [np.rint(onsets * sfreq).astype(np.int64), np.zeros(len(onsets), dtype=np.int64), labels + 1]
    )
    event_id = {f"W{i}": i for i in range(1, 6)}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-1.0,
        tmax=11.0 - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )
    data = epochs.get_data(picks="eeg").astype(np.float32, copy=False)
    expected_shape_tail = (63, sfreq * 12)
    if data.shape[1:] != expected_shape_tail:
        raise RuntimeError(f"{subject_id} unexpected epoch shape: {data.shape}")

    lo, hi = np.percentile(data, [0.1, 99.9])
    data = np.clip(data, lo, hi).astype(np.float32, copy=False)

    info = mne.create_info(list(epochs.ch_names), sfreq, "eeg")
    set_montage(info, args.montage)
    epo = mne.EpochsArray(data, info, verbose=False, tmin=-1.0)
    epo.filter(args.filter_l_freq, args.filter_h_freq, verbose=False)
    epo.set_eeg_reference("average", projection=True, verbose=False)
    epo.apply_proj(verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=min(args.ica_components, len(epo.ch_names) - 1),
        random_state=42,
        max_iter="auto",
    )
    ica.fit(epo, verbose=False)

    if args.plot_components:
        import matplotlib.pyplot as plt

        ica.plot_components(res=20, ncols=8, nrows=5, show=False)
        plt.savefig(fig_dir / f"{subject_id}.jpg", dpi=120)
        plt.close("all")

    try:
        eog_idx, _ = ica.find_bads_eog(
            epo, ch_name=["Fp1", "Fp2"], threshold=args.eog_threshold, verbose=False
        )
    except Exception:
        eog_idx = []
    eog_idx = [int(v) for v in eog_idx]

    try:
        muscle_all_idx, muscle_scores = ica.find_bads_muscle(epo, verbose=False)
    except Exception:
        muscle_all_idx, muscle_scores = [], np.array([])
    muscle_all_idx = [int(v) for v in muscle_all_idx]
    if len(muscle_all_idx) > args.muscle_topk:
        scores = np.asarray(muscle_scores, dtype=float)
        muscle_idx = sorted(muscle_all_idx, key=lambda idx: scores[idx], reverse=True)[: args.muscle_topk]
    else:
        muscle_idx = muscle_all_idx

    exclude_idx = sorted(set(eog_idx).union(muscle_idx))
    ica.exclude = exclude_idx
    ica.apply(epo, verbose=False)

    x_uv = epo.get_data(units="uV").astype(np.float32, copy=False)
    if x_uv.shape[1:] != expected_shape_tail:
        raise RuntimeError(f"{subject_id} unexpected cleaned shape: {x_uv.shape}")

    np.savez_compressed(
        out_npz,
        X=x_uv,
        Y=labels.astype(np.int64),
        block=blocks.astype(np.int64),
        subject_id=np.array(subject_id),
        dataset=np.array(args.dataset),
        sfreq=np.array(sfreq),
        ch_names=np.array(epo.ch_names, dtype=object),
        time_vec=epo.times.astype(np.float64),
        eog_idx=np.array(eog_idx, dtype=np.int64),
        muscle_idx=np.array(muscle_idx, dtype=np.int64),
        exclude_idx=np.array(exclude_idx, dtype=np.int64),
        clip_limits=np.array([lo, hi], dtype=np.float64),
        filter_l_freq=np.array(args.filter_l_freq, dtype=np.float64),
        filter_h_freq=np.array(args.filter_h_freq, dtype=np.float64),
    )
    savemat(
        out_eeg,
        {
            "eeg_data": x_uv,
            "sfreq": sfreq,
            "ch_names": np.array(epo.ch_names, dtype=object),
            "time_vec": epo.times.astype(np.float64),
            "baseline_applied": np.array(False),
            "filter_l_freq": np.array(args.filter_l_freq, dtype=np.float64),
            "filter_h_freq": np.array(args.filter_h_freq, dtype=np.float64),
            "units": "uV",
            "source": f"fast_benchmark_{args.dataset}",
        },
        do_compression=True,
    )
    savemat(
        out_lbl,
        {"word_label": labels.astype(np.int32) + 1, "block": blocks.astype(np.int32)},
        do_compression=True,
    )

    return SubjectSummary(
        subject_id=subject_id,
        dataset=args.dataset,
        raw_file=str(vhdr),
        x_shape=tuple(int(v) for v in x_uv.shape),
        y_counts=np.bincount(labels, minlength=5).astype(int).tolist(),
        block_counts={int(b): int(np.sum(blocks == b)) for b in np.unique(blocks)},
        eog_idx=eog_idx,
        muscle_idx=muscle_idx,
        exclude_idx=exclude_idx,
        clip_limits=(float(lo), float(hi)),
        filter_l_freq=float(args.filter_l_freq),
        filter_h_freq=float(args.filter_h_freq),
        out_npz=str(out_npz),
        out_mat=str(out_eeg),
    )


def default_dataset_name(args: argparse.Namespace) -> str:
    threshold = str(args.eog_threshold).replace(".", "p")
    return f"fast_benchmark_k{args.muscle_topk}_eogt{threshold}_{args.dataset}"


def combine_subjects(args: argparse.Namespace, summaries: list[SubjectSummary]) -> Path | None:
    if args.no_combine:
        return None
    out_root = Path(args.out_root)
    dataset_path = out_root / f"{args.dataset_name or default_dataset_name(args)}.npz"
    xs, ys, blocks, trial_subjects = [], [], [], []
    for summary in summaries:
        with np.load(out_root / "subjects" / f"{summary.subject_id}.npz", allow_pickle=True) as data:
            x = data["X"]
            xs.append(x)
            ys.append(data["Y"])
            blocks.append(data["block"])
            trial_subjects.extend([summary.subject_id] * x.shape[0])
    x_all = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    block_all = np.concatenate(blocks, axis=0).astype(np.int64, copy=False)
    np.savez_compressed(
        dataset_path,
        X=x_all,
        Y=y_all,
        block=block_all,
        subject_ids=np.array([s.subject_id for s in summaries], dtype=object),
        trial_subject=np.array(trial_subjects, dtype=object),
        trials_per_subject=np.array([s.x_shape[0] for s in summaries], dtype=np.int32),
        sfreq=np.array(args.resample),
        filter_l_freq=np.array(args.filter_l_freq, dtype=np.float64),
        filter_h_freq=np.array(args.filter_h_freq, dtype=np.float64),
        units="uV",
        preprocessing=f"fast_benchmark_{args.dataset}",
    )
    return dataset_path


def main() -> None:
    args = parse_args()
    subjects = selected_subjects(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    print(f"Subjects: {len(subjects)}")
    print(f"Raw root: {args.raw_root}")
    print(f"Out root: {args.out_root}")
    print(f"Workers: {args.workers}")
    print(f"Filter: {args.filter_l_freq:g}-{args.filter_h_freq:g} Hz")

    if args.workers <= 1:
        summaries = [process_subject(args, subject_id) for subject_id in subjects]
    else:
        with mp.Pool(args.workers) as pool:
            summaries = pool.map(partial(process_subject, args), subjects)

    dataset_path = combine_subjects(args, summaries)
    summary_payload = {
        "dataset_path": str(dataset_path) if dataset_path else None,
        "n_subjects": len(summaries),
        "filter_l_freq": args.filter_l_freq,
        "filter_h_freq": args.filter_h_freq,
        "subjects": [asdict(s) for s in summaries],
    }
    summary_path = out_root / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    if dataset_path:
        print(f"Saved dataset: {dataset_path}")


if __name__ == "__main__":
    main()
