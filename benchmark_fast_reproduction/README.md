# FAST Benchmark Reproduction

This auxiliary folder contains the benchmark-reproduction workflow for the COSMO
standalone EEG covert-speech benchmark. It follows the preprocessing and
evaluation setting used in the original FAST study and is provided as an
additional reference benchmark for the dataset.

This folder contains code only. Benchmark NPZ derivatives are distributed
with the dataset package, not in this GitHub repository.

## Workflow

```text
released raw standalone EEG
-> FAST-compatible whole-trial preprocessing
-> subject-level benchmark derivatives
-> FAST aggregation
-> SVM and EEGNet comparison
-> benchmark summary table
```

## Expected Reference Result

For the 58-participant five-class covert-speech benchmark:

| Model | Mean balanced accuracy | SD |
|---|---:|---:|
| FAST fine-tuned | 34.86% | 10.89% |
| Linear SVM | 26.88% | 7.85% |
| EEGNet | 25.95% | 6.92% |

Chance level is 20%.

## Directory Map

| Directory | Purpose |
|---|---|
| `preprocess/` | Raw BrainVision EEG to whole-trial NPZ/MAT benchmark derivatives. |
| `reproduce_fast/` | FAST aggregation helpers, including the 58-participant aggregation patch. |
| `benchmarks/` | SVM, EEGNet, and benchmark-summary scripts. |
| `config/` | Example path configuration for remote execution. |

## Main Entry Points

| Script | Use |
|---|---|
| `preprocess/benchmark_eeg_preprocess.py` | General benchmark preprocessing entrypoint. |
| `preprocess/benchmark_st_preprocess.py` | Standalone EEG wrapper. |
| `preprocess/run_benchmark_st_aa.sh` | Example remote wrapper for standalone benchmark preprocessing. |
| `reproduce_fast/launch_fast_58_benchmark_chain.sh` | Example raw-to-FAST benchmark launch script. |
| `reproduce_fast/run_fast_zipparams_npz.py` | Runs FAST zip parameters and patches final aggregation to 58 participants. |
| `reproduce_fast/aggregate_fast_all_subjects.py` | Aggregate helper for FAST outputs. |
| `benchmarks/aa_svm_100trial_blocks.py` | Block-aware SVM benchmark. |
| `benchmarks/aa_eegnet_100trial_folds.py` | Block-aware EEGNet benchmark. |
| `benchmarks/summarize_benchmark_results.py` | Writes the benchmark summary table. |

## Notes

- The workflow uses de-identified participant IDs only.
- The benchmark derivatives are 12-second whole-trial epochs for the standalone
  EEG decoding benchmark.
- FAST model training requires the FAST implementation and its dependencies.
