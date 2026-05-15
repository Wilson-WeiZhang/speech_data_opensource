# COSMO Open Code

This repository contains the public code accompanying the data descriptor:

**COSMO, a covert and overt speech multimodal open dataset with EEG and fMRI**

The repository contains code only. Raw data, metadata tables, and benchmark NPZ
files are distributed through the dataset
repository described in the manuscript.

## Repository Structure

```text
.
|-- code/
|   |-- standalone_eeg_prep_stage1.m
|   |-- standalone_eeg_prep_stage2.m
|   |-- standalone_eeg_prep_stage3.m
|   |-- simultaneous_eeg_prep_stage1.m
|   |-- simultaneous_eeg_prep_stage2.m
|   |-- simultaneous_eeg_prep_stage3.m
|   |-- gen_fig2.m
|   |-- gen_fig3.m
|   |-- gen_fig4.py
|   |-- gen_fig5.m
|   `-- ...
`-- benchmark_fast_reproduction/    # optional benchmark reproduction
    |-- preprocess/
    |-- reproduce_fast/
    |-- benchmarks/
    `-- config/
```

## Main Manuscript Code

`code/` contains the primary manuscript-facing code:

- ICA-based standalone EEG preprocessing.
- ICA-based simultaneous EEG-fMRI preprocessing after AAS/BCG correction.
- ICLabel-based component rejection and release cleaning.
- EEG signal-quality and ERP figure scripts.
- fMRI, audio, and metadata validation scripts.

These scripts support the main Methods, Technical Validation, and figure results
in the manuscript.

## Optional FAST Benchmark

`benchmark_fast_reproduction/` is an auxiliary folder for reproducing the
standalone EEG decoding benchmark reported in Technical Validation. It follows
the preprocessing and evaluation setting used in the original FAST study,
starting from the released raw standalone EEG recordings and generating
benchmark derivatives for FAST, SVM, and EEGNet evaluation.

Under this reference setting, the 58-participant five-class covert-speech
benchmark produced:

- FAST fine-tuned balanced accuracy: 34.86 +/- 10.89%.
- Linear SVM balanced accuracy: 26.88 +/- 7.85%.
- EEGNet balanced accuracy: 25.95 +/- 6.92%.

Chance level is 20% for the five-class task.

## Requirements

Main manuscript code:

- MATLAB R2024b or newer.
- EEGLAB with ICLabel.
- SPM12 for fMRI preprocessing.
- Python 3.8+ with NumPy, Matplotlib, nibabel, pandas, and SciPy for selected
  scripts.

Optional FAST benchmark:

- Python 3.10+.
- MNE-Python, NumPy, SciPy, scikit-learn, PyTorch, and the dependencies required
  by the FAST implementation.
- Access to the original FAST codebase for FAST model training.

## Usage

For manuscript preprocessing and figures, see:

```text
code/README.md
```

For the optional FAST benchmark, see:

```text
benchmark_fast_reproduction/README.md
```

Each script contains path placeholders that should be changed to match the local
data layout before running.

## Citation

If you use this code or dataset, please cite the associated COSMO data
descriptor manuscript. A permanent dataset DOI will be added after acceptance.

## Contact

For questions about the dataset or code, contact the corresponding author listed
in the manuscript.
