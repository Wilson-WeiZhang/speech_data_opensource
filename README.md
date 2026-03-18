# COSMO: Covert and Overt Speech Multimodal Open Dataset

This repository contains the preprocessing pipelines, figure generation scripts, and analysis code for the data descriptor paper:

**COSMO, a covert and overt speech multimodal open dataset with EEG and fMRI**

Wei Zhang<sup>1,2,☨</sup>, Muyun Jiang<sup>1,☨</sup>, Kok Ann Colin Teo<sup>2,3,4,5</sup>, Raghavan Bhuvanakantham<sup>2</sup>, Chuan Huat Vince Foo<sup>6</sup>, Jia Lu<sup>6,7</sup>, Balázs Gulyás<sup>8,9</sup>, & Cuntai Guan<sup>1,10,\*</sup>

<sup>1</sup> College of Computing and Data Science, Nanyang Technological University, Singapore
<sup>2</sup> Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore
<sup>3</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore
<sup>4</sup> IGP-Neuroscience, Interdisciplinary Graduate Programme, Nanyang Technological University, Singapore
<sup>5</sup> Division of Neurosurgery, National University Health System, Singapore
<sup>6</sup> DSO National Laboratories, Singapore
<sup>7</sup> Yong Loo Lin School of Medicine, National University of Singapore, Singapore
<sup>8</sup> Hungarian Research Network (HUN-REN), Hungary
<sup>9</sup> Department of Clinical Neuroscience, Karolinska Institutet, Stockholm, Sweden
<sup>10</sup> Center of AI in Medicine (C-AIM), Nanyang Technological University, Singapore

<sup>☨</sup> These authors contributed equally to this study
<sup>\*</sup> Corresponding author

## Overview

COSMO is a publicly available multimodal dataset of covert and overt speech production from 58 healthy adults performing a five-phrase command task. The dataset includes:

- **Standalone EEG** (N = 58) — 64-channel, 1 kHz
- **Simultaneous EEG–fMRI** (N = 51) — 64-channel MR-compatible EEG inside a 3T scanner
- **Task fMRI** (N = 49) — gradient-echo EPI
- **T1 structural MRI** (N = 58) — 1 mm isotropic MPRAGE
- **Speech timing annotations** (N = 53) — onset/offset from overt trials

Each participant produced up to 1,000 covert and 1,000 overt utterances across two experiments, yielding more than 108,000 labelled utterances across the dataset.

## Citation

If you use this code or dataset in your research, please cite our paper:

```
Zhang, W., Jiang, M., Teo, K.A.C., Bhuvanakantham, R., Foo, C.H.V., Lu, J., Gulyás, B., & Guan, C.
COSMO, a covert and overt speech multimodal open dataset with EEG and fMRI. Scientific Data (2026).
```

## Data Availability

The full dataset is publicly available at:

- **Dataset repository**: [to be added upon publication]

Related code repositories:
1. **This repository** — Preprocessing pipelines and data quality validation
2. **EEG–fMRI source localization**: https://github.com/Wilson-WeiZhang/Covert-Speech-Encoding
3. **EEG phrase classification (FAST)**: https://github.com/Jiang-Muyun/FAST

## Code Description

### EEG Preprocessing Pipeline

The standalone and simultaneous EEG data are preprocessed in three stages using EEGLAB in MATLAB:

| Stage | Standalone EEG | Simultaneous EEG |
|-------|----------------|-------------------|
| **Stage 1** | Resample 250 Hz → bandpass 1–100 Hz → notch 49–51 Hz → epoch 5 utterances | Resample 250 Hz → bandpass 1–30 Hz (no notch) → epoch 5 utterances |
| **Stage 2** | Extended infomax ICA → ICLabel → MARA | Same |
| **Stage 3** | Remove artifact > 0.9 and brain < 0.1 ICs → 56 channels → re-reference | Keep brain > 0.9 ICs only → 56 channels → re-reference |

| Script | Description |
|--------|-------------|
| `standalone_eeg_prep_stage1.m` | ST: raw → filtered, epoched trials |
| `standalone_eeg_prep_stage2.m` | ST: ICA decomposition + ICLabel + MARA |
| `standalone_eeg_prep_stage3.m` | ST: artifact IC removal → clean 56-channel data |
| `simultaneous_eeg_prep_stage1.m` | SI: BCG-corrected → filtered, epoched trials |
| `simultaneous_eeg_prep_stage2.m` | SI: ICA decomposition + ICLabel + MARA |
| `simultaneous_eeg_prep_stage3.m` | SI: brain IC selection → clean 56-channel data |
| `run_eeg_prep_stages2_3.m` | Orchestration script: runs Stage 2 (parallel) + Stage 3 for both ST and SI |

### fMRI Preprocessing Pipeline

| Script | Description |
|--------|-------------|
| `fmri_preprocess_spm12.m` | SPM12 pipeline: slice timing → realignment → coregistration → segmentation → normalization → smoothing (4 mm FWHM) |

### Audio Onset Detection

| Script | Description |
|--------|-------------|
| `gen_onset_detection.m` | Three-stage pipeline: HDF5→MAT conversion → Hilbert envelope + outlier rejection → threshold-based onset/offset detection |

### Technical Validation Figures

| Script | Description | Output |
|--------|-------------|--------|
| `gen_fig2.m` | EEG signal quality: grand-mean PSD, topomaps, sample ICA components | Figure 2 |
| `gen_fig3.m` | Phrase-specific ERPs: GFP and Cz waveforms for 5 command phrases | Figure 3 |
| `gen_fig4.py` | fMRI data quality: framewise displacement + temporal SNR maps | Figure 4 |
| `gen_fig5.m` | Audio recording quality: onset/offset distributions + waveform envelopes | Figure 5 |

### Analysis Utilities

| Script | Description |
|--------|-------------|
| `gen_iclabel_stats.m` | ICLabel classification statistics per subject |
| `dso_dataset.py` | Dataset loader for the [FAST](https://github.com/Jiang-Muyun/FAST) model (Jiang et al., JBHI 2024); separate 10 s epoch pipeline for EEG classification |

## Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **MATLAB** | R2024b+ | EEG/fMRI preprocessing, figure generation |
| **EEGLAB** | 2024.0+ | EEG data loading, filtering, ICA |
| **ICLabel** | (EEGLAB plugin) | Automated IC classification |
| **MARA** | (EEGLAB plugin) | IC artifact detection |
| **SPM12** | v7771+ | fMRI preprocessing |
| **Python** | 3.8+ | fMRI QC figures, dataset loader |
| **NumPy, Matplotlib, nibabel** | — | Python dependencies for `gen_fig4.py` |
| **MNE-Python, SciPy, pandas** | — | Python dependencies for `dso_dataset.py` |

## Usage

1. Set the data and EEGLAB paths in each script (marked with `<-- SET YOUR ...`)
2. Run the EEG preprocessing pipeline:
   ```matlab
   addpath('/path/to/eeglab2024');
   cd('/path/to/code');

   % Stage 1 (run separately for ST and SI)
   standalone_eeg_prep_stage1;
   simultaneous_eeg_prep_stage1;

   % Stages 2-3 (orchestrated, uses parfor for ICA)
   run_eeg_prep_stages2_3;
   ```
3. Run the fMRI preprocessing pipeline:
   ```matlab
   fmri_preprocess_spm12;
   ```
4. Generate validation figures:
   ```matlab
   gen_fig2;
   gen_fig3;
   gen_fig5;
   ```
   ```bash
   python gen_fig4.py
   ```

## Acknowledgements

This research is supported by the National Research Foundation, Singapore, and DSO National Laboratories under the AI Singapore Programme (AISG Award No: AISG2-RP-2020-016) and the National Medical Research Council under the OF-YIRG Grant (OFYIRG21jun-0058).

Thanks also to Prof. Victoria Leong for mentorship, Wei Khang Jeremy Sim for writing the script for stimuli, LaiGuan Fong for helping with data collection, and Rong Hui Jonathan Chua and Parasuraman Padmanabhan for project administration.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2026 Zhang, Jiang et al.

You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit and indicate if changes were made.

The full license text is available at: https://creativecommons.org/licenses/by-nc/4.0/

## Contact

For any questions regarding this code repository, please contact:

**Corresponding author**: Prof. Cuntai Guan
College of Computing and Data Science
Nanyang Technological University, Singapore
Email: ctguan@ntu.edu.sg

**First author**: Dr. Wei Zhang
Cognitive Neuroimaging Centre
Nanyang Technological University, Singapore
Email: wilson.zhangwei@ntu.edu.sg
