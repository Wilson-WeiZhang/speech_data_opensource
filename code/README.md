# code: manuscript preprocessing, validation, and figures

This directory contains the primary manuscript code for the released dataset.
It covers the EEG preprocessing described in the main text, the Technical
Validation analyses, and the figure-generation scripts used for the manuscript
results.

## Main Scope

- Standalone EEG preprocessing, including ICLabel-based component rejection,
  ECG-channel removal, average reference, and release-format EEGLAB outputs.
- Simultaneous EEG-fMRI preprocessing after BrainVision Analyzer AAS/BCG
  correction, including the SI-specific 1--30 Hz filtering and component
  rejection used for the manuscript signal-quality figures.
- Figure 2 and Figure 3 generation from the manuscript release derivatives.
- fMRI, audio, metadata, and table-generation scripts used by the Scientific
  Data manuscript.

## Key EEG Entry Points

| File | Use |
|---|---|
| `standalone_eeg_prep_stage1.m` | Standalone EEG raw-to-epoch preprocessing for five-utterance trials. |
| `standalone_eeg_prep_stage2.m` | Standalone EEG ICA stage. |
| `standalone_eeg_prep_stage3.m` | Standalone EEG IC rejection and 63-channel release cleaning; ECG is the only channel removed. |
| `simultaneous_eeg_prep_stage1.m` | Simultaneous EEG-fMRI post-AAS/BCG epoch preprocessing. |
| `simultaneous_eeg_prep_stage2.m` | Simultaneous EEG-fMRI ICA stage. |
| `simultaneous_eeg_prep_stage3.m` | Simultaneous EEG-fMRI IC rejection and 63-channel release cleaning; ECG is the only channel removed. |
| `gen_fig2.m` | Figure 2 PSD/topomap source panels from the 63-channel release derivatives. |
| `gen_fig3.m` | Figure 3 grand-mean GFP and Cz ERP source panels from the 63-channel release derivatives. |
| `gen_iclabel_stats.m` | ICLabel distribution summary for Technical Validation. |

## Data Roots Used by Manuscript Figures

- Standalone EEG release derivative: `prep_st_eeg/`
- Simultaneous EEG-fMRI release derivative: `prep_si_eeg/`
- Figure 2 and Figure 3 source panels are generated from the release
  derivatives using `gen_fig2.m` and `gen_fig3.m`.

Set these paths inside the scripts before running them on a local copy of the
dataset.

## Dependencies

- MATLAB R2024b+ or R2025b
- EEGLAB with ICLabel
- BrainVision Analyzer output for simultaneous EEG-fMRI after AAS/BCG
- Python 3 for selected table, audio, and metadata utilities
