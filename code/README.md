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
| `aa_prep_st_5u_c1.m` | Standalone EEG raw-to-epoch preprocessing for five-utterance trials. |
| `aa_prep_st_5u_c2.m` | Standalone EEG ICA stage. |
| `aa_prep_st_5u_c3.m` / `aa_prep_st_5u_c3b.m` | Standalone EEG IC rejection and release cleaning. |
| `aa_prep_si_5u_c1.m` | Simultaneous EEG-fMRI post-AAS/BCG epoch preprocessing. |
| `aa_prep_si_5u_c2.m` | Simultaneous EEG-fMRI ICA stage. |
| `aa_prep_si_5u_c3.m` / `aa_prep_si_5u_c3b.m` | Simultaneous EEG-fMRI IC rejection and release cleaning. |
| `local_psd_4plots.m` | Figure 2 PSD/topomap panels from the ICA-based release derivatives. |
| `local_s0016_ic_gfp.m` | Representative S0016 ICA panels for Figure 2. |
| `local_erp_5phrases_C3_4panels.m` | Phrase ERP panels for Figure 3. |
| `local_grandmean_gfp.m` | Grand-mean GFP panels for Figure 3. |
| `scan_iclabel_all_subjects.m` | ICLabel distribution scan and representative-subject selection. |
| `QC_PLAN.md` | Technical Validation figure and QC plan. |

## Data Roots Used by Manuscript Figures

- Standalone EEG release derivative: `prep_st_5u/`
- Simultaneous EEG-fMRI release derivative: `prep_si_5u/`
- Figure 2 source panels are generated from the release derivatives using the
  scripts in this folder.

Set these paths inside the scripts before running them on a local copy of the
dataset.

## Dependencies

- MATLAB R2024b+ or R2025b
- EEGLAB with ICLabel
- BrainVision Analyzer output for simultaneous EEG-fMRI after AAS/BCG
- Python 3 for selected table, audio, and metadata utilities
