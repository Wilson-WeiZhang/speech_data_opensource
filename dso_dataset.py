"""
DSO EEG Dataset Loader — used by the FAST model (Jiang et al., JBHI 2024).
GitHub: https://github.com/Jiang-Muyun/FAST

Converts raw BrainVision EEG to epoched .mat files (10 s trials) and provides
a load_subject() function for training/evaluation. This is a separate pipeline
from the MATLAB preprocessing scripts (2 s utterance-level epochs).

Requirements: scipy, numpy, mne, pandas, matplotlib
"""
from functools import partial
import scipy
import numpy as np
import mne
import os
import glob
import pandas as pd
import multiprocessing as mp

def find_available_path(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            return folder
    raise FileNotFoundError('None of the given paths exist: ' + str(folder_list))

DSO_ROOT = find_available_path([
    './data/',                  # <-- SET YOUR DSO DATA PATH
    '../data/',
])

EVENTS_ID = {
    'Loud/Go-there': 0,
    'Loud/Distract-target': 1,
    'Loud/Follow-me': 2,
    'Loud/Explore-here': 3,
    'Loud/Terminate': 4,
    'Silent/Go-there': 5,
    'Silent/Distract-target': 6,
    'Silent/Follow-me': 7,
    'Silent/Explore-here': 8,
    'Silent/Terminate': 9,
}

CH_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5',
    'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7',
    'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz'
]

CH_NAMES_NO_ECG = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5',
    'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7',
    'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz'
]

def load_subject(mat_path, task, kind, bandpass=(1, 70), slide_window=False, whitening=False):
    """Load a single subject's epoched data from .mat file."""
    assert task in ['loud', 'silent', 'all'], 'task must be one of [loud, silent, all]'
    assert kind in ['EEG-Only', 'EEG-fMRI', 'EEG-fMRI-BCG'], 'kind must be one of [EEG-Only, EEG-fMRI, EEG-fMRI-BCG]'

    sfreq = 250
    data = scipy.io.loadmat(mat_path)
    usid = os.path.basename(mat_path)[:-4]
    epoched = data['epoched']
    label = data['label'].reshape(-1)

    if task == 'loud':
        epoched = epoched[label < 5]
        label = label[label < 5]
    elif task == 'silent':
        epoched = epoched[label >= 5]
        label = label[label >= 5]
        label -= 5

    alpha = 0.5
    epoched = np.clip(epoched, np.percentile(epoched, alpha), np.percentile(epoched, 100 - alpha))

    if bandpass is not None:
        n_seg, n_channels, n_times = epoched.shape
        ch_names = ['ch%i' % i for i in range(n_channels)]
        ch_types = 'eeg'
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        events = np.array([[i, 0, 1] for i in range(n_seg)])
        epochs = mne.EpochsArray(epoched, info, events=events)
        epochs.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
        epochs.resample(128, npad='auto')
        epoched = epochs.get_data().astype(np.float32)
        sfreq = 128

    if slide_window:
        mini_Epoched, mini_Label = [], []
        for i in range(epoched.shape[0]):
            mini_Epoched.append(epoched[i, :, 0:sfreq*2])
            mini_Epoched.append(epoched[i, :, sfreq*2:sfreq*4])
            mini_Epoched.append(epoched[i, :, sfreq*4:sfreq*6])
            mini_Epoched.append(epoched[i, :, sfreq*6:sfreq*8])
            mini_Epoched.append(epoched[i, :, sfreq*8:sfreq*10])
            for ii in range(5):
                mini_Label.append(label[i])
        mini_Epoched, mini_Label = np.array(mini_Epoched), np.array(mini_Label)
        epoched, label = mini_Epoched, mini_Label

    if whitening:
        # Preserve zero-point; normalize by std only
        epoched = epoched / epoched.std(axis=(0, -1), keepdims=True)

    return usid, epoched, label


def load_vmrk(fn, sfreq):
    """Parse BrainVision .vmrk marker file."""
    assert fn.endswith('.vmrk')
    _marker, _onset, _pos = [], [], []
    last = None
    with open(fn, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('Mk'):
                continue
            if line.startswith('Mk1='):
                continue
            _, marker, pos, _, _ = line.split(',')
            if marker[0] == 'S':
                if marker == last:
                    continue
                else:
                    last = marker
            _marker.append(marker)
            _pos.append(int(pos))
            _onset.append(int(pos) / sfreq)
    events = pd.DataFrame(zip(_onset, _pos, _marker), columns=['onset', 'pos', 'marker'])
    return events


def process_marker(mkr):
    """Assign overt/covert labels based on block order (odd=overt, even=covert)."""
    buf = []
    marker_count = 0
    mapping = {
        'S  1': 'Go-there',
        'S  2': 'Distract-target',
        'S  3': 'Follow-me',
        'S  4': 'Explore-here',
        'S  5': 'Terminate',
    }
    for index, row in mkr.iterrows():
        if row.marker.startswith('S  '):
            block_id = marker_count // 20
            marker_count += 1
            if block_id in [0, 2, 4, 6, 8]:
                buf.append([row.onset, row.pos, 'Loud/' + mapping[row.marker], int(row.marker[-1]) - 1])
            else:
                buf.append([row.onset, row.pos, 'Silent/' + mapping[row.marker], int(row.marker[-1]) - 1 + 5])
    sessions = pd.DataFrame(buf, columns=['onset', 'pos', 'marker', 'label'])
    return sessions


def convert_subject_to_mat(i, kind, return_raw=False, return_epochs=False):
    """Convert one subject's raw BrainVision EEG to an epoched .mat file."""
    assert kind in ['EEG-Only', 'EEG-fMRI', 'EEG-fMRI-BCG']
    version = 'V2'
    if kind == 'EEG-Only' and i == 24:
        return

    ID = 'S00%02d' % (i)
    if kind == 'EEG-Only':
        base_name = '_Filters'
    if kind == 'EEG-fMRI':
        base_name = '-fmri_SAC_BR_Filters'
    if kind == 'EEG-fMRI-BCG':
        base_name = '-fmri_BCG_Correction'

    vhdr = f'{DSO_ROOT}/EEG_PP/{version}/{ID}{base_name}.vhdr'
    vmrk = f'{DSO_ROOT}/EEG_PP/{version}/{ID}{base_name}.vmrk'
    mat_path = f'{DSO_ROOT}/EEG_PP/{version}_Epoched/{ID}-{kind}.mat'

    if not os.path.exists(vhdr):
        return

    mkr = load_vmrk(vmrk, sfreq=1000)
    try:
        assert mkr[mkr.marker == 'S 42'].shape[0] == 200, vmrk
        assert mkr[mkr.marker == 'S  1'].shape[0] == 40, vmrk
        assert mkr[mkr.marker == 'S  2'].shape[0] == 40, vmrk
        assert mkr[mkr.marker == 'S  3'].shape[0] == 40, vmrk
        assert mkr[mkr.marker == 'S  4'].shape[0] == 40, vmrk
        assert mkr[mkr.marker == 'S  5'].shape[0] == 40, vmrk
    except AssertionError as err:
        import warnings
        warnings.warn(f'Marker count mismatch in {vmrk}: {err}')

    mkr = process_marker(mkr)

    raw = mne.io.read_raw_brainvision(vhdr, preload=True).resample(sfreq=250)
    # Optional montage: BrainVision BC-MR-64 electrode positions (.bvef file)
    montage_file = os.path.join(os.path.dirname(__file__), '..', 'meta_data', 'BC-MR-64.bvef')
    if os.path.exists(montage_file):
        raw.set_montage(mne.channels.read_custom_montage(montage_file), verbose=False)
    sfreq = int(raw.info['sfreq'])
    mkr.pos = (sfreq * mkr.onset).astype(int)

    if return_raw:
        return raw

    events = np.array([[row.pos, 0, row.label] for _, row in mkr.iterrows()], dtype=int)
    tmin, tmax = 0, 10 - 1 / 250
    epochs = mne.Epochs(raw, events, EVENTS_ID, tmin, tmax, baseline=None, preload=True)

    if return_epochs:
        return epochs

    if not os.path.exists(mat_path):
        save_dict = {
            'sfreq': sfreq,
            'name': f'{ID}-{kind}',
            'epoched': epochs.get_data().astype(np.float32),
            'marker': np.array(mkr.marker),
            'label': np.array(mkr.label).astype(np.uint8),
            'onset': np.array(mkr.onset).astype(np.float32),
            'marker_legend': EVENTS_ID,
            'channel_names': np.array(raw.ch_names),
        }
        scipy.io.savemat(mat_path, save_dict)


if __name__ == '__main__':
    n_workers = min(24, os.cpu_count() or 1)
    with mp.Pool(n_workers) as pool:
        pool.map(partial(convert_subject_to_mat, kind='EEG-Only'), range(9, 75))
    with mp.Pool(n_workers) as pool:
        pool.map(partial(convert_subject_to_mat, kind='EEG-fMRI'), range(9, 75))
        pool.map(partial(convert_subject_to_mat, kind='EEG-fMRI-BCG'), range(9, 75))
