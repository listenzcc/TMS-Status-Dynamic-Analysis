"""
File: source-estimation-on-eeg.py
Author: Chuncheng Zhang
Date: 2026-06-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation on EEG dataset.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-20 ------------------------
# Requirements and constants
from tqdm.auto import tqdm
import pandas as pd
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

import mat73

from util.easy_imports import *

# %%
DATA_DIR = Path('./data/eeg-mat')
OUTPUT_DIR = Path('./data/eeg-stc')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%

# %%
# Prepare mne
fs_dir = fetch_fsaverage()
subjects_dir = os.path.dirname(fs_dir)

logger.info(f'Got {fs_dir=}, {subjects_dir=}')

# %% ---- 2026-06-20 ------------------------
# Function and class
ch_names = [e.split('\t')[1]
            for e in open(DATA_DIR / 'chan_labels.txt').read().split('\n') if '\t' in e]
times = [float(e) / 1000 for e in open(DATA_DIR /
                                       'times_ms.txt').read().split()]
sfreq = 1000  # Hz


def read_eeg_mat_to_epochs(src):
    logger.info(f'Read {src=}')
    mat = mat73.loadmat(src)

    # raw_data shape is (n_channels, n_times, n_epochs)
    raw_data = mat['data']
    # data shape convert into (n_epochs, n_channels, n_times)
    data = raw_data.transpose([2, 0, 1])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    events = np.array([[i, 0, 1] for i in range(len(data))])  # 全部标为事件1
    epochs = mne.EpochsArray(data, info, events=events, tmin=times[0])

    # 设置标准 10-20 系统蒙太奇
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    return epochs


def source_estimation(epochs, method='MNE', snr=3.0, loose=0.2, depth=0.8, baseline=(None, 0.0), use_eye_cov=True):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""

    trans = 'fsaverage'
    src_fname = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem_fname = os.path.join(
        fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    # Determine channel types for the forward model.
    # print(mne.pick_types(evoked.info))
    # print(evoked.info)
    # use_eeg = bool(mne.pick_types(evoked.info, eeg=True, meg=False))
    # use_meg = bool(mne.pick_types(evoked.info, meg=True, eeg=False))
    use_eeg = len(mne.pick_types(epochs.info, eeg=True, meg=False)) > 0
    use_meg = len(mne.pick_types(epochs.info, meg=True, eeg=False)) > 0
    print(f'{use_eeg=}, {use_meg=}')

    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(
        epochs.info,
        trans=trans,
        src=src,
        bem=bem_fname,
        eeg=use_eeg,
        meg=use_meg,
    )

    # Use baseline data to estimate noise covariance.
    tmin, tmax = baseline
    if tmin is None:
        tmin = epochs.tmin
    if tmax is None:
        tmax = min(0.0, epochs.times[-1])

    # Use eye covariance if no baseline data is available (not recommended).
    # Used for SSVEP in all times.
    if use_eye_cov:
        cov = mne.Covariance(
            data=np.eye(len(epochs.info['ch_names'])),  # --- IGNORE ---
            names=epochs.info['ch_names'],
            bads=[],
            projs=[],
            nfree=len(epochs.info['ch_names'])
        )
    else:
        baseline_evoked = epochs.average().crop(tmin, tmax)
        baseline_data = baseline_evoked.data
        cov = mne.Covariance(
            data=np.cov(baseline_data),
            names=epochs.info['ch_names'],
            bads=[],
            projs=[],
            nfree=baseline_data.shape[1],
        )

    inverse_operator = make_inverse_operator(
        epochs.info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
    )

    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
    )

    return stcs


# %% ---- 2026-06-20 ------------------------
# Play ground
set_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .set files: {len(set_files)=}')

# src = set_files[0]

for src in tqdm(set_files, 'Main loop'):
    npy_fpath = (OUTPUT_DIR / src.relative_to(DATA_DIR)).with_suffix('.npy')
    npy_fpath.parent.mkdir(exist_ok=True, parents=True)

    if npy_fpath.is_file():
        continue

    epochs = read_eeg_mat_to_epochs(src)

    # Prevent Customized Reference in EEG since it is not allowed in inverse solution.
    epochs.set_eeg_reference('average', projection=True)

    # Source estimation
    stcs = source_estimation(epochs)

    # Make parc
    fname = 'aparc_sub.json'
    fpath = OUTPUT_DIR / fname
    try:
        labels_pars_df = pd.read_json(fpath)
        logger.info(f'Read parc df from {fpath=}')
    except:
        parc = 'aparc_sub'
        labels_parc = mne.read_labels_from_annot(
            'fsaverage', parc=parc, subjects_dir=subjects_dir)
        labels_parc_df = pd.DataFrame([(e.name, e)
                                       for e in labels_parc], columns=['name', 'label'])
        labels_parc_df.set_index('name', inplace=True)
        labels_parc_df.to_json(fpath)
        logger.info(f'Write parc df to {fpath=}')
    labels_parc_df

    # Fetch data
    array = []
    for i, row in tqdm(labels_parc_df.iterrows(), 'Parc loop'):
        label = row['label']
        data = np.array([np.mean(stc.in_label(label).data, axis=0)
                        for stc in stcs])
        array.append(data)

    data = np.array(array)
    data.dump(npy_fpath)
    logger.info(f'Write stc average data to {npy_fpath=}')


# %%
