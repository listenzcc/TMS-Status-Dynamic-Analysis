"""
File: plot-singlemap-1.py
Author: Chuncheng Zhang
Date: 2026-09-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the single map evoked.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-01 ------------------------
# Requirements and constants
import numpy as np
from itertools import product

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

from util.io import read_eeg_mat
from util.easy_imports import *

# %%
FS_DIR = fetch_fsaverage()
MAT_FOLDER = Path('./data/eeg-mat')

methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
method = 'MNE'

OUTPUT_DIR = Path(f'./output/singlemap-{method}-1-pre')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHAN_LABELS_FILE = './data/singlemap/chanloc.txt'
CH_NAMES = [e.split('\'')[1]
            for e in open(CHAN_LABELS_FILE).read().split('\n')]
print(CH_NAMES)

SFREQ = 1000  # Hz

montage = mne.channels.make_standard_montage('standard_1020')

# %% ---- 2026-09-01 ------------------------
# Function and class


def read_eeg_map(condition='T80', state=0):
    """
    Read EEG map data for a specific condition and state.

    Parameters:
    - condition: str, the condition name (default is 'T80')
    - state: int, the state index (default is 0)

    Returns:
    - eeg_map: np.ndarray, the EEG map data
    """
    # Construct the file path based on the condition and state
    fname = f'./data/singlemap/{condition}.txt'
    values = np.loadtxt(fname)[:, state]

    return values[:, np.newaxis]  # Ensure the output is a 2D array


def create_evoked(x, info):
    evoked = mne.EvokedArray(x, info=info, tmin=0.0)
    return evoked


def source_estimation(evoked, method='MNE', cov=None):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""
    snr = 3.0
    loose = 0.2
    depth = 0.8
    pick_ori = None

    trans = 'fsaverage'
    src_fname = Path(FS_DIR, 'bem', 'fsaverage-ico-5-src.fif')
    bem_fname = Path(
        FS_DIR, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    info = evoked.info

    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem_fname,
        eeg=True,
        meg=False,
    )

    if cov is None:
        cov = mne.Covariance(
            data=np.eye(len(info['ch_names'])),  # --- IGNORE ---
            names=info['ch_names'],
            bads=[],
            projs=[],
            nfree=len(info['ch_names'])
        )

    inverse_operator = make_inverse_operator(
        info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
    )

    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
    )

    return stc


# %% ---- 2026-09-01 ------------------------
# Play ground
for condition, state in product(['T80', 'T100', 'T120'], [0, 1, 2, 3]):

    # cov
    mat_files = list(
        (MAT_FOLDER / condition / 'pre').glob(f'{condition}*.mat'))
    mat_files = mat_files[:1]
    print(mat_files)
    epochs_buf = [read_eeg_mat(mat_file) for mat_file in mat_files]
    cov_buf = [mne.compute_covariance(epochs) for epochs in epochs_buf]
    print(epochs_buf)
    print(cov_buf)
    cov = cov_buf[0]
    print(cov)
    # exit(1)

    title = f'{condition}-{state}'
    print(f'{montage=}')
    values = read_eeg_map(condition=condition, state=state)
    print(f'{values.shape=}')
    print(f'{values=}')
    evoked = create_evoked(values, info=mne.create_info(
        ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg'))
    evoked.set_montage(montage)
    evoked.set_eeg_reference('average', projection=True)
    print(f'{evoked=}')

    fig = evoked.plot_topomap(
        times=0.0, ch_type='eeg', size=6, show_names=True)
    plt.savefig(OUTPUT_DIR / f'{title}-topomap.svg')
    plt.close(fig)

    fname = OUTPUT_DIR / f'{title}-brain.png'
    if fname.exists():
        continue
    stc = source_estimation(evoked, method=method, cov=cov)
    print(stc)

    alpha = 1.0
    brain_kwargs = dict(alpha=alpha, background="white", cortex="low_contrast")
    brain = stc.plot(
        hemi="both",
        views=['dorsal'],
        # surface='pial',
        surface='inflated',
        transparent=True,
        brain_kwargs=brain_kwargs)

    brain.add_text(0.5, 0.9, title, 'title', justification='center')
    brain.save_image(fname)
    brain.close()


# %% ---- 2026-09-01 ------------------------
# Pending

# input()


# %% ---- 2026-09-01 ------------------------
# Pending
