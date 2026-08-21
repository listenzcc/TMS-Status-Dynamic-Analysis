"""
File: source-on-conditions.py
Author: Chuncheng Zhang
Date: 2026-07-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read EEG and conditions, source estimation and plot.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-07-30 ------------------------
# Requirements and constants
import numpy as np
from rich import print
from pathlib import Path
from tqdm.auto import tqdm

from util.io import read_cls_mat, read_eeg_mat
from util.source_estimation import source_estimation

# %%
# Folder with EEG mat files
EEG_FOLDER = Path('./data/eeg-mat')

# Folder with Micro State Class mat files
CLS_FOLDER = Path('./data/MSClass_labels')

# Output folder
OUTPUT_FOLDER = Path('./data/20484-eLORETA-filter/')

# %% ---- 2026-07-30 ------------------------
# Function and class


def select_pairs():
    efiles = sorted(EEG_FOLDER.rglob('*sub*.mat'))
    pairs = []
    for en in efiles:
        rel = en.relative_to(EEG_FOLDER)
        parts = [rel.parent, rel.name.replace(
            'sub', '').replace('.mat', '_MSClass_labels.mat')]
        cn = CLS_FOLDER.joinpath(*parts)
        assert cn.is_file(), f'File does not exist, {cn}'
        pairs.append({
            'name': en.stem,
            'eegfile': en,
            'clsfile': cn
        })
    return pairs


# %% ---- 2026-07-30 ------------------------
# Play ground
pairs = select_pairs()
print(pairs)

for pair in tqdm(pairs):

    # Do nothing if the output files already exist
    output_fnames = [
        (OUTPUT_FOLDER / f'{pair["name"]}-{_cls}').with_suffix('.npy') for _cls in [1, 2, 3, 4]]
    if all([e.exists() for e in output_fnames]):
        continue

    # Read
    # cls_values shape is (n_trials, n_times)
    cls_values = read_cls_mat(pair['clsfile'])
    epochs = read_eeg_mat(pair['eegfile'])

    # Lower samples for debug
    # cls_values = cls_values[:2]
    # epochs = epochs[:2]

    # Prevent Customized Reference in EEG since it is not allowed in inverse solution.
    epochs.set_eeg_reference('average', projection=True)

    epochs = epochs.filter(l_freq=0.5, h_freq=30.5, n_jobs=-1)

    # Source estimation
    stcs = source_estimation(epochs, 'eLORETA')

    # Print
    print(epochs, cls_values.shape)
    print(stcs)

    # source_values shape is (n_trials, n_vertices, n_times)
    source_values = np.array([stc.data for stc in stcs])
    print(source_values.shape)

    # mask shape is (n_trials, n_times, n_vertices)
    # Align it with value
    value = source_values.transpose([0, 2, 1])
    mask = cls_values
    print(value.shape, mask.shape)

    for _cls in [1, 2, 3, 4]:
        output_fname = (OUTPUT_FOLDER /
                        f'{pair["name"]}-{_cls}').with_suffix('.npy')
        output_fname.parent.mkdir(exist_ok=True, parents=True)

        # Do nothing if it already exists.
        if output_fname.exists():
            continue

        # Get all the _cls
        # d shape is (n_samples, n_vertices)
        d = value[mask == _cls]
        # dd shape is (n_vertices,)
        dd = np.mean(d, axis=0)

        np.save(open(output_fname, 'wb'), dd)


# %% ---- 2026-07-30 ------------------------
# Pending


# %% ---- 2026-07-30 ------------------------
# Pending

# %%
