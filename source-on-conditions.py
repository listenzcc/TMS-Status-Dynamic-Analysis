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
    # Read
    cls_values = read_cls_mat(pair['clsfile'])
    epochs = read_eeg_mat(pair['eegfile'])

    # Prevent Customized Reference in EEG since it is not allowed in inverse solution.
    epochs.set_eeg_reference('average', projection=True)

    # Source estimation
    stcs = source_estimation(epochs)

    # Print
    print(epochs, cls_values.shape)
    print(stcs)

    break

stcs

# %% ---- 2026-07-30 ------------------------
# Pending


# %% ---- 2026-07-30 ------------------------
# Pending
