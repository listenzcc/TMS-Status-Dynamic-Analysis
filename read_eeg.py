"""
File: read_eeg.py
Author: Chuncheng Zhang
Date: 2026-06-17
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read eeg data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-17 ------------------------
# Requirements and constants
import mne
from util.easy_imports import *

# %%
DATA_DIR = Path('./data/set')

# %% ---- 2026-06-17 ------------------------
# Function and class


# %% ---- 2026-06-17 ------------------------
# Play ground
set_files = sorted(DATA_DIR.rglob('*.set'))
logger.info(f'Found .set files: {len(set_files)=}')

src = set_files[0]
src = Path('data/set/T100/post/10.set')

logger.info(f'Read {src=}')

epochs = mne.io.read_epochs_eeglab(src)
print(epochs)


# %% ---- 2026-06-17 ------------------------
# Pending


# %% ---- 2026-06-17 ------------------------
# Pending
