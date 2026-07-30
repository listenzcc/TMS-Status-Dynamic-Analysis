"""
File: clear_aparc_sub.py
Author: Chuncheng Zhang
Date: 2026-07-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Clear aparc_sub.json, now it is too big.

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
import pandas as pd
from pathlib import Path


# %% ---- 2026-07-30 ------------------------
# Function and class
df = pd.read_json(Path('./data/eeg-stc/aparc_sub.json'))
print(df.head())

names = df.index.to_list()
print(f'{names=}, {len(names)=}')


# %%
fname = next(Path('./data/eeg-stc').rglob('*.npy'))
array = np.load(fname, allow_pickle=True)
print(type(array), array.shape)

# %% ---- 2026-07-30 ------------------------
# Play ground


# %% ---- 2026-07-30 ------------------------
# Pending


# %% ---- 2026-07-30 ------------------------
# Pending
