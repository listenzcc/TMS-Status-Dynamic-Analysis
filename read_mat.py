"""
File: read_mat.py
Author: Chuncheng Zhang
Date: 2026-06-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read mat files in data folder.



Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-10 ------------------------
# Requirements and constants
import mat73
from util.easy_imports import *

# %%
DATA_DIR = Path('./data/MSClass_labels')

# %% ---- 2026-06-10 ------------------------
# Function and class


# %% ---- 2026-06-10 ------------------------
# Play ground
mat_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .mat files: {len(mat_files)=}')

src = mat_files[0]

logger.info(f'Read {src=}')
mat = mat73.loadmat(src)
for k, v in mat.items():
    print('-'*80)
    print(k, type(v))
    if isinstance(v, np.ndarray):
        print(k, v.shape, np.unique(v))

# %% ---- 2026-06-10 ------------------------
# Pending
plt.imshow(mat['MSClass'])


# %% ---- 2026-06-10 ------------------------
# Pending
