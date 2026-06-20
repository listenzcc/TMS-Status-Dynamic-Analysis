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
import mat73
from util.easy_imports import *

# %%
DATA_DIR = Path('./data/eeg-mat')

# %%
ch_names = [e.split('\t')[1]
            for e in open(DATA_DIR / 'chan_labels.txt').read().split('\n') if '\t' in e]
print(f'{ch_names=}')

times = [float(e) / 1000 for e in open(DATA_DIR /
                                       'times_ms.txt').read().split()]
print(f'{times=}')

sfreq = 1000  # Hz


# %% ---- 2026-06-17 ------------------------
# Function and class


# %% ---- 2026-06-17 ------------------------
# Play ground
set_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .set files: {len(set_files)=}')

src = set_files[0]

logger.info(f'Read {src=}')
mat = mat73.loadmat(src)

# raw_data shape is (n_channels, n_times, n_epochs)
raw_data = mat['data']
# data shape convert into (n_epochs, n_channels, n_times)
data = raw_data.transpose([2, 0, 1])

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
events = np.array([[i, 0, 1] for i in range(len(data))])  # 全部标为事件1
epochs = mne.EpochsArray(data, info, events=events, tmin=times[0])
# ===== 关键代码：设置标准 10-20 系统蒙太奇 =====
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)
print(epochs)

# %% ---- 2026-06-17 ------------------------
# Pending
evoked = epochs.average()
evoked.plot_joint()


# %% ---- 2026-06-17 ------------------------
# Pending
