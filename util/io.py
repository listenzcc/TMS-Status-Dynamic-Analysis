# %%
import mne
import mat73
import numpy as np
from pathlib import Path

# %%
CHAN_LABELS_FILE = './data/eeg-mat/chan_labels.txt'
CH_NAMES = [e.split('\t')[1]
            for e in open(CHAN_LABELS_FILE).read().split('\n') if '\t' in e]

TIMES_FILE = './data/eeg-mat/times_ms.txt'
TIMES = [float(e) / 1000 for e in open(TIMES_FILE).read().split()]

SFREQ = 1000  # Hz

# %%


def read_cls_mat(src: Path):
    '''
    Read Micro State Class markers from the mat file.
    '''

    mat = mat73.loadmat(src)

    # cls shape is (n_times, n_trials)
    # values are 0, 1, 2, 3, 4
    # Convert into (n_trials, n_times)
    cls = mat['MSClass'].transpose([1, 0])

    return cls


def read_eeg_mat(src: Path):
    '''
    Read EEG epochs from the mat file.
    '''

    mat = mat73.loadmat(src)

    # raw_data shape is (n_channels, n_times, n_epochs)
    raw_data = mat['data']
    # data shape convert into (n_epochs, n_channels, n_times)
    data = raw_data.transpose([2, 0, 1])

    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    events = np.array([[i, 0, 1] for i in range(len(data))])  # 全部标为事件1
    epochs = mne.EpochsArray(data, info, events=events, tmin=TIMES[0])

    # 设置标准 10-20 系统蒙太奇
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    return epochs
