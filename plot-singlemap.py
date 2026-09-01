"""
File: plot-singlemap.py
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
from itertools import product
import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from util.easy_imports import *

# %%
FS_DIR = fetch_fsaverage()

OUTPUT_DIR = Path('./output/singlemap')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHAN_LABELS_FILE = './data/singlemap/chanloc.txt'
CH_NAMES = [e.split('\'')[1]
            for e in open(CHAN_LABELS_FILE).read().split('\n')]
print(CH_NAMES)

SFREQ = 1000  # Hz

montage = mne.channels.make_standard_montage('standard_1020')

# %% ---- 2026-09-01 ------------------------
# Function and class


def create_montage():

    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    # 从文本数据中提取电极信息
    # 数据格式: 名称, [], theta, radius, x, y, z, phi, radius_2d, radius_3d, 序号, 'average'

    electrode_data = [
        ('F3', -39.9, 0.3444, 57.576, 48.141, 39.905),
        ('F4', 39.9, 0.3444, 57.576, -48.141, 39.905),
        ('C3', -90, 0.2667, 3.868e-15, 63.167, 56.876),
        ('C4', 90, 0.2667, 3.868e-15, -63.167, 56.876),
        ('P3', -140, 0.3444, -57.492, 48.242, 39.905),
        ('P4', 140, 0.3444, -57.492, -48.242, 39.905),
        ('Fz', 0, 0.2533, 60.730, 0, 59.471),
        ('Cz', 0, 0, 5.205e-15, 0, 85),
        ('Pz', 180, 0.2533, -60.730, -7.437e-15, 59.471),
        ('FC1', -44.9, 0.1811, 32.439, 32.326, 71.608),
        ('FC2', 44.9, 0.1811, 32.439, -32.326, 71.608),
        ('CP1', -135, 0.1811, -32.382, 32.382, 71.608),
        ('CP2', 135, 0.1811, -32.382, -32.382, 71.608),
        ('FC5', -69.3, 0.4083, 28.808, 76.238, 24.141),
        ('FC6', 69.3, 0.4083, 28.808, -76.238, 24.141),
        ('CP5', -111, 0.4083, -29.207, 76.087, 24.141),
        ('CP6', 111, 0.4083, -29.207, -76.087, 24.141),
        ('F1', -23.5, 0.2789, 59.888, 26.040, 54.409),
        ('F2', 23.5, 0.2789, 59.888, -26.040, 54.409),
        ('C1', -90, 0.1333, 2.117e-15, 34.573, 77.651),
        ('C2', 90, 0.1333, 2.117e-15, -34.573, 77.651),
        ('P1', -157, 0.2789, -60.113, 25.516, 54.409),
        ('P2', 157, 0.2789, -60.113, -25.516, 54.409),
        ('FC3', -62.4, 0.2883, 30.990, 59.278, 52.448),
        ('FC4', 62.4, 0.2883, 30.990, -59.278, 52.448),
        ('CP3', -118, 0.2883, -31.403, 59.060, 52.448),
        ('CP4', 118, 0.2883, -31.403, -59.060, 52.448),
        ('F5', -49.4, 0.4317, 54.046, 63.057, 18.108),
        ('F6', 49.4, 0.4311, 54.025, -63.033, 18.253),
        ('C5', -90, 0.4, 4.950e-15, 80.840, 26.266),
        ('C6', 90, 0.4, 4.950e-15, -80.840, 26.266),
        ('P5', -131, 0.4317, -54.485, 62.678, 18.108),
        ('P6', 131, 0.4311, -54.464, -62.654, 18.253),
        ('CPz', 180, 0.1267, -32.939, -4.034e-15, 78.358),
    ]

    # 提取名称和3D坐标
    ch_names = [item[0] for item in electrode_data]
    positions = np.array([[item[4], item[5], item[6]]
                         for item in electrode_data])  # x, y, z

    # 创建蒙太奇
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(ch_names, positions)),
        coord_frame='head'
    )

    # 打印验证
    print(f"蒙太奇包含 {len(montage.ch_names)} 个通道")
    print("前5个通道位置:")
    for name in montage.ch_names[:5]:
        pos = montage.get_positions()['ch_pos'][name]
        print(f"  {name}: {pos}")

    return montage


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


def source_estimation(evoked, method='MNE'):
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

    # Determine channel types for the forward model.
    # print(mne.pick_types(evoked.info))
    # print(evoked.info)
    # use_eeg = bool(mne.pick_types(evoked.info, eeg=True, meg=False))
    # use_meg = bool(mne.pick_types(evoked.info, meg=True, eeg=False))
    use_eeg = len(mne.pick_types(info, eeg=True, meg=False)) > 0
    use_meg = len(mne.pick_types(info, meg=True, eeg=False)) > 0

    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem_fname,
        eeg=use_eeg,
        meg=use_meg,
    )

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
    # fig.save(OUTPUT_DIR / f'{title}-topomap.png')
    plt.savefig(OUTPUT_DIR / f'{title}-topomap.svg')
    plt.close(fig)

    fname = OUTPUT_DIR / f'{title}-brain.png'
    if fname.exists():
        continue
    # stc = source_estimation(evoked, method='sLORETA')
    # stc = source_estimation(evoked, method='eLORETA')
    stc = source_estimation(evoked, method='dSPM')
    # stc = source_estimation(evoked, method='MNE')
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
