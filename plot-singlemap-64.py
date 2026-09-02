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

methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
method = 'MNE'

OUTPUT_DIR = Path(f'./output/singlemap-{method}-64')
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


def source_estimation(evoked, method):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""
    snr = 3.0
    # snr = 30.0
    loose = 0.2
    depth = 0.8
    pick_ori = None
    pick_ori = 'normal'

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


def interpolate_to_montage(
    values,
    ch_names,
    sfreq,
    montage,
):
    """
    Interpolate EEG data from a subset of channels to all channels
    available in the target montage.

    Parameters
    ----------
    values : array, shape (n_channels, n_times) or (n_channels, 1)
        Original EEG values.

    ch_names : list[str]
        Names of the original channels.

    sfreq : float
        Sampling frequency.

    montage : mne.channels.DigMontage
        Target montage, e.g. a 64-channel montage.

    Returns
    -------
    evoked : mne.Evoked
        Evoked containing all channels in the target montage.
    """

    # ------------------------------------------------------------
    # 1. Target channels = channels defined by the montage
    # ------------------------------------------------------------

    target_ch_names = montage.ch_names

    # Keep only EEG channels from montage
    target_ch_names = [
        ch for ch in target_ch_names
        if ch in montage.get_positions()["ch_pos"]
    ]

    # ------------------------------------------------------------
    # 2. Create a 64-channel Info
    # ------------------------------------------------------------

    info = mne.create_info(
        ch_names=target_ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    info.set_montage(montage)

    # ------------------------------------------------------------
    # 3. Create target data array
    #
    # Existing channels -> original values
    # Missing channels  -> NaN
    # ------------------------------------------------------------

    values = np.asarray(values)

    if values.ndim == 1:
        values = values[:, np.newaxis]

    n_times = values.shape[1]

    data = np.full(
        (len(target_ch_names), n_times),
        np.nan,
        dtype=float,
    )

    source_index = {
        ch: i
        for i, ch in enumerate(ch_names)
    }

    for i, ch in enumerate(target_ch_names):
        if ch in source_index:
            data[i] = values[source_index[ch]]

    # ------------------------------------------------------------
    # 4. Create Evoked
    # ------------------------------------------------------------

    evoked = mne.EvokedArray(
        data,
        info,
        tmin=0.0,
    )

    # ------------------------------------------------------------
    # 5. Mark missing channels as bad
    #
    # IMPORTANT:
    # These channels now exist in evoked.info,
    # so they can safely be put into info["bads"].
    # ------------------------------------------------------------

    missing_channels = [
        ch for ch in target_ch_names
        if ch not in source_index
    ]

    evoked.info["bads"] = missing_channels

    print(f"Original channels: {len(ch_names)}")
    print(f"Target channels:   {len(target_ch_names)}")
    print(f"Missing channels:  {len(missing_channels)}")
    print(f"Missing: {missing_channels}")

    # ------------------------------------------------------------
    # 6. Spatial interpolation
    # ------------------------------------------------------------

    evoked.interpolate_bads(
        reset_bads=True,
    )

    return evoked


def remove_duplicate_montage_positions(montage, tol=1e-6):
    pos = montage.get_positions()
    ch_pos = pos["ch_pos"]

    kept = {}
    removed = []

    for ch, xyz in ch_pos.items():
        duplicate = False

        for kept_ch, kept_xyz in kept.items():
            if np.linalg.norm(xyz - kept_xyz) < tol:
                removed.append((ch, kept_ch))
                duplicate = True
                break

        if not duplicate:
            kept[ch] = xyz

    montage_clean = mne.channels.make_dig_montage(
        ch_pos=kept,
        nasion=pos["nasion"],
        lpa=pos["lpa"],
        rpa=pos["rpa"],
        hpi=pos["hpi"],
        hsp=pos["hsp"],
        coord_frame=pos["coord_frame"],
    )

    print("Removed duplicate electrodes:")
    for a, b in removed:
        print(f"  {a} == {b}")

    return montage_clean


# %% ---- 2026-09-01 ------------------------
# Play ground
for condition, state in product(['T80', 'T100', 'T120', 'Sham'], [0, 1, 2, 3]):
    title = f'{condition}-{state}'
    print(f'{montage=}')
    values = read_eeg_map(condition=condition, state=state)
    print(f'{values.shape=}')
    print(f'{values=}')

    # CH_NAMES has 34 channels, values are their value in shape (34, 1)
    montage = remove_duplicate_montage_positions(montage)

    evoked = interpolate_to_montage(
        values=values,
        ch_names=CH_NAMES,
        sfreq=SFREQ,
        montage=montage,
    )
    evoked.set_eeg_reference('average', projection=True)

    fig = evoked.plot_topomap(
        times=0.0, ch_type='eeg', size=6, show_names=True)
    plt.savefig(OUTPUT_DIR / f'{title}-topomap.svg')
    plt.close(fig)

    fname = OUTPUT_DIR / f'{title}-stc'
    try:
        stc = mne.read_source_estimate(fname)
        print(f"Loaded existing STC from {fname}")
    except:
        stc = source_estimation(evoked, method=method)
        stc.save(fname, overwrite=True)
        print(stc)

    fname = OUTPUT_DIR / f'{title}-brain.png'
    if fname.exists():
        continue

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
