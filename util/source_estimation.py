# %%
import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

from pathlib import Path

# %%
FS_DIR = fetch_fsaverage()
SUBJECTS_DIR = Path(FS_DIR).parent.as_posix()

# %%


def source_estimation(epochs, method='MNE', pick_ori=None, snr=3.0, loose=0.2, depth=0.8, baseline=(None, 0.0), use_eye_cov=True):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""

    trans = 'fsaverage'
    src_fname = Path(FS_DIR, 'bem', 'fsaverage-ico-5-src.fif')
    bem_fname = Path(
        FS_DIR, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    # Determine channel types for the forward model.
    # print(mne.pick_types(evoked.info))
    # print(evoked.info)
    # use_eeg = bool(mne.pick_types(evoked.info, eeg=True, meg=False))
    # use_meg = bool(mne.pick_types(evoked.info, meg=True, eeg=False))
    use_eeg = len(mne.pick_types(epochs.info, eeg=True, meg=False)) > 0
    use_meg = len(mne.pick_types(epochs.info, meg=True, eeg=False)) > 0

    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(
        epochs.info,
        trans=trans,
        src=src,
        bem=bem_fname,
        eeg=use_eeg,
        meg=use_meg,
    )

    # Use baseline data to estimate noise covariance.
    tmin, tmax = baseline
    if tmin is None:
        tmin = epochs.tmin
    if tmax is None:
        tmax = min(0.0, epochs.times[-1])

    # Use eye covariance if no baseline data is available (not recommended).
    # Used for SSVEP in all times.
    if use_eye_cov:
        cov = mne.Covariance(
            data=np.eye(len(epochs.info['ch_names'])),  # --- IGNORE ---
            names=epochs.info['ch_names'],
            bads=[],
            projs=[],
            nfree=len(epochs.info['ch_names'])
        )
    else:
        baseline_evoked = epochs.average().crop(tmin, tmax)
        baseline_data = baseline_evoked.data
        cov = mne.Covariance(
            data=np.cov(baseline_data),
            names=epochs.info['ch_names'],
            bads=[],
            projs=[],
            nfree=baseline_data.shape[1],
        )

    inverse_operator = make_inverse_operator(
        epochs.info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
    )

    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
    )

    return stcs
