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

    # Estimate a diagonal noise covariance from channel-wise variance.
    # Each channel is assigned its own noise variance; off-diagonal
    # sensor noise correlations are assumed to be zero.
    if use_eye_cov:
        # epochs.get_data() shape:
        # (n_epochs, n_channels, n_times)
        data = epochs.get_data()

        # 每个 channel 的 variance
        # 把 epoch 和 time 两个维度合并
        channel_var = np.var(
            data,
            axis=(0, 2),
            ddof=1
        )

        cov = mne.Covariance(
            data=np.diag(channel_var),
            names=epochs.info['ch_names'],
            bads=epochs.info['bads'],
            projs=epochs.info['projs'],
            nfree=data.shape[0] * data.shape[2] - 1
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
