# %%
import mne
import mat73
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse
from util.easy_imports import *

# %%
DATA_DIR = Path('./data/eeg-mat')
FS_DIR = fetch_fsaverage()

# %%
set_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .set files: {len(set_files)=}')

# %%
ch_names = [e.split('\t')[1]
            for e in open(DATA_DIR / 'chan_labels.txt').read().split('\n') if '\t' in e]
times = [float(e) / 1000 for e in open(DATA_DIR /
                                       'times_ms.txt').read().split()]
sfreq = 1000  # Hz


def read_eeg_mat_to_epochs(src):
    logger.info(f'Read {src=}')
    mat = mat73.loadmat(src)

    # raw_data shape is (n_channels, n_times, n_epochs)
    raw_data = mat['data']
    # data shape convert into (n_epochs, n_channels, n_times)
    data = raw_data.transpose([2, 0, 1])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    events = np.array([[i, 0, 1] for i in range(len(data))])  # 全部标为事件1
    epochs = mne.EpochsArray(data, info, events=events, tmin=times[0])

    # 设置标准 10-20 系统蒙太奇
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    return epochs


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


# %%
src = set_files[0]
logger.debug(f'{src=}')

epochs = read_eeg_mat_to_epochs(src)
epochs.set_eeg_reference('average', projection=True)

n_trials, n_channels, n_times = epochs.get_data().shape
print(n_trials, n_channels, n_times)

# %%
x = np.random.randn(n_channels, 1)  # (n_channels, n_times)
evoked = mne.EvokedArray(x, info=epochs.info, tmin=0.0)
evoked

stc = source_estimation(evoked)
print(stc)

alpha = 1.0
brain_kwargs = dict(alpha=alpha, background="white", cortex="low_contrast")
stc.plot(
    hemi="both",
    views=['dorsal'],
    # surface='pial',
    surface='inflated',
    transparent=True,
    brain_kwargs=brain_kwargs)

input()

# %%
