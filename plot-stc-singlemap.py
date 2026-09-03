"""
File: plot-stc-singlemap.py
Author: Chuncheng Zhang
Date: 2026-09-02
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the stc of singlemap.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-09-02 ------------------------
# Requirements and constants
import mne
from mne.datasets import fetch_fsaverage

import argparse
from util.easy_imports import *

# %%
argparser = argparse.ArgumentParser(description='Plot the stc of singlemap.')
argparser.add_argument('-m', '--method', type=str, default='MNE',
                       help='The method to use for source estimation (default: MNE).')
argparser.add_argument('-c', '--condition', type=str, default='T80',
                       help='The condition to process T80 | T100 | T120 | Sham (default: T80).')
argparser.add_argument('-s', '--state', type=int, default=0,
                       help='The state index to process 0 | 1 | 2 | 3 (default: 0).')
argparser.add_argument('-p', '--plot', action='store_true',
                       help='Whether to plot the stc (default: False).')


args = argparser.parse_args()

METHOD = args.method
CONDITION = args.condition
STATE = args.state
PLOT_FLAG = args.plot

# %%
FS_DIR = fetch_fsaverage()

# %% ---- 2026-09-02 ------------------------
# Function and class
fname = Path(f'./output/singlemap-{METHOD}/{CONDITION}-{STATE}-stc')
title = fname.name
stc = mne.read_source_estimate(fname.as_posix())
stc.subject = 'fsaverage'
stc.data = stc.data / np.std(stc.data)  # Normalize the data

alpha = 1.0
clim = dict(kind='value', lims=[1, 1.5, 2])
brain_kwargs = dict(alpha=alpha, background="white", cortex="low_contrast")

single_view_kwargs = dict(
    views=['dorsal'],
    size=(720, 540),
)

split_view_kwargs = dict(
    views=['dorsal', 'ventral'],
    view_layout='horizontal',
    size=(1440, 540),
)

brain = stc.plot(
    clim=clim,
    hemi="both",
    surface='pial',
    # surface='inflated',
    transparent=True,
    brain_kwargs=brain_kwargs,
    **split_view_kwargs
)

brain.add_text(0.0, 0.9, title, 'title', justification='left')

# According to the flag to show the image or save it directly
if PLOT_FLAG:
    # Wait to show
    input("Press Enter to continue...")
else:
    # Just save it
    brain.save_image(fname.with_suffix('.jpg'))
    brain.close()


# %% ---- 2026-09-02 ------------------------
# Play ground


# %% ---- 2026-09-02 ------------------------
# Pending


# %% ---- 2026-09-02 ------------------------
# Pending
