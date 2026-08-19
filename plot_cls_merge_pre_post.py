"""
File: plot_cls_merge_pre_post.py
Author: Chuncheng Zhang
Date: 2026-07-31
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot for cls.

Require 3D support:
    # 在终端或命令行中执行
    # 安装pyvista和qt后端
    pip install pyvista pyvistaqt qtpy PyQt5

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-07-31 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger

import mne
from mne.viz import plot_source_estimates
from mne.datasets import fetch_fsaverage

# %%
FS_DIR = fetch_fsaverage()
SUBJECTS_DIR = Path(FS_DIR).parent.as_posix()

# %%
vertex_value_files = sorted(
    Path('./data/20484-sLORETA-baseline').glob('*.npy'))
logger.info(f'{len(vertex_value_files)=}')

# %%
OUTPUT_DIR = Path('./output-sLORETA-baseline/')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-07-31 ------------------------
# Function and class


def parse_fpath(path: Path):
    stem = path.stem
    stim, stage, sub, cls = stem.replace('_', '-').split('-')
    return {
        'stim': stim,
        'stage': stage,
        'sub': sub,
        'cls': cls,
        'path': path
    }
# %%


def visualize_single_vector_stc(data_vector, title='title', subject='fsaverage', subjects_dir=SUBJECTS_DIR,
                                clim=None, smoothing_steps=10):
    """
    将单个20484维向量可视化为源估计

    Parameters:
    -----------
    data_vector : np.ndarray
        形状为 (20484,) 的数据向量
    subject : str
        被试名称
    subjects_dir : str
        被试目录
    clim : dict or None
        颜色图范围，如 dict(kind='value', lims=[0.5, 1.0, 2.0])
    smoothing_steps : int
        平滑步数
    """
    # 确保数据是列向量 (20484, 1)
    if len(data_vector.shape) == 1:
        data_2d = data_vector.reshape(-1, 1)
    else:
        data_2d = data_vector

    # 加载 fsaverage 的源空间信息
    src_fname = Path(FS_DIR, 'bem', 'fsaverage-ico-5-src.fif')
    src = mne.read_source_spaces(src_fname)

    # 获取每个半球的顶点数
    n_vertices_lh = len(src[0]['vertno'])
    n_vertices_rh = len(src[1]['vertno'])
    total_vertices = n_vertices_lh + n_vertices_rh

    print(f"左半球顶点数: {n_vertices_lh}")
    print(f"右半球顶点数: {n_vertices_rh}")
    print(f"总顶点数: {total_vertices}")
    print(f"数据向量长度: {len(data_vector)}")

    # 检查数据长度是否匹配
    if len(data_vector) != total_vertices:
        print(f"警告: 数据长度 ({len(data_vector)}) 与总顶点数 ({total_vertices}) 不匹配!")
        # 尝试截断或填充
        if len(data_vector) < total_vertices:
            padded_data = np.zeros(total_vertices)
            padded_data[:len(data_vector)] = data_vector
            data_2d = padded_data.reshape(-1, 1)
            print(f"数据已填充到 {total_vertices} 个顶点")
        else:
            data_2d = data_vector[:total_vertices].reshape(-1, 1)
            print(f"数据已截断到 {total_vertices} 个顶点")

    # 提取左右半球的数据
    lh_data = data_2d[:n_vertices_lh, :]
    rh_data = data_2d[n_vertices_lh:total_vertices, :]

    # 创建 SourceEstimate 对象
    stc = mne.SourceEstimate(
        data=data_2d,
        vertices=[np.arange(n_vertices_lh), np.arange(n_vertices_rh)],
        tmin=0.0,
        tstep=1.0,
        subject=subject
    )

    # assume the values are in N(0, 1)
    # clim = dict(kind='value', lims=[0.5, 3, 5])
    # alpha = 0.8
    clim = dict(kind='value', lims=[1, 2, 4])
    alpha = 1.0

    brain_kwargs = dict(alpha=alpha, background="white", cortex="low_contrast")
    brain = stc.plot(
        initial_time=0,
        # hemi="split",
        hemi="both",
        views=['dorsal'],
        # surface='pial',
        surface='inflated',
        subjects_dir=subjects_dir,
        transparent=True,
        clim=clim,
        # show_traces=False,
        # time_label=None,
        brain_kwargs=brain_kwargs
    )
    brain.add_text(0.5, 0.9, title, 'title', justification='center')

    brain.save_image(OUTPUT_DIR / f'{title}.png')

    return stc, brain


# %% ---- 2026-07-31 ------------------------
# Play ground
table = pd.DataFrame([parse_fpath(e) for e in vertex_value_files])
print(table.head())

# %%
df = table.copy()
group = df.groupby(['stim', 'cls'])
print(group.count())
query_pairs = sorted(group.groups.keys())
print(query_pairs)

# %%
if False:
    df = table.copy()
    vertex_values = np.vstack(
        [np.load(open(path, 'rb'), allow_pickle=True) for path in tqdm(df['path'])])
    _mean = np.mean(vertex_values)
    _std = np.std(vertex_values)

# %% ---- 2026-07-31 ------------------------
# Pending
_stim, _cls = 'T100', '2'

for pair in query_pairs:
    stim, cls = pair
    # if not all([stim in ['T100'], cls in ['2']]):
    #     continue

    query = [f'stim=="{stim}"', f'cls=="{cls}"']

    df = table.copy()
    df = df.query(' & '.join(query))
    print(df)

    title = ', '.join(query) + f' ({len(df)})'

    vertex_values = np.vstack(
        [np.load(open(path, 'rb'), allow_pickle=True) for path in tqdm(df['path'])])
    print(vertex_values.shape)

    data_vector = np.mean(vertex_values, axis=0)
    data_vector = (data_vector - np.mean(data_vector)) / np.std(data_vector)
    stc, brain = visualize_single_vector_stc(data_vector, title)


# %% ---- 2026-07-31 ------------------------
# Pending

input('Press Enter to Quit.')

# %%
