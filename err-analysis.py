# %%
import pandas as pd
import argparse
import mat73
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from pathlib import Path

from util_err.tools import microsynt_err, STATES

# %%
# =========================================================
# 1. 参数
# =========================================================
# N_STATES = 5
WORD_SIZE = 5
N_SURROGATES = 1000
MIN_RUN = 5
RANDOM_SEED = np.random.randint(65536)


# %%
# =========================================================
# 2. Load data
DATA_DIR = Path('./data/MSClass_labels')

mat_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .mat files: {len(mat_files)=}')

parser = argparse.ArgumentParser(
    prog='Compute Entropy Representation Ratio',
    description='Compute Entropy Representation Ratio',
    epilog='Wish me a good luck'
)

parser.add_argument('-p', '--path', help='File path of .mat',
                    default='./data/MSClass_labels/sham/post/sham_post_10_MSClass_labels.mat')
parser.add_argument('-d', '--display', help='Whether to draw the matplotlib plot',
                    action='store_true')
args = parser.parse_args()
logger.info(f'Start with {args=}')

MAT_FPATH = Path(args.path)
OUTPUT_DIR = Path('output-err', args.path).with_name(MAT_FPATH.stem)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logger.info(f'{OUTPUT_DIR=}')

logger.info(f'Read {MAT_FPATH=}')
mat = mat73.loadmat(MAT_FPATH)
for k, v in mat.items():
    print('-'*80)
    print(k, type(v))
    if isinstance(v, np.ndarray):
        print(k, v.shape, np.unique(v))

# values shape is (2000, 205), (n_windows, n_trials)
values = mat['MSClass']
sequence = ''
for trial in values.T:
    for v in trial:
        if v == 0:
            continue
        s = STATES[int(v-1)]
        sequence += s

# print(f'{sequence=}')
# print(f'{len(sequence)=}')

# %%
# =========================================================
# 3. Compute
# 实际序列

seq = list(sequence)

results, surrogate, processed = microsynt_err(
    seq,
    n=WORD_SIZE,
    n_surrogates=N_SURROGATES,
    min_run=MIN_RUN,
    already_preprocessed=False,
    seed=RANDOM_SEED
)

# print("预处理后序列：")
# print("".join(processed))

print("\nEntropy Representation Ratio：")
print(results.to_string(index=False))

fpath = OUTPUT_DIR / 'results.json'
results.to_json(fpath)
logger.info(f'Saved into {fpath=}')

# %%

# %%
# Plot
# 绘制 ERR


def _plot():
    x = results["Class"].to_numpy()
    y = results["ERR"].to_numpy()

    lower = results["Surrogate_Lower"].to_numpy()
    upper = results["Surrogate_Upper"].to_numpy()

    plt.figure(figsize=(9, 5))

    plt.bar(
        x,
        y,
        color="steelblue",
        alpha=0.8,
        label="Real ERR"
    )

    plt.fill_between(
        x,
        lower,
        upper,
        color="gray",
        alpha=0.3,
        label="Surrogate 95% interval"
    )

    plt.axhline(
        1,
        color="black",
        linestyle="--",
        label="Theoretical expectation"
    )

    plt.yscale("log")

    plt.xlabel("Entropy Class")
    plt.ylabel("Entropy Representation Ratio")
    plt.title("Microsynt Entropy Representation Ratio")

    plt.legend()
    plt.tight_layout()
    plt.show()


if args.display:
    _plot()

# %%

# %%
