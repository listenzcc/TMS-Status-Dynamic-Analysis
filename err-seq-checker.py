# %%
import mat73
import numpy as np
import argparse
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from collections import Counter

from loguru import logger
from pathlib import Path

from util_err.tools import microsynt_err, STATES, extract_words

# %%
# =========================================================
# 1. 参数
# =========================================================
# N_STATES = 5
WORD_SIZE = 5
N_SURROGATES = 1000  # 1000
MIN_RUN = 5
RANDOM_SEED = np.random.randint(65536)

MIN_RUN = 1

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
                    default='./data/MSClass_labels/T120/post/T120_post_10_MSClass_labels.mat')
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
print(f'{values.shape=}')
sequence = []
for trial in tqdm(values.T):
    for v in trial:
        if v == 0:
            continue
        s = STATES[int(v-1)]
        sequence.append(s)

print(trial)
print(f'{trial.shape=}')
# print(f'{sequence=}')
# print(f'{len(sequence)=}')

# %%
# =========================================================
# 3. Compute
# 实际序列

seq = [e for e in sequence]

results, results_chars, surrogate, processed, word_to_class = microsynt_err(
    seq,
    n=WORD_SIZE,
    n_surrogates=N_SURROGATES,
    min_run=MIN_RUN,
    already_preprocessed=False,
    seed=RANDOM_SEED
)

for k in ['results', 'results_chars', 'surrogate']:
    print(f'\n==== {k} ====')
    print(eval(k))

# %%
print('\n==== seq ====')
print(''.join(seq[:80]) + '...')
print(''.join(processed[:80]) + '...')
print(Counter(processed))

# %%
print('\n==== Words of Classes ====')
# print([(''.join(k), v) for k, v in word_to_class.items()])

# %%
# words = extract_words(processed, n=WORD_SIZE)
# print(words)
# print([''.join(w) for w in words])

# %%

# %%
