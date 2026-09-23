# %%
import argparse

from util_err.tools_chars_detail import microsynt_err, STATES
from util.easy_imports import *

# %%
WORD_SIZE = 5
N_SURROGATES = 1000
MIN_RUN = 5
RANDOM_SEED = np.random.randint(65536)
MIN_RUN = 1

# %%
parser = argparse.ArgumentParser(
    prog='Compute Entropy Representation Ratio',
    description='Compute Entropy Representation Ratio',
    epilog='Wish me a good luck'
)

parser.add_argument('-p', '--path', help='File path of .txt',
                    default='./data/seq-data-20260920/T120_Sub1.txt')
parser.add_argument('-d', '--display', help='Whether to draw the matplotlib plot',
                    action='store_true')
args = parser.parse_args()
logger.info(f'Start with {args=}')

# %%
OUTPUT_DIR = Path('./output-err-20260920',
                  args.path).with_name(Path(args.path).stem)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
logger.info(f'{OUTPUT_DIR=}')

# %%
# content has 900 lines and 770 columns
# for 900 timepoints and 770 trials
content = open(args.path).read()
data = np.array([[e for e in line.split()] for line in content.split('\n')[:900]])
# Convert into 770 lines
data = data.T
print(f'{data.shape=}')
seq = ''
for d in data:
    seq += ''.join(d).replace('0', '')
logger.info(f'{len(seq)=}, {seq[:20]=}')

i = 'concat'

logger.debug(f'seq len is {len(seq)}')
results_chars, results_word = microsynt_err(
    seq,
    n=WORD_SIZE,
    n_surrogates=N_SURROGATES,
    min_run=MIN_RUN,
    already_preprocessed=False,
    seed=RANDOM_SEED
)


print("\nEntropy Representation Ratio：")
print(results_chars.to_string(index=False))
print(results_word.to_string(index=False))


fpath = OUTPUT_DIR / f'results-chars-detail-{i}.json'
results_chars.to_json(fpath)
logger.info(f'Saved into {fpath=}')

fpath = OUTPUT_DIR / f'results-word-detail-{i}.json'
results_word.to_json(fpath)
logger.info(f'Saved into {fpath=}')


# %%
