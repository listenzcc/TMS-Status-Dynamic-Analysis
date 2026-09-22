# %%
import argparse

from util_err.tools import microsynt_err, STATES
from util.easy_imports import *

# %%
WORD_SIZE = 5
N_SURROGATES = 1000
MIN_RUN = 5
RANDOM_SEED = np.random.randint(65536)
# MIN_RUN = 1

# %%
parser = argparse.ArgumentParser(
    prog='Compute Entropy Representation Ratio',
    description='Compute Entropy Representation Ratio',
    epilog='Wish me a good luck'
)

parser.add_argument('-p', '--path', help='File path of .txt',
                    default='./data/seq-data-20260920/T120.txt')
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
content = open(args.path).read()
content = ''.join(content.split())
segments = [e for e in content.split('0') if e]
segments = [e for e in segments if len(set(e)) > 1]
segments = [e for e in segments if len(e) > 100]
lens = [len(e) for e in segments]
lens = sorted(set(lens))

logger.info(
    f'Found segments is {len(segments)}, lengthRange is ({lens[0]}, {lens[-1]})')

# %%
# segments = [''.join(segments)]
# for i, seq in enumerate(segments):

for i in range(0, len(segments), 100):
    seq = ''.join(segments[i: i+100])
    logger.debug(f'seq len is {len(seq)}')
    try:
        results_path = OUTPUT_DIR / f'results-{i}.json'
        results_chars_path = OUTPUT_DIR / f'results-chars-{i}.json'

        results, results_chars, surrogate, processed, word_to_class = microsynt_err(
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

        results.to_json(results_path)
        logger.info(f'Saved into {results_path=}')

        results_chars.to_json(results_chars_path)
        logger.info(f'Saved into {results_chars_path=}')
    except Exception as err:
        logger.error(err)


# %%
