# %%
import mat73
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from itertools import product

from loguru import logger


# =========================================================
# 1. 参数
# =========================================================

N_STATES = 5
WORD_SIZE = 6
WORD_SIZE = 5
N_SURROGATES = 1000
MIN_RUN = 5
RANDOM_SEED = np.random.randint(65536)

STATES = list("ABCD")

# =========================================================
# Load data
# %%
DATA_DIR = Path('./data/MSClass_labels')

mat_files = sorted(DATA_DIR.rglob('*.mat'))
logger.info(f'Found .mat files: {len(mat_files)=}')

src = mat_files[0]

logger.info(f'Read {src=}')
mat = mat73.loadmat(src)
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

print(f'{sequence=}')
print(f'{len(sequence)=}')

# %%

# =========================================================
# 2. 预处理：删除短微状态，并去除连续重复状态
# =========================================================


def preprocess_sequence(sequence, min_run=5):
    """
    先按连续 run 分段，保留长度 >= min_run 的 run，
    再合并过滤后相邻且状态相同的 run，保证无连续重复状态。
    """
    seq = list(sequence)
    if not seq:
        return []

    # Run-length encoding
    runs = []
    current = seq[0]
    length = 1

    for state in seq[1:]:
        if state == current:
            length += 1
        else:
            runs.append((current, length))
            current = state
            length = 1
    runs.append((current, length))

    # 保留达到阈值的 run
    kept = [state for state, run_len in runs if run_len >= min_run]

    # 过滤后可能出现相邻同状态，再压缩一次
    processed = []
    for state in kept:
        if not processed or processed[-1] != state:
            processed.append(state)

    return processed


# =========================================================
# 3. 提取最大重叠窗口
# =========================================================

def extract_words(seq, n=6):
    """
    例如：
    ABCAB -> ABC, BCA, CAB
    """

    if len(seq) < n:
        return []

    return [
        tuple(seq[i:i+n])
        for i in range(len(seq) - n + 1)
    ]


# =========================================================
# 4. 计算 Shannon entropy
# =========================================================

def word_entropy(word):
    """
    H = -sum(p * log2(p))
    """

    counts = Counter(word)
    probabilities = np.array(list(counts.values())) / len(word)

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


# =========================================================
# 5. 构建理论词典及熵类别
# =========================================================

def build_theoretical_dictionary(states, n=6):
    """
    构建无连续重复状态的全部理论词典。

    例如：
    AA 不允许
    AB 允许

    理论词典大小：
    K * (K-1)^(n-1)
    """

    dictionary = []

    for word in product(states, repeat=n):
        if all(
            word[i] != word[i-1]
            for i in range(1, n)
        ):
            dictionary.append(word)

    # 按熵值归类
    entropy_values = np.array([
        word_entropy(word)
        for word in dictionary
    ])

    # 浮点数误差处理
    rounded_entropy = np.round(entropy_values, 10)

    unique_entropies = np.sort(
        np.unique(rounded_entropy)
    )
    print(f'{unique_entropies=}')

    # 类别编号从 1 开始
    entropy_to_class = {
        h: i + 1
        for i, h in enumerate(unique_entropies)
    }

    word_to_class = {
        word: entropy_to_class[h]
        for word, h in zip(dictionary, rounded_entropy)
    }

    # 理论上每个熵类别有多少个词
    theoretical_counts = Counter(word_to_class.values())

    total_words = len(dictionary)

    theoretical_proportions = {
        cls: count / total_words
        for cls, count in theoretical_counts.items()
    }

    return (
        dictionary,
        word_to_class,
        unique_entropies,
        theoretical_counts,
        theoretical_proportions
    )


# =========================================================
# 6. 计算真实序列的 Entropy Representation Ratio
# =========================================================

def calculate_err(
    seq,
    word_to_class,
    theoretical_proportions,
    n=6
):
    """
    返回：
    每个熵类别的真实占比、理论占比、ERR。
    """

    words = extract_words(seq, n)

    if len(words) == 0:
        raise ValueError(
            f"序列长度必须至少为 {n}"
        )

    # 统计各类别的词频
    class_counts = Counter(
        word_to_class[word]
        for word in words
    )

    total_windows = len(words)

    results = []

    for cls in sorted(theoretical_proportions):

        real_count = class_counts.get(cls, 0)

        real_prop = real_count / total_windows

        theoretical_prop = theoretical_proportions[cls]

        err = real_prop / theoretical_prop

        results.append({
            "Class": cls,
            "Real_Count": real_count,
            "Real_Proportion": real_prop,
            "Theoretical_Proportion": theoretical_prop,
            "ERR": err
        })

    return pd.DataFrame(results)


# =========================================================
# 7. Surrogate bootstrap
# =========================================================

def calculate_surrogate_err(
    original_seq,
    word_to_class,
    theoretical_proportions,
    n=6,
    n_surrogates=1000,
    min_run=5,
    seed=42,
    already_preprocessed=False
):
    """
    对原始微状态序列随机打乱顺序。

    每次打乱后重新执行相同的预处理，
    然后计算各熵类别的 ERR。
    """

    rng = np.random.default_rng(seed)

    original_seq = list(original_seq)

    surrogate_results = []

    for _ in range(n_surrogates):

        # 随机打乱原始微状态标签
        shuffled = rng.permutation(original_seq).tolist()

        if already_preprocessed:
            processed = shuffled
        else:
            processed = preprocess_sequence(
                shuffled,
                min_run=min_run
            )

        # 长度不足时跳过该次
        if len(processed) < n:
            continue

        df = calculate_err(
            processed,
            word_to_class,
            theoretical_proportions,
            n=n
        )

        surrogate_results.append(
            df["ERR"].to_numpy()
        )

    if len(surrogate_results) == 0:
        raise ValueError(
            "没有有效 surrogate，请检查序列长度和预处理参数。"
        )

    return np.array(surrogate_results)


# =========================================================
# 8. 主函数
# =========================================================

def microsynt_err(
    seq,
    n=6,
    n_surrogates=1000,
    min_run=5,
    already_preprocessed=False,
    seed=42
):
    """
    单条序列的完整 ERR 分析。
    """

    seq = list(seq)

    if already_preprocessed:
        processed = seq
    else:
        processed = preprocess_sequence(
            seq,
            min_run=min_run
        )

    print(f'{len(processed)=}')

    if len(processed) < n:
        raise ValueError(
            f"预处理后序列长度为 {len(processed)}，"
            f"小于词长 {n}"
        )

    (
        dictionary,
        word_to_class,
        entropy_values,
        theoretical_counts,
        theoretical_proportions
    ) = build_theoretical_dictionary(
        STATES,
        n=n
    )

    print(f'{word_to_class=}')

    # 真实序列 ERR
    real_df = calculate_err(
        processed,
        word_to_class,
        theoretical_proportions,
        n=n
    )

    # Surrogate ERR
    surrogate_err = calculate_surrogate_err(
        seq,
        word_to_class,
        theoretical_proportions,
        n=n,
        n_surrogates=n_surrogates,
        min_run=min_run,
        seed=seed,
        already_preprocessed=already_preprocessed
    )

    # 95% 双侧 surrogate 区间
    lower = np.percentile(surrogate_err, 2.5, axis=0)
    upper = np.percentile(surrogate_err, 97.5, axis=0)

    real_df["Surrogate_Lower"] = lower
    real_df["Surrogate_Upper"] = upper

    real_df["Significant"] = (
        (real_df["ERR"] < lower) |
        (real_df["ERR"] > upper)
    )

    return real_df, surrogate_err, processed


# %%
# =========================================================
# 9. 示例：替换成你自己的序列
# =========================================================

if __name__ == "__main__":

    # 示例数据：这里替换为你自己的序列
    # seq = list("AAABBBBBCCCCCAAAAABBBBBCCCCC")
    seq = list(sequence)

    results, surrogate, processed = microsynt_err(
        seq,
        n=WORD_SIZE,
        n_surrogates=N_SURROGATES,
        min_run=MIN_RUN,
        already_preprocessed=False,
        seed=RANDOM_SEED
    )

    print("预处理后序列：")
    print("".join(processed))

    print("\nEntropy Representation Ratio：")
    print(results.to_string(index=False))

    # results = results.copy().sort_values(by='Theoretical_Proportion')
    # results['class'] = [1, 2, 3, 4, 5]

    # 绘制 ERR
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

# %%
# %%

# %%
