
# %%
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from itertools import product
from tqdm.auto import tqdm
from collections import Counter

# %%
STATES = list("ABCD")

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
# ============================================================
# Worker globals
# ============================================================

_WORKER_SEQ = None
_WORKER_SHM = None
_WORKER_LABELS = None

_WORKER_WORD_TO_CLASS = None
_WORKER_THEORETICAL_PROPORTIONS = None
_WORKER_N = None
_WORKER_MIN_RUN = None
_WORKER_ALREADY_PREPROCESSED = None


def _init_worker(
    shm_name,
    seq_shape,
    seq_dtype,
    labels,
    word_to_class,
    theoretical_proportions,
    n,
    min_run,
    already_preprocessed
):
    global \
        _WORKER_SEQ, \
        _WORKER_SHM, \
        _WORKER_LABELS, \
        _WORKER_WORD_TO_CLASS, \
        _WORKER_THEORETICAL_PROPORTIONS, \
        _WORKER_N, \
        _WORKER_MIN_RUN, \
        _WORKER_ALREADY_PREPROCESSED

    # 连接 shared memory
    _WORKER_SHM = shared_memory.SharedMemory(
        name=shm_name
    )

    _WORKER_SEQ = np.ndarray(
        seq_shape,
        dtype=np.dtype(seq_dtype),
        buffer=_WORKER_SHM.buf
    )

    # int -> 原始字符串标签
    _WORKER_LABELS = labels

    _WORKER_WORD_TO_CLASS = word_to_class
    _WORKER_THEORETICAL_PROPORTIONS = theoretical_proportions
    _WORKER_N = n
    _WORKER_MIN_RUN = min_run
    _WORKER_ALREADY_PREPROCESSED = already_preprocessed


def _calculate_one_surrogate(seed):
    """
    单个 surrogate。
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # permutation
    # --------------------------------------------------------

    shuffled_codes = rng.permutation(_WORKER_SEQ)

    # --------------------------------------------------------
    # 转回原始标签
    #
    # preprocess_sequence / calculate_err 如果要求
    # ['A', 'B', 'C', ...]，这里恢复成字符串。
    # --------------------------------------------------------

    shuffled = _WORKER_LABELS[shuffled_codes].tolist()

    # --------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------

    if _WORKER_ALREADY_PREPROCESSED:
        processed = shuffled

    else:
        processed = preprocess_sequence(
            shuffled,
            min_run=_WORKER_MIN_RUN
        )

    # --------------------------------------------------------
    # 长度检查
    # --------------------------------------------------------

    if len(processed) < _WORKER_N:
        return None

    # --------------------------------------------------------
    # ERR
    # --------------------------------------------------------

    df = calculate_err(
        processed,
        _WORKER_WORD_TO_CLASS,
        _WORKER_THEORETICAL_PROPORTIONS,
        n=_WORKER_N
    )

    return df["ERR"].to_numpy()


def calculate_surrogate_err(
    original_seq,
    word_to_class,
    theoretical_proportions,
    n=6,
    n_surrogates=1000,
    min_run=5,
    seed=42,
    already_preprocessed=False,
    n_jobs=32
):
    """
    并行计算 permutation surrogate ERR。

    支持字符串微状态标签，例如：

        ['A', 'B', 'C', 'D', 'A', ...]

    使用 shared memory 保存整数编码后的序列。
    """

    # ========================================================
    # n_jobs
    # ========================================================

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()

    if n_jobs < 1:
        raise ValueError(
            "n_jobs 必须 >= 1，或者使用 -1。"
        )

    # ========================================================
    # 原始序列
    # ========================================================

    original_seq = np.asarray(
        original_seq
    )

    if original_seq.ndim != 1:
        raise ValueError(
            "original_seq 必须是一维序列。"
        )

    if len(original_seq) < n:
        raise ValueError(
            f"original_seq 长度 ({len(original_seq)}) "
            f"小于 n ({n})。"
        )

    # ========================================================
    # 字符串标签 → integer codes
    # ========================================================
    #
    # 例如：
    #
    # A B C D A C
    #
    # ↓
    #
    # 0 1 2 3 0 2
    #
    # labels = ['A', 'B', 'C', 'D']
    #
    # ========================================================

    labels, encoded_seq = np.unique(
        original_seq,
        return_inverse=True
    )

    encoded_seq = encoded_seq.astype(
        np.int8 if len(labels) <= 127 else np.int16
    )

    # ========================================================
    # shared memory
    # ========================================================

    shm = shared_memory.SharedMemory(
        create=True,
        size=encoded_seq.nbytes
    )

    try:

        shared_seq = np.ndarray(
            encoded_seq.shape,
            dtype=encoded_seq.dtype,
            buffer=shm.buf
        )

        shared_seq[:] = encoded_seq

        # ====================================================
        # 独立 random seeds
        # ====================================================

        seed_sequence = np.random.SeedSequence(seed)

        child_sequences = seed_sequence.spawn(
            n_surrogates
        )

        seeds = [
            int(
                child.generate_state(
                    1,
                    dtype=np.uint64
                )[0]
            )
            for child in child_sequences
        ]

        # ====================================================
        # Process pool
        # ====================================================

        surrogate_results = []

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(
                shm.name,
                encoded_seq.shape,
                encoded_seq.dtype,
                labels,
                word_to_class,
                theoretical_proportions,
                n,
                min_run,
                already_preprocessed,
            )
        ) as executor:

            futures = [
                executor.submit(
                    _calculate_one_surrogate,
                    s
                )
                for s in seeds
            ]

            # =================================================
            # 实时进度
            # =================================================

            for future in tqdm(
                as_completed(futures),
                total=n_surrogates,
                desc="Permutation",
                unit="it"
            ):

                result = future.result()

                if result is not None:
                    surrogate_results.append(result)

    finally:

        # ====================================================
        # 清理 shared memory
        # ====================================================

        shm.close()
        shm.unlink()

    # ========================================================
    # 最终结果
    # ========================================================

    if len(surrogate_results) == 0:
        raise ValueError(
            "没有有效 surrogate，请检查序列长度和预处理参数。"
        )

    return np.asarray(
        surrogate_results
    )


def calculate_surrogate_err_1(
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

    for _ in tqdm(range(n_surrogates), 'Permutation'):

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

    # print(f'{word_to_class=}')

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
