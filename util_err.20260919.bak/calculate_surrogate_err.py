import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory


# ============================================================
# Worker 全局变量
# ============================================================

_WORKER_SEQ = None
_WORKER_SHM = None
_WORKER_WORD_TO_CLASS = None
_WORKER_THEORETICAL_PROPORTIONS = None
_WORKER_N = None
_WORKER_MIN_RUN = None
_WORKER_ALREADY_PREPROCESSED = None


def _init_worker(
    shm_name,
    seq_shape,
    seq_dtype,
    word_to_class,
    theoretical_proportions,
    n,
    min_run,
    already_preprocessed
):
    """
    每个 worker 进程初始化一次。

    original_seq 不通过 pickle 传递，
    而是通过 shared memory 访问。
    """

    global \
        _WORKER_SEQ, \
        _WORKER_SHM, \
        _WORKER_WORD_TO_CLASS, \
        _WORKER_THEORETICAL_PROPORTIONS, \
        _WORKER_N, \
        _WORKER_MIN_RUN, \
        _WORKER_ALREADY_PREPROCESSED

    # 连接到主进程创建的 shared memory
    _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)

    _WORKER_SEQ = np.ndarray(
        seq_shape,
        dtype=np.dtype(seq_dtype),
        buffer=_WORKER_SHM.buf
    )

    # 这些对象只在 worker 初始化时传一次
    _WORKER_WORD_TO_CLASS = word_to_class
    _WORKER_THEORETICAL_PROPORTIONS = theoretical_proportions
    _WORKER_N = n
    _WORKER_MIN_RUN = min_run
    _WORKER_ALREADY_PREPROCESSED = already_preprocessed


def _calculate_one_surrogate(seed):
    """
    worker 中执行一次 permutation。
    """

    rng = np.random.default_rng(seed)

    # 在 shared memory 上的原始序列进行 permutation
    shuffled = rng.permutation(_WORKER_SEQ)

    if _WORKER_ALREADY_PREPROCESSED:
        processed = shuffled.tolist()
    else:
        processed = preprocess_sequence(
            shuffled.tolist(),
            min_run=_WORKER_MIN_RUN
        )

    if len(processed) < _WORKER_N:
        return None

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

    Parameters
    ----------
    original_seq : array-like
        原始微状态序列。推荐使用整数标签，例如 0,1,2,3。

    word_to_class : dict
        word -> entropy class 的映射。

    theoretical_proportions : array-like
        理论比例。

    n : int
        entropy word 长度。

    n_surrogates : int
        surrogate 数量。

    min_run : int
        preprocess_sequence 的 min_run 参数。

    seed : int
        随机种子。

    already_preprocessed : bool
        是否已经预处理。

    n_jobs : int
        并行进程数，例如 32。
        -1 表示使用全部 CPU。
    """

    # --------------------------------------------------------
    # 参数处理
    # --------------------------------------------------------

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()

    if n_jobs < 1:
        raise ValueError("n_jobs 必须 >= 1，或者使用 -1。")

    # 不要保留 Python list
    original_seq = np.asarray(original_seq)

    if original_seq.ndim != 1:
        raise ValueError(
            "original_seq 必须是一维序列。"
        )

    if len(original_seq) < n:
        raise ValueError(
            f"original_seq 长度 ({len(original_seq)}) 小于 n ({n})。"
        )

    # --------------------------------------------------------
    # 检查 dtype
    # --------------------------------------------------------

    if original_seq.dtype.kind not in "biuf":
        raise TypeError(
            "为了使用 shared memory，original_seq 应该是数值型，"
            "例如 int8/int16/int32/int64。"
            f"当前 dtype={original_seq.dtype}"
        )

    # --------------------------------------------------------
    # 创建 shared memory
    # --------------------------------------------------------

    shm = shared_memory.SharedMemory(
        create=True,
        size=original_seq.nbytes
    )

    try:

        # 把原始序列放进 shared memory
        shared_seq = np.ndarray(
            original_seq.shape,
            dtype=original_seq.dtype,
            buffer=shm.buf
        )

        shared_seq[:] = original_seq

        # ----------------------------------------------------
        # 生成独立、可复现的 random seeds
        # ----------------------------------------------------

        seed_sequence = np.random.SeedSequence(seed)

        child_sequences = seed_sequence.spawn(
            n_surrogates
        )

        # 用 uint64 seed，pickle 更轻
        seeds = [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in child_sequences
        ]

        # ----------------------------------------------------
        # 启动 worker
        # ----------------------------------------------------

        surrogate_results = []

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(
                shm.name,
                original_seq.shape,
                original_seq.dtype,
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

            # as_completed 能让 tqdm 实时更新
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
        # ----------------------------------------------------
        # 非常重要：释放 shared memory
        # ----------------------------------------------------

        shm.close()
        shm.unlink()

    # --------------------------------------------------------
    # 检查结果
    # --------------------------------------------------------

    if len(surrogate_results) == 0:
        raise ValueError(
            "没有有效 surrogate，请检查序列长度和预处理参数。"
        )

    return np.asarray(surrogate_results)
