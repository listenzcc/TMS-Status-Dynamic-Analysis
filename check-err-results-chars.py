# %%
from itertools import combinations
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from rich import print
from loguru import logger
from pathlib import Path
from collections import defaultdict
from IPython.display import display

# %%
DATA_DIR = Path('./output-err/data/MSClass_labels')

# %%
files = sorted(DATA_DIR.rglob('*/results_chars.json'))
print(files)

dct = defaultdict(list)
for p in files:
    tag = f'{p.parent.parent.parent.name}/{p.parent.parent.name}'
    dct[tag].append(p)

json_files_dct = dict(dct)
print(json_files_dct)

# %%
dfs = []
for key, value in json_files_dct.items():
    logger.debug(f'Working with {key}')
    for p in value:
        df = pd.read_json(p)
        df['tag'] = key
        dfs.append(df)
df_chars = pd.concat(dfs)
display(df_chars)


# %%
# Display
query = 'Class==1'
df = df_chars.copy().query(query)
display(df)

# %%
if True:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxenplot(df, x='Chars', y='Real_Proportion', hue='tag', ax=ax)
    plt.title(query)
    plt.show()


# exit(0)

# %%


def analyze_chars_by_tag(df, tag_col='tag', group_col='Chars', value='Real_Proportion'):
    tags = sorted(df[tag_col].unique())
    anova_rows = []
    pairwise_rows = []

    for tg in tags:
        sub = df[df[tag_col] == tg]
        groups = {g: sub[sub[group_col] == g][value].dropna()
                  for g in sub[group_col].unique()}
        groups = {g: v for g, v in groups.items() if len(v) >= 2}
        keys = sorted(groups.keys())

        # ---- 1) 整体检验：ANOVA + Kruskal-Wallis ----
        if len(keys) >= 2:
            f_stat, p_anova = stats.f_oneway(*[groups[k] for k in keys])
            h_stat, p_kw = stats.kruskal(*[groups[k] for k in keys])
        else:
            f_stat = p_anova = h_stat = p_kw = np.nan

        anova_rows.append({
            'tag': tg,
            'n_groups': len(keys),
            'groups': ','.join(keys),
            'F': f_stat,
            'p_anova': p_anova,
            'H': h_stat,
            'p_kruskal': p_kw,
        })

        # ---- 2) 两两 t 检验（Welch） ----
        for g1, g2 in combinations(keys, 2):
            a, b = groups[g1], groups[g2]
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

            # Cohen's d
            n1, n2 = len(a), len(b)
            s1, s2 = a.std(ddof=1), b.std(ddof=1)
            sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan

            pairwise_rows.append({
                'tag': tg,
                'group1': g1,
                'group2': g2,
                'n1': n1,
                'n2': n2,
                'mean1': a.mean(),
                'mean2': b.mean(),
                'diff': a.mean() - b.mean(),
                't': t_stat,
                'p': p_val,
                'cohen_d': d,
            })

    anova_df = pd.DataFrame(anova_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    # ---- 3) FDR 校正（每个 tag 内部校正） ----
    def bh(p):
        p = np.asarray(p, dtype=float)
        n = len(p)
        if n == 0:
            return p
        order = np.argsort(p)
        ranked = np.empty(n)
        ranked[order] = np.arange(1, n+1)
        adj = p * n / ranked
        # 单调化
        adj = np.minimum.accumulate(adj[order][::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(adj, 0, 1)
        return out

    if not pairwise_df.empty:
        pairwise_df['p_adj'] = (pairwise_df.groupby('tag')['p']
                                .transform(lambda s: bh(s.values)))

        def stars(p):
            if pd.isna(p):
                return ''
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return 'ns'
        pairwise_df['sig'] = pairwise_df['p_adj'].apply(stars)

    return anova_df, pairwise_df


anova_df, pairwise_df = analyze_chars_by_tag(df)

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)

print('===== 每个 tag 的整体检验 (ANOVA + Kruskal-Wallis) =====')
print(anova_df.to_string(index=False))

print('\n===== 每个 tag 内 Chars 两两比较 =====')
print(pairwise_df.query('sig != "ns"').to_string(index=False))

# 导出
# anova_df.to_csv('anova_by_tag.csv', index=False)
# pairwise_df.to_csv('pairwise_by_tag.csv', index=False)

# %% -------------------------------------


def pre_post_by_chars(df, tag_col='tag', chars_col='Chars', value='Real_Proportion'):
    # ---- 1) 从 tag 解析出 treatment 和 phase ----
    tmp = df.copy()

    parts = tmp[tag_col].str.split('/', expand=True)
    tmp['treatment'] = parts[0]
    tmp['phase'] = parts[1]

    rows = []
    for (chars, treat), sub in tmp.groupby([chars_col, 'treatment']):
        pre = sub[sub['phase'] == 'pre'][value].dropna()
        post = sub[sub['phase'] == 'post'][value].dropna()

        # ? Convert into entropy?
        # pre = -np.log2(pre)
        # post = -np.log2(post)

        if len(pre) < 2 or len(post) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(
            pre, post, equal_var=False, alternative='two-sided')

        # Cohen's d (pooled)
        n1, n2 = len(pre), len(post)
        s1, s2 = pre.std(ddof=1), post.std(ddof=1)
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = (pre.mean() - post.mean()) / sp if sp > 0 else np.nan

        rows.append({
            'Chars': chars,
            'treatment': treat,
            'n_pre': n1,
            'n_post': n2,
            'mean_pre': pre.mean(),
            'mean_post': post.mean(),
            'diff_pre_minus_post': pre.mean() - post.mean(),
            't': t_stat,
            'p': p_val,
            'cohen_d': d,
        })

    out = pd.DataFrame(rows)

    # ---- 2) 多重比较校正（BH-FDR，按 treatment 内校正） ----
    def bh(p):
        p = np.asarray(p, dtype=float)
        n = len(p)
        if n == 0:
            return p
        order = np.argsort(p)
        ranked = np.empty(n)
        ranked[order] = np.arange(1, n + 1)
        adj = p * n / ranked
        adj = np.minimum.accumulate(adj[order][::-1])[::-1]
        res = np.empty(n)
        res[order] = np.clip(adj, 0, 1)
        return res

    if not out.empty:
        out['p_adj'] = (out.groupby('treatment')['p']
                        .transform(lambda s: bh(s.values)))

        def stars(p):
            if pd.isna(p):
                return ''
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return 'ns'
        out['sig'] = out['p_adj'].apply(stars)

    # 排序，便于阅读
    out = out.sort_values(['treatment', 'Chars']).reset_index(drop=True)
    return out


result = pre_post_by_chars(df)

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 220)

print(result.to_string(index=False))

# result.to_csv('pre_post_by_chars.csv', index=False)
# %%
