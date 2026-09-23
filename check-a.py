# %%
from IPython.display import display
import pandas as pd
from collections import defaultdict
from util.easy_imports import *

# %%
DATA_DIR = Path('./output-err-20260920/data/seq-data-20260920')

# %%
files = sorted(DATA_DIR.rglob('*/results-concat.json'))

print(files[:8])

dct = defaultdict(list)
for p in files:
    tag = p.parent.name.split('_')[0]
    dct[tag].append(p)

json_files_dct = dict(dct)
for k, v in json_files_dct.items():
    print(k, len(v), v[:2])
# print(json_files_dct)

# %%

dfs = []
for key, value in tqdm(json_files_dct.items()):
    logger.debug(f'Working with {key}')
    for p in tqdm(value, key):
        df = pd.read_json(p)
        df['tag'] = key
        dfs.append(df)
df = pd.concat(dfs)
display(df)

# %%
group = df.groupby(['Class', 'tag'])
display(group.mean(numeric_only=True))

# %%
fig, ax = plt.subplots(figsize=(12, 6))

sns.boxenplot(data=df, x='Class', y='ERR', hue='tag', ax=ax)

tags = df['tag'].unique()
classes = sorted(df['Class'].unique())
n_tags = len(tags)
total_width = 0.8
width = total_width / n_tags
offsets = {tag: (i - (n_tags - 1) / 2) * width for i, tag in enumerate(tags)}

for i, cls in enumerate(classes):
    for tag in tags:
        subset = df[(df['Class'] == cls) & (df['tag'] == tag)]
        if subset.empty:
            continue
        x_center = i + offsets[tag]
        lower = subset['Surrogate_Lower'].values[0]
        upper = subset['Surrogate_Upper'].values[0]
        half_w = width * 0.35

        ax.fill_between([x_center - half_w, x_center + half_w],
                        lower, upper,
                        color='gray', alpha=0.4, zorder=4)

plt.tight_layout()
plt.show()

fpath = DATA_DIR / 'err-results-20260920.csv'
df.to_csv(fpath)
logger.info(f'Saved into {fpath=}')

# %%
