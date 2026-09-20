# %%
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
files = sorted(DATA_DIR.rglob('*/results.json'))
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
df = pd.concat(dfs)
display(df)

group = df.groupby(['Class', 'tag'])
print(group.mean(numeric_only=True))

group = df.groupby('tag')
print(group.sum(numeric_only=True))

# %%
# exit(0)


# %%
# fig = plt.figure(figsize=(12, 6))
# sns.boxenplot(df, x='Class', y='ERR', hue='tag')
# plt.show()


# %%
fig, ax = plt.subplots(figsize=(12, 6))

sns.boxenplot(data=df, x='Class', y='ERR', hue='tag', ax=ax)

tags = df['tag'].unique()
classes = sorted(df['Class'].unique())
n_tags = len(tags)
total_width = 0.8
width = total_width / n_tags
offsets = {tag: (i - (n_tags - 1) / 2) * width for i, tag in enumerate(tags)}

# for i, cls in enumerate(classes):
#     for tag in tags:
#         subset = df[(df['Class'] == cls) & (df['tag'] == tag)]
#         if subset.empty:
#             continue
#         x_center = i + offsets[tag]
#         lower = subset['Surrogate_Lower'].values[0]
#         upper = subset['Surrogate_Upper'].values[0]
#         half_w = width * 0.35

#         ax.fill_between([x_center - half_w, x_center + half_w],
#                         lower, upper,
#                         color='gray', alpha=0.4, zorder=4)

plt.tight_layout()
plt.show()

# %%
