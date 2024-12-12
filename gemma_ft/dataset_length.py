# データセットの質問文と回答文の長さの分布を可視化
# %%
# %%
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from gemma_ft.jsonl_dataset import JSONLDataset
from scraping.g_category_info import category_dict

sys.path.append("..")
# %%
q_lengths = []
a_lengths = []

for category in category_dict.keys():
    dataset = JSONLDataset(f"../data/goo_q_n_a/{category}.jsonl")
    for data in dataset:
        q_lengths.append(len(data["q_text"]))
        a_lengths.append(len(str(data["a_texts"])))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=[q_lengths, a_lengths])
plt.xticks([0, 1], ["q_text", "a_texts"])
plt.ylim(0, 5000)
plt.title("Boxplot of q_text and a_texts Lengths")

plt.subplot(1, 2, 2)
sns.histplot(q_lengths, kde=True, color="blue", label="q_text")
sns.histplot(a_lengths, kde=True, color="green", label="a_texts")
plt.legend()
plt.xlim(0, 5000)
plt.title("Distribution of q_text and a_texts Lengths")

plt.tight_layout()
plt.show()

# %%
