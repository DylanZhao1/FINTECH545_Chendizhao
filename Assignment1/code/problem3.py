"""
Problem 3: Pearson Against Spearman
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "problem3.csv")
cols = list(df.columns)
pairs = list(itertools.combinations(cols, 2))

# Pairwise scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (a, b) in zip(axes.flatten(), pairs):
    ax.scatter(df[a], df[b], alpha=0.4, s=12, color="#4C72B0")
    ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{a} vs {b}")
plt.tight_layout()
plt.savefig("problem3_scatter_matrix.png", dpi=150)

# Pearson and Spearman correlation matrices
pearson_corr = df.corr(method="pearson")
spearman_corr = df.corr(method="spearman")
print("PEARSON:\n", pearson_corr.round(4))
print("\nSPEARMAN:\n", spearman_corr.round(4))

# Gap between the two measures
gap_df = pd.DataFrame(
    [(a, b, pearson_corr.loc[a, b], spearman_corr.loc[a, b],
      abs(pearson_corr.loc[a, b] - spearman_corr.loc[a, b])) for a, b in pairs],
    columns=["var1", "var2", "pearson", "spearman", "abs_gap"]
).sort_values("abs_gap", ascending=False).reset_index(drop=True)
print("\nPAIRWISE GAPS:\n", gap_df.round(4))

# plot for the pair with the largest gap: raw values vs. ranks
a, b = gap_df.iloc[0][["var1", "var2"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(df[a], df[b], alpha=0.5, s=18, color="#C44E52")
axes[0].set_title(f"Raw values: {a} vs {b}\nPearson r = {gap_df.iloc[0]['pearson']:.3f}")
axes[0].set_xlabel(a); axes[0].set_ylabel(b)
axes[1].scatter(df[a].rank(), df[b].rank(), alpha=0.5, s=18, color="#55A868")
axes[1].set_title(f"Ranks: rank({a}) vs rank({b})\nSpearman rho = {gap_df.iloc[0]['spearman']:.3f}")
axes[1].set_xlabel(f"rank({a})"); axes[1].set_ylabel(f"rank({b})")
plt.tight_layout()
plt.savefig("problem3_top_gap_pair.png", dpi=150)
