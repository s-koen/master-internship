import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pandas as pd

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

data = pd.read_csv("scripts/w22/Ba_star_orbits.csv")

# %%
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.3), constrained_layout=True
)

d = dwarf["M"].dropna().to_numpy().astype(np.float64)

print(d)

plt.scatter(d, d * 0, s=1.5, marker="o", color="k")
# plt.scatter(mild["[Fe/H]"], mild["[Fe/H]"] * 0 + 1, s=1.5, marker="o", color="k")
# plt.scatter(strong["[Fe/H]"], strong["[Fe/H]"] * 0 + 2, s=1.5, marker="o", color="k")

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
# plt.xlim(-1, 0.25)
# plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25])

plt.xlabel("[Fe/H]")
labels = ["dwarfs", "mild giants", "strong giant"]
plt.yticks([0, 1, 2], labels)
plt.ylim(-0.5, 2.5)
plt.savefig("/home/koen/LaTeX-setup/plots/w22-scatter.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.3), constrained_layout=True
)

data = [
    dwarf["[Fe/H]"].dropna().to_numpy(),
    mild["[Fe/H]"].dropna().to_numpy(),
    strong["[Fe/H]"].dropna().to_numpy(),
]
labels = ["dwarfs", "mild giants", "strong giant"]
boxprops = dict(linestyle="-", linewidth=0, color="white")
whiskerprops = dict(linestyle="-", linewidth=0.8, color="k", zorder=-100)
meanprops = dict(linestyle="-", linewidth=0.8, color="white")
medianprops = dict(linestyle="-", linewidth=2, color="white", zorder=100)
flierprops = dict(marker=".", markerfacecolor="k", markersize=3, markeredgecolor="k")

bplot = axs.boxplot(
    data,
    orientation="horizontal",
    whis=[10, 90],
    showcaps=True,
    flierprops=flierprops,
    boxprops=boxprops,
    meanprops=meanprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    patch_artist=True,
    labels=labels,
    showmeans=True,
    meanline=True,
)

print(bplot)
# fill with colors
for patch in bplot["boxes"]:
    patch.set_facecolor("C0")
axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
plt.xlim(-1, 0.25)
plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25])
plt.xlabel("[Fe/H]")
plt.savefig("/home/koen/LaTeX-setup/plots/w21-boxplot.pgf", format="pgf")
plt.show()
plt.close()


# %%
data = pd.read_csv("presentation-1/Ba_star_orbits.csv")
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
giant = data[
    data["class"].isin(
        [
            "Ba 0",
            "Ba 0.5",
            "Ba 1",
            "Ba 2",
            "Ba mild",
            "sgCH",
            "sgCH / Ba 1",
            "Ba 3",
            "Ba 4",
            "Ba 5",
            "Ba strong",
            "Ba 5 / eS",
        ]
    )
]

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.25), constrained_layout=True
)

d = dwarf["M"].dropna().to_numpy().astype(np.float64)
g = giant["M"].dropna().to_numpy().astype(np.float64)
plt.scatter(d, d * 0, s=1.5, marker="o", color="k")
plt.scatter(g, g * 0 + 1, s=1.5, marker="o", color="k")

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
# plt.xlim(-1, 0.25)
plt.xticks([0.5, 1, 2, 3, 4, 5, 5.5])

plt.xlabel("Ba-star mass ($M_\odot$)")
labels = ["dwarfs", "giants"]
plt.yticks([0, 1], labels)
plt.ylim(-0.5, 1.5)
plt.savefig("/home/koen/LaTeX-setup/plots/w22-scatter-2.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.25), constrained_layout=True
)

data = [
    dwarf["M"].dropna().to_numpy().astype(np.float64),
    giant["M"].dropna().to_numpy().astype(np.float64),
]
labels = ["dwarfs", "giants"]
boxprops = dict(linestyle="-", linewidth=0, color="white")
whiskerprops = dict(linestyle="-", linewidth=0.8, color="k", zorder=-100)
meanprops = dict(linestyle="-", linewidth=0.8, color="white")
medianprops = dict(linestyle="-", linewidth=2, color="white", zorder=100)
flierprops = dict(marker=".", markerfacecolor="k", markersize=3, markeredgecolor="k")

bplot = axs.boxplot(
    data,
    orientation="horizontal",
    whis=[10, 90],
    showcaps=True,
    flierprops=flierprops,
    boxprops=boxprops,
    meanprops=meanprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    patch_artist=True,
    labels=labels,
    showmeans=True,
    meanline=True,
    widths=0.3,
)

print(bplot)
# fill with colors
for patch in bplot["boxes"]:
    patch.set_facecolor("C0")
axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
# plt.xlim(-1, 0.25)
# plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25])
plt.xticks([0.5, 1, 2, 3, 4, 5, 5.5])
plt.xlabel("Ba-star mass ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-boxplot-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

print(
    np.percentile(
        dwarf["M"].dropna().to_numpy().astype(np.float64), [10, 25, 50, 75, 90]
    )
)
print(np.mean(dwarf["M"].dropna().to_numpy().astype(np.float64)))
# %%
df = dwarf[["M", "Mc"]].dropna().astype(np.float64)

print(df)
plt.scatter(df["M"], df["Mc"])
plt.show()
# %%
