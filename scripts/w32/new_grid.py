import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.plot_helpers import *
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.accretor import *
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

with open(
    f"/home/koen/master-internship/scripts/w31/cache/df.pkl",
    "rb",
) as f:
    df_results = pickle.load(f)
df = AbundanceTables()


def get_ls(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 3


def get_hs(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += ab.iron_abundance[i]

    return fe / 4


def get_s(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


def get_ba(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba"]:
            fe += ab.iron_abundance[i]

    return fe


# %%

from matplotlib.colors import TwoSlopeNorm

subset = df_results[(np.abs(df_results["eps"] - 0.11288379) < 0.01)]


norm = TwoSlopeNorm(
    vcenter=0.25,
    vmin=np.min(subset["s"]),
    vmax=np.max(subset["s"]),
)


fig, axs = plt.subplots(
    5,
    4,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1.5),
    constrained_layout=True,
)

axs = axs.flatten()

qs = np.unique(df_results["qi"])
for i, ax in enumerate(axs):
    if i == 19:
        ax.axis("off")
        continue

    subset = df_results[
        (np.abs(df_results["eps"] - 0.11288379) < 0.01) & (df_results["qi"] == qs[i])
    ]

    s_grid = subset.pivot(
        index="m1i",
        columns="r_init",
        values="s",
    )

    im = ax.pcolormesh(
        s_grid.columns,
        s_grid.index,
        s_grid.values,
        norm=norm,
        cmap="coolwarm",
        rasterized=True,
    )

    ax.set_xscale("log")
    ax.set_title(f"$q = {qs[i]:.2f}$")
# ax.set_yscale("log")

cb = fig.colorbar(
    im,
    ax=axs[1:3],
    label=r"$[\mathrm{s}/\mathrm{Fe}]$",
    orientation="horizontal",
    location="top",
)
cb.ax.set_xscale("linear")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.supxlabel("$R_\\textrm{RL}$ ($R_\\odot$)", fontsize=10)
fig.supylabel("$M_\\textrm{TPAGB,i}$ ($M_\\odot$)", fontsize=10)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w32-R-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

qs = [0.40, 0.55, 0.70, 0.85, 1]
ms = [1.4, 1.8, 2.2, 2.6, 3.0]

eps = [0, 0.1]

# %%

from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    4,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1.5),
    constrained_layout=True,
)

subset = df_results[(np.abs(df_results["eps"] - 0.11288379) < 0.01)]


norm = TwoSlopeNorm(
    vcenter=0.25,
    vmin=np.min(subset["s"]),
    vmax=np.max(subset["s"]),
)


axs = axs.flatten()

qs = np.unique(df_results["qi"])
print(qs)

qs = np.array([0.40, 0.55, 0.70, 0.85, 1])
ms = [1.4, 1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):
    if i in [5, 11]:
        ax.axis("off")
        continue
    if i >= 6:
        continue

    subset = df_results[
        (np.abs(df_results["eps"] - 0.11288379) < 0.01)
        & (np.isclose(df_results["m1i"], ms[i]))
    ]

    s_grid = subset.pivot(
        index="qi",
        columns="r_init",
        values="s",
    )

    im = ax.pcolormesh(
        s_grid.columns,
        s_grid.index,
        s_grid.values,
        norm=norm,
        cmap="coolwarm",
        rasterized=True,
    )

    if np.isclose(ms[i], 1.4):
        rs14 = np.logspace(np.log10(350), np.log10(775), 5)
        rr, qq = np.meshgrid(rs14, qs)
        ax.scatter(rr, qq, c="k", marker=".")
        axs[i + 6].scatter(rr, qq, c="k", marker=".", zorder=1000)

    if np.isclose(ms[i], 1.8):
        rs18 = np.logspace(np.log10(337.5), np.log10(1.05e3), 5)
        rr, qq = np.meshgrid(rs18, qs)
        ax.scatter(rr, qq, c="k", marker=".")
        axs[i + 6].scatter(rr, qq, c="k", marker=".", zorder=1000)

    if np.isclose(ms[i], 2.2):
        rs22 = np.logspace(np.log10(325), np.log10(1.25e3), 5)
        rr, qq = np.meshgrid(rs22, qs)
        ax.scatter(rr, qq, c="k", marker=".")
        axs[i + 6].scatter(rr, qq, c="k", marker=".", zorder=1000)

    if np.isclose(ms[i], 2.6):
        rs26 = np.logspace(np.log10(300), np.log10(1.3e3), 5)
        rr, qq = np.meshgrid(rs26, qs)
        ax.scatter(rr, qq, c="k", marker=".")
        axs[i + 6].scatter(rr, qq, c="k", marker=".", zorder=1000)

    if np.isclose(ms[i], 3.0):
        rs30 = np.logspace(np.log10(337.5), np.log10(1.5e3), 5)
        rr, qq = np.meshgrid(rs30, qs)
        ax.scatter(rr, qq, c="k", marker=".")
        axs[i + 6].scatter(rr, qq, c="k", marker=".", zorder=1000)

    ax.set_xscale("log")
    ax.set_title(f"$M = {ms[i]:.2f}$")
# ax.set_yscale("log")

cb = fig.colorbar(
    im,
    ax=axs[0:3],
    label=r"$[\mathrm{s}/\mathrm{Fe}]$",
    orientation="horizontal",
    location="top",
)
cb.ax.set_xscale("linear")

subset = df_results[(np.abs(df_results["eps"] - 0.001) < 0.0001)]


norm = TwoSlopeNorm(
    vcenter=0.25,
    vmin=np.min(subset["s"]),
    vmax=np.max(subset["s"]),
)


axs = axs.flatten()

qs = np.unique(df_results["qi"])
print(qs)

qs = np.array([0.40, 0.55, 0.70, 0.85, 1])
ms = [1.4, 1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):
    j = i + 6
    if i >= 5:
        continue

    subset = df_results[
        (np.abs(df_results["eps"] - 0.001) < 0.0001)
        & (np.isclose(df_results["m1i"], ms[i]))
    ]

    s_grid = subset.pivot(
        index="qi",
        columns="r_init",
        values="s",
    )

    im = axs[j].pcolormesh(
        s_grid.columns,
        s_grid.index,
        s_grid.values,
        norm=norm,
        cmap="coolwarm",
        rasterized=True,
    )

    axs[j].set_xscale("log")
    axs[j].set_title(f"$M = {ms[i]:.2f}$")
# ax.set_yscale("log")

cb = fig.colorbar(
    im,
    ax=axs[6:9],
    label=r"$[\mathrm{s}/\mathrm{Fe}]$",
    orientation="horizontal",
    location="top",
)
cb.ax.set_xscale("linear")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.supxlabel("$R_\\textrm{RL}$ ($R_\\odot$)", fontsize=10)
fig.supylabel("$M_\\textrm{TPAGB,i}$ ($M_\\odot$)", fontsize=10)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w32-R-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

rs14 = np.logspace(np.log10(350), np.log10(775), 5)
rs18 = np.logspace(np.log10(337.5), np.log10(1.05e3), 5)
rs22 = np.logspace(np.log10(325), np.log10(1.25e3), 5)
rs26 = np.logspace(np.log10(300), np.log10(1.5e3), 5)
rs30 = np.logspace(np.log10(337.5), np.log10(1.5e3), 5)
rr, qq = np.meshgrid(rs26, qs)

# %%
print(qs)
# %%
