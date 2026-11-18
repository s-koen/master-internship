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
plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

from matplotlib.colors import TwoSlopeNorm

# %%

from pathlib import Path
import pandas as pd


def load_grid(file):
    return pd.read_pickle(file)


df_old = load_grid("/home/koen/master-internship/scripts/w36/cache/df2.pkl")
df_new = load_grid("/home/koen/master-internship/scripts/w36/cache/df3.pkl")

df_results = pd.concat([df_old, df_new], ignore_index=True)


import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator


def _sample_power_law(n, m_min, m_max, alpha, rng):
    """Sample from p(m) proportional to m**(-alpha)."""
    u = rng.random(n)

    if np.isclose(alpha, 1.0):
        return m_min * (m_max / m_min) ** u

    exponent = 1.0 - alpha
    return (m_min**exponent + u * (m_max**exponent - m_min**exponent)) ** (
        1.0 / exponent
    )


def sample_salpeter(n, m_min: float = 0.08, m_max: float = 100, rng=None):
    """
    Sample stellar masses from the Salpeter (1955) IMF.
    """
    if rng is None:
        rng = np.random.default_rng()

    return _sample_power_law(n, m_min, m_max, alpha=2.35, rng=rng)


def build_interpolator(df, eps, wind, quantity):
    """
    Build a trilinear interpolator for one outcome quantity
    at a fixed mass-transfer efficiency.

    Values outside the grid are evaluated at the nearest grid boundary.
    """

    data = df[np.isclose(df["eps"], eps) & (df["wind"] == wind)].copy()

    data["logP"] = np.log10(data["p_init"])

    axes = (
        np.sort(data["m1i"].unique()),
        np.sort(data["qi"].unique()),
        np.sort(data["logP"].unique()),
    )

    index = pd.MultiIndex.from_product(
        axes,
        names=["m1i", "qi", "logP"],
    )

    values = (
        data.set_index(["m1i", "qi", "logP"])[quantity]
        .reindex(index)
        .to_numpy()
        .reshape(tuple(len(axis) for axis in axes))
    )

    interpolator = RegularGridInterpolator(
        axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def interpolate(points):
        points = np.asarray(points)
        points = np.clip(
            points,
            [axis[0] for axis in axes],
            [axis[-1] for axis in axes],
        )
        return interpolator(points)

    return interpolate


df = df_results
interpolators = {
    eps: {
        wind: {
            quantity: build_interpolator(df, eps, wind, quantity)
            for quantity in [
                "ba",
                "hs",
                "ls",
                "s",
                "m_env_rlof",
                "m_star_rlof",
                "rlof_tpagb",
                # "m1i",
                # "p_init",
                "m2f",
            ]
        }
        for wind in ["fast", "BHL", "Saladino"]
    }
    for eps in [0, 0.1, 0.2, 0.3, 0.4]
}

n_repeat = 1000
chunk_size = 1_000_000

bins = 30


H_bins = np.zeros((bins, bins), dtype=np.int64)
H_trips = np.zeros((bins, bins), dtype=np.int64)
H_quads = np.zeros((bins, bins), dtype=np.int64)


with open(
    f"/home/koen/master-internship/scripts/w37/cache/q-bins-dump-clip.pkl",
    "rb",
) as f:
    qq_bins = np.array(pickle.load(f)).flatten().astype(np.float32)

with open(
    f"/home/koen/master-internship/scripts/w37/cache/q-trips-dump-clip.pkl",
    "rb",
) as f:
    qq_trips = np.array(pickle.load(f)).flatten().astype(np.float32)

with open(
    f"/home/koen/master-internship/scripts/w37/cache/q-quads-dump-clip.pkl",
    "rb",
) as f:
    qq_quads = np.array(pickle.load(f)).flatten().astype(np.float32)


with open(
    f"/home/koen/master-internship/scripts/w37/cache/m-p-dump-clip.pkl",
    "rb",
) as f:
    mp = pickle.load(f)

mm_sing = np.array(mp["m_sing"])
mm_bins = np.array(mp["m_bins"])
pp_bins = np.array(mp["p_bins"])
mm_trips = np.array(mp["m_trips"])
pp_trips = np.array(mp["p_trips"])
mm_quads = np.array(mp["m_quads"])
pp_quads = np.array(mp["p_quads"])


# %%

fig, axs = plt.subplots(
    5,
    4,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1.25),
    constrained_layout=True,
)

axs = axs.flatten()

qs = np.unique(df_results["qi"])
for i, ax in enumerate(axs):
    values = None
    if i == 19:
        ax.axis("off")
        continue

    for wind in ["Saladino", "BHL", "fast"]:
        subset = df_results[
            (np.abs(df_results["eps"] - 0.1) < 0.01)
            & (df_results["qi"] == qs[i])
            & (df_results["wind"] == wind)
        ]

        s_grid = subset.pivot(
            index="m1i",
            columns="p_init",
            values="rlof_tpagb",
        )

        ss_grid = subset.pivot(
            index="m1i",
            columns="p_init",
            values="s",
        )

        ss_grid = ss_grid.values > 0.25

        if type(values) == type(None):
            values = s_grid.values.T.astype(np.float64) + 1 / 3 * ss_grid.T.astype(
                np.float64
            )
        else:
            values += s_grid.values.T.astype(np.float64) + 1 / 3 * ss_grid.T.astype(
                np.float64
            )

    print(values)
    im = ax.pcolormesh(
        s_grid.index,
        s_grid.columns,
        values,
        # norm=norm,
        cmap="Spectral",
        rasterized=True,
    )

    ax.set_yscale("log")
    ax.set_title(f"$q = {qs[i]:.2f}$")
# ax.set_yscale("log")

# cb = fig.colorbar(
#     im,
#     ax=axs[1:3],
#     label=r"$[\mathrm{s}/\mathrm{Fe}]$",
#     orientation="horizontal",
#     location="top",
# )
# cb.ax.set_xscale("linear")

axs[-1].annotate(
    "no Ba-star; no RLOF",
    xy=(1, 200),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    va="center",
    fontsize=8,
)

axs[-1].annotate(
    "no Ba-star; RLOF",
    xy=(1, 1.1e3),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    va="center",
    fontsize=8,
)

axs[-1].annotate(
    "Ba-star; RLOF",
    xy=(1, 3000),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    va="center",
    fontsize=8,
)

axs[-1].annotate(
    "Ba-star; Saladino RLOF",
    xy=(1, 7e3),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    va="center",
    fontsize=8,
)

axs[-1].annotate(
    "Ba-star; no RLOF",
    xy=(1, 1.5e4),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    va="center",
    fontsize=8,
)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.supylabel("Period (days)", fontsize=10)
fig.supxlabel("$M_\\textrm{TPAGB,i}$ ($M_\\odot$)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w37-winds.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("bids:fake_parula").to_mpl()
fig, axs = plt.subplots(
    3, 3, sharex="col", figsize=set_size(column, height=0.85), constrained_layout=True
)


axs[0, 0].hist(
    mm_bins,
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[1, 0].hist2d(
    np.repeat(mm_bins, 1000),
    qq_bins,
    bins=200,
    cmap=cmap,
    density=True,
    rasterized=True,
)
axs[2, 0].hist2d(
    mm_bins, np.log10(pp_bins), bins=50, rasterized=True, cmap=cmap, density=True
)
axs[2, 1].hist2d(
    qq_bins,
    np.repeat(np.log10(pp_bins), 1000),
    bins=200,
    rasterized=True,
    cmap=cmap,
    density=True,
)
axs[1, 1].hist(
    qq_bins,
    bins=200,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[2, 2].hist(
    np.log10(pp_bins),
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)

axs[0, 1].axis("off")
axs[0, 2].axis("off")
axs[1, 2].axis("off")


m_salpeter = sample_salpeter(1000000, 1, 3)
axs[0, 0].hist(m_salpeter, 100, density=True, color="k", histtype="step")
axs[0, 0].annotate(
    "Salpeter IMF",
    xy=(1.15, 1.5),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="left",
    va="center",
    fontsize=10,
)

axs[0, 0].tick_params(labelbottom=False)
axs[0, 0].set_yticks([])
axs[1, 0].tick_params(labelbottom=False)
axs[1, 1].tick_params(labelbottom=False)
axs[1, 1].set_yticks([])
axs[2, 1].tick_params(labelleft=False)
axs[2, 2].set_yticks([])

axs[1, 0].set_ylabel("$q$ ($M_2 / M_1$)")
axs[2, 0].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
axs[2, 0].set_xlabel("$M_1$ ($M_\\odot$)")
axs[2, 1].set_xlabel("$q$ ($M_2 / M_1$)")
for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-population-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

N = len(mm_bins)
n_repeat = 1000
chunk_size = 10_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 6, 2 * bins + 1)

H_fast = np.zeros((bins, 2 * bins), dtype=np.int64)
H_BHL = np.zeros((bins, 2 * bins), dtype=np.int64)
H_Saladino = np.zeros((bins, 2 * bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_bins[j],
            qq_bins[idx],
            np.log10(pp_bins[j]),
        )
    ).astype(np.float32)

    # mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points) > 0.5
    # mask2 = interpolators[0.1]["fast"]["s"](points) > 0.25
    # mask = mask1 * mask2
    #
    # if np.any(mask):
    #     h, _, _ = np.histogram2d(
    #         mm_bins[j[mask]],
    #         np.log10(pp_bins[j[mask]]),
    #         bins=(xedges, yedges),
    #     )
    #     H_fast += h.astype(np.int64)
    #
    # mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points) > 0.5
    # mask2 = interpolators[0.1]["BHL"]["s"](points) > 0.25
    # mask = mask1 * mask2
    #
    # if np.any(mask):
    #     h, _, _ = np.histogram2d(
    #         mm_bins[j[mask]],
    #         np.log10(pp_bins[j[mask]]),
    #         bins=(xedges, yedges),
    #     )
    #     H_BHL += h.astype(np.int64)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.25
    mask = mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm_bins[j[mask]],
            np.log10(pp_bins[j[mask]]),
            bins=(xedges, yedges),
        )
        H_Saladino += h.astype(np.int64)

# %%
import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("colorcet:CET_L19").to_mpl()
# cmap=Colormap("cmasher:chroma_r").to_mpl()


fig, axs = plt.subplot_mosaic(
    [["histx", "."], ["scatter", "histy"]],
    figsize=set_size(column, height=1),
    width_ratios=(3, 1),
    height_ratios=(1, 3),
    layout="constrained",
)

axs["scatter"].pcolormesh(xedges, yedges, H_Saladino.T, rasterized=True)

ax_histx = axs["histx"]
ax_histx.spines[["right", "top", "left"]].set_visible(False)
ax_histx.tick_params(labelbottom=False)
ax_histx.set_yticks([])
ax_histy = axs["histy"]
ax_histy.spines[["right", "top", "bottom"]].set_visible(False)
ax_histy.set_xticks([])
ax_histy.tick_params(labelleft=False)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_fast.T, axis=0) / np.sum(np.sum(H_fast.T, axis=0)) * bins / (2),
    c="C9",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_BHL.T, axis=0) / np.sum(np.sum(H_BHL.T, axis=0)) * bins / (2),
    c="C9",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / (2),
    c="k",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / 2,
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm_bins > xedges[0])
mms = mm_bins[args][np.argwhere(mm_bins[args] < 3)]
hist, edges = np.histogram(mms, bins=20)
print(edges)
ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
# ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    0 * np.sum(H_Saladino.T, axis=1),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp_bins) > yedges[0])
pps = np.log10(pp_bins)[args][np.argwhere(np.log10(pp_bins)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with Saladino Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w37-compare-distribution-sal.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

N = len(mm_bins)
n_repeat = 1000
chunk_size = 1_000_000

bins = 40

# Define fixed bin edges
xedges = np.linspace(0, 5, bins + 1)
yedges = np.linspace(0.2, 8, 2 * bins + 1)

H_s = np.zeros((bins, 2 * bins), dtype=np.int64)
H_all = np.zeros((bins, 2 * bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_bins[j],
            qq_bins[idx],
            np.log10(pp_bins[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_bins[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_bins[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_bins[idx][low_mass] * mm_bins[j][low_mass]
    high_mass = mm_bins[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_bins[idx][high_mass] * mm_bins[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_bins[j][mask]),
            bins=(xedges, yedges),
        )
        H_s += h.astype(np.int64)

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals,
            np.log10(pp_bins[j]),
            bins=(xedges, yedges),
        )
        H_all += h.astype(np.int64)

H_s_trips = np.zeros((bins, 2 * bins), dtype=np.int64)

N = len(mm_trips)
for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_trips[j],
            qq_trips[idx],
            np.log10(pp_trips[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_trips[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_trips[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_trips[idx][low_mass] * mm_trips[j][low_mass]
    high_mass = mm_trips[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_trips[idx][high_mass] * mm_trips[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_trips[j][mask]),
            bins=(xedges, yedges),
        )
        H_s_trips += h.astype(np.int64)


H_s_quads = np.zeros((bins, 2 * bins), dtype=np.int64)

N = len(mm_quads)
for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_quads[j],
            qq_quads[idx],
            np.log10(pp_quads[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_quads[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_quads[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_quads[idx][low_mass] * mm_quads[j][low_mass]
    high_mass = mm_quads[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_quads[idx][high_mass] * mm_quads[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_quads[j][mask]),
            bins=(xedges, yedges),
        )
        H_s_quads += h.astype(np.int64)


# %%

import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("colorcet:CET_L19").to_mpl()
# cmap=Colormap("cmasher:chroma_r").to_mpl()


fig, axs = plt.subplot_mosaic(
    [["histx", "."], ["scatter", "histy"]],
    figsize=set_size(column, height=1),
    width_ratios=(3, 1),
    height_ratios=(1, 3),
    layout="constrained",
)

axs["scatter"].pcolormesh(xedges, yedges, H_s.T, rasterized=True)

ax_histx = axs["histx"]
ax_histx.spines[["right", "top", "left"]].set_visible(False)
ax_histx.tick_params(labelbottom=False)
ax_histx.set_yticks([])
ax_histy = axs["histy"]
ax_histy.spines[["right", "top", "bottom"]].set_visible(False)
ax_histy.set_xticks([])
ax_histy.tick_params(labelleft=False)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(np.sum(H_s.T, axis=0)) * bins / (5),
    c="k",
)
ax_histx.set_ylim(
    0,
)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(np.sum(H_s.T, axis=0)) * bins / 5,
    0 * np.sum(H_s.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(0, 5)

# mms = np.repeat(mm_bins, 1000) *qq_bins
# hist, edges = np.histogram(mms, bins=70)
# print(edges)
# ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_s.T, axis=1) / np.sum(np.sum(H_s.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
# ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_s.T, axis=1) / np.sum(np.sum(H_s.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_s.T, axis=1),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp_bins) > yedges[0])
pps = np.log10(pp_bins)[args][np.argwhere(np.log10(pp_bins)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with Saladino Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_\\textrm{Ba;f}$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-m2-distr.pgf", format="pgf")
plt.show()
plt.close()

# %%

import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("colorcet:CET_L19").to_mpl()
# cmap=Colormap("cmasher:chroma_r").to_mpl()


fig, axs = plt.subplot_mosaic(
    [["histx", "."], ["scatter", "histy"]],
    figsize=set_size(column, height=1),
    width_ratios=(3, 1),
    height_ratios=(1, 3),
    layout="constrained",
)

axs["scatter"].pcolormesh(xedges, yedges, H_s.T / H_all.T, rasterized=True)

ax_histx = axs["histx"]
ax_histx.spines[
    [
        "right",
        "top",
    ]
].set_visible(False)
ax_histy = axs["histy"]
ax_histy.spines[
    [
        "right",
        "top",
    ]
].set_visible(False)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(H_all.T, axis=0),
    c="k",
)
ax_histx.set_ylim(
    0,
)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(H_all.T, axis=0),
    0 * np.sum(H_all.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(0, 4)

# mms = np.repeat(mm_bins, 1000) *qq_bins
# hist, edges = np.histogram(mms, bins=70)
# print(edges)
# ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_s.T, axis=1) / np.sum(H_all.T, axis=1),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
# ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_all.T, axis=1) / np.sum(np.sum(H_all.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_all.T, axis=1),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp_bins) > yedges[0])
pps = np.log10(pp_bins)[args][np.argwhere(np.log10(pp_bins)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with Saladino Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_\\textrm{Ba;f}$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-m2-distr-all.pgf", format="pgf")
plt.show()
plt.close()

# %%

import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("colorcet:CET_L19").to_mpl()
# cmap=Colormap("cmasher:chroma_r").to_mpl()


fig, axs = plt.subplot_mosaic(
    [["histx", "."], ["scatter", "histy"]],
    figsize=set_size(column, height=1),
    width_ratios=(3, 1),
    height_ratios=(1, 3),
    layout="constrained",
)

axs["scatter"].pcolormesh(xedges, yedges, np.log10(H_all.T), rasterized=True)

ax_histx = axs["histx"]
ax_histx.spines[
    [
        "right",
        "top",
    ]
].set_visible(False)
ax_histy = axs["histy"]
ax_histy.spines[
    [
        "right",
        "top",
    ]
].set_visible(False)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(H_all.T, axis=0),
    c="k",
)
ax_histx.set_ylim(
    0,
)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_s.T, axis=0) / np.sum(H_all.T, axis=0),
    0 * np.sum(H_all.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(0, 5)

# mms = np.repeat(mm_bins, 1000) *qq_bins
# hist, edges = np.histogram(mms, bins=70)
# print(edges)
# ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_s.T, axis=1) / np.sum(H_all.T, axis=1),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
# ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_all.T, axis=1) / np.sum(np.sum(H_all.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_all.T, axis=1),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp_bins) > yedges[0])
pps = np.log10(pp_bins)[args][np.argwhere(np.log10(pp_bins)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with Saladino Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_\\textrm{Ba;f}$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-m2-distr-all.pgf", format="pgf")
plt.show()
plt.close()

# %%

sum1 = np.sum(H_all.T, axis=0)
sum2 = np.sum(H_s.T, axis=0)

for a, b in zip(sum1, sum2):
    print(a, b, b / a)

# %%

hist, edges = np.histogram(mm_sing, bins=xedges)

plt.plot(
    (edges[0:-1] + edges[1:]) / 2,
    np.sum(H_s.T, axis=0) / (1000 * hist + np.sum(H_all.T, axis=0)),
    c="k",
)
#
# plt.plot(
#     (xedges[0:-1] + xedges[1:]) / 2,
#     np.sum(H_s.T, axis=0),
#     c="k",
# )
#
# plt.plot(
#     (xedges[0:-1] + xedges[1:]) / 2,
#     np.sum(H_all.T, axis=0),
#     c="k",
# )

plt.show()
# %%

N = 100_000
m_salpeter = sample_kroupa(N, 0.01, 5)
p_bins = []
p_trips = []
p_quads = []

m_sing = []
m_bins = []
m_trips = []
m_quads = []

for i, m in enumerate(m_salpeter):
    if i % 1000 == 0:
        print(i // 1000)

    periods = sample_companion_periods(np.clip(m, 0.8, 40))
    if len(periods) == 0:
        m_sing.append(m)
    if len(periods) == 1:
        p_bins.append(periods[0])
        m_bins.append(m)
    elif len(periods) == 2:
        for per in periods:
            p_trips.append(per)
            m_trips.append(m)
    elif len(periods) == 3:
        for per in periods:
            p_quads.append(per)
            m_quads.append(m)
# %%
results = {
    "m_sing": m_sing,
    "m_bins": m_bins,
    "p_bins": p_bins,
    "m_trips": m_trips,
    "p_trips": p_trips,
    "m_quads": m_quads,
    "p_quads": p_quads,
}

with open(
    f"/home/koen/master-internship/scripts/w37/cache/m-p-dump-clip.pkl",
    "wb",
) as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mm_total = np.concatenate(
    (
        np.repeat(mm_sing, 1000),
        np.repeat(mm_bins, 1000),
        qq_bins * np.repeat(mm_bins, 1000),
        np.repeat(mm_trips, 500),
        qq_trips * np.repeat(mm_trips, 1000),
        np.repeat(mm_quads, 333),
        qq_quads * np.repeat(mm_quads, 1000),
    )
)
hist, edges = np.histogram(mm_total, bins=edges)

plt.step(
    (edges[0:-1] + edges[1:]) / 2,
    (
        np.sum(H_s.T, axis=0)
        + np.sum(H_s_trips.T, axis=0) / 2
        + np.sum(H_s_quads.T, axis=0) / 3
    )
    / hist,
    c="k",
)

# plt.plot(
#     (xedges[0:-1] + xedges[1:]) / 2,
#     np.sum(H_s.T, axis=0),
#     c="k",
# )

# plt.plot(
#     (xedges[0:-1] + xedges[1:]) / 2,
#     np.sum(H_all.T, axis=0),
#     c="k",
# )

plt.grid(axis="y")

plt.ylim(0, 0.04)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M$ ($M_\\odot$)")
plt.ylabel("$N_\\textrm{Ba stars} / N_\\textrm{all stars}$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w37-probs.pgf", format="pgf")
plt.show()
plt.close()

# %%
print(len(mm_sing))
print(len(mm_bins))
print(len(mm_trips) / 2)
print(len(mm_quads) / 3)

print(len(mm_sing) + len(mm_bins) + len(mm_trips) / 2 + len(mm_quads) / 3)
# %%

bins = np.logspace(-2, 2, 50)
N = 100_000
m_kroupa = sample_kroupa(N, 0.01, 5)

plt.hist(mm_total, bins=bins, histtype="step")
plt.hist(np.repeat(m_kroupa, 1000), bins=bins, histtype="step")
plt.show()
# %%

m_cut = np.linspace(0, 5, 100)

for m in m_cut:
    print(len(np.where(mm_total < m)[0]))
    print(len(np.where(mm_total < m)[0]))
# %%

N = len(mm_bins)
n_repeat = 1000
chunk_size = 1_000_000

bins = 100

# Define fixed bin edges
xedges = np.linspace(0, 5, bins + 1)
yedges = np.linspace(0.2, 8, 2 * bins + 1)

H_s = np.zeros((bins, 2 * bins), dtype=np.int64)
H_all = np.zeros((bins, 2 * bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_bins[j],
            qq_bins[idx],
            np.log10(pp_bins[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_bins[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_bins[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_bins[idx][low_mass] * mm_bins[j][low_mass]
    high_mass = mm_bins[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_bins[idx][high_mass] * mm_bins[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_bins[j][mask]),
            bins=(xedges, yedges),
        )
        H_s += h.astype(np.int64)

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals,
            np.log10(pp_bins[j]),
            bins=(xedges, yedges),
        )
        H_all += h.astype(np.int64)

H_s_trips = np.zeros((bins, 2 * bins), dtype=np.int64)

N = len(mm_trips)
for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_trips[j],
            qq_trips[idx],
            np.log10(pp_trips[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_trips[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_trips[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_trips[idx][low_mass] * mm_trips[j][low_mass]
    high_mass = mm_trips[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_trips[idx][high_mass] * mm_trips[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_trips[j][mask]),
            bins=(xedges, yedges),
        )
        H_s_trips += h.astype(np.int64)


H_s_quads = np.zeros((bins, 2 * bins), dtype=np.int64)

N = len(mm_quads)
for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm_quads[j],
            qq_quads[idx],
            np.log10(pp_quads[j]),
        )
    ).astype(np.float32)

    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.5
    mask3 = qq_quads[idx] > 0.1
    mask4 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > -1
    mask = mask2 * mask3 * mask4
    xvals = interpolators[0.1]["Saladino"]["m2f"](points)
    low_mass = mm_quads[j] < 1
    xvals[low_mass] = xvals[low_mass] * qq_quads[idx][low_mass] * mm_quads[j][low_mass]
    high_mass = mm_quads[j] > 3
    xvals[high_mass] = (
        xvals[high_mass] / 3 * qq_quads[idx][high_mass] * mm_quads[j][high_mass]
    )

    if np.any(mask):
        h, _, _ = np.histogram2d(
            xvals[mask],
            np.log10(pp_quads[j][mask]),
            bins=(xedges, yedges),
        )
        H_s_quads += h.astype(np.int64)


# %%

m_cut = np.linspace(0, 5, 101)
m_cut
# %%
n_tots = []
for m in m_cut:
    n_tots.append(len(np.where(mm_total < m)[0]))


# %%
m_cut = np.linspace(0, 5, 101)

bins, _ = np.histogram(mm_total, bins=m_cut)

centers = (m_cut[0:-1] + m_cut[1:]) / 2

n_s = (
    np.sum(H_s.T, axis=0)
    + np.sum(H_s_trips.T, axis=0) / 2
    + np.sum(H_s_quads.T, axis=0) / 3
)

plt.plot(centers, np.array(n_s) / bins)
plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(centers, np.array(n_s) / np.nanmax(bins))
plt.plot(centers, bins / np.nanmax(bins))

plt.annotate(
    "All stars\n(singles + binaries +\ntriples + quadruples)",
    xy=(2.5, 2e-3),
    xycoords="data",
    xytext=(0, 40),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C9"),
    fontsize=8,
    va="center",
    ha="center",
)

plt.annotate(
    "Barium stars",
    xy=(2.5, 3e-5),
    xycoords="data",
    xytext=(0, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C9"),
    fontsize=8,
    va="center",
    ha="center",
)


plt.yscale("log")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M$ ($M_\\odot$)")
plt.ylabel("Normalized count")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-frac-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%

# %%
from scipy.interpolate import griddata

levels = 8
cmap = "viridis"

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)

val2d = np.zeros((100, 100))
val2d[val2d == 0] = np.nan
x = []
y = []
z = []

for i in range(100):
    for j in range(i + 7, 100):
        val2d[i, j] = np.sum(n_s[i:j]) / np.sum(bins[i:j])
        x.append(centers[i])
        y.append(centers[j])
        z.append(np.sum(n_s[i:j]) / np.sum(bins[i:j]))

# plt.pcolormesh(centers, centers,val2d.T, cmap="viridis", rasterized=True)

plt.grid(color="w", alpha=0.5)

xi = np.linspace(min(x), max(x), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (x, y),
    z,
    (X, Y),
    method="cubic",
)

c = plt.contourf(
    X,
    Y,
    Z,
    levels=levels,
    cmap=cmap,
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

axs.spines[["right", "bottom", "left", "top"]].set_visible(False)
axs.tick_params(color="w")

plt.colorbar(
    c, orientation="horizontal", location="top", aspect=30, label="Probability $p$"
)
plt.xlim(0)
plt.ylim(0)

plt.xlabel("$M_\\textrm{min}$ ($M_\\odot$)")
plt.ylabel("$M_\\textrm{max}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-probs-cuts.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
