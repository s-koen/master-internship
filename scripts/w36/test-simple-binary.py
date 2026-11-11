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

with open(
    f"/home/koen/master-internship/scripts/w36/cache/df2.pkl",
    "rb",
) as f:
    df_results = pickle.load(f)


import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator


def build_interpolator(df, eps, wind, quantity):
    """
    Build a trilinear interpolator for one outcome quantity
    at a fixed mass-transfer efficiency.
    """

    data = df[(np.isclose(df["eps"], eps)) & (df_results["wind"] == wind)].copy()

    # Interpolate in log10(period)
    data["logP"] = np.log10(data["p_init"])

    axes = (
        np.sort(data["m1i"].unique()),
        np.sort(data["qi"].unique()),
        np.sort(data["logP"].unique()),
    )

    # Construct the full Cartesian grid
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

    return RegularGridInterpolator(
        axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


wind = "Saladino"
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
                "m1i",
                "p_init",
            ]
        }
        for wind in ["fast", "BHL", "Saladino"]
    }
    for eps in [0, 0.1, 0.2, 0.3, 0.4]
}

n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_bins = np.zeros((bins, bins), dtype=np.int64)
H_trips = np.zeros((bins, bins), dtype=np.int64)
H_quads = np.zeros((bins, bins), dtype=np.int64)


with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-dump.pkl",
    "rb",
) as f:
    qq_bins = np.array(pickle.load(f)).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-trips-dump.pkl",
    "rb",
) as f:
    qq_trips = np.array(pickle.load(f)).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-quads-dump.pkl",
    "rb",
) as f:
    qq_quads = np.array(pickle.load(f)).flatten()


with open(
    f"/home/koen/master-internship/scripts/w36/cache/m-p-dump.pkl",
    "rb",
) as f:
    mp = pickle.load(f)

mm_bins = np.array(mp["m_bins"])
pp_bins = np.array(mp["p_bins"])
mm_trips = np.array(mp["m_trips"])
pp_trips = np.array(mp["p_trips"])
mm_quads = np.array(mp["m_quads"])
pp_quads = np.array(mp["p_quads"])

# %%


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
    "/home/koen/LaTeX-setup/plots/w36-R-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()

# %%

qs = np.linspace(0.1, 1.0, 19)

Ps = np.logspace(np.log10(100), np.log10(20000), 100)

Ps
# %%
from tqdm import tqdm
import pandas as pd

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


def period_to_RL(P, q, m1):

    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a * r_L / R_sun


def period_to_a(P, q, m1):

    q_donor = 1 / q

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a / R_sun


def get_rlof_properties(star, bin):
    """
    Get stellar properties at the moment of RLOF.

    Returns
    -------
    m_env_rlof : float
        Envelope mass at RLOF, or nan if no RLOF occurred.
    m_star_rlof : float
        Stellar mass at RLOF, or nan if no RLOF occurred.
    rlof_tpagb : bool
        Whether RLOF occurred during the TP-AGB.
    """

    t_rlof = bin.age[-1]

    # Check whether the binary evolved to the end without RLOF
    if np.isclose(t_rlof, star.age[-1]):
        return np.nan, np.nan, False

    # Interpolate stellar properties onto the binary time
    m_env_rlof = np.interp(t_rlof, star.age, star.m_env)
    m_star_rlof = np.interp(t_rlof, star.age, star.mass)

    # Check whether RLOF happened during the TP-AGB
    t_tpagb = star.age[star.ntpagb]
    rlof_tpagb = t_rlof >= t_tpagb

    return m_env_rlof, m_star_rlof, rlof_tpagb


class Result:
    def __init__(self):
        pass


from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
qs = np.linspace(0.1, 1.0, 19)
Ps = np.logspace(np.log10(100), np.log10(30000), 100)
epss = [1e-20, 0.1, 0.2, 0.3, 1]

wind_methods = ["Saladino", "BHL", "fast"]


def run_model(pi):

    results = []

    for mi in ms:
        star = get_star(m=mi)

        for qi in qs:

            for method in wind_methods:
                a_init = period_to_a(pi, qi, mi)
                ri = period_to_RL(pi, qi, mi)
                star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    star, qi, a_init, simple_only=True, wind_model=method
                )
                bin = Bins[0]
                m_env_rlof, m_star_rlof, rlof_tpagb = get_rlof_properties(star, bin)

                for eps in epss:

                    ab = Abundances(
                        None,
                        df,
                        mass=mi,
                        sb=bin,
                        sampling=33,
                        mass_transfer_efficiency=eps,
                    )
                    if ab.m_env[-1] > 0.04:
                        m2f = bin.m2[-1] + ab.m_env[-1] * eps
                    else:
                        m2f = bin.m2[-1]

                    results.append(
                        {
                            "p_init": pi,
                            "r_init": ri,
                            "a_init": a_init,
                            "m1i": mi,
                            "m1f": star.mass[-1],
                            "m_env_rlof": m_env_rlof,
                            "m_star_rlof": m_star_rlof,
                            "rlof_tpagb": rlof_tpagb,
                            "qi": qi,
                            "eps": eps,
                            "m2f": m2f,
                            "s": get_s(ab),
                            "hs": get_hs(ab),
                            "ls": get_ls(ab),
                            "ba": get_ba(ab),
                            "wind": method,
                        }
                    )

    with open(
        f"/home/koen/master-internship/scripts/w36/cache/df_{pi:.1f}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results


from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":

    np.random.shuffle(Ps)
    print(Ps)
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_model, pi) for pi in Ps]

        results = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
            )
        ]

    results = [result for r_results in results for result in r_results]

df_results = pd.DataFrame(results)
print(df_results)

with open(
    f"/home/koen/master-internship/scripts/w36/cache/df.pkl",
    "wb",
) as f:
    pickle.dump(df_results, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%

df_results.columns
# %%

subset = df_results[
    (np.abs(df_results["eps"] - 0.1) < 0.01) & (df_results["wind"] == "fast")
]

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
        (np.abs(df_results["eps"] - 0.1) < 0.01)
        & (df_results["qi"] == qs[i])
        & (df_results["wind"] == "fast")
    ]
    print(subset)

    s_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="rlof_tpagb",
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
    "/home/koen/LaTeX-setup/plots/w36-P-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()

# %%

subset = df_results[
    (np.abs(df_results["eps"] - 0.1) < 0.01) & (df_results["wind"] == "Saladino")
]

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
        (np.abs(df_results["eps"] - 0.1) < 0.01)
        & (df_results["qi"] == qs[i])
        & (df_results["wind"] == "Saladino")
    ]
    print(subset)

    rlof_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="rlof_tpagb",
    ).astype(np.float64)

    s_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="s",
    )

    ba_grid = (s_grid > 0.25).astype(np.float64)

    im = ax.pcolormesh(
        s_grid.columns,
        s_grid.index,
        (ba_grid * rlof_grid + (ba_grid + rlof_grid)).values,
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
    "/home/koen/LaTeX-setup/plots/w36-P-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()


# %%

subset = df_results[
    (np.abs(df_results["eps"] - 0.1) < 0.01) & (df_results["wind"] == "Saladino")
]

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
        (np.abs(df_results["eps"] - 0.1) < 0.01)
        & (df_results["qi"] == qs[i])
        & (df_results["wind"] == "Saladino")
    ]
    print(subset)

    rlof_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="rlof_tpagb",
    ).astype(np.float64)

    s_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="s",
    )

    m_rlof_grid = subset.pivot(
        index="m1i",
        columns="p_init",
        values="m_env_rlof",
    )

    ba_grid = (s_grid > 0.25).astype(np.float64)

    im = ax.pcolormesh(
        s_grid.columns,
        s_grid.index,
        (ba_grid * m_rlof_grid).values,
        # norm=norm,
        cmap="viridis",
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
    "/home/koen/LaTeX-setup/plots/w36-P-M-s-q-panels.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator


def build_interpolator(df, eps, wind, quantity):
    """
    Build a trilinear interpolator for one outcome quantity
    at a fixed mass-transfer efficiency.
    """

    data = df[(np.isclose(df["eps"], eps)) & (df_results["wind"] == wind)].copy()

    # Interpolate in log10(period)
    data["logP"] = np.log10(data["p_init"])

    axes = (
        np.sort(data["m1i"].unique()),
        np.sort(data["qi"].unique()),
        np.sort(data["logP"].unique()),
    )

    # Construct the full Cartesian grid
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

    return RegularGridInterpolator(
        axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


# %%
wind = "Saladino"
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
                "m2f",
                "mi"
            ]
        }
        for wind in ["fast", "BHL", "Saladino"]
    }
    for eps in [0, 0.1, 0.2, 0.3, 0.4]
}
# %%


with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-dump.pkl",
    "rb",
) as f:
    qq = np.array(pickle.load(f)).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/m_bin-dump.pkl",
    "rb",
) as f:
    mm = np.array(pickle.load(f))

with open(
    f"/home/koen/master-internship/scripts/w36/cache/p_bin-dump.pkl",
    "rb",
) as f:
    pp = np.array(pickle.load(f))

# %%

# with open(
#     f"/home/koen/master-internship/scripts/w36/cache/m_bin-dump.pkl",
#     "wb",
# ) as f:
#     pickle.dump(m_bin,f, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open(
#     f"/home/koen/master-internship/scripts/w36/cache/p_bin-dump.pkl",
#     "wb",
# ) as f:
#     pickle.dump(p_bin,f, protocol=pickle.HIGHEST_PROTOCOL)
#
# %%

plt.scatter(m_bin, p_bin)
plt.show()
# %%

N = len(mm)

mm = np.array(np.repeat(mm[:N], 1000)).astype(np.float16)
pp = np.log10(np.repeat(pp[:N], 1000)).astype(np.float16)
points = np.column_stack((mm, qq, pp)).astype(np.float16)

# %%
print(points[:10])
# %%
res2 = interpolators[0.1]["s"](points).astype(np.float16)
res2 = res2 > 0.25

res2 = np.where(res2)

plt.hist2d(mm[re2], pp[res2], bins=50, cmap="viridis")
plt.show()
# %%

ms = m_bin[:10]
print(np.repeat(ms, 3))
# %%


plt.hist2d(mm[args], pp[args], bins=50, cmap="viridis")
plt.show()
# %%
N = len(mm)
n_repeat = 1000
chunk_size = 50_000

res2 = np.empty(N * n_repeat, dtype=bool)

for i in range(0, N * n_repeat, chunk_size):
    print(i)
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))

    j = np.array(idx // n_repeat)

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    )

    res2[i : i + len(idx)] = interpolators[0.1]["s"](points) > 0.25
# %%
print(np.size(res2))
# %%
plt.hist2d(mm[res2], pp[res2], bins=50, cmap="viridis")
plt.show()
# %%
N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 50

# Define fixed bin edges
xedges = np.linspace(mm.min(), mm.max(), bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    mask = interpolators[0.1]["s"](points) > 0.25

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[mask]],
            np.log10(pp[j[mask]]),
            bins=(xedges, yedges),
        )
        H += h.astype(np.int64)
# %%

plt.pcolormesh(xedges, yedges, H.T, cmap="viridis")
plt.show()
# %%
plt.plot((xedges[0:-1] + xedges[1:]) / 2, np.sum(H.T, axis=0))
plt.xlim(1, 3)
plt.show()
# %%

plt.plot((yedges[0:-1] + yedges[1:]) / 2, np.sum(H.T, axis=1))
# plt.xlim(1,3)
plt.show()
# %%
N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_fast = np.zeros((bins, bins), dtype=np.int64)
H_BHL = np.zeros((bins, bins), dtype=np.int64)
H_Saladino = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points) > 0.5
    mask2 = interpolators[0.1]["fast"]["s"](points) > 0.25
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[mask]],
            np.log10(pp[j[mask]]),
            bins=(xedges, yedges),
        )
        H_fast += h.astype(np.int64)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points) > 0.5
    mask2 = interpolators[0.1]["BHL"]["s"](points) > 0.25
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[mask]],
            np.log10(pp[j[mask]]),
            bins=(xedges, yedges),
        )
        H_BHL += h.astype(np.int64)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points) > 0.25
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[mask]],
            np.log10(pp[j[mask]]),
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

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
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
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-sal.pgf", format="pgf"
)
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

axs["scatter"].pcolormesh(xedges, yedges, H_BHL.T, rasterized=True)

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
    c="k",
    zorder=1000,
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / (2),
    c="C9",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_BHL.T, axis=0) / np.sum(np.sum(H_BHL.T, axis=0)) * bins / 2,
    0 * np.sum(H_BHL.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
    c="k",
    zorder=1000,
    label="Ba-star primary",
)

ax_histy.plot(
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_BHL.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with BHL Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-bhl.pgf", format="pgf"
)
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

axs["scatter"].pcolormesh(xedges, yedges, H_fast.T, rasterized=True)

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
    c="k",
    zorder=1000,
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_BHL.T, axis=0) / np.sum(np.sum(H_BHL.T, axis=0)) * bins / (2),
    c="C9",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / (2),
    c="C9",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_fast.T, axis=0) / np.sum(np.sum(H_fast.T, axis=0)) * bins / 2,
    0 * np.sum(H_fast.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
hist, edges = np.histogram(mms, bins=20)
print(edges)
ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    zorder=1000,
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

ax_histy.plot(
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_fast.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with fast Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-fast.pgf", format="pgf"
)
plt.show()
plt.close()


# %%

N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H2 = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points) > -99
    mask2 = interpolators[0.1]["fast"]["s"](points) > -99
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[mask]],
            np.log10(pp[j[mask]]),
            bins=(xedges, yedges),
        )
        H2 += h.astype(np.int64)


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

axs["scatter"].pcolormesh(xedges, yedges, H.T / H2.T, rasterized=True)

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
    np.sum(H.T, axis=0) / np.sum(np.sum(H.T, axis=0)) * bins / (2),
    c="k",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H.T, axis=0) / np.sum(np.sum(H.T, axis=0)) * bins / 2,
    0 * np.sum(H.T, axis=0),
    alpha=0.5,
    color="C9",
)

plt.xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
hist, edges = np.histogram(mms, bins=20)
print(edges)
ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H.T, axis=1) / np.sum(np.sum(H.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    label="Ba-star primary",
)
ax_histy.set_xlim(0)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H.T, axis=1) / np.sum(np.sum(H.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(loc="upper right", handles=[m, l])

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-compare-distribution-3.pgf", format="pgf")
plt.show()
plt.close()

# %%

N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_fast = np.zeros((bins, bins), dtype=np.int64)
H_BHL = np.zeros((bins, bins), dtype=np.int64)
H_Saladino = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["fast"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_fast += h.astype(np.int64)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["BHL"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_BHL += h.astype(np.int64)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
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

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
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
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-sal-high-q.pgf", format="pgf"
)
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

axs["scatter"].pcolormesh(xedges, yedges, H_BHL.T, rasterized=True)

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
    c="k",
    zorder=1000,
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / (2),
    c="C9",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_BHL.T, axis=0) / np.sum(np.sum(H_BHL.T, axis=0)) * bins / 2,
    0 * np.sum(H_BHL.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
    c="k",
    zorder=1000,
    label="Ba-star primary",
)

ax_histy.plot(
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_BHL.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with BHL Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-bhl-high-q.pgf", format="pgf"
)
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

axs["scatter"].pcolormesh(xedges, yedges, H_fast.T, rasterized=True)

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
    c="k",
    zorder=1000,
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_BHL.T, axis=0) / np.sum(np.sum(H_BHL.T, axis=0)) * bins / (2),
    c="C9",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / (2),
    c="C9",
)
ax_histx.set_ylim(0, 1.4)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_fast.T, axis=0) / np.sum(np.sum(H_fast.T, axis=0)) * bins / 2,
    0 * np.sum(H_fast.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
hist, edges = np.histogram(mms, bins=20)
print(edges)
ax_histx.plot((edges[0:-1] + edges[1:]) / 2, hist / np.sum(hist) / np.diff(edges))
ax_histx.set_ylabel("$p$")

(m,) = ax_histy.plot(
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="k",
    zorder=1000,
    label="Ba-star primary",
)

(m,) = ax_histy.plot(
    np.sum(H_BHL.T, axis=1) / np.sum(np.sum(H_BHL.T, axis=1)) / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)

ax_histy.plot(
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="C9",
    label="Ba-star primary",
)


ax_histy.set_xlim(0)
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_fast.T, axis=1) / np.sum(np.sum(H_fast.T, axis=1)) / np.diff(yedges),
    0 * np.sum(H_fast.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
hist, edges = np.histogram(pps, bins=20)
(l,) = ax_histy.plot(
    hist / np.sum(hist) / np.diff(edges),
    (edges[0:-1] + edges[1:]) / 2,
    label="Moe \\& Di Stefano",
)
ax_histy.set_xlabel("$p$")

fig.legend(handles=[m, l], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
fig.suptitle(
    "Moe \\& Di Stefano Binary population with fast Winds",
    x=0.1,
    ha="left",
    fontsize=10,
)

axs["scatter"].spines[["right", "top"]].set_visible(False)
axs["scatter"].set_xlabel("$M_1$ ($M_\\odot$)")
axs["scatter"].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-fast-high-q.pgf",
    format="pgf",
)
plt.show()
plt.close()


# %%

N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_fast = np.zeros((bins, bins), dtype=np.int64)
H_BHL = np.zeros((bins, bins), dtype=np.int64)
H_Saladino = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["fast"]["s"](points[args]) > 0.5
    mask3 = interpolators[0.1]["fast"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0.1]["fast"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_fast += h.astype(np.int64)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["BHL"]["s"](points[args]) > 0.5
    mask3 = interpolators[0.1]["BHL"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0.1]["BHL"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_BHL += h.astype(np.int64)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points[args]) > 0.5
    mask3 = interpolators[0.1]["Saladino"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0.1]["Saladino"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
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
ax_histx.set_ylim(0, 1.8)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / 2,
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
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
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-sal-high-q-correct-mass.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%

N = len(mm)
n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_fast = np.zeros((bins, bins), dtype=np.int64)
H_BHL = np.zeros((bins, bins), dtype=np.int64)
H_Saladino = np.zeros((bins, bins), dtype=np.int64)

for i in range(0, N * n_repeat, chunk_size):
    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0]["fast"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0]["fast"]["s"](points[args]) > 0.5
    mask3 = interpolators[0]["fast"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0]["fast"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_fast += h.astype(np.int64)

    mask1 = interpolators[0]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0]["BHL"]["s"](points[args]) > 0.5
    mask3 = interpolators[0]["BHL"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0]["BHL"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_BHL += h.astype(np.int64)

    mask1 = interpolators[0]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0]["Saladino"]["s"](points[args]) > 0.5
    mask3 = interpolators[0]["Saladino"]["m2f"](points[args]) > 0.7
    mask4 = interpolators[0]["Saladino"]["m2f"](points[args]) < 1.5
    mask = mask1 * mask2 * mask3 * mask4

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
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
ax_histx.set_ylim(0, 1.8)
ax_histx.fill_between(
    (xedges[0:-1] + xedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=0) / np.sum(np.sum(H_Saladino.T, axis=0)) * bins / 2,
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

ax_histx.set_xlim(1, 3)

args = np.where(mm > xedges[0])
mms = mm[args][np.argwhere(mm[args] < 3)]
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
ax_histy.set_ylim(2.8, 4.4)
ax_histy.fill_betweenx(
    (yedges[0:-1] + yedges[1:]) / 2,
    np.sum(H_Saladino.T, axis=1)
    / np.sum(np.sum(H_Saladino.T, axis=1))
    / np.diff(yedges),
    0 * np.sum(H_Saladino.T, axis=0),
    alpha=0.5,
    color="C9",
)

args = np.where(np.log10(pp) > yedges[0])
pps = np.log10(pp)[args][np.argwhere(np.log10(pp)[args] < yedges[-1])]
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
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-sal-high-q-correct-mass-eps0.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%


n_repeat = 1000
chunk_size = 1_000_000

bins = 30

# Define fixed bin edges
xedges = np.linspace(1, 3, bins + 1)
yedges = np.linspace(2.8, 4.4, bins + 1)

H_bins = np.zeros((bins, bins), dtype=np.int64)
H_trips = np.zeros((bins, bins), dtype=np.int64)
H_quads = np.zeros((bins, bins), dtype=np.int64)


with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-dump.pkl",
    "rb",
) as f:
    qq_bins = np.array(pickle.load(f)).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-trips-dump.pkl",
    "rb",
) as f:
    qq_trips = np.array(pickle.load(f)).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-quads-dump.pkl",
    "rb",
) as f:
    qq_quads = np.array(pickle.load(f)).flatten()


with open(
    f"/home/koen/master-internship/scripts/w36/cache/m-p-dump.pkl",
    "rb",
) as f:
    mp = pickle.load(f)
# %%

mm_bins = np.array(mp["m_bins"])
pp_bins = np.array(mp["p_bins"])
mm_trips = np.array(mp["m_trips"])
pp_trips = np.array(mp["p_trips"])
mm_quads = np.array(mp["m_quads"])
pp_quads = np.array(mp["p_quads"])

# %%

N = len(mm_bins)
for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_bins += h.astype(np.int64)


N = len(mm_trips)
for i in range(0, N * n_repeat, chunk_size):

    mm = mm_trips
    pp = pp_trips
    qq = qq_trips

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_trips += h.astype(np.int64)


N = len(mm_quads)
for i in range(0, N * n_repeat, chunk_size):

    mm = mm_quads
    pp = pp_quads
    qq = qq_quads

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    mask2 = interpolators[0.1]["Saladino"]["s"](points[args]) > 0.5
    mask = mask1 * mask2

    if np.any(mask):
        h, _, _ = np.histogram2d(
            mm[j[args][mask]],
            np.log10(pp[j[args][mask]]),
            bins=(xedges, yedges),
        )
        H_quads += h.astype(np.int64)

H_tot = H_bins + H_trips + H_quads
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


H = np.stack(
    [H_quads.T, H_trips.T, H_bins.T],
    axis=-1,
).astype(float)

H /= H.max()
H *=0.8

# Start with white and subtract the other two channels
H_rgb = np.ones_like(H)

H_rgb[..., 0] -= H[..., 1] + H[..., 2]  # red
H_rgb[..., 1] -= H[..., 0] + H[..., 2]  # green
H_rgb[..., 2] -= H[..., 0] + H[..., 1]  # blue

H_rgb = np.clip(H_rgb, 0, 1)

mesh = axs["scatter"].pcolormesh(
    xedges,
    yedges,
    H_bins.T,
    shading="flat",
    cmap="gray",
    rasterized=True,
)

mesh.set_array(None)
mesh.set_facecolor(H_rgb.reshape(-1, 3))
mesh.set_edgecolor("none")


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
    # np.mean(H_quads.T, axis=0) / np.mean(H_tot.T, axis=0) * bins / (2),
    np.sum(H_quads.T, axis=0),
    c="r",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    # np.mean(H_trips.T, axis=0) / np.mean(H_tot.T, axis=0) * bins / (2),
    np.sum(H_trips.T, axis=0),
    c="g",
)

ax_histx.plot(
    (xedges[0:-1] + xedges[1:]) / 2,
    # np.mean(H_bins.T, axis=0) / np.mean(H_tot.T, axis=0) * bins / (2),
    np.sum(H_bins.T, axis=0),
    c="b",
)
ax_histx.set_ylim(0)

ax_histx.set_xlim(1, 3)

ax_histx.set_ylabel("$p$")

(k,) = ax_histy.plot(
    np.sum(H_quads.T, axis=1),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="r",
    label="Quadruples",
)

(l,) = ax_histy.plot(
    np.sum(H_trips.T, axis=1),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="g",
    label="Triples",
)

(m,) = ax_histy.plot(
    np.sum(H_bins.T, axis=1),
    (yedges[0:-1] + yedges[1:]) / 2,
    c="b",
    label="Binaries",
)


ax_histy.set_xlim(0)
ax_histy.set_ylim(2.8, 4.4)


fig.legend(handles=[m, l, k], bbox_to_anchor=(0.5, 0.85, 0.5, 0.1), frameon=False)
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
    "/home/koen/LaTeX-setup/plots/w36-compare-distribution-bins-trips-quads.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins = np.concatenate(s)
m_env_bins = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_trips)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_trips
    pp = pp_trips
    qq = qq_trips

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_trips = np.concatenate(s)
m_env_trips = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_quads)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_quads
    pp = pp_quads
    qq = qq_quads

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_quads = np.concatenate(s)
m_env_quads = np.concatenate(m_env)

# %%

fig, axs = plt.subplots(2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True)


counts, edges = np.histogram(s_bins, bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Binaries")
axs[0].axhline((np.cumsum(counts) / np.sum(counts))[207], c="C9", zorder=-100, linewidth=0.75)

counts, edges = np.histogram(s_bins, bins=200)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts)

counts, edges = np.histogram(s_trips, bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Triples")
axs[0].axhline((np.cumsum(counts) / np.sum(counts))[208], c="C9", zorder=-100, linewidth=0.75)

counts, edges = np.histogram(s_trips, bins=200)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

counts, edges = np.histogram(s_quads, bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Quadruples")
axs[0].axvline(0.25, c="C9", zorder=-100, linewidth=0.75)
axs[1].axvline(0.25, c="C9", zorder=-100, linewidth=0.75)
axs[0].axhline((np.cumsum(counts) / np.sum(counts))[208], c="C9", zorder=-100, linewidth=0.75)

axs[1].set_yscale("log")
axs[1].set_ylabel("Counts")
axs[0].set_ylabel("CDF")
axs[1].set_xlabel("[s/Fe]")

axs[0].legend()

counts, edges = np.histogram(s_quads, bins=200)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-fe.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True)


counts, edges = np.histogram(m_env_bins[np.isfinite(m_env_bins)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Binaries")

counts, edges = np.histogram(m_env_bins[np.isfinite(m_env_bins)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts)

counts, edges = np.histogram(m_env_trips[np.isfinite(m_env_trips)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Triples")

counts, edges = np.histogram(m_env_trips[np.isfinite(m_env_trips)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

counts, edges = np.histogram(m_env_quads[np.isfinite(m_env_quads)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Quadruples")

axs[1].set_yscale("log")
axs[1].set_ylabel("Counts")
axs[0].set_ylabel("CDF")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")

axs[0].legend()

counts, edges = np.histogram(m_env_quads[np.isfinite(m_env_quads)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-m_env.pgf", format="pgf")
plt.show()
plt.close()


# %%
np.min(m_env_bins)
# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

plt.hist2d(m_env_bins[np.isfinite(m_env_bins)], s_bins[np.isfinite(m_env_bins)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)

plt.colorbar(label="Counts", extend="max")
axs.set_facecolor("k")


axs.spines[["right", "top"]].set_visible(False)
plt.ylabel("[s/Fe]")
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

df_results.columns
# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_bhl = np.concatenate(s)
m_env_bins_bhl = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_trips)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_trips
    pp = pp_trips
    qq = qq_trips

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_trips_bhl = np.concatenate(s)
m_env_trips_bhl = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_quads)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_quads
    pp = pp_quads
    qq = qq_quads

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_quads_bhl = np.concatenate(s)
m_env_quads_bhl = np.concatenate(m_env)

# %%

fig, axs = plt.subplots(2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True)


counts, edges = np.histogram(m_env_bins_bhl[np.isfinite(m_env_bins_bhl)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Binaries")

counts, edges = np.histogram(m_env_bins_bhl[np.isfinite(m_env_bins_bhl)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts)

counts, edges = np.histogram(m_env_trips_bhl[np.isfinite(m_env_trips_bhl)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Triples")

counts, edges = np.histogram(m_env_trips_bhl[np.isfinite(m_env_trips_bhl)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

counts, edges = np.histogram(m_env_quads_bhl[np.isfinite(m_env_quads_bhl)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Quadruples")

axs[1].set_yscale("log")
axs[1].set_ylabel("Counts")
axs[0].set_ylabel("CDF")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")

axs[0].legend()

counts, edges = np.histogram(m_env_quads_bhl[np.isfinite(m_env_quads_bhl)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-m_env_bhl.pgf", format="pgf")
plt.show()
plt.close()


# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

plt.hist2d(m_env_bins_bhl[np.isfinite(m_env_bins_bhl)], s_bins_bhl[np.isfinite(m_env_bins_bhl)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)

plt.colorbar(label="Counts", extend="max")
axs.set_facecolor("k")


axs.spines[["right", "top"]].set_visible(False)
plt.ylabel("[s/Fe]")
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env_bhl.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_f = np.concatenate(s)
m_env_bins_f = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_trips)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_trips
    pp = pp_trips
    qq = qq_trips

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_trips_f = np.concatenate(s)
m_env_trips_f = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_quads)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_quads
    pp = pp_quads
    qq = qq_quads

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_quads_f = np.concatenate(s)
m_env_quads_f = np.concatenate(m_env)

# %%

fig, axs = plt.subplots(2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True)


counts, edges = np.histogram(m_env_bins_f[np.isfinite(m_env_bins_f)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Binaries")

counts, edges = np.histogram(m_env_bins_f[np.isfinite(m_env_bins_f)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts)

counts, edges = np.histogram(m_env_trips_f[np.isfinite(m_env_trips_f)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Triples")

counts, edges = np.histogram(m_env_trips_f[np.isfinite(m_env_trips_f)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

counts, edges = np.histogram(m_env_quads_f[np.isfinite(m_env_quads_f)], bins=1000)
centers = (edges[0:-1] + edges[1:])/2
axs[0].plot(centers, np.cumsum(counts) / np.sum(counts), label="Quadruples")

axs[1].set_yscale("log")
axs[1].set_ylabel("Counts")
axs[0].set_ylabel("CDF")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")

axs[0].legend()

counts, edges = np.histogram(m_env_quads_f[np.isfinite(m_env_quads_f)], bins=100)
centers = (edges[0:-1] + edges[1:])/2
axs[1].plot(centers, counts )

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-m_env_f.pgf", format="pgf")
plt.show()
plt.close()


# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

plt.hist2d(m_env_bins_f[np.isfinite(m_env_bins_f)], s_bins_f[np.isfinite(m_env_bins_f)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)

plt.colorbar(label="Counts", extend="max")
axs.set_facecolor("k")


axs.spines[["right", "top"]].set_visible(False)
plt.ylabel("[s/Fe]")
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env_f.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins = np.concatenate(s)
m_env_bins = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins0 = np.concatenate(s)
m_env_bins0 = np.concatenate(m_env)

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)

h = axs[0].hist2d(m_env_bins[np.isfinite(m_env_bins)], s_bins[np.isfinite(m_env_bins)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[0].set_title("$\\varepsilon=0.1$")
axs[1].hist2d(m_env_bins0[np.isfinite(m_env_bins0)], s_bins0[np.isfinite(m_env_bins0)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("$\\varepsilon=0.0$")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("[s/Fe]")
axs[0].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_bhl = np.concatenate(s)
m_env_bins_bhl = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0]["BHL"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_bhl0 = np.concatenate(s)
m_env_bins_bhl0 = np.concatenate(m_env)

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)

h = axs[0].hist2d(m_env_bins_bhl[np.isfinite(m_env_bins_bhl)], s_bins_bhl[np.isfinite(m_env_bins_bhl)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[0].set_title("$\\varepsilon=0.1$")
axs[1].hist2d(m_env_bins_bhl0[np.isfinite(m_env_bins_bhl0)], s_bins_bhl0[np.isfinite(m_env_bins_bhl0)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("$\\varepsilon=0.0$")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("[s/Fe]")
axs[0].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env-bhl.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0.1]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0.1]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_f = np.concatenate(s)
m_env_bins_f = np.concatenate(m_env)


s = []
m_env = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0]["fast"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    mm_env = interpolators[0]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))


s_bins_f0 = np.concatenate(s)
m_env_bins_f0 = np.concatenate(m_env)

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)

axs[1].hist2d(m_env_bins_f0[np.isfinite(m_env_bins_f0)], s_bins_f0[np.isfinite(m_env_bins_f0)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
h = axs[0].hist2d(m_env_bins_f[np.isfinite(m_env_bins_f)], s_bins_f[np.isfinite(m_env_bins_f)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("$\\varepsilon=0.0$")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("[s/Fe]")
axs[0].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs[1].set_xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m_env-f.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

s = []
m = []
m_env = []
p = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5

    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))

    m.append(np.array(mm[j][args][mask1]))
    p.append(np.array(np.log10(pp[j])[args][mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))

s_bins = np.concatenate(s)
m_bins = np.concatenate(m)
p_bins = np.concatenate(p)
m_env_bins = np.concatenate(m_env)


s = []
m = []
m_env = []
p = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0]["Saladino"]["rlof_tpagb"](points[args]) > 0.5
    ss = interpolators[0]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))
    m.append(np.array(mm[j][args][mask1]))
    p.append(np.array(np.log10(pp[j])[args][mask1]))
    mm_env = interpolators[0]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))



s_bins0 = np.concatenate(s)
m_bins0 = np.concatenate(m)
p_bins0 = np.concatenate(p)
m_env_bins0 = np.concatenate(m_env)

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)


h = axs[0].hist2d(m_bins[np.isfinite(m_env_bins)], s_bins[np.isfinite(m_env_bins)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[0].set_title("$\\varepsilon=0.1$")
axs[1].hist2d(m_bins[np.isfinite(m_env_bins0)], s_bins0[np.isfinite(m_env_bins0)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("$\\varepsilon=0.0$")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("[s/Fe]")
axs[0].set_xlabel("$M_\\textrm{star;i}$ ($M_\\odot$)")
axs[1].set_xlabel("$M_\\textrm{star;i}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-m.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)


h = axs[0].hist2d(p_bins[np.isfinite(m_env_bins)], s_bins[np.isfinite(m_env_bins)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[0].set_title("$\\varepsilon=0.1$")
axs[1].hist2d(p_bins[np.isfinite(m_env_bins0)], s_bins0[np.isfinite(m_env_bins0)], bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("$\\varepsilon=0.0$")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("[s/Fe]")
axs[0].set_xlabel("log Initial period (days)")
axs[1].set_xlabel("log Initial period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-s-p.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)


h = axs[0].hist2d(p_bins[np.isfinite(m_env_bins)], np.log10(m_env_bins[np.isfinite(m_env_bins)]), bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].hist2d(m_bins[np.isfinite(m_env_bins0)], np.log10(m_env_bins[np.isfinite(m_env_bins0)]), bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs[0].set_xlabel("log Initial period (days)")
axs[1].set_xlabel("$M_\\textrm{star;i}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-m-m_env-p.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

s = []
m = []
m_env = []
p = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["Saladino"]["rlof_tpagb"](points[args]) > 0.5

    ss = interpolators[0.1]["Saladino"]["s"](points[args])
    s.append(np.array(ss[mask1]))

    m.append(np.array(mm[j][args][mask1]))
    p.append(np.array(np.log10(pp[j])[args][mask1]))
    mm_env = interpolators[0.1]["Saladino"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))

s_bins_sal = np.concatenate(s)
m_bins_sal = np.concatenate(m)
p_bins_sal = np.concatenate(p)
m_env_bins_sal = np.concatenate(m_env)

s = []
m = []
m_env = []
p = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["BHL"]["rlof_tpagb"](points[args]) > 0.5

    ss = interpolators[0.1]["BHL"]["s"](points[args])
    s.append(np.array(ss[mask1]))

    m.append(np.array(mm[j][args][mask1]))
    p.append(np.array(np.log10(pp[j])[args][mask1]))
    mm_env = interpolators[0.1]["BHL"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))

s_bins_bhl = np.concatenate(s)
m_bins_bhl = np.concatenate(m)
p_bins_bhl = np.concatenate(p)
m_env_bins_bhl = np.concatenate(m_env)


s = []
m = []
m_env = []
p = []
N = len(mm_bins)

for i in range(0, N * n_repeat, chunk_size):

    mm = mm_bins
    pp = pp_bins
    qq = qq_bins

    print(f"{100*i / (N * n_repeat):.1f}%")
    idx = np.arange(i, min(i + chunk_size, N * n_repeat))
    j = idx // n_repeat

    points = np.column_stack(
        (
            mm[j],
            qq[idx],
            np.log10(pp[j]),
        )
    ).astype(np.float32)

    args = np.where(qq[idx] > 0.5)

    mask1 = interpolators[0.1]["fast"]["rlof_tpagb"](points[args]) > 0.5

    ss = interpolators[0.1]["fast"]["s"](points[args])
    s.append(np.array(ss[mask1]))

    m.append(np.array(mm[j][args][mask1]))
    p.append(np.array(np.log10(pp[j])[args][mask1]))
    mm_env = interpolators[0.1]["fast"]["m_env_rlof"](points[args])
    m_env.append(np.array(mm_env[mask1]))

s_bins_fast = np.concatenate(s)
m_bins_fast = np.concatenate(m)
p_bins_fast = np.concatenate(p)
m_env_bins_fast = np.concatenate(m_env)

# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmap import Colormap
cmap=Colormap("cmasher:rainforest").to_mpl()
fig, axs = plt.subplots(1, 3, sharey=True, figsize=set_size(full, height=0.4), constrained_layout=True)


h = axs[0].hist2d(m_bins_sal[np.isfinite(m_env_bins_sal)], np.log10(m_env_bins_sal[np.isfinite(m_env_bins_sal)]), bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[0].set_title("Saladino winds")
axs[1].hist2d(m_bins_bhl[np.isfinite(m_env_bins_bhl)], np.log10(m_env_bins_bhl[np.isfinite(m_env_bins_bhl)]), bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[1].set_title("BHL winds")
axs[2].hist2d(m_bins_fast[np.isfinite(m_env_bins_fast)], np.log10(m_env_bins_fast[np.isfinite(m_env_bins_fast)]), bins=150, norm=LogNorm(vmin=1, vmax=20000), cmap=cmap, rasterized=True)
axs[2].set_title("Fast winds")

plt.colorbar(h[3], ax = axs, label="Counts", extend="max")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("k")
axs[0].set_ylabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs[0].set_xlabel("log Initial period (days)")
axs[1].set_xlabel("$M_\\textrm{star;i}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-m-m_env-winds.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

