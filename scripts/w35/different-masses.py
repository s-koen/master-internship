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

# %%

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps1.pkl",
    "rb",
) as f:
    eps1 = pickle.load(f)

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps0.pkl",
    "rb",
) as f:
    eps0 = pickle.load(f)

# %%


def period_days(q, M1, R_L):
    """Compute the orbital period from the mass ratio and donor Roche lobe radius.

    Parameters
    ----------
    q : float or array-like
        Mass ratio M2 / M1.
    M1 : float
        Donor mass in solar masses.
    R_L : float or array-like
        Donor Roche lobe radius in solar radii.

    Returns
    -------
    float or ndarray
        Orbital period in days.
    """
    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    a = R_L / r_L * 695508000  # orbital separation in solar radii
    M_total = M1 * (1 + q) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T


# %%

ms = np.linspace(1, 3, 100)
plt.plot(
    ms,
    period_days(0.75, ms, 100),
    c="C0",
)

plt.show()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
print(eps1["m1i"])
plt.plot(
    eps1["m1i"],
    period_days(0.75, eps1["m1i"], eps1["r_init"]),
    label="$\\varepsilon=1$",
    c="C0",
)
plt.scatter(
    eps1["m1i"],
    period_days(0.75, eps1["m1i"], eps1["r_init"]),
    c="C0",
    zorder=100,
    marker=".",
    s=50,
)
plt.scatter(
    eps1["m1i"],
    period_days(0.75, eps1["m1i"], eps1["r_init"]),
    c="w",
    zorder=99,
    marker=".",
    s=100,
)
plt.plot(
    eps0["m1i"],
    period_days(0.75, eps1["m1i"], eps0["r_init"]),
    label="$\\varepsilon=0$",
    c="C1",
)
plt.scatter(
    eps0["m1i"],
    period_days(0.75, eps1["m1i"], eps0["r_init"]),
    c="C1",
    zorder=100,
    marker=".",
    s=50,
)
plt.scatter(
    eps0["m1i"],
    period_days(0.75, eps1["m1i"], eps0["r_init"]),
    c="w",
    zorder=99,
    s=100,
    marker=".",
)

axs.annotate(
    "$\\varepsilon_\\textrm{RLOF} = 1$",
    xy=(2.1, 440),
    xycoords="data",
    xytext=(0, -25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="center",
)

axs.annotate(
    "$\\varepsilon_\\textrm{RLOF} = 0$",
    xy=(2.1, 310),
    xycoords="data",
    xytext=(0, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="center",
)

for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    ms = np.linspace(1, 3, 100)
    plt.plot(ms, period_days(0.5, ms, i), c="C3", linewidth=0.75, zorder=-1)

for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    ms = np.linspace(1, 3, 100)
    plt.plot(ms, period_days(0.75, ms, i), c="C9", linewidth=0.75, zorder=-1)

plt.title("$q = 0.75$", fontsize=10)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w35-min-R_RL-for-barium-star.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
# %%
for m in grid.models:
    print(m)

# %%

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


m_mesa = np.arange(1, 3.1, 0.1)
for m in m_mesa:
    star = get_star(m=m)
    plt.plot(
        (star.age[star.ntpagb :] - star.age[star.ntpagb])
        / (star.age[-1] - star.age[star.ntpagb]),
        star.Mdot[star.ntpagb :],
        c=cmap(norm(m)),
    )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")

plt.show()
# %%
from matplotlib import colors

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

m_mesa = np.arange(1, 3.1, 0.1)

cmap = plt.cm.viridis
bounds = np.arange(0.95, 3.06, 0.1)
norm = colors.BoundaryNorm(bounds, cmap.N)


for m in m_mesa:
    star = get_star(m=m)
    args = np.argwhere(np.diff(star.TP_count) > 0)
    ages = np.array(star.age[args] - star.age[star.ntpagb]).flatten()
    print(ages)
    counts = np.array(star.TP_count[args]).flatten()
    diffs = np.diff(ages)
    print(counts, diffs)
    plt.plot(
        counts[2:] - counts[-1],
        diffs[1:],
        c=cmap(norm(m)),
    )

plt.xlim(-24, 1)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


cbar = plt.colorbar(
    sm,
    ax=axs,
    location="left",
    boundaries=bounds,
    ticks=m_mesa[::2],
    spacing="uniform",
)
cbar.set_label(r"$M_\textrm{TPAGB}$ $(M_\odot)$")

axs.yaxis.set_label_position("right")
axs.yaxis.tick_right()
axs.spines[["left", "top"]].set_visible(False)
plt.xlabel("TP-count $-$ final TP-count")
plt.ylabel("$t_\\textrm{interpulse}$ (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-interpulse-time.pgf", format="pgf")
plt.show()
plt.close()
# %%

for m in m_mesa:
    star = get_star(m=m)
    args = np.argwhere(np.diff(star.TP_count) > 0)
    ages = np.array(star.m_env[args]).flatten()
    print(ages)
    counts = np.array(star.TP_count[args]).flatten()
    print(counts, diffs)
    plt.plot(
        counts[1:],
        ages[1:],
        c=cmap(norm(m)),
    )

plt.show()
# %%

m200 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/fixes/M2.0/LOGS/EAGB/history.data"
)
m201 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/new/M2.01/LOGS/EAGB/history.data"
)
m210 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/fixes/M2.1/LOGS/EAGB/history.data"
)
# %%

plt.plot(m200.star_age, m200.R)
plt.plot(m201.star_age, m201.R)
plt.plot(m210.star_age, m210.R)
plt.show()
# %%

plt.plot(m200.log_Teff, m200.log_L)
plt.plot(m201.log_Teff, m201.log_L)
plt.plot(m210.log_Teff, m210.log_L)
plt.show()
# %%

m_mesa = np.arange(1, 3.1, 0.1)

for m in m_mesa:
    star = get_star(m=m)
    plt.plot(
        (star.age[star.ntpagb :] - star.age[star.ntpagb])
        / (star.age[-1] - star.age[star.ntpagb]),
        star.log_LHe[star.ntpagb :],
    )
    plt.show()
# %%
df = AbundanceTables()


def evaluate_RLOF(ri, mi, qi=0.9):
    star = get_star(m=mi)

    a_init = inv_roche_lobe(ri, qi)
    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    # Find stellar timestep corresponding to the end of binary evolution
    age_rlof = bin.age[-1]
    i = np.searchsorted(star.age, age_rlof, side="right") - 1

    # Determine whether RLOF starts during the last interpulse
    tp_max = np.max(star.TP_count)

    return star.TP_count[i] == tp_max - 1 and star.log_LHe[i] < 4


def binary_search_boundary(mi, R_low, R_high, transition, tol=1.0, qi=0.9):
    """
    Find a boundary between False and True (or True and False).

    transition='rising':  False -> True
    transition='falling': True -> False
    """
    while R_high - R_low > tol:
        R_mid = (R_low + R_high) / 2
        result = evaluate_RLOF(R_mid, mi, qi)

        if transition == "rising":
            if result:
                R_high = R_mid
            else:
                R_low = R_mid
        else:
            if result:
                R_low = R_mid
            else:
                R_high = R_mid
        print(R_low, R_high)

    return (R_low + R_high) / 2


def find_RLOF_range(mi, R_min=50, R_max=1000, n_coarse=10, tol=1.0, qi=0.9):
    """
    Find the minimum and maximum initial RL for which RLOF
    starts during the last interpulse.

    First performs a coarse scan to bracket both boundaries,
    then refines them using binary searches.
    """

    # Coarse scan
    R_values = np.linspace(R_min, R_max, n_coarse)
    conditions = []

    print(f"Coarse scan for M={mi:.2f}:")

    for R in R_values:
        condition = evaluate_RLOF(R, mi, qi)
        conditions.append(condition)
        print(f"R={R:.1f}: {condition}")

    conditions = np.array(conditions)

    # Find transitions
    transitions = np.diff(conditions.astype(int))

    rising = np.where(transitions == 1)[0]
    falling = np.where(transitions == -1)[0]

    if len(rising) == 0 or len(falling) == 0:
        raise ValueError(
            f"Could not bracket both boundaries for M={mi:.2f}. "
            "Try increasing R_max or n_coarse."
        )

    # Use the first rising and last falling transition
    i_lower = rising[0]
    i_upper = falling[-1]

    # Binary searches
    print("lower:")
    R_lower = binary_search_boundary(
        mi,
        R_values[i_lower],
        R_values[i_lower + 1],
        transition="rising",
        tol=tol,
        qi=qi,
    )

    print("upper:")
    R_upper = binary_search_boundary(
        mi,
        R_values[i_upper],
        R_values[i_upper + 1],
        transition="falling",
        tol=tol,
        qi=qi,
    )

    return R_lower, R_upper


# %%
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import os

ms = np.arange(3, 3.1, 0.1)


def process_mass(mi):
    print(f"starting M={mi:.2f} in PID={os.getpid()}", flush=True)

    try:
        R_lower, R_upper = find_RLOF_range(
            mi,
            R_min=50,
            R_max=1500,
            n_coarse=10,
            tol=1.0,
        )
    except ValueError:
        R_lower, R_upper = find_RLOF_range(
            mi,
            R_min=50,
            R_max=1500,
            n_coarse=30,
            tol=1.0,
        )

    return {
        "m1i": mi,
        "R_min": R_lower,
        "R_max": R_upper,
    }


if __name__ == "__main__":
    results2 = []

    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=10,
        mp_context=ctx,
    ) as executor:
        futures = [executor.submit(process_mass, mi) for mi in ms]

        for future in tqdm(futures):
            results2.append(future.result())

    results2 = pd.DataFrame(results2).sort_values("m1i").reset_index(drop=True)
# %%

fig, axs = plt.subplots(
    6, 3, sharey="col", figsize=set_size(full, height=2), constrained_layout=True
)

axs = axs.flatten()


def plot(ax, ri, mi, c, zorder, alpha, linewidth):

    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1

    rl = bin.a * (
        0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log1p(q ** (-1 / 3)))
    )

    ax.plot(
        bin.age[star.ntpagb :],
        rl[star.ntpagb :],
        c=c,
        zorder=zorder,
        alpha=alpha,
        linewidth=linewidth,
    )


for i, result in results.iterrows():
    mi = result["m1i"]
    print(f"computing {mi}")
    R_max = result["R_max"]
    R_min = result["R_min"]
    qi = 0.9

    r = R_max - (R_max - R_min) / 4
    ax = axs[i]

    plot(ax, r, mi, c="C0", zorder=1, alpha=1, linewidth=1)

    for j, result in results.iterrows():
        R_max = result["R_max"]
        R_min = result["R_min"]
        qi = 0.9
        r = R_max - (R_max - R_min) / 6
        if i != j:
            plot(ax, r, mi, c="C9", zorder=-1, alpha=0.5, linewidth=0.5)

    star = get_star(m=mi)
    ax.plot(
        star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", linewidth=1
    )
    ax.text(
        0.05,
        0.95,
        f"${mi:.1f}\\; M_\\odot$",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

axs[-2].set_xlabel("Star age (yr)")
# fig.supylabel(r"$R_\textrm{RL}$ $(R_\odot)$", fontsize=10)

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    if i % 3 == 0:
        ax.set_ylabel(r"$R_\textrm{RL}$ $(R_\odot)$")

plt.savefig("/home/koen/LaTeX-setup/plots/w35-roche-radii.pgf", format="pgf")
plt.show()
plt.close()
# %%
print(results)
# %%

# with open(
#     f"/home/koen/master-internship/scripts/w35/cache/results1.pkl",
#     "wb",
# ) as f:
#     pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
print(results)
# %%

rs = []
for j, result in results.iterrows():
    R_max = result["R_max"]
    R_min = result["R_min"]
    qi = 0.9
    r = R_max - (R_max - R_min) / 6
    rs.append(r)
rs.sort()
for r in rs:
    print(r)
# %%

rr = [
    500,
    525,
    550,
    575,
    600,
    650,
    700,
    750,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
]

ms = np.arange(1.3, 3.1, 0.1)

fig, axs = plt.subplots(
    6, 3, sharey="row", figsize=set_size(full, height=2), constrained_layout=True
)

axs = axs.flatten()


def plot(ax, ri, mi, c, zorder, alpha, linewidth):

    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1

    rl = bin.a * (
        0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log1p(q ** (-1 / 3)))
    )

    ax.plot(
        bin.age[star.ntpagb :],
        rl[star.ntpagb :],
        c=c,
        zorder=zorder,
        alpha=alpha,
        linewidth=linewidth,
    )


for i, m in enumerate(ms):
    mi = m
    print(f"\ncomputing {mi}")
    ax = axs[i]

    for r in rr:
        print(".", end="", flush=True)
        qi = 0.9
        plot(ax, r, mi, c="k", zorder=-1, alpha=1, linewidth=0.5)

    star = get_star(m=mi)
    ax.plot(
        star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", linewidth=1
    )
    ax.text(
        0.95,
        0.05,
        f"${mi:.1f}\\; M_\\odot$",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlim(star.age[-1] - (star.age[-1] - star.age[star.ntpagb]) / 4, star.age[-1])
    ax.set_ylim(0, np.max([ax.get_ylim()[1], np.max(1.2 * 10**star.log_R)]))
axs[-2].set_xlabel("Star age (yr)")
# fig.supylabel(r"$R_\textrm{RL}$ $(R_\odot)$", fontsize=10)

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    if i % 3 == 0:
        ax.set_ylabel(r"$R_\textrm{RL}$ $(R_\odot)$")

plt.savefig("/home/koen/LaTeX-setup/plots/w35-roche-radii-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

g1 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
g2 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
g3 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M2.2-2026-10-09/")
g4 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M2.6-2026-10-09/")
g5 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M3.0-2026-10-09/")

# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
grid.merge(g2)
grid.merge(g3)
grid.merge(g4)
grid.merge(g5)


# %%
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.1):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
plt.tricontour(x, y, z, levels=5, linewidths=0.75 / 2, colors="w")
cntr2 = plt.tricontourf(
    x,
    y,
    z,
    levels=5,
    cmap="viridis",
)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")

plt.yticks([1.4, 1.8, 2.2, 2.6, 3.0])

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(np.min(y) - 0.1, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.1):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=6, cmap="viridis")
axs.contour(10**X, Y, Z, levels=6, colors="w", linewidths=0.75 / 2)

# plt.colorbar(c)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.annotate(
    "[14\\,000 -- 16\\,000] days",
    xy=(1150, 2.9),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
)

plt.annotate(
    "[12\\,000 -- 14\\,000]",
    xy=(1300, 2.4),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(950, 1.325),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(820, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(500, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(350, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.2):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=6, cmap="viridis")
axs.contour(10**X, Y, Z, levels=6, colors="w", linewidths=0.75 / 2)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.annotate(
    "[12\\,000 -- 14\\,000] days",
    xy=(1150, 2.9),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(950, 1.325),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(820, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(650, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(450, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.3):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=6, cmap="viridis")
axs.contour(10**X, Y, Z, levels=6, colors="w", linewidths=0.75 / 2)

# plt.colorbar(c)
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.annotate(
    "[12\\,000 -- 14\\,000] days",
    xy=(1300, 2.4),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(970, 1.3),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)
plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(850, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(700, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(520, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[0 -- 2\\,000]",
    xy=(370, 1.05),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-3.pgf", format="pgf")
plt.show()
plt.close()


# %%


def get_final_period(mi, ri, qi):
    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1
    a = bin.a * 695508000

    M_total = (bin.m1 + bin.m2) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T[-1]


from scipy.interpolate import griddata

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.3):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


ms = np.arange(1.4, 3.1, 0.2)
print(len(ms))
rrr = [
    [1000, 1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
]
for i, m in enumerate(ms):
    print(m)
    for r in rrr[i]:
        x.append(r)
        y.append(m)
        z.append(get_final_period(m, r, 0.85))

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x, y, c=np.log10(z), cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2
)
c = axs.contourf(10**X, Y, np.log10(Z), levels=15, cmap="viridis")
c.set_rasterized(True)
axs.contour(10**X, Y, np.log10(Z), levels=15, colors="w", linewidths=0.75 / 2)

plt.colorbar(c, label="$\\log(\\textrm{period} / \\textrm{days})$")
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 200)
plt.ylim(1.35, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%


def get_final_period(mi, ri, qi):
    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1
    a = bin.a * 695508000

    M_total = (bin.m1 + bin.m2) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T[-1]


from scipy.interpolate import griddata

x = []
y = []
z = []
for r in grid.filter(q=0.85, eps=0.1, f_delta=0.1):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


ms = np.arange(1.4, 3.1, 0.2)
print(len(ms))
rrr = [
    [1000, 1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
]
for i, m in enumerate(ms):
    print(m)
    for r in rrr[i]:
        x.append(r)
        y.append(m)
        z.append(get_final_period(m, r, 0.85))

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x, y, c=np.log10(z), cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2
)
c = axs.contourf(10**X, Y, np.log10(Z), levels=15, cmap="viridis")
c.set_rasterized(True)
axs.contour(10**X, Y, np.log10(Z), levels=15, colors="w", linewidths=0.75 / 2)

plt.colorbar(c, label="$\\log(\\textrm{period} / \\textrm{days})$")
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 200)
plt.ylim(1.35, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-5.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.7, eps=0.1, f_delta=0.1):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=5, cmap="viridis")
axs.contour(10**X, Y, Z, levels=5, colors="w", linewidths=0.75 / 2)

# plt.colorbar(c)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


# plt.annotate(
#     "[14\\,000 -- 16\\,000] days",
#     xy=(1150, 2.9),
#     xycoords="data",
#     xytext=(0, 0),
#     textcoords="offset points",
#     arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
#     zorder=0,
#     c="k",
#     va="center",
#     fontsize=8,
# )

plt.annotate(
    "[12\\,000 -- 14\\,000] days",
    xy=(1300, 2.4),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(950, 1.325),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(820, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(650, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(450, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-1-q07.pgf", format="pgf")
plt.show()
plt.close()


# %%
from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.7, eps=0.1, f_delta=0.2):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=6, cmap="viridis")
axs.contour(10**X, Y, Z, levels=6, colors="w", linewidths=0.75 / 2)
# plt.colorbar(c)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.annotate(
    "[14\\,000 -- 16\\,000] days",
    xy=(1520, 2.75),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-20,
)

plt.annotate(
    "[12\\,000 -- 14\\,000]",
    xy=(1375, 2.3),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-20,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(930, 1.35),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(820, 1.25),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(700, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(500, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[0 -- 2\\,000]",
    xy=(375, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-2-q07.pgf", format="pgf")
plt.show()
plt.close()


# %%

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.filter(q=0.7, eps=0.1, f_delta=0.3):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(x, y, c=z, cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2)
c = axs.contourf(10**X, Y, Z, levels=6, cmap="viridis")
axs.contour(10**X, Y, Z, levels=6, colors="w", linewidths=0.75 / 2)

# plt.colorbar(c)
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.annotate(
    "[14\\,000 -- 16\\,000] days",
    xy=(1520, 2.75),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-20,
)

plt.annotate(
    "[12\\,000 -- 14\\,000]",
    xy=(1375, 2.4),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-20,
)

plt.annotate(
    "[10\\,000 -- 12\\,000]",
    xy=(1130, 1.65),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-40,
)

plt.annotate(
    "[8\\,000 -- 10\\,000]",
    xy=(950, 1.325),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)
plt.annotate(
    "[6\\,000 -- 8\\,000]",
    xy=(830, 1.25),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[4\\,000 -- 6\\,000]",
    xy=(720, 1.15),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[2\\,000 -- 4\\,000]",
    xy=(570, 1),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.annotate(
    "[0 -- 2\\,000]",
    xy=(400, 1.05),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="w", linewidth=0.75),
    zorder=0,
    c="k",
    va="center",
    fontsize=8,
    rotation=-55,
)

plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-3-q07.pgf", format="pgf")
plt.show()
plt.close()


# %%


def get_final_period(mi, ri, qi):
    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1
    a = bin.a * 695508000

    M_total = (bin.m1 + bin.m2) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T[-1]


from scipy.interpolate import griddata

x = []
y = []
z = []
for r in grid.filter(q=0.7, eps=0.1, f_delta=0.3):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


ms = np.arange(1.4, 3.1, 0.2)
print(len(ms))
rrr = [
    [1000, 1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
]
for i, m in enumerate(ms):
    print(m)
    for r in rrr[i]:
        x.append(r)
        y.append(m)
        z.append(get_final_period(m, r, 0.7))

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x, y, c=np.log10(z), cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2
)
c = axs.contourf(10**X, Y, np.log10(Z), levels=15, cmap="viridis")
c.set_rasterized(True)
axs.contour(10**X, Y, np.log10(Z), levels=15, colors="w", linewidths=0.75 / 2)

plt.colorbar(c, label="$\\log(\\textrm{period} / \\textrm{days})$")
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 200)
plt.ylim(1.35, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-4-q07.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%


def get_final_period(mi, ri, qi):
    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1
    a = bin.a * 695508000

    M_total = (bin.m1 + bin.m2) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T[-1]


from scipy.interpolate import griddata

x = []
y = []
z = []
for r in grid.filter(q=0.7, eps=0.1, f_delta=0.1):
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.period_days[-1])


ms = np.arange(1.4, 3.1, 0.2)
print(len(ms))
rrr = [
    [1000, 1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [1500, 2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
    [2000, 3000, 4000],
]
for i, m in enumerate(ms):
    print(m)
    for r in rrr[i]:
        x.append(r)
        y.append(m)
        z.append(get_final_period(m, r, 0.7))

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x, y, c=np.log10(z), cmap="viridis", zorder=1000, edgecolor="w", linewidth=0.75 / 2
)
c = axs.contourf(10**X, Y, np.log10(Z), levels=15, cmap="viridis")
c.set_rasterized(True)
axs.contour(10**X, Y, np.log10(Z), levels=15, colors="w", linewidths=0.75 / 2)

plt.colorbar(c, label="$\\log(\\textrm{period} / \\textrm{days})$")
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 200)
plt.ylim(1.35, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-5-q07.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
from matplotlib.colors import TwoSlopeNorm

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    3,
    2,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = 1e-99

deltas = [0.1, 0.2, 0.3]
epss = [0, 0.1]

for i, ax in enumerate(axs):
    delta = deltas[i // 2]
    eps = epss[i % 2]

    x = []
    y = []
    z = []
    for r in grid.filter(q=0.7, eps=eps, f_delta=delta):
        if r.envelope_mass[-1] > 0.01:
            continue

        x.append(r.params["R"])
        y.append(r.params["m"])
        z.append(r.s)

    xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
    yi = np.linspace(min(y), max(y), 1000)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata(
        (np.log10(x), y),
        z,
        (X, Y),
        method="cubic",
    )

    if np.min(z) < minn:
        minn = np.min(z)
    if np.max(z) > maxx:
        maxx = np.max(z)

for i, ax in enumerate(axs):
    delta = deltas[i // 2]
    eps = epss[i % 2]

    x = []
    y = []
    z = []
    for r in grid.filter(q=0.7, eps=eps, f_delta=delta):
        if r.envelope_mass[-1] > 0.01:
            continue

        x.append(r.params["R"])
        y.append(r.params["m"])
        z.append(r.s)

    xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
    yi = np.linspace(min(y), max(y), 1000)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata(
        (np.log10(x), y),
        z,
        (X, Y),
        method="cubic",
    )

    norm = TwoSlopeNorm(0.25, minn, maxx)
    ax.scatter(
        x,
        y,
        c=z,
        cmap="coolwarm",
        zorder=1000,
        edgecolor="w",
        linewidth=0.75 / 2,
        norm=norm,
    )
    c = ax.contourf(10**X, Y, Z, levels=10, cmap="coolwarm", norm=norm)
    c.set_rasterized(True)
    ax.contour(10**X, Y, Z, levels=10, colors="w", linewidths=0.75 / 2, norm=norm)

    ax.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
    ax.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")
    ax.tick_params(color="w", axis="both", which="both")
    ax.text(
        0.95,
        0.05,
        f"$\\varepsilon = {eps},\\;f_\\delta = {delta}$",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
    )

fig.colorbar(c, ax=axs[-3], label="[s/Fe]")
# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
plt.xscale("log")

for ax in axs:
    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
axs[-1].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[-2].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[-2].set_ylabel("Initial mass ($M_\\odot$)")
axs[0].set_ylabel("Initial mass ($M_\\odot$)")
axs[2].set_ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-mass-s-1.pgf", format="pgf")
plt.show()
plt.close()
# %%
