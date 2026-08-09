import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from tqdm import tqdm
import sys
import pickle
from pathlib import Path
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M1.5"

cache_dir = Path(f"{proj_dir}/scripts/w23/cache")
cache_dir.mkdir(exist_ok=True)


def compute_epsilon(barium_mass, tpagb_mass, core_mass, q):
    """
    computes the mass transfer efficiency assuming no wind mass transfer.
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = barium_mass - tpagb_mass * q

    return delta_accreted / delta_donated


def compute_final_mass(epsilon, tpagb_mass, core_mass, q):
    """
    computes the final mass of the barium star for a given epsilon
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = epsilon * delta_donated

    return delta_accreted + tpagb_mass * q


for m_i in np.arange(1.0, 2.3, 0.1):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        print("est")
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%

R = 450
barium_mass_min = 0.73
barium_mass_max = 1.5
eps_max = 0.5

masses = np.arange(2.3, 2.45, 0.1)

qs = np.arange(0.05, 1.05, 0.05)
rs = np.arange(100, 1260, 10)
possible_matrix = np.full((len(rs), len(qs)), np.nan)
eps_min_matrix = np.full((len(rs), len(qs)), np.nan)
eps_max_matrix = np.full((len(rs), len(qs)), np.nan)
m_DUP_matrix = np.full((len(rs), len(qs)), np.nan)
CO_matrix = np.full((len(rs), len(qs)), np.nan)
min_mass_matrix = np.full((len(rs), len(qs)), np.nan)
max_mass_matrix = np.full((len(rs), len(qs)), np.nan)

# %%
for m_i in masses:

    for i, R in enumerate(tqdm(rs)):

        single_star_dir = (
            f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
        )
        try:
            with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
                Star = pickle.load(f)
        except:

            Star = read_stellar_models(single_star_dir)[0]
            with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
                pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

        for j, q in enumerate(qs):

            a_init = inv_roche_lobe(R, q)

            cache_file = cache_dir / f"m{m_i:.1f}_R{R:.2f}_q{q:.3f}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                if len(result) == 6:
                    # old cache
                    Star, Options, q_init, a_init, e_init, Bins = result
                elif len(result) == 5:
                    # new cache (Star omitted)
                    Options, q_init, a_init, e_init, Bins = result
                else:
                    raise ValueError(
                        f"Unexpected cache format ({len(result)} elements)"
                    )
            else:
                Star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    Star, q, a_init, simple_only=True
                )

                with open(cache_file, "wb") as f:
                    pickle.dump(
                        (Options, q_init, a_init, e_init, Bins),
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            R_star = 10**Star.log_R
            core_mass = Star.m_core
            ages_star = Star.age
            bin = Bins[0]

            q_evolve = bin.m2 / bin.m1
            RL = roche_lobe(1 / q_evolve) * bin.a

            q_begin = q_evolve[-1]
            tpagb_mass = bin.m1[-1]

            # interpolate stellar radius onto binary ages
            core_mass_interp = np.interp(bin.age, ages_star, core_mass)
            try:
                TP_count = int(
                    np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1]
                )
            except ValueError:
                TP_count = int(
                    np.interp(
                        bin.age, Star.age[Star.ntpagb :], Star.TP_count[Star.ntpagb :]
                    )[-1]
                )
            CO_ratio = np.interp(
                bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
            )[-1]

            eps_min_mass = compute_epsilon(
                barium_mass_min, tpagb_mass, core_mass_interp[-1], q_begin
            )
            eps_max_mass = compute_epsilon(
                barium_mass_max, tpagb_mass, core_mass_interp[-1], q_begin
            )

            min_mass = compute_final_mass(0, tpagb_mass, core_mass_interp[-1], q_begin)
            max_mass = compute_final_mass(
                0.5, tpagb_mass, core_mass_interp[-1], q_begin
            )

            if eps_min_mass < 0 and eps_max_mass < 0:
                possible = False
            elif eps_min_mass > eps_max and eps_max_mass > eps_max:
                possible = False
            elif bin.age[-1] - ages_star[-1] == 0:
                possible = 0.5
            else:
                possible = True

            possible_matrix[i, j] = possible
            min_mass_matrix[i, j] = min_mass
            max_mass_matrix[i, j] = max_mass
            eps_min_matrix[i, j] = eps_min_mass
            eps_max_matrix[i, j] = eps_max_mass
            m_DUP_matrix[i, j] = np.sum(Star.m_DUP[:TP_count])
            CO_matrix[i, j] = CO_ratio

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "wb") as f:
        pickle.dump(possible_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "wb") as f:
        pickle.dump(eps_min_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "wb") as f:
        pickle.dump(eps_max_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "wb") as f:
        pickle.dump(min_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "wb") as f:
        pickle.dump(max_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "wb") as f:
        pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "wb") as f:
        pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
m_i = 1.6
R = 1000

for i, m_i in enumerate(masses):

    if i != 6:
        continue

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

# %%
from matplotlib.collections import LineCollection
import numpy as np


def draw_region_boundaries(
    ax,
    region,
    x,
    y,
    colors=None,
    linewidth=2,
):
    """
    Draw blocky boundaries between discrete regions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    region : 2D ndarray
        Region labels (e.g. 0, 0.5, 1).
        Shape must match the pcolormesh array.
    x, y : 1D arrays
        Grid coordinates used by pcolormesh.
    colors : dict, optional
        Dictionary mapping region pairs to colors.
        Example:
        {
            (0, 0.5): "red",
            (0.5, 1): "blue",
            (0, 1): "green",
        }
    linewidth : float
        Line width.
    """

    if colors is None:
        colors = {
            (0, 0.5): "red",
            (0.5, 1): "blue",
            (0, 1): "black",
        }

    segments = []
    segment_colors = []

    ny, nx = region.shape

    # vertical boundaries
    for j in range(ny):
        for i in range(nx - 1):
            a = region[j, i]
            b = region[j, i + 1]

            if a != b:
                pair = tuple(sorted((a, b)))
                if pair in colors:
                    segments.append(
                        [
                            (0.5 * (x[i] + x[i + 1]), y[j] - 0.5 * (y[1] - y[0])),
                            (0.5 * (x[i] + x[i + 1]), y[j] + 0.5 * (y[1] - y[0])),
                        ]
                    )
                    segment_colors.append(colors[pair])

    # horizontal boundaries
    for j in range(ny - 1):
        for i in range(nx):
            a = region[j, i]
            b = region[j + 1, i]

            if a != b:
                pair = tuple(sorted((a, b)))
                if pair in colors:
                    segments.append(
                        [
                            (x[i] - 0.5 * (x[1] - x[0]), 0.5 * (y[j] + y[j + 1])),
                            (x[i] + 0.5 * (x[1] - x[0]), 0.5 * (y[j] + y[j + 1])),
                        ]
                    )
                    segment_colors.append(colors[pair])

    lc = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=linewidth,
        capstyle="butt",
        joinstyle="miter",
        rasterized=True,
        label="test",
    )

    ax.add_collection(lc)

    return lc


# %%


from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    5,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    if np.nanmin(CO_matrix) < minn:
        minn = np.nanmin(CO_matrix)

    if np.nanmax(CO_matrix) > maxx:
        maxx = np.nanmax(CO_matrix)

norm = TwoSlopeNorm(1, minn, maxx)
cmap = plt.cm.coolwarm_r


axs = axs.flatten()

for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        CO_matrix.T,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )

    lc = draw_region_boundaries(
        axs[i],
        eps_matrix.T,  # important: match CO_matrix.T
        rs,
        qs,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    axs[i].contourf(
        R,
        Q,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )
    # axs[i].text(
    #     0.1,
    #     0.9,
    #     f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    #     transform=axs[i].transAxes,
    #     ha="left",
    #     va="top",
    #     bbox=dict(
    #         boxstyle="square",
    #         fc=(1.0, 1, 1,0.7),
    #         ec=(1.0, 1, 1, 0.7),
    #     ),
    # )
    axs[i].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

axs[0].set_yticks([0.1, 0.5, 1])

cbar = plt.colorbar(sm, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"CO-ratio at end of binary evolution")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind regime with\n possible and impossible final masses",
        "Border between RLOF and\nwind only regime",
        "Border between RLOF regime with\n possible and impossible final masses",
        "Impossible region",
    ],
    loc="lower right",
    ncols=1,
)

axs[-3].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Initial mass ratio $q$")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-CO-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

eps_mins = []
eps_maxs = []


for m, m_i in enumerate(masses):
    if m != len(masses) - 5:
        continue

    for i, R in enumerate(tqdm(rs)):

        single_star_dir = (
            f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
        )
        try:
            with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
                Star = pickle.load(f)
        except:

            Star = read_stellar_models(single_star_dir)[0]
            with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
                pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # if i != 72:
        #     continue

        for j, q in enumerate(qs):

            print(j)
            if j != 15:
                continue

            a_init = inv_roche_lobe(R, q)

            cache_file = cache_dir / f"m{m_i:.1f}_R{R:.2f}_q{q:.3f}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                if len(result) == 6:
                    # old cache
                    Star, Options, q_init, a_init, e_init, Bins = result
                elif len(result) == 5:
                    # new cache (Star omitted)
                    Options, q_init, a_init, e_init, Bins = result
                else:
                    raise ValueError(
                        f"Unexpected cache format ({len(result)} elements)"
                    )
            else:
                Star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    Star, q, a_init, simple_only=True
                )

            R_star = 10**Star.log_R
            core_mass = Star.m_core
            ages_star = Star.age
            bin = Bins[0]

            q_evolve = bin.m2 / bin.m1
            RL = roche_lobe(1 / q_evolve) * bin.a

            q_begin = q_evolve[-1]
            tpagb_mass = bin.m1[-1]
            core_mass_interp = np.interp(bin.age, ages_star, core_mass)

            eps_min_mass = compute_epsilon(
                barium_mass_min, tpagb_mass, core_mass_interp[-1], q_begin
            )
            eps_max_mass = compute_epsilon(
                barium_mass_max, tpagb_mass, core_mass_interp[-1], q_begin
            )

            print(q, eps_min_mass, eps_max_mass)

            axs[0].plot(bin.age, RL)

            eps_mins.append(eps_min_mass)
            eps_maxs.append(eps_max_mass)
            # axs[1].plot(bin.age, bin.m2)
            # axs[1].plot(bin.age, bin.m1)

# axs[0].plot(Star.age, 10**Star.log_R)

# axs[1].plot(rs, eps_mins)
# axs[1].plot(rs, eps_maxs)
#
# axs[0].set_xlim(921396580.533341, 922654840.0238882)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
# plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()

# %%


from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    5,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    if np.nanmin(m_DUP_matrix) < minn:
        minn = np.nanmin(m_DUP_matrix)

    if np.nanmax(m_DUP_matrix) > maxx:
        maxx = np.nanmax(m_DUP_matrix)


axs = axs.flatten()

for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        m_DUP_matrix.T,
        shading="nearest",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )

    lc = draw_region_boundaries(
        axs[i],
        eps_matrix.T,  # important: match CO_matrix.T
        rs,
        qs,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    axs[i].contourf(
        R,
        Q,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )
    # axs[i].text(
    #     0.1,
    #     0.9,
    #     f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    #     transform=axs[i].transAxes,
    #     ha="left",
    #     va="top",
    #     bbox=dict(
    #         boxstyle="square",
    #         fc=(1.0, 1, 1,0.7),
    #         ec=(1.0, 1, 1, 0.7),
    #     ),
    # )
    axs[i].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


axs[0].set_yticks([0.1, 0.5, 1])

cbar = plt.colorbar(colors, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"$M_\textrm{dup}$ at end of binary evolution ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind regime with\n possible and impossible final masses",
        "Border between RLOF and\nwind only regime",
        "Border between RLOF regime with\n possible and impossible final masses",
        "Impossible region",
    ],
    loc="lower right",
    ncols=1,
)

axs[-3].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Initial mass ratio $q$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-m_DUP-region-all-masses.pgf", format="pgf"
)
plt.show()
plt.close()

# %%


from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    5,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

my_cmap = plt.get_cmap(name="viridis", lut=None).copy()
my_cmap.set_under("white")
my_cmap.set_over("white")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    matrix = eps_min_matrix

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)


axs = axs.flatten()

for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        eps_min_matrix.T,
        shading="nearest",
        cmap=my_cmap,
        vmin=0,
        vmax=0.5,
        rasterized=True,
    )

    lc = draw_region_boundaries(
        axs[i],
        eps_matrix.T,  # important: match CO_matrix.T
        rs,
        qs,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    axs[i].contourf(
        R,
        Q,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )
    # axs[i].text(
    #     0.1,
    #     0.9,
    #     f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    #     transform=axs[i].transAxes,
    #     ha="left",
    #     va="top",
    #     bbox=dict(
    #         boxstyle="square",
    #         fc=(1.0, 1, 1,0.7),
    #         ec=(1.0, 1, 1, 0.7),
    #     ),
    # )
    axs[i].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


axs[0].set_yticks([0.1, 0.5, 1])

cbar = plt.colorbar(colors, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"$M_\textrm{dup}$ at end of binary evolution ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind regime with\n possible and impossible final masses",
        "Border between RLOF and\nwind only regime",
        "Border between RLOF regime with\n possible and impossible final masses",
        "Impossible region",
    ],
    loc="lower right",
    ncols=1,
)

axs[-3].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Initial mass ratio $q$")

# plt.savefig(
#     "/home/koen/LaTeX-setup/plots/w23-m_DUP-region-all-masses.pgf", format="pgf"
# )
plt.show()
plt.close()
# %%


from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    5,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99


for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    matrix = min_mass_matrix

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)
        print(minn)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)
        print(maxx)


axs = axs.flatten()

norm = TwoSlopeNorm(1.5, minn, maxx)
print(minn, maxx)
cmap = plt.cm.coolwarm


for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        min_mass_matrix.T,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        # vmin=0,
        # vmax=1,
        rasterized=True,
    )

    lc = draw_region_boundaries(
        axs[i],
        eps_matrix.T,  # important: match CO_matrix.T
        rs,
        qs,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    axs[i].contourf(
        R,
        Q,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )
    # axs[i].text(
    #     0.1,
    #     0.9,
    #     f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    #     transform=axs[i].transAxes,
    #     ha="left",
    #     va="top",
    #     bbox=dict(
    #         boxstyle="square",
    #         fc=(1.0, 1, 1,0.7),
    #         ec=(1.0, 1, 1, 0.7),
    #     ),
    # )
    axs[i].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


axs[0].set_yticks([0.1, 0.5, 1])

cbar = plt.colorbar(colors, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"Mass with $\varepsilon = 0$ ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind regime with\n possible and impossible final masses",
        "Border between RLOF and\nwind only regime",
        "Border between RLOF regime with\n possible and impossible final masses",
        "Impossible region",
    ],
    loc="lower right",
    ncols=1,
)

axs[-3].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Initial mass ratio $q$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-min-mass-region-all-masses.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    5,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99


for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    matrix = max_mass_matrix

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)
        print(minn)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)
        print(maxx)


axs = axs.flatten()

norm = TwoSlopeNorm(0.73, minn, maxx)
print(minn, maxx)
cmap = plt.cm.coolwarm_r


for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    with open(f"scripts/w23/const_mass_possible_{m_i}_small.pkl", "rb") as f:
        eps_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_max_{m_i}_small.pkl", "rb") as f:
        eps_min_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_eps_min_{m_i}_small.pkl", "rb") as f:
        eps_max_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_co_{m_i}_small.pkl", "rb") as f:
        CO_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_dup_{m_i}_small.pkl", "rb") as f:
        m_DUP_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_min_mass_{m_i}_small.pkl", "rb") as f:
        min_mass_matrix = pickle.load(f)
    with open(f"scripts/w23/const_mass_max_mass_{m_i}_small.pkl", "rb") as f:
        max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        max_mass_matrix.T,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        # vmin=0,
        # vmax=1,
        rasterized=True,
    )

    lc = draw_region_boundaries(
        axs[i],
        eps_matrix.T,  # important: match CO_matrix.T
        rs,
        qs,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    axs[i].contourf(
        R,
        Q,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )
    # axs[i].text(
    #     0.1,
    #     0.9,
    #     f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    #     transform=axs[i].transAxes,
    #     ha="left",
    #     va="top",
    #     bbox=dict(
    #         boxstyle="square",
    #         fc=(1.0, 1, 1,0.7),
    #         ec=(1.0, 1, 1, 0.7),
    #     ),
    # )
    axs[i].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


axs[0].set_yticks([0.1, 0.5, 1])

cbar = plt.colorbar(colors, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"Mass with $\varepsilon = 0.5$ ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind regime with\n possible and impossible final masses",
        "Border between RLOF and\nwind only regime",
        "Border between RLOF regime with\n possible and impossible final masses",
        "Impossible region",
    ],
    loc="lower right",
    ncols=1,
)

axs[-3].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Initial mass ratio $q$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-max-mass-region-all-masses.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    5,
    3,
    sharex=False,
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

eps_mins = []
eps_maxs = []

axs = axs.flatten()


for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")


for m, m_i in enumerate(masses):

    for i, R in enumerate(tqdm(rs[::2])):

        single_star_dir = (
            f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
        )
        try:
            with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
                Star = pickle.load(f)
        except:

            Star = read_stellar_models(single_star_dir)[0]
            with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
                pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # if i != 72:
        #     continue

        for j, q in enumerate(qs):

            if j != 15:
                continue

            a_init = inv_roche_lobe(R, q)

            cache_file = cache_dir / f"m{m_i:.1f}_R{R:.2f}_q{q:.3f}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                if len(result) == 6:
                    # old cache
                    Star, Options, q_init, a_init, e_init, Bins = result
                elif len(result) == 5:
                    # new cache (Star omitted)
                    Options, q_init, a_init, e_init, Bins = result
                else:
                    raise ValueError(
                        f"Unexpected cache format ({len(result)} elements)"
                    )
            else:
                Star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    Star, q, a_init, simple_only=True
                )

            R_star = 10**Star.log_R
            core_mass = Star.m_core
            ages_star = Star.age
            bin = Bins[0]

            q_evolve = bin.m2 / bin.m1
            RL = roche_lobe(1 / q_evolve) * bin.a

            q_begin = q_evolve[-1]
            tpagb_mass = bin.m1[-1]
            core_mass_interp = np.interp(bin.age, ages_star, core_mass)

            eps_min_mass = compute_epsilon(
                barium_mass_min, tpagb_mass, core_mass_interp[-1], q_begin
            )
            eps_max_mass = compute_epsilon(
                barium_mass_max, tpagb_mass, core_mass_interp[-1], q_begin
            )

            if bin.age[-1] == Star.age[-1]:
                axs[m].plot(bin.age, RL, c="C2", linewidth=0.4, rasterized=True)
            else:
                axs[m].plot(bin.age, RL, c="C3", linewidth=0.4, rasterized=True)

            eps_mins.append(eps_min_mass)
            eps_maxs.append(eps_max_mass)
            # axs[1].plot(bin.age, bin.m2)
            # axs[1].plot(bin.age, bin.m1)

        if i == 0:
            min = bin.age[-1]

    axs[m].plot(Star.age, 10**Star.log_R, c="k")
    max = Star.age[-1]
    min = np.max([min, Star.age[Star.ntpagb]])
    delta = max - min
    axs[m].set_xlim(min - delta / 5, max + delta / 5)

    axs[m].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


custom_lines = [
    Line2D([0], [0], color="C2", lw=2),
    Line2D([0], [0], color="C3", lw=2),
    Line2D([0], [0], color="k", lw=2),
]

fig.legend(
    custom_lines,
    [
        "Roche lobe radius (No RLOF)",
        "Roche lobe radius (RLOF)",
        "Star radius",
    ],
    loc="lower right",
    ncols=1,
)


plt.ylim(0, 1300)

axs[-3].set_xlabel("Age (yr)")
axs[6].set_ylabel("Radius ($R_\odot$)")

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-wind.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    5,
    3,
    sharex=False,
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

eps_mins = []
eps_maxs = []

axs = axs.flatten()


for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")


for m, m_i in enumerate(masses):

    for i, R in enumerate(tqdm(rs[::2])):

        single_star_dir = (
            f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
        )
        try:
            with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
                Star = pickle.load(f)
        except:

            Star = read_stellar_models(single_star_dir)[0]
            with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
                pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # if i != 72:
        #     continue

        for j, q in enumerate(qs):

            if j != 15:
                continue

            a_init = inv_roche_lobe(R, q)

            cache_file = cache_dir / f"m{m_i:.1f}_R{R:.2f}_q{q:.3f}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                if len(result) == 6:
                    # old cache
                    Star, Options, q_init, a_init, e_init, Bins = result
                elif len(result) == 5:
                    # new cache (Star omitted)
                    Options, q_init, a_init, e_init, Bins = result
                else:
                    raise ValueError(
                        f"Unexpected cache format ({len(result)} elements)"
                    )
            else:
                Star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    Star, q, a_init, simple_only=True
                )

            R_star = 10**Star.log_R
            core_mass = Star.m_core
            ages_star = Star.age
            bin = Bins[0]

            q_evolve = bin.m2 / bin.m1
            RL = roche_lobe(1 / q_evolve) * bin.a

            q_begin = q_evolve[-1]
            tpagb_mass = bin.m1[-1]
            core_mass_interp = np.interp(bin.age, ages_star, core_mass)

            eps_min_mass = compute_epsilon(
                barium_mass_min, tpagb_mass, core_mass_interp[-1], q_begin
            )
            eps_max_mass = compute_epsilon(
                barium_mass_max, tpagb_mass, core_mass_interp[-1], q_begin
            )

            if bin.age[-1] == Star.age[-1]:
                axs[m].plot(bin.age, RL, c="C2", linewidth=0.4, rasterized=True)
            else:
                axs[m].plot(bin.age, RL, c="C3", linewidth=0.4, rasterized=True)

            eps_mins.append(eps_min_mass)
            eps_maxs.append(eps_max_mass)
            # axs[1].plot(bin.age, bin.m2)
            # axs[1].plot(bin.age, bin.m1)

    axs[m].plot(Star.age, 10**Star.log_R, c="k")
    max = Star.age[-1]
    min = (Star.age[Star.ntpagb] + 2 * max) / 3
    delta = max - min
    axs[m].set_xlim(min - delta / 5, max + delta / 5)

    axs[m].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$",
    )


custom_lines = [
    Line2D([0], [0], color="C2", lw=2),
    Line2D([0], [0], color="C3", lw=2),
    Line2D([0], [0], color="k", lw=2),
]

fig.legend(
    custom_lines,
    [
        "Roche lobe radius (No RLOF)",
        "Roche lobe radius (RLOF)",
        "Star radius",
    ],
    loc="lower right",
    ncols=1,
)


plt.ylim(0, 1300)

axs[-3].set_xlabel("Age (yr)")
axs[6].set_ylabel("Radius ($R_\odot$)")

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-wind-zoom.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)


norm = plt.Normalize(np.min(masses), np.max(masses))
cmap = plt.cm.Spectral
# color = cmap(norm(x))


for m, m_i in enumerate(masses):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

    ind = np.argmax(Star.log_R)

    plt.plot(
        Star.m_env[ind:],
        10 ** Star.log_R[ind:] / np.max(10 ** Star.log_R[ind:]),
        c=cmap(norm(m_i)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\textrm{TPAGB}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Envelope mass ($M_\odot$)")
plt.xscale("log")
plt.ylabel(r"$R / R_\textrm{max}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-bump.pgf", format="pgf")
plt.show()
plt.close()
# %%
