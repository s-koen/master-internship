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


for m_i in np.arange(1.0, 2.55, 0.1):

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

masses = np.arange(1.0, 2.65, 0.1)
# masses = np.arange(2.2, 2.45, 0.1)
# masses = [2.5] 
# masses = [1.5,1.7] 
print(masses)

qs = np.arange(0.05, 1.05, 0.05)
rs = np.arange(100, 1260, 10)
possible_matrix = np.full((len(rs), len(qs)), np.nan)
eps_min_matrix = np.full((len(rs), len(qs)), np.nan)
eps_max_matrix = np.full((len(rs), len(qs)), np.nan)
m_DUP_matrix = np.full((len(rs), len(qs)), np.nan)
CO_matrix = np.full((len(rs), len(qs)), np.nan)
min_mass_matrix = np.full((len(rs), len(qs)), np.nan)
max_mass_matrix = np.full((len(rs), len(qs)), np.nan)
wind_accretion_matrix = np.full((len(rs), len(qs)), np.nan)

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
            wind_accretion_matrix[i, j] = bin.m2[-1] - bin.m2[0]
            CO_matrix[i, j] = CO_ratio

    with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(possible_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(eps_min_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(eps_max_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(min_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(max_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(wind_accretion_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
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
    6,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    if np.nanmin(CO_matrix) < minn:
        minn = np.nanmin(CO_matrix)

    if np.nanmax(CO_matrix) > maxx:
        maxx = np.nanmax(CO_matrix)

norm = TwoSlopeNorm(1, minn, maxx)
cmap = plt.cm.coolwarm_r


axs = axs.flatten()

for ax in axs:
    if ax == axs[-1]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    try:
        colors = axs[i].pcolormesh(
            R,
            Q,
            CO_matrix.T,
            shading="nearest",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
    except:
        pass

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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
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
    6,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):
    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    try:
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
    except:
        pass

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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
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
    6,
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

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
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
    6,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99


for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
    if ax == axs[-1]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
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
    6,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99


for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
    if ax == axs[-1]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
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
    if ax == axs[-1] :
        ax.axis("off")

masses_strict = np.arange(1,2.35,0.1)

for m, m_i in enumerate(masses_strict):

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
        "Roche lobe radius\n(No RLOF)",
        "Roche lobe radius\n(RLOF)",
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
    if ax == axs[-1]:
        ax.axis("off")


for m, m_i in enumerate(masses_strict):

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
        "Roche lobe radius\n(No RLOF)",
        "Roche lobe radius\n(RLOF)",
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

from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    6,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    matrix = eps_matrix

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)

print(minn, maxx)
norm = TwoSlopeNorm(0.5, minn, maxx)
cmap = plt.cm.coolwarm_r


axs = axs.flatten()

for ax in axs:
    if ax == axs[-1] or ax == axs[-2]:
        ax.axis("off")

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
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
        eps_matrix.T,
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

# plt.savefig("/home/koen/LaTeX-setup/plots/w23-CO-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%

for m_i in [2.5]:

    for i, R in enumerate([1000]):

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

            try:
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
            except EOFError:
                print(q)
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
            print(bin.age[-1] - ages_star[-1])

            plt.plot(bin.age, RL)

plt.plot(ages_star, R_star)
plt.show()
# %%

q = 0.05
R = 960

a_init = inv_roche_lobe(R, q)
Star, Options, q_init, a_init, e_init, Bins = call_evolution(
    Star, q, a_init, simple_only=True
)

R_star = 10**Star.log_R
core_mass = Star.m_core
ages_star = Star.age
bin = Bins[0]

q_evolve = bin.m2 / bin.m1
RL = roche_lobe(1 / q_evolve) * bin.a

plt.plot(bin.age, RL)
plt.plot(Star.age, 10**Star.log_R)
plt.show()
# %%
print(type(bin.m2[-1]))
# %%
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    1,
    2,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=0.6),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    if np.nanmin(CO_matrix) < minn:
        minn = np.nanmin(CO_matrix)

    if np.nanmax(CO_matrix) > maxx:
        maxx = np.nanmax(CO_matrix)

norm = TwoSlopeNorm(1, minn, maxx)
cmap = plt.cm.coolwarm_r

# masses = [2.5]
masses = np.arange(1.0, 3.05, 0.1)
cmap = plt.cm.viridis
norm = Normalize(np.min(masses), np.max(masses))

for i, m_i in enumerate(masses):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot == 0)] = 0
    eps_plot[(eps_plot > 0) | (eps_plot == 0.5)] = m_i

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    # colors = axs.pcolormesh(
    #     R,
    #     Q,
    #     CO_matrix.T,
    #     shading="nearest",
    #     cmap=cmap,
    #     norm=norm,
    #     rasterized=True,
    # )
    #
    # lc = draw_region_boundaries(
    #     axs,
    #     eps_matrix.T,  # important: match CO_matrix.T
    #     rs,
    #     qs,
    #     colors={
    #         (0, 0.5): "magenta",
    #         (0.5, 1): "black",
    #         (0, 1): "lime",
    #     },
    # )
    #
    print(eps_plot.T)
    # plt.contour(R,Q,eps_plot.T, levels=[m_i/2], colors=[cmap(norm(m_i))])
    axs[0].contourf(
    R,
    Q,
    eps_plot.T,
    levels=[m_i / 2, m_i + 1],
    colors=[cmap(norm(m_i))],
    alpha=1
    )
    # plt.pcolormesh(R,Q,eps_plot.T,cmap="viridis", vmin=np.min(masses),vmax=np.max(masses))

    # axs.contourf(
    #     R,
    #     Q,
    #     eps_matrix.T,
    #     colors="none",
    #     hatches=["||", None, None, None],
    #     corner_mask=False,
    #     rasterized=True,
    # )
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
for i, m_i in enumerate(masses[::-1]):

    try:
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
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot == 0)] = 0
    eps_plot[(eps_plot > 0) | (eps_plot == 0.5)] = m_i

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    # colors = axs.pcolormesh(
    #     R,
    #     Q,
    #     CO_matrix.T,
    #     shading="nearest",
    #     cmap=cmap,
    #     norm=norm,
    #     rasterized=True,
    # )
    #
    # lc = draw_region_boundaries(
    #     axs,
    #     eps_matrix.T,  # important: match CO_matrix.T
    #     rs,
    #     qs,
    #     colors={
    #         (0, 0.5): "magenta",
    #         (0.5, 1): "black",
    #         (0, 1): "lime",
    #     },
    # )
    #
    print(eps_plot.T)
    # plt.contour(R,Q,eps_plot.T, levels=[m_i/2], colors=[cmap(norm(m_i))])
    cs = axs[1].contourf(
    R,
    Q,
    eps_plot.T,
    levels=[m_i / 2, m_i + 1],
    colors=[cmap(norm(m_i))],
    alpha=1,
    zorder=-20
    )

    # plt.pcolormesh(R,Q,eps_plot.T,cmap="viridis", vmin=np.min(masses),vmax=np.max(masses))

    # axs.contourf(
    #     R,
    #     Q,
    #     eps_matrix.T,
    #     colors="none",
    #     hatches=["||", None, None, None],
    #     corner_mask=False,
    #     rasterized=True,
    # )
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


for ax in axs:
    ax.set_rasterization_zorder(-10)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs, orientation="horizontal", location="top", aspect=50)
cbar.ax.set_xscale("linear")
cbar.set_label(r"Mass of TPAGB-star ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)

axs[0].set_title("Maximum possible mass")
axs[1].set_title("Minimum possible mass")

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# custom_lines = [
#     Line2D([0], [0], color="magenta", lw=2),
#     Line2D([0], [0], color="black", lw=2),
#     Line2D([0], [0], color="lime", lw=2),
#     Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
# ]
#
# fig.legend(
#     custom_lines,
#     [
#         "Border between wind regime with\n possible and impossible final masses",
#         "Border between RLOF and\nwind only regime",
#         "Border between RLOF regime with\n possible and impossible final masses",
#         "Impossible region",
#     ],
#     loc="lower right",
#     ncols=1,
# )

plt.savefig("/home/koen/LaTeX-setup/plots/w23-min-max-mass.pgf", format="pgf")
plt.show()
plt.close()

# %%


RRs = [460,460,490,540,730,710,860,780,960,900,960,1000,1120,1140,1160,1180, 1180]

fig, axs = plt.subplots(
    6,
    3,
    sharex=False,
    sharey=False,
    figsize=set_size(full, height=1.5),
    constrained_layout=True,
)

eps_mins = []
eps_maxs = []

axs = axs.flatten()

norm = plt.Normalize(0.05, 1)
cmap = plt.cm.viridis
# color = cmap(norm(x))




for ax in axs:
    if ax == axs[-1]:
        ax.axis("off")


for m, m_i in enumerate(tqdm(masses)):

    for i, R in enumerate([RRs[m]]):

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

            axs[m].plot(bin.age, RL, c=cmap(norm(q)), linewidth=0.4, rasterized=True)

            eps_mins.append(eps_min_mass)
            eps_maxs.append(eps_max_mass)
            # axs[1].plot(bin.age, bin.m2)
            # axs[1].plot(bin.age, bin.m1)

    axs[m].plot(Star.age, 10**Star.log_R, c="k")
    max = Star.age[-1]
    min = (Star.age[Star.ntpagb] + 5 * max) / 6
    delta = max - min
    axs[m].set_xlim(min - delta / 10, max + delta / 10)

    axs[m].set_title(
        f"$M_\\textrm{{TPAGB}} = {m_i:.1f}$, $R_\\textrm{{RL,i}} = {R:.0f}$",
    )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])



cbar = plt.colorbar(sm, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"Initial binary mass ratio ($q$)")


# plt.ylim(0, 1300)

axs[-3].set_xlabel("Age (yr)")
axs[6].set_ylabel("Radius ($R_\odot$)")

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-q-wind.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    1,
    1,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=0.6),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

masses = [2.0]
for i, m_i in enumerate(masses):

    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)

    if np.nanmin(CO_matrix) < minn:
        minn = np.nanmin(CO_matrix)

    if np.nanmax(CO_matrix) > maxx:
        maxx = np.nanmax(CO_matrix)

norm = TwoSlopeNorm(1, minn, maxx)
cmap = plt.cm.coolwarm_r

masses = [2.0]
# masses = np.arange(1.0, 2.55, 0.1)
cmap = plt.cm.viridis
norm = Normalize(np.min(masses), np.max(masses))

for i, m_i in enumerate(masses):

    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot == 0)] = 0
    eps_plot[(eps_plot > 0) | (eps_plot == 0.5)] = m_i

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    # colors = axs.pcolormesh(
    #     R,
    #     Q,
    #     CO_matrix.T,
    #     shading="nearest",
    #     cmap=cmap,
    #     norm=norm,
    #     rasterized=True,
    # )
    #
    # lc = draw_region_boundaries(
    #     axs,
    #     eps_matrix.T,  # important: match CO_matrix.T
    #     rs,
    #     qs,
    #     colors={
    #         (0, 0.5): "magenta",
    #         (0.5, 1): "black",
    #         (0, 1): "lime",
    #     },
    # )
    #
    print(eps_plot.T)
    # plt.contour(R,Q,eps_plot.T, levels=[m_i/2], colors=[cmap(norm(m_i))])
    axs.pcolormesh(
    R,
    Q,
    wind_accretion_matrix.T,
    )
    # plt.pcolormesh(R,Q,eps_plot.T,cmap="viridis", vmin=np.min(masses),vmax=np.max(masses))

    # axs.contourf(
    #     R,
    #     Q,
    #     eps_matrix.T,
    #     colors="none",
    #     hatches=["||", None, None, None],
    #     corner_mask=False,
    #     rasterized=True,
    # )
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


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs, orientation="horizontal", location="top", aspect=50)
cbar.ax.set_xscale("linear")
cbar.set_label(r"Mass of TPAGB-star ($M_\odot$)")
# plt.colorbar(label=r"$\epsilon$")
# plt.xlabel(r"initial mass [$M_\odot$]")
# plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)


from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# custom_lines = [
#     Line2D([0], [0], color="magenta", lw=2),
#     Line2D([0], [0], color="black", lw=2),
#     Line2D([0], [0], color="lime", lw=2),
#     Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
# ]
#
# fig.legend(
#     custom_lines,
#     [
#         "Border between wind regime with\n possible and impossible final masses",
#         "Border between RLOF and\nwind only regime",
#         "Border between RLOF regime with\n possible and impossible final masses",
#         "Impossible region",
#     ],
#     loc="lower right",
#     ncols=1,
# )

# plt.savefig("/home/koen/LaTeX-setup/plots/w23-min-max-mass.pgf", format="pgf")
plt.show()
plt.close()


# %%


from matplotlib.colors import TwoSlopeNorm

R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    7,
    3,
    sharex="col",
    sharey=True,
    figsize=set_size(full, height=1.8),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):
    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)

    matrix = wind_accretion_matrix

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)


axs = axs.flatten()

for i, m_i in enumerate(masses):

    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    colors = axs[i].pcolormesh(
        R,
        Q,
        wind_accretion_matrix.T,
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

    try:
        axs[i].contourf(
             R,
             Q,
             eps_matrix.T,
             colors="none",
             hatches=["||", None, None, None],
             corner_mask=False,
             rasterized=True,
        )
    except:
        pass

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
cbar.set_label(r"$\Delta M_\textrm{wind}$ at end of binary evolution ($M_\odot$)")
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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-wind-accretion-region-all-masses.pgf", format="pgf"
)
plt.show()
plt.close()


# %%

single_star_dir = (
    f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M3.0"
)
try:
    with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
        Star = pickle.load(f)
except:

    Star = read_stellar_models(single_star_dir)[0]
    with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
        pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

q = 0.3
R = 1300 

a_init = inv_roche_lobe(R, q)
Star, Options, q_init, a_init, e_init, Bins = call_evolution(
    Star, q, a_init, simple_only=True
)

R_star = 10**Star.log_R
core_mass = Star.m_core
ages_star = Star.age
bin = Bins[0]

q_evolve = bin.m2 / bin.m1
RL = roche_lobe(1 / q_evolve) * bin.a

plt.plot(bin.age, RL)
plt.plot(Star.age, 10**Star.log_R)
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

masses = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7, 2.8,2.9, 3.0]

norm = plt.Normalize(np.min(masses), np.max(masses))
cmap = plt.cm.Spectral
# color = cmap(norm(x))

print(masses)


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

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

masses = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7, 2.8,2.9, 3.0]

min_mass = 1e99
max_mass = -1e99

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
    
    mass = Star.mass[ind]
    if mass < min_mass:
        min_mass = mass
    if mass > max_mass:
        max_mass = mass
    



norm = plt.Normalize(min_mass, max_mass)
cmap = plt.cm.Spectral
# color = cmap(norm(x))

print(masses)


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
        c=cmap(norm(Star.mass[ind])),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\textrm{TPAGB}$ at largest radius ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Envelope mass ($M_\odot$)")
plt.xscale("log")
plt.ylabel(r"$R / R_\textrm{max}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-bump-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

from matplotlib.colors import TwoSlopeNorm

masses = np.arange(1.0, 3.05, 0.1)
R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    7,
    3,
    sharex=False,
    sharey=False,
    figsize=set_size(full, height=1.8),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e99

for i, m_i in enumerate(masses):
    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)


    wind_masked = np.where(eps_matrix, wind_accretion_matrix, np.nan)
    matrix = wind_masked

    if np.nanmin(matrix) < minn:
        minn = np.nanmin(matrix)

    if np.nanmax(matrix) > maxx:
        maxx = np.nanmax(matrix)


axs = axs.flatten()

for i, m_i in enumerate(masses):

    try:
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
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)
    except FileNotFoundError:
        with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "rb") as f:
            eps_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "rb") as f:
            eps_min_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "rb") as f:
            eps_max_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "rb") as f:
            CO_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "rb") as f:
            m_DUP_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "rb") as f:
            min_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "rb") as f:
            max_mass_matrix = pickle.load(f)
        with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "rb") as f:
            wind_accretion_matrix = pickle.load(f)

    eps_plot = eps_matrix.copy()
    eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan

    RR, QQ = np.meshgrid(
        R,
        Q,
    )

    wind_masked = np.where(eps_matrix, wind_accretion_matrix, np.nan)

    colors = axs[i].pcolormesh(
        R,
        Q,
        wind_masked.T,
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

    try:
        axs[i].contourf(
             R,
             Q,
             eps_matrix.T,
             colors="none",
             hatches=["||", None, None, None],
             corner_mask=False,
             rasterized=True,
        )
    except:
        pass

    for max_count,arr in enumerate(wind_masked.T[::-1]):
        if not np.all(np.isnan(arr)):
            break


    for min_count,arr in enumerate(wind_masked.T):
        if not np.all(np.isnan(arr)):
            break

    for min_xcount,arr in enumerate(wind_masked):
        if np.any(arr > 0.005):
            break


    axs[i].set_ylim(qs[min_count]-0.025,qs[-(max_count+1)]+0.025)
    axs[i].set_xlim(rs[min_xcount]+0.025, axs[i].get_xlim()[1])





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


cbar = plt.colorbar(colors, ax=axs[1], orientation="horizontal", location="top")
cbar.ax.set_xscale("linear")
cbar.set_label(r"$\Delta M_\textrm{wind}$ accreted at end of binary evolution ($M_\odot$)")
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
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-wind-accretion-region-all-masses-zoom.pgf", format="pgf"
)
plt.show()
plt.close()


# %%

for i,arr in enumerate(wind_masked):
    if np.any(arr > 0.005):
        break


# %%
from pathlib import Path
import pickle

DATA_DIR = Path("scripts/w23")


# helpers

def load_mass_data(mass):
    """Load all W23 grid data for a given TP-AGB mass."""

    names = [
        "possible",
        "eps_max",
        "eps_min",
        "co",
        "dup",
        "min_mass",
        "max_mass",
        "wind_accretion"
    ]

    for fmt in [f"{mass}", f"{mass:.1f}"]:
        try:
            data = {}

            for name in names:
                path = DATA_DIR / f"const_mass_{name}_{fmt}_small.pkl"

                with open(path, "rb") as f:
                    data[name] = pickle.load(f)

            return data

        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"could not find data for mass {mass}")

def plot_panel(ax, co_matrix, eps_matrix, x, y, norm, cmap):
    """Plot one CO-ratio / epsilon-region panel."""

    ax.pcolormesh(
        x,
        y,
        co_matrix.T,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )

    draw_region_boundaries(
        ax,
        eps_matrix.T,
        x,
        y,
        colors={
            (0, 0.5): "magenta",
            (0.5, 1): "black",
            (0, 1): "lime",
        },
    )

    ax.contourf(
        x,
        y,
        eps_matrix.T,
        colors="none",
        hatches=["||", None, None, None],
        corner_mask=False,
        rasterized=True,
    )

def plot_by_mass(data,variable, masses, rs, qs, norm, cmap, figsize=full, figheight=1.8, ncols=3):
    ncols = ncols
    nrows = int(np.ceil(len(masses) / ncols))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=True,
        figsize=set_size(figsize, height=figheight),
        constrained_layout=True,
    )

    axs = np.atleast_1d(axs).flatten()

    for ax, mass in zip(axs, masses):
        d = data[mass]

        plot_panel(
            ax,
            d[variable],
            d["possible"],
            rs,
            qs,
            norm,
            cmap,
        )

        ax.set_title(f"$M_{{\\rm TPAGB}} = {mass:.1f}$")

    for ax in axs[len(masses):]:
        ax.axis("off")

    return fig, axs

def get_q_slice(data, variable, masses, q_index):
    """Return CO and epsilon grids for a fixed q."""

    co = np.array([
        data[mass][variable][:, q_index]
        for mass in masses
    ]).T

    eps = np.array([
        data[mass]["possible"][:, q_index]
        for mass in masses
    ]).T

    return co, eps

def plot_by_q(data, variable, masses, qs, rs, norm, cmap, figsize=full, figheight=1.8,ncols=3):
    ncols = ncols 
    nrows = int(np.ceil(len(qs) / ncols))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=True,
        figsize=set_size(figsize, height=figheight),
        constrained_layout=True,
    )

    axs = np.atleast_1d(axs).flatten()

    for ax, q_index, q in zip(axs, range(len(qs)), qs):

        co, eps = get_q_slice(data, variable, masses, q_index)

        plot_panel(
            ax,
            co,
            eps,
            rs,
            masses,
            norm,
            cmap,
        )

        ax.set_title(f"$q = {q:.2f}$")

    for ax in axs[len(qs):]:
        ax.axis("off")

    return fig, axs

def add_colorbar(fig, ax, norm, cmap, label):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
    )
    cbar.ax.set_xscale("linear")

    cbar.set_label(label)

    return cbar
# %%
        # "possible",
        # "eps_max",
        # "eps_min",
        # "co",
        # "dup",
        # "min_mass",
        # "max_mass",
        # "wind_accretion"

variable = "max_mass"

# loading data
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

print(minn, maxx)
norm = TwoSlopeNorm(vcenter=0.73, vmin=minn, vmax=maxx)
# norm = plt.Normalize(minn, maxx)
cmap = plt.cm.coolwarm_r


fig, axs = plot_by_mass(
    data,
    variable,
    masses,
    rs,
    qs,
    norm,
    cmap,
    ncols=3
)
custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1], norm, cmap, label=r"Mass with $\varepsilon = 0.5$ ($M_\odot$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-max-mass-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%

        # "possible",
        # "eps_max",
        # "eps_min",
        # "co",
        # "dup",
        # "min_mass",
        # "max_mass",
        # "wind_accretion"

variable = "min_mass"

# loading data
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

print(minn, maxx)
norm = TwoSlopeNorm(vcenter=1.5, vmin=minn, vmax=maxx)
# norm = plt.Normalize(minn, maxx)
cmap = plt.cm.coolwarm


fig, axs = plot_by_mass(
    data,
    variable,
    masses,
    rs,
    qs,
    norm,
    cmap,
    ncols=3
)
custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1], norm, cmap, label=r"Mass with $\varepsilon = 0.0$ ($M_\odot$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-min-mass-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%

        # "possible",
        # "eps_max",
        # "eps_min",
        # "co",
        # "dup",
        # "min_mass",
        # "max_mass",
        # "wind_accretion"

variable = "co"

# loading data
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

print(minn, maxx)
norm = TwoSlopeNorm(vcenter=1, vmin=minn, vmax=maxx)
# norm = plt.Normalize(minn, maxx)
cmap = plt.cm.coolwarm_r


fig, axs = plot_by_mass(
    data,
    variable,
    masses,
    rs,
    qs,
    norm,
    cmap,
    ncols=3
)
custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1], norm, cmap, label=r"CO-ratio at the end of \texttt{evolve.py}")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-CO-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%

        # "possible",
        # "eps_max",
        # "eps_min",
        # "co",
        # "dup",
        # "min_mass",
        # "max_mass",
        # "wind_accretion"

variable = "dup"

# loading data
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

print(minn, maxx)
# norm = TwoSlopeNorm(vcenter=1, vmin=minn, vmax=maxx)
norm = plt.Normalize(minn, maxx)
cmap = plt.cm.viridis


fig, axs = plot_by_mass(
    data,
    variable,
    masses,
    rs,
    qs,
    norm,
    cmap,
    ncols=3
)
custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1], norm, cmap, label=r"$M_\textrm{dup}$ the end of \texttt{evolve.py} ($M_\odot$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-m_DUP-region-all-masses.pgf", format="pgf")
plt.show()
plt.close()

# %%




variable = "min_mass"

# loading data
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

norm = TwoSlopeNorm(vcenter=1.5, vmin=minn, vmax=maxx)
# norm = plt.Normalize(minn, maxx)
cmap = plt.cm.coolwarm

fig, axs = plot_by_q(
    data,
    variable,
    masses,
    qs,
    rs,
    norm,
    cmap,
    ncols=4
)
fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1:3], norm, cmap, label=r"Mass with $\varepsilon = 0.0$ ($M_\odot$)")



plt.savefig("/home/koen/LaTeX-setup/plots/w23-mspace-m_min.pgf", format="pgf")

plt.show()
plt.close()

# %%
variable = "max_mass"

# loading data
masses = np.arange(1.0, 3.05, 0.1)
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

norm = TwoSlopeNorm(vcenter=0.73, vmin=minn, vmax=maxx)
cmap = plt.cm.coolwarm_r

fig, axs = plot_by_q(
    data,
    variable,
    masses,
    qs,
    rs,
    norm,
    cmap,
    ncols=4
)
fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1:3], norm, cmap, label=r"Mass with $\varepsilon = 0.5$ ($M_\odot$)")



plt.savefig("/home/koen/LaTeX-setup/plots/w23-mspace-m_max.pgf", format="pgf")



plt.show()


# %%
variable = "co"

# loading data
masses = np.arange(1.0, 3.05, 0.1)
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

norm = TwoSlopeNorm(vcenter=1, vmin=minn, vmax=maxx)
# norm = plt.Normalize(minn, maxx)
cmap = plt.cm.coolwarm_r

fig, axs = plot_by_q(
    data,
    variable,
    masses,
    qs,
    rs,
    norm,
    cmap,
    ncols=4
)

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.supxlabel(r"Initial Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel(r"TPAGB mass ($M_\odot$)", fontsize=10)

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1:3], norm, cmap, label=r"CO-ratio at end of \texttt{evolve.py}")


plt.savefig("/home/koen/LaTeX-setup/plots/w23-mspace-co.pgf", format="pgf")
plt.show()


# %%
variable = "dup"

# loading data
masses = np.arange(1.0, 3.05, 0.1)
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

# norm = TwoSlopeNorm(vcenter=1, vmin=minn, vmax=maxx)
norm = plt.Normalize(minn, maxx)
cmap = plt.cm.viridis

fig, axs = plot_by_q(
    data,
    variable,
    masses,
    qs,
    rs,
    norm,
    cmap,
    ncols=4
)

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.supxlabel(r"Initial Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel(r"TPAGB mass ($M_\odot$)", fontsize=10)

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1:3], norm, cmap, label=r"$M_\textrm{DUP}$ at end of \texttt{evolve.py}")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-mspace-dup.pgf", format="pgf")
plt.show()


# %%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def build_q_region_grid(data, masses, qs, which="min"):
    """
    Construct a mass x R grid containing the minimum or maximum
    q for which the final mass is possible.

    Returns
    -------
    q_grid : ndarray
        Shape (len(masses), len(rs)).
        Contains q where possible, NaN where impossible.
    """

    q_grid = np.full((len(masses), len(rs)), np.nan)

    for i, mass in enumerate(masses):
        possible = data[mass]["possible"]

        for j in range(len(rs)):
            possible_q = qs[possible[j] > 0]

            if len(possible_q) == 0:
                continue

            if which == "min":
                q_grid[i, j] = possible_q.min()
            elif which == "max":
                q_grid[i, j] = possible_q.max()
            else:
                raise ValueError("which must be 'min' or 'max'")

    return q_grid
# %%
masses = np.arange(1.0, 3.05, 0.1)

cmap = plt.cm.viridis
norm = Normalize(
    vmin=np.min(qs),
    vmax=np.max(qs),
)

fig, axs = plt.subplots(
    1,
    2,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=0.6),
    constrained_layout=True,
)

for ax, which, title in zip(
    axs,
    ["max", "min"],
    ["Maximum possible $q$", "Minimum possible $q$"],
):
    q_grid = build_q_region_grid(
        data,
        masses,
        qs,
        which=which,
    )

    pcm = ax.pcolormesh(
        rs,
        masses,
        q_grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )

    ax.set_title(title)

axs[0].set_ylabel("Initial mass ($M_\\odot$)")
axs[0].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")
axs[1].set_xlabel("Initial Roche lobe radius ($R_\\odot$)")

cbar = fig.colorbar(
    pcm,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
)
cbar.set_label("Initial mass ratio $q$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w23-min-max-q.pgf",
    format="pgf",
)

plt.show()
plt.close()

# %%
variable = "wind_accretion"

# loading data
masses = np.arange(1.0, 3.05, 0.1)
data = {
    m: load_mass_data(m)
    for m in masses
}

# color normalisation
minn = np.min([np.nanmin(d[variable]) for d in data.values()])
maxx = np.max([np.nanmax(d[variable]) for d in data.values()])

# norm = TwoSlopeNorm(vcenter=1, vmin=minn, vmax=maxx)
norm = plt.Normalize(minn, maxx)
cmap = plt.cm.viridis

fig, axs = plot_by_q(
    data,
    variable,
    masses,
    qs,
    rs,
    norm,
    cmap,
    ncols=4
)

custom_lines = [
    Line2D([0], [0], color="magenta", lw=2),
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="lime", lw=2),
    Rectangle((0, 0), 2, 2, fill=False, hatch="|||"),
]

fig.supxlabel(r"Initial Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel(r"TPAGB mass ($M_\odot$)", fontsize=10)

fig.legend(
    custom_lines,
    [
        "Border between wind\nregime with possible and\nimpossible final masses",
        "Border between RLOF\nand wind only regime",
        "Border between RLOF\nregime with possible and\nimpossible final masses",
        "Impossible region",
    ],
    loc="outside upper center",
    ncols=4,
)

fig.supylabel("Initial mass ratio ($q$)", fontsize=10)
fig.supxlabel("Initial Roche lobe radius ($R_\odot$)", fontsize=10)

add_colorbar(fig, axs[1:3], norm, cmap, label=r"$\Delta M_\textrm{wind}$ accreted at end of \texttt{evolve.py}")

plt.savefig("/home/koen/LaTeX-setup/plots/w23-mspace-wind-accretion.pgf", format="pgf")
plt.show()


# %%
def get_q_wind_accretion_slice(data, masses, qs, q_index):
    """
    Construct the wind-accretion grid for a fixed q.

    Returns
    -------
    wind : ndarray
        Shape (len(masses), len(rs)).
        Invalid regions are NaN.
    """

    wind = np.full((len(masses), len(rs)), np.nan)

    for i, mass in enumerate(masses):
        possible = data[mass]["possible"]
        accretion = data[mass]["wind_accretion"]

        wind[i] = np.where(
            possible[:, q_index],
            accretion[:, q_index],
            np.nan,
        )

    return wind
# %%
from matplotlib.colors import Normalize

masses = np.arange(1.0, 3.05, 0.1)

for mass in masses:
    mask = data[mass]["possible"].astype(bool)

# determine global colour scale using only valid regions
valid_accretion = np.concatenate([
    data[mass]["wind_accretion"][data[mass]["possible"].astype(bool)]
    for mass in masses
])

norm = Normalize(
    vmin=np.nanmin(valid_accretion),
    vmax=np.nanmax(valid_accretion),
)

cmap = plt.cm.viridis


fig, axs = plt.subplots(
    5,
    4,
    sharex=False,
    sharey=False,
    figsize=set_size(full, height=1.8),
    constrained_layout=True,
)

axs = axs.flatten()

for i, q_index in enumerate(range(len(qs))):

    wind = get_q_wind_accretion_slice(
        data,
        masses,
        qs,
        q_index,
    )

    pcm = axs[i].pcolormesh(
        rs,
        masses,
        wind,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )

    axs[i].set_title(f"$q = {qs[q_index]:.2f}$")
    valid = ~np.isnan(wind)
    
    if np.any(valid):
        valid_mass = np.any(valid, axis=1)
        valid_r = np.any(valid, axis=0)
    
        mass_indices = np.where(valid_mass)[0]
        r_indices = np.where(valid_r)[0]
    
        axs[i].set_ylim(
            masses[mass_indices[0]] - 0.05,
            masses[mass_indices[-1]] + 0.05,
        )
    
        axs[i].set_xlim(
            rs[r_indices[0]] - 0.025,
            rs[r_indices[-1]] + 0.025,
        )


for ax in axs[len(qs):]:
    ax.axis("off")


# colourbar
cbar = fig.colorbar(
    pcm,
    ax=axs[1:3],
    orientation="horizontal",
    location="top",
)

cbar.set_label(
    r"$\Delta M_\mathrm{wind}$ accreted at end of binary evolution "
    r"($M_\odot$)"
)


fig.supxlabel("Initial Roche lobe radius ($R_\odot$)",fontsize=10)
fig.supylabel(r"$M_\mathrm{TPAGB}$ ($M_\odot$)",fontsize=10)


plt.savefig(
    "/home/koen/LaTeX-setup/plots/"
    "w23-wind-accretion-region-all-q.pgf",
    format="pgf",
)

plt.show()
plt.close()

# %%

