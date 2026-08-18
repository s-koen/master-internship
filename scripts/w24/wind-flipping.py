import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star

plt.cplot = cplot

# %%


fig, axs = plt.subplots(
    7, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


# ms = np.arange(1.0,3.01, 0.1)
ms = np.arange(1.0, 3.01, 0.1)[:7]

norm = plt.Normalize(ms.min(), ms.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


for ax, m in zip(axs, ms[:7]):
    star = get_star(m=m)
    ax.plot(
        star.age[star.ntpagb :] - star.age[star.ntpagb],
        np.log10(-star.dM_dt[star.ntpagb - 1 :]),
        c=cmap(norm(m)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


ms = np.arange(1.0, 3.01, 0.1)

norm = plt.Normalize(ms.min(), ms.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m in ms:
    star = get_star(m=m)
    plt.plot(
        star.m_env[star.ntpagb :],
        np.log10(-star.dM_dt[star.ntpagb - 1 :] / star.mass[star.ntpagb :]),
        c=cmap(norm(m)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()


# %%


def v_orbit(Mtot, a):
    """
    Relative orbital (circular) velocity [km/s].

        Args:
        Mtot = binary mass [Msun]
        a = semi-major axis [Rsun]
    """
    vorb = np.sqrt(const.G * (Mtot * const.Msun) / (a * const.Rsun))
    vorb_kms = vorb * 1e-5
    return vorb_kms


# %%
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
import pickle

from pathlib import Path

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


from tqdm import tqdm

RRs = [
    460,
    460,
    490,
    540,
    730,
    710,
    860,
    780,
    960,
    900,
    960,
    1000,
    1120,
    1140,
    1160,
    1180,
    1180,
]

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

masses = np.arange(1.1, 2.65, 0.1)


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
