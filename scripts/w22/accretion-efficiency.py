import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from tqdm import tqdm
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

import pickle

# %%


def compute_epsilon(barium_mass, tpagb_mass, core_mass, q):
    """
    computes the mass transfer efficiency assuming no wind mass transfer.
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = barium_mass - tpagb_mass * q

    return delta_accreted / delta_donated


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

proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M1.5"

# WARNING: SETTINGS FOR THE GRID

qs = np.linspace(0.4, 1, 7)

# %%

Star = read_stellar_models(single_star_dir)[0]


# %%
for m_i in np.arange(1.0, 2.3, 0.1):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%

R = 450
barium_mass = 1.29

masses = np.arange(1.0, 2.3, 0.1)
qs = np.arange(0.4, 1.025, 0.05)

eps_matrix = np.full((len(masses), len(qs)), np.nan)
m_DUP_matrix = np.full((len(masses), len(qs)), np.nan)
CO_matrix = np.full((len(masses), len(qs)), np.nan)

for i, m_i in enumerate(masses):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

    for j, q in enumerate(qs):
        print(m_i, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
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

        # interpolate stellar radius onto binary ages
        core_mass_interp = np.interp(bin.age, ages_star, core_mass)
        TP_count = int(np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1])
        CO_ratio = np.interp(
            bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
        )[-1]

        print(np.sum(Star.m_DUP[:TP_count]))
        print(CO_ratio)

        eps = compute_epsilon(barium_mass, tpagb_mass, core_mass_interp[-1], q)
        eps_matrix[i, j] = eps
        m_DUP_matrix[i, j] = np.sum(Star.m_DUP[:TP_count])
        CO_matrix[i, j] = CO_ratio

with open(f"scripts/w22/eps_{int(R)}.pkl", "wb") as f:
    pickle.dump(eps_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/co_{int(R)}.pkl", "wb") as f:
    pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/dup_{int(R)}.pkl", "wb") as f:
    pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
R = 450
masses = np.arange(1.0, 2.3, 0.1)
qs = np.arange(0.4, 1.025, 0.05)


with open(f"scripts/w22/eps_{int(R)}.pkl", "rb") as f:
    eps_matrix = pickle.load(f)
with open(f"scripts/w22/co_{int(R)}.pkl", "rb") as f:
    CO_matrix = pickle.load(f)
with open(f"scripts/w22/dup_{int(R)}.pkl", "rb") as f:
    m_DUP_matrix = pickle.load(f)
# %%

eps_plot = eps_matrix.copy()
eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan


M, Q = np.meshgrid(masses, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(
    M, Q, eps_plot.T, shading="nearest", vmin=0, vmax=1, cmap="viridis", rasterized=True
)
plt.xlabel(r"initial mass [$M_\odot$]")
plt.ylabel(r"$q$")
plt.colorbar(label=r"$\epsilon$")


plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-350.pgf", format="pgf")
plt.show()
plt.close()
# %%


M, Q = np.meshgrid(masses, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(M, Q, CO_matrix.T, shading="nearest", cmap="viridis", rasterized=True)
plt.xlabel(r"initial mass ($M_\odot$)")
plt.ylabel(r"$q$")
plt.colorbar(label=r"C/O-ratio")

plt.savefig("/home/koen/LaTeX-setup/plots/w22-co-350.pgf", format="pgf")
plt.show()
plt.close()

# %%
M, Q = np.meshgrid(masses, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(M, Q, m_DUP_matrix.T, shading="nearest", cmap="viridis", rasterized=True)
plt.xlabel(r"initial mass [$M_\odot$]")
plt.ylabel(r"$q$")
plt.colorbar(label=r"$M_\textrm{DUP}$ ($M_\odot$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w22-grid-dup-350.pgf", format="pgf")
plt.show()
plt.close()
# %%


plt.plot(m_DUP)
plt.show()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(2.2, 2.3, 0.5)
for i, m_i in enumerate(masses):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

    # plt.plot(range(len(Star.m_DUP))[1:], np.cumsum(Star.m_DUP[1:]))

    plt.plot(
        np.array(range(len(Star.m_DUP))) + 1,
        np.cumsum(Star.m_DUP),
        c="k",
        linewidth=0.8,
    )
    plt.scatter(
        np.array(range(len(Star.m_DUP))) + 1,
        np.cumsum(Star.m_DUP),
        c="w",
        marker=".",
        s=200,
        zorder=10,
    )
    plt.scatter(
        np.array(range(len(Star.m_DUP))) + 1,
        np.cumsum(Star.m_DUP),
        c="k",
        marker=".",
        zorder=11,
    )


axs.spines[["right", "top"]].set_visible(False)

plt.xlim(0, 25)
plt.ylim(-0.005, 0.09)
plt.xticks([0, 5, 10, 15, 20, 25])
plt.xlabel("TP-count")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-M_DUP.pgf", format="pgf")
plt.show()
plt.close()


# %%

plt.plot(Star.age[Star.ntpagb :], Star.TP_count / Star.TP_count[-1])
plt.plot(Star.age, Star.m_core / Star.m_core[-1])
plt.plot(Star.age, Star.lambda_DUP)

plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
masses = np.arange(1.0, 2.3, 0.1)
m_dup = []
for i, m_i in enumerate(masses):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

    m_dup.append(np.sum(Star.m_DUP[1:]))

# plt.ylim(0,0.1)
plt.plot(masses, m_dup, c="k", linewidth=0.8)
plt.scatter(
    masses,
    m_dup,
    c="w",
    marker=".",
    s=200,
    zorder=10,
)
plt.scatter(
    masses,
    m_dup,
    c="k",
    marker=".",
    zorder=11,
)


axs.spines[["right", "top"]].set_visible(False)

plt.ylim(-0.005, 0.09)
plt.xlabel(r"$M_\textrm{star}$ ($M_\odot$)")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-M_DUP-2.pgf", format="pgf")
plt.show()
plt.close()


# %%


R = 450
barium_mass = 1.29

masses = np.arange(1.0, 2.3, 1.1)
qs = np.arange(0.4, 1.010, 0.02)

rs = np.arange(300, 702.5, 5)
eps_matrix = np.full((len(rs), len(qs)), np.nan)
m_DUP_matrix = np.full((len(rs), len(qs)), np.nan)
CO_matrix = np.full((len(rs), len(qs)), np.nan)
m_i = 1.5

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
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
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

        # interpolate stellar radius onto binary ages
        core_mass_interp = np.interp(bin.age, ages_star, core_mass)
        TP_count = int(np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1])
        CO_ratio = np.interp(
            bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
        )[-1]

        eps = compute_epsilon(barium_mass, tpagb_mass, core_mass_interp[-1], q_begin)
        eps_matrix[i, j] = eps
        m_DUP_matrix[i, j] = np.sum(Star.m_DUP[:TP_count])
        CO_matrix[i, j] = CO_ratio

with open(f"scripts/w22/const_mass_eps_{m_i}.pkl", "wb") as f:
    pickle.dump(eps_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/const_mass_co_{m_i}.pkl", "wb") as f:
    pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/const_mass_dup_{m_i}.pkl", "wb") as f:
    pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
m_i = 1.5
R = 1000
qs = np.arange(0.4, 1.010, 0.02)
rs = np.arange(300, 702.5, 5)

with open(f"scripts/w22/const_mass_eps_{m_i}.pkl", "rb") as f:
    eps_matrix = pickle.load(f)
with open(f"scripts/w22/const_mass_co_{m_i}.pkl", "rb") as f:
    CO_matrix = pickle.load(f)
with open(f"scripts/w22/const_mass_dup_{m_i}.pkl", "rb") as f:
    m_DUP_matrix = pickle.load(f)
# %%

eps_plot = eps_matrix.copy()
eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan


R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

plt.pcolormesh(
    R,
    Q,
    eps_plot.T,
    shading="nearest",
    vmin=0,
    vmax=1,
    cmap="viridis",
    rasterized=True,
)
plt.colorbar(label=r"$\epsilon$")
plt.scatter(
    [450, 462.5, 475, 487.5, 500, 512.5, 525, 537.5, 550, 562.5],
    [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
    ],
    c="w",
    s=100,
    marker=".",
)
plt.scatter(
    [450, 462.5, 475, 487.5, 500, 512.5, 525, 537.5, 550, 562.5],
    [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
    ],
    c=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    cmap="tab10",
    marker=".",
    vmin=1,
    vmax=10,
)

plt.xlabel(r"initial mass [$M_\odot$]")
plt.ylabel(r"$q$")
plt.ylim(0.39, 0.9)

plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-1000.pgf", format="pgf")
plt.show()
plt.close()
# %%

plt.plot(Star.age[Star.ntpagb :], 100 * Star.mass[Star.ntpagb :])
plt.plot(Star.age[Star.ntpagb :], 10 ** Star.log_R[Star.ntpagb :])
plt.show()
# %%


R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(
    R,
    Q,
    m_DUP_matrix.T,
    shading="nearest",
    # vmin=0,
    # vmax=1,
    cmap="viridis",
    rasterized=True,
)
plt.xlabel(r"initial mass [$M_\odot$]")
plt.ylabel(r"$q$")
plt.colorbar(label=r"$\epsilon$")


plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-1000.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)
q = 0.6
for R in [450, 462.5, 475, 487.5, 500, 512.5, 525, 537.5, 550, 562.5]:
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

    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
        Star, q, a_init, simple_only=True
    )
    R_star = 10**Star.log_R
    core_mass = Star.m_core
    ages_star = Star.age
    bin = Bins[0]

    q_evolve = bin.m2 / bin.m1
    RL = roche_lobe(1 / q_evolve) * bin.a
    plt.plot(bin.age, RL, linewidth=0.8)
plt.xlim(2271780961.673242, 2272072657.9535985)
plt.ylim(250, 600)
plt.plot(Star.age, R_star, c=f"C{9}", linewidth=3, zorder=-1)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-mass-loss.pgf", format="pgf")
plt.show()
plt.close()

# %%


R = 450
barium_mass = 1.29

masses = np.arange(1.0, 2.3, 1.1)

qs = np.linspace(0.4, 0.8, 5)
rs = np.linspace(400, 600, 5)
print(rs)
eps_matrix = np.full((len(rs), len(qs)), np.nan)
m_DUP_matrix = np.full((len(rs), len(qs)), np.nan)
CO_matrix = np.full((len(rs), len(qs)), np.nan)
m_i = 1.5

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
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
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

        # interpolate stellar radius onto binary ages
        core_mass_interp = np.interp(bin.age, ages_star, core_mass)
        TP_count = int(np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1])
        CO_ratio = np.interp(
            bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
        )[-1]

        eps = compute_epsilon(barium_mass, tpagb_mass, core_mass_interp[-1], q_begin)
        eps_matrix[i, j] = eps
        m_DUP_matrix[i, j] = np.sum(Star.m_DUP[:TP_count])
        CO_matrix[i, j] = CO_ratio

with open(f"scripts/w22/const_mass_eps_{m_i}_small.pkl", "wb") as f:
    pickle.dump(eps_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/const_mass_co_{m_i}_small.pkl", "wb") as f:
    pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"scripts/w22/const_mass_dup_{m_i}_small.pkl", "wb") as f:
    pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
m_i = 1.5
R = 1000
qs = np.linspace(0.4, 0.8, 5)
rs = np.linspace(400, 600, 5)


with open(f"scripts/w22/const_mass_eps_{m_i}_small.pkl", "rb") as f:
    eps_matrix = pickle.load(f)
with open(f"scripts/w22/const_mass_co_{m_i}_small.pkl", "rb") as f:
    CO_matrix = pickle.load(f)
with open(f"scripts/w22/const_mass_dup_{m_i}_small.pkl", "rb") as f:
    m_DUP_matrix = pickle.load(f)
# %%

eps_plot = eps_matrix.copy()
eps_plot[(eps_plot < 0) | (eps_plot > 1)] = np.nan


R, Q = np.meshgrid(rs, qs)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(
    R,
    Q,
    eps_plot.T,
    shading="nearest",
    vmin=0,
    vmax=1,
    cmap="viridis",
    rasterized=True,
)
plt.colorbar(label=r"$\epsilon$")
plt.xlabel(r"initial mass [$M_\odot$]")
plt.ylabel(r"$q$")
# plt.ylim(0.39, 0.9)

plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-1000-small.pgf", format="pgf")
plt.show()
plt.close()
# %%
