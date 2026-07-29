import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"
proj_dir = "/home/koen/master-internship"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

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

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


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

    plt.plot(Star.age, Star.log_R, color=cmap(norm(m_i)))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Time (yr)")
plt.ylabel("$\log(R / R_\odot)$ ")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-naive-time-radius.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


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

    plt.plot(Star.log_Teff, Star.log_L, color=cmap(norm(m_i)))


plt.gca().invert_xaxis()
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.ylabel(r"$\log(L / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-naive-HR.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


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

    plt.plot(Star.mass, Star.lambda_DUP)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Time (yr)")
plt.ylabel("$\log(R / R_\odot)$ ")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-time-radius.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


max_TP = []
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

    max_TP.append(Star.TP_count[-1])

plt.scatter(masses, max_TP, color="w", s=200, zorder=10, marker=".")
plt.scatter(masses, max_TP, color="k", zorder=11, marker=".")
plt.plot(masses, max_TP, c="k", linewidth=0.8)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Initial mass ($M_\odot$)")
plt.ylabel("Number of thermal pulses ")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-mass-tp.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


max_TP = []
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

    max_TP.append(Star.envelope_c12[-1] / Star.envelope_o16[-1] * 16 / 12)

plt.scatter(masses, max_TP, color="w", s=200, zorder=10, marker=".")
plt.scatter(masses, max_TP, color="k", zorder=11, marker=".")
plt.plot(masses, max_TP, c="k", linewidth=0.8)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Initial mass ($M_\odot$)")
plt.ylabel("Final envelope C/O number ratio ")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-mass-co.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


m_DUP = []
for i, m_i in enumerate([1.5, 1.8]):

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

    plt.scatter(
        range(3, len(Star.m_DUP) + 3),
        np.cumsum(Star.m_DUP),
        color="w",
        s=200,
        zorder=10,
        marker=".",
    )
    plt.scatter(
        range(3, len(Star.m_DUP) + 3),
        np.cumsum(Star.m_DUP),
        color="k",
        zorder=11,
        marker=".",
    )

    plt.plot(range(3, len(Star.m_DUP) + 3), np.cumsum(Star.m_DUP), c="k", linewidth=0.8)

# plt.scatter(masses, max_TP, color="w", s=200, zorder=10, marker=".")
# plt.scatter(masses, max_TP, color="k", zorder=11, marker=".")
# plt.plot(masses, max_TP, c="k", linewidth=0.8)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-M_DUP.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


m_DUP = []
for i, m_i in enumerate([1.8]):

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

    plt.scatter(
        range(3, len(Star.m_DUP) + 3),
        np.cumsum(Star.m_DUP) / np.cumsum(Star.m_DUP)[-1],
        color="w",
        s=200,
        zorder=10,
        marker=".",
    )
    plt.scatter(
        range(3, len(Star.m_DUP) + 3),
        np.cumsum(Star.m_DUP) / np.cumsum(Star.m_DUP)[-1],
        color="k",
        zorder=11,
        marker=".",
    )

    plt.plot(
        range(3, len(Star.m_DUP) + 3),
        np.cumsum(Star.m_DUP) / np.cumsum(Star.m_DUP)[-1],
        c="k",
        linewidth=0.8,
    )

    plt.plot(
        Star.TP_count,
        Star.envelope_c12[-len(Star.TP_count) :]
        / Star.envelope_o16[-len(Star.TP_count) :]
        * 16
        / 12,
        c="k",
        linewidth=0.8,
    )

# plt.scatter(masses, max_TP, color="w", s=200, zorder=10, marker=".")
# plt.scatter(masses, max_TP, color="k", zorder=11, marker=".")
# plt.plot(masses, max_TP, c="k", linewidth=0.8)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-M_DUP.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


max_TP = []
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

    plt.plot(
        Star.age[Star.ntpagb :] - Star.age[Star.ntpagb],
        Star.m_env[Star.ntpagb :],
        color=cmap(norm(m_i)),
    )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.yscale("log")
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Time since TPAGB (yr)")
plt.ylabel("Envelope mass ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-env-mass-time.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = np.arange(1.0, 2.3, 0.1)

n = len(masses)
norm = plt.Normalize(masses.min(), masses.max())
cmap = plt.cm.viridis
# color = cmap(norm(x))


max_TP = []
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

    plt.plot(
        Star.m_env[Star.ntpagb :],
        10 ** Star.log_R[Star.ntpagb :],
        color=cmap(norm(m_i)),
    )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

axs.invert_xaxis()
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\mathrm{initial}$ ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-env-mass-radius.pgf", format="pgf")
plt.show()
plt.close()


# %%
