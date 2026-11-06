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

# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/grid-M3.0-2026-10-09/")
# %%

for m in grid.models:
    if np.isclose(m.params["R"], 1500):
        break

profiles = m.profiles
# %%

print(profiles)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0, 1.9)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


for p in profiles:
    print(p.mass[0] - p.he_core_mass)
    plt.plot(p.R, p.mass, c=cmap(norm(p.mass[0] - p.he_core_mass)))
    print()

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Envelope mass ($M_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$r$ ($R_\\odot$)")
plt.ylabel("$m$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-r-profile.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

rs = []
ms = []

ubs = []
ofs = [1.01, 1.1, 1.25, 1.5, 1.75, 2, 3, 10]
for p in profiles:
    rs.append(p.R[0])
    ms.append(p.mass[0] - p.he_core_mass)
    us = []
    for of in ofs:
        ind = np.argwhere(p.R < p.R[0] / (of))[0][0]

        # us.append(p.mass[0] - p.mass[ind])
        us.append((p.mass[0] - p.mass[ind]) / (p.mass[0] - p.he_core_mass))

    ubs.append(us)

ubs = np.array(ubs)


for i in range(np.shape(ubs)[1]):
    plt.plot(ms, ubs[:, i])
    plt.text(ms[0] + 0.1, ubs[0, i], ofs[i], va="center", ha="left", fontsize=8)

plt.text(
    ms[0] + 0.15,
    1.2,
    "$R_\\textrm{star} / R_\\textrm{RL}$",
    va="bottom",
    ha="center",
    fontsize=8,
)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("$M_\\textrm{unb} / M_\\textrm{env}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-unbound-frac-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.85), constrained_layout=True
)

rs = []
ms = []

ubs = []
ofs = [1.01, 1.1, 1.25, 1.5, 1.75, 2, 3, 10]
for p in profiles:
    rs.append(p.R[0])
    ms.append(p.mass[0] - p.he_core_mass)
    us = []
    for of in ofs:
        ind = np.argwhere(p.R < p.R[0] / (of))[0][0]

        us.append(p.mass[0] - p.mass[ind])
        # us.append((p.mass[0] - p.mass[ind]) / (p.mass[0] - p.he_core_mass))

    ubs.append(us)

ubs = np.array(ubs)


for i in range(np.shape(ubs)[1]):
    plt.plot(ms, ubs[:, i])
    plt.text(ms[0] + 0.1, ubs[0, i], ofs[i], va="center", ha="left", fontsize=8)

plt.text(
    ms[0] + 0.15,
    ms[0] + 0.5,
    "$R_\\textrm{star} / R_\\textrm{RL}$",
    va="bottom",
    ha="center",
    fontsize=8,
)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("$M_\\textrm{unb}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-unbound-1.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.85), constrained_layout=True
)

norm = plt.Normalize(0, 1.9)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

rs = []
ms = []

ubs = []
for p in profiles:
    rs.append(p.R[0])
    ms.append(p.mass[0] - p.he_core_mass)
    us = []
    ofs = np.logspace(0.001, 1, 100)
    print(ofs)
    for of in ofs:
        ind = np.argwhere(p.R < p.R[0] / (of))[0][0]

        us.append(p.mass[0] - p.mass[ind])

    plt.plot(ofs, us, c=cmap(norm(p.mass[0] - p.he_core_mass)))
    # ubs.append(us)

ubs = np.array(ubs)

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Envelope mass ($M_\odot$)")

# for i in range(np.shape(ubs)[1]):
#     plt.plot(ms, ubs[:,i])
#     plt.text(ms[0]+0.1, ubs[0,i], ofs[i], va="center", ha="left", fontsize=8)

# plt.text(ms[0]+0.15, ms[0] +0.5, "$R_\\textrm{star} / R_\\textrm{RL}$", va="bottom", ha="center", fontsize=8)

plt.xscale("log")
# plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.ylabel("$M_\\textrm{unb}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-unbound-switch.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.85), constrained_layout=True
)

norm = plt.Normalize(0, 1.9)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

rs = []
ms = []

ubs = []
for p in profiles:
    rs.append(p.R[0])
    ms.append(p.mass[0] - p.he_core_mass)
    us = []
    ofs = np.linspace(1.001, 3.5, 100)
    print(ofs)
    for of in ofs:
        ind = np.argwhere(p.R < p.R[0] / (of))[0][0]

        us.append((p.mass[0] - p.mass[ind]) / (p.mass[0]))

    plt.plot(ofs, us, c=cmap(norm(p.mass[0] - p.he_core_mass)))
    # ubs.append(us)

ubs = np.array(ubs)

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Envelope mass ($M_\odot$)")

# for i in range(np.shape(ubs)[1]):
#     plt.plot(ms, ubs[:,i])
#     plt.text(ms[0]+0.1, ubs[0,i], ofs[i], va="center", ha="left", fontsize=8)

# plt.text(ms[0]+0.15, ms[0] +0.5, "$R_\\textrm{star} / R_\\textrm{RL}$", va="bottom", ha="center", fontsize=8)

# plt.xscale("log")
# plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.ylabel("$M_\\textrm{unb} / M_\\textrm{star}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-unbound-switch-2.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

rs = []
ms = []

ubs = []
ofs = [1.01, 1.1, 1.25, 1.5, 1.75, 2, 3, 10]
for p in profiles:
    rs.append(p.R[0])
    ms.append(p.mass[0] - p.he_core_mass)
    us = []
    for of in ofs:
        ind = np.argwhere(p.R < p.R[0] / (of))[0][0]

        # us.append(p.mass[0] - p.mass[ind])
        us.append((p.mass[0] - p.mass[ind]) / (p.mass[0]))

    ubs.append(us)

ubs = np.array(ubs)


for i in range(np.shape(ubs)[1]):
    plt.plot(ms, ubs[:, i])
    plt.text(ms[0] + 0.1, ubs[0, i], ofs[i], va="center", ha="left", fontsize=8)

plt.text(
    ms[0] + 0.15,
    1.2,
    "$R_\\textrm{star} / R_\\textrm{RL}$",
    va="bottom",
    ha="center",
    fontsize=8,
)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("$M_\\textrm{unb} / M_\\textrm{star}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-m-unbound-frac-2.pgf", format="pgf")
plt.show()
plt.close()
