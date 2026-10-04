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

grid = MesaGrid(f"{MASTER}/grid-zoom-R-2026-10-03")
df = AbundanceTables()

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

# %%

for model in grid_sorted:
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
    else:
        c = "C2"
    plt.plot(model.age, model.R, c=c)
    plt.plot(model.age, model.rl_1, c=c)

star = get_star(m=2.2)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%

for model in grid_sorted:
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
    else:
        c = "C2"
    plt.plot(model.envelope_mass, model.min_S, c=c)
    plt.plot(model.envelope_mass, model.min_S, c=c)

plt.show()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column, height=1), constrained_layout=True
)


for model in grid_sorted:
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
    else:
        c = "C2"
    axs[0].plot(model.age, model.rl_1, c=c, alpha=0.2)
    axs[0].plot(model.age, model.R, c=c, alpha=0.2)

    axs[1].plot(model.envelope_mass, model.sad_s_m_max, c=c, alpha=0.2)
    axs[1].plot(model.envelope_mass, model.sad_s_m_max, c=c, alpha=0.2)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
axs[0].set_ylabel("Radius ($R_\\odot$)")
axs[0].set_xlim(920066943.6601021, 920070626.5807078)
axs[0].set_ylim(67.76559604475392, 304.43567240955775)
axs[1].set_ylabel("Minimum entropy in envelope")
axs[0].set_xlabel("Age (yr)")
axs[1].set_xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-entropy-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

norm = plt.Normalize(230, 240)
cmap = plt.cm.viridis

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    print(model.params["R"])
    plt.plot(
        model.envelope_mass,
        (model.sad_m_max - model.sad_m_min) / model.envelope_mass,
        c=cmap(norm(model.params["R"])),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$R_\textrm{RL,i}$ ($R_\odot$)")
plt.yscale("log")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$m_\textrm{env}$ ($M_\odot$)")
plt.ylabel("$M_\\textrm{sad} / M_\\textrm{env}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-sad-m-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

norm = plt.Normalize(230, 240)
cmap = plt.cm.viridis

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    print(model.params["R"])
    plt.plot(model.envelope_mass, model.lg_mstar_dot_1, c=cmap(norm(model.params["R"])))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$R_\textrm{RL,i}$ ($R_\odot$)")
# plt.yscale("log")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$m_\textrm{env}$ ($M_\odot$)")
plt.ylabel("$M_\\textrm{sad} / M_\\textrm{env}$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-sad-m-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

model.bulk_names
# %%

norm = plt.Normalize(230, 240)
cmap = plt.cm.viridis

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    print(model.params["R"])
    plt.plot(model.R, model.Teff, c=cmap(norm(model.params["R"])))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$R_\textrm{RL,i}$ ($R_\odot$)")
# plt.yscale("log")


plt.xlim(62.875313458654446, 292.786850782851)
plt.ylim(3273.442609705736, 4945.924447728545)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$R_\textrm{star}$ ($R_\odot$)")
plt.ylabel("$T_\\textrm{eff}$ (K)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-sad-m-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

norm = plt.Normalize(230, 240)
cmap = plt.cm.viridis

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    print(model.params["R"])
    plt.plot(
        model.R,
        model.thermal_time_min_S / 3600 / 24 / 365,
        c=cmap(norm(model.params["R"])),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$R_\textrm{RL,i}$ ($R_\odot$)")
# plt.yscale("log")


plt.xlim(62.875313458654446, 292.786850782851)
# plt.ylim(3273.442609705736, 4945.924447728545)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$R_\textrm{star}$ ($R_\odot$)")
plt.ylabel("$T_\\textrm{eff}$ (K)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-sad-m-2.pgf", format="pgf")
plt.show()
plt.close()
# %%
model.bulk_names

# %%

norm = plt.Normalize(230, 240)
cmap = plt.cm.viridis

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    print(model.params["R"])
    plt.plot(model.R, model.surface_S - model.min_S, c=cmap(norm(model.params["R"])))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$R_\textrm{RL,i}$ ($R_\odot$)")
# plt.yscale("log")


plt.xlim(62.875313458654446, 292.786850782851)
# plt.ylim(3273.442609705736, 4945.924447728545)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$R_\textrm{star}$ ($R_\odot$)")
plt.ylabel("$T_\\textrm{eff}$ (K)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-sad-m-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

norm = plt.Normalize(np.log10(0.01), np.log10(1.6))
cmap = plt.cm.viridis
# color = cmap(norm(x))

model = grid_sorted[15]

for p in model.profiles:
    plt.plot(
        p.logT,
        p.gradT - p.grada,
        c=cmap(norm(np.log10(p.mass[0] - p.he_core_mass))),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")
plt.yscale("log")
plt.ylim(1e-3, 1e1)

plt.xlim(3.4, 5.4)
plt.show()

# %%
