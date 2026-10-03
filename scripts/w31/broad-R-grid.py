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
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

# %%

grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
df = AbundanceTables()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

for model in grid_sorted[36:][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c3 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Blues",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
for model in grid_sorted[:36][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c2 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Greens",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )

plt.colorbar(c1, label="log$(M_\\textrm{env} / M_\\odot)$", pad=0.01, aspect=50)
cbar = plt.colorbar(c3, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])
cbar = plt.colorbar(c2, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star}$ ($R_\\odot$)")
plt.ylabel("$\\dot{M} / \\dot{M}_\\textrm{crit}$")
# plt.savefig(
#     "/home/koen/LaTeX-setup/plots/w30-failing-models-6.pgf", format="pgf", dpi=600
# )
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    per.append(m.params["R"])
    s.append(m.star_2_mass[-1] - m.params["m"] * m.params["q"])

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25 * m.envelope_mass[0], c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("$\\Delta M_\\textrm{acc}$ ($M_\\odot$)")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-macc-RL.pgf", format="pgf")
plt.show()
plt.close()
# %%

for model in grid_sorted[5:6]:
    profiles = model.profiles
    for p in profiles:
        plt.plot(
            p.mass[0] - p.mass,
            10**p.log_thermal_time_to_surface / 3600 / 24 / 365,
            c="C0",
        )


for model in grid_sorted[40:41]:
    profiles = model.profiles
    for p in profiles:
        plt.plot(
            p.mass[0] - p.mass,
            10**p.log_thermal_time_to_surface / 3600 / 24 / 365,
            c="C1",
        )


plt.show()


# %%

norm = plt.Normalize(150, 404)
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:26][::-1]:
    print(model.params["R"])
    ents = []
    ms = []
    ts = []

    profiles = model.profiles
    for p in profiles:
        args = np.argmax(p.entropy)
        mass = p.mass[:args]
        entropy = p.entropy[:args]
        argmin = np.argmin(entropy)
        ents.append(entropy[argmin])
        ms.append(mass[argmin] - p.he_core_mass)
        # ents.append(p.entropy[0])
        # ms.append(p.mass[0] - p.he_core_mass)
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(ms, ents, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(model.params["R"]))
    plt.plot(ms, ents, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("Minimum entropy in envelope")

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w31-min-entropy.pgf", format="pgf")
plt.show()
plt.close()

# %%
