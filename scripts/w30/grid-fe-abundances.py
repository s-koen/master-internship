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

sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid3 = MesaGrid(f"{MASTER}/grid-masses-3-2026-08-24")
grid4 = MesaGrid(f"{MASTER}/grid-masses-4-2026-08-25")
grid.merge(grid2)
grid.merge(grid3, overwrite=True)
grid.merge(grid4, overwrite=True)

# %%

q = 0.6
Star = get_star(m=2.2)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


Rs = np.logspace(np.log10(125), np.log10(1250), 50)
Rslog = np.log10(Rs)

norm = plt.Normalize(np.min(Rslog), np.max(Rslog))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for R in Rs:
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
        Star, q, a_init, simple_only=True
    )

    R_star = 10**Star.log_R
    ages_star = Star.age
    bin = Bins[0]

    q_evolve = bin.m2 / bin.m1
    RL = roche_lobe(1 / q_evolve) * bin.a
    plt.plot(bin.age, RL, color=cmap(norm(np.log10(R))))

plt.xlim(918387515.8370682, 922499995.2403557)
plt.ylim(53.50374353284981, 1303.2451535583723)
plt.yscale("log")
plt.plot(Star.age, 10**Star.log_R, c="C9", zorder=-1)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$\textrm{log}(R_\textrm{RL} / R_\odot)$")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel(r"$R$ ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-broad-R-grid-setup.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(Star.age, 10**Star.log_R)
plt.show()
# %%

df = AbundanceTables()


def get_ba(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df)
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_ba, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)

# %%

fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(
    c,
    ax=axs,
    label="$[s/\\textrm{Fe}]$",
)

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-grid-s-fe.pgf", format="pgf")
plt.show()
plt.close()


# %%
def get_ls(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df)
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 3


def get_hs(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df)
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


def get_s(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df)
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


s_fes = []
ls_fes = []
hs_fes = []
colors = []
markers = []

for m in grid.models:
    mass = m.params["m"]
    match mass:
        case 1.8:
            marker = "*"
        case 2.2:
            marker = "s"
        case 2.6:
            marker = "^"
        case 3.0:
            marker = "o"
    s_fe = get_s(m)
    s_fes.append(s_fe)
    ls_fe = get_ls(m)
    ls_fes.append(ls_fe)
    hs_fe = get_hs(m)
    hs_fes.append(hs_fe)
    markers.append(marker)
    colors.append(m.params["R"])

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
labels = [
    "$M_\\textrm{TPAGB} = 1.8\\;M_\\odot$",
    "$M_\\textrm{TPAGB} = 2.2\\;M_\\odot$",
    "$M_\\textrm{TPAGB} = 2.6\\;M_\\odot$",
    "$M_\\textrm{TPAGB} = 3.0\\;M_\\odot$",
]

for i, marker in enumerate(np.unique(markers)):
    mask = np.array(markers) == marker

    plt.scatter(
        np.array(s_fes)[mask],
        (np.array(hs_fes) - np.array(ls_fes))[mask],
        c=np.array(colors)[mask],
        cmap="viridis",
        vmin=np.min(colors),
        vmax=np.max(colors),
        marker=marker,
        label=labels[i],
        s=10,
    )

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("[$s$/Fe]")
plt.ylabel("[hs/ls]")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-hslss.pgf", format="pgf")
plt.show()
plt.close()

# %%
for m in grid.filter(m=3, R=[500, 600]):
    if m.envelope_mass[-1] > 0.01:
        continue
    plt.plot(m.age, m.rl_1)
    plt.plot(m.age, m.R)

star = get_star(m=3)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%
