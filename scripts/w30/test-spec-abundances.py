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

df = AbundanceTables()
ab = Abundances(None, df, mass=2, m_acc=1, mass_transfer_efficiency=1.0)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

ab2 = Abundances(None, df, mass=2, m_acc=1, mass_transfer_efficiency=1)

removes = [39, 57, -1]

epss = np.logspace(-4, 0, 10)

norm = plt.Normalize(np.log10(np.min(epss)), np.log10(np.max(epss)))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for eps in epss:

    ab = Abundances(None, df, mass=2, m_acc=1, mass_transfer_efficiency=eps)
    plt.plot(
        np.delete(ab.elements_mass, removes),
        np.delete(ab.iron_abundance, removes),
        c=cmap(norm(np.log10(eps))),
        linewidth=1,
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca(), aspect=50)
cbar.set_label(r"$\epsilon$ mass-transfer")

element_labels(
    fig,
    np.delete(ab2.elements_mass, removes),
    np.delete(ab2.elements_name, removes),
    axs,
)

plt.axhline(0, c="C9", zorder=-10, linewidth=0.75)

plt.title("Constant MS mass ($1\\;M_\\odot$) and TPAGB mass ($2\\;M_\\odot$)")
plt.xlabel("Elements")
plt.ylabel("[X/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-fe-ab.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

removes = [39, 57, -1]
epss = np.arange(1, 3.01, 0.2)

norm = plt.Normalize(np.min(epss), np.max(epss))
cmap = plt.cm.viridis

for eps in epss:

    ab = Abundances(None, df, mass=eps, m_acc=1, mass_transfer_efficiency=0.25)
    plt.plot(
        np.delete(ab.elements_mass, removes),
        np.delete(ab.iron_abundance, removes),
        c=cmap(norm(eps)),
        linewidth=1,
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca(), aspect=50)
cbar.set_label(r"$M_\textrm{TPAGB}$ ($M_\odot$)")

element_labels(
    fig,
    np.delete(ab2.elements_mass, removes),
    np.delete(ab2.elements_name, removes),
    axs,
)

plt.axhline(0, c="C9", zorder=-10, linewidth=0.75)

plt.title(
    "Constant MS mass ($1\\;M_\\odot$) and constant $\\epsilon$ mass-transfer (0.25)"
)
plt.xlabel("Elements")
plt.ylabel("[X/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-fe-ab-mass.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

removes = [39, 57, -1]
epss = np.arange(1, 3.01, 0.2)

norm = plt.Normalize(np.min(epss), np.max(epss))
cmap = plt.cm.viridis

for eps in epss:

    ab = Abundances(None, df, mass=eps, m_acc=eps, mass_transfer_efficiency=0.25)
    plt.plot(
        np.delete(ab.elements_mass, removes),
        np.delete(ab.iron_abundance, removes),
        c=cmap(norm(eps)),
        linewidth=1,
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca(), aspect=50)
cbar.set_label(r"$M_\textrm{TPAGB}$ ($M_\odot$)")

element_labels(
    fig,
    np.delete(ab2.elements_mass, removes),
    np.delete(ab2.elements_name, removes),
    axs,
)

plt.axhline(0, c="C9", zorder=-10, linewidth=0.75)

plt.title("Constant $q$ ($1$) and constant $\\epsilon$ mass-transfer (0.25)")
plt.xlabel("Elements")
plt.ylabel("[X/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-fe-ab-mass-q.pgf", format="pgf")
plt.show()
plt.close()
# %%
