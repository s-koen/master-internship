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

df = AbundanceTables()

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue
    print(m.params["m"])
    mass = m.params["m"]
    ab = Abundances(model=m, df=df, sampling=1)
    m.ab = ab.iron_abundance
    m.elements_mass = ab.elements_mass
    m.elements_name = ab.elements_name
    m.t_av = ab.t_av
    m.m_dup_av = ab.m_dup_av


def get_ls(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 3


def get_hs(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += r.ab[i]

    return fe / 4


def get_s(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 7


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:

    per.append(m.params["R"])
    s.append(get_s(m))

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25, c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-RL.pgf", format="pgf")
plt.show()
plt.close()

# %%
