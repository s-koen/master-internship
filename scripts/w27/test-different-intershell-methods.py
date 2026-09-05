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
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
    Element,
    Isotope,
)

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

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

from matplotlib.colors import TwoSlopeNorm

df = AbundanceTables()


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    ab = Abundances(model=r, df=df)
    ac1 = ab.ba.m_accreted[-1]
    ab = Abundances(model=r, df=df, method="tp")
    ac2 = ab.ba.m_accreted[-1]
    return np.log10(ac1 - ac2)
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


# R, q, ratio = grid.array(z, x="R", y="q")


ms = [1.8, 2.2, 2.6, 3.0]

ratios = []
for m in ms:
    R, q, ratio = grid.array(z, x="R", y="q", m=m)
    ratios.append(ratio)


for ratio in ratios:
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

print(minn, maxx)

# norm = TwoSlopeNorm(0, minn, -minn)

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    ratio = ratios[i]

    c = ax.pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)
plt.colorbar(
    c,
    ax=axs,
    orientation="vertical",
    location="right",
    label=r"Difference of $\Delta M_\textrm{Ba}$ accreted ($M_\odot$)",
)
fig.suptitle("Parameterized by $M_\\textrm{DUP}$ vs TP count", fontsize=10)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w27-grid-diff-tp.pgf", format="pgf")
plt.show()
plt.close()


# %%

from matplotlib.colors import TwoSlopeNorm

df = AbundanceTables()


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    ab = Abundances(model=r, df=df)
    ac1 = ab.ba.m_accreted[-1]
    ab = Abundances(model=r, df=df, method="tp")
    ac2 = ab.ba.m_accreted[-1]
    return np.log10(ac2 / ac1)
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


# R, q, ratio = grid.array(z, x="R", y="q")


ms = [1.8, 2.2, 2.6, 3.0]

ratios = []
for m in ms:
    R, q, ratio = grid.array(z, x="R", y="q", m=m)
    ratios.append(ratio)


for ratio in ratios:
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

print(minn, maxx)

norm = TwoSlopeNorm(0, minn, 1e-10)

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    ratio = ratios[i]

    c = ax.pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)
car = plt.colorbar(
    c,
    ax=axs,
    orientation="vertical",
    location="right",
    label=r"Log divided of $\Delta M_\textrm{Ba}$ accreted ($M_\odot$)",
)
car.ax.set_yscale("linear")

fig.suptitle("Parameterized by $M_\\textrm{DUP}$ vs TP count", fontsize=10)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w27-grid-diff-tp-2.pgf", format="pgf")
plt.show()
plt.close()
# %%
