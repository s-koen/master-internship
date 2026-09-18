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
df = AbundanceTables()


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    # ab = Abundances(model=r, df=df)
    # return np.log10(ab.ba.envelope[-1])

    return (r.star_2_mass[-1] - r.params["q"] * r.params["m"]) / r.star_2_mass[-1]
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full),
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

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    ratio = ratios[i]
    for k, r in enumerate(R):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{ratio[k, j]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass

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
    label=r"$\Delta M_2 / M_\textrm{2,f}$",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w28-accretion-1.pgf", format="pgf")
plt.show()
plt.close()


# %%

df = AbundanceTables()


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    # ab = Abundances(model=r, df=df)
    # return np.log10(ab.ba.envelope[-1])

    return np.max(r.R / r.rl_1)
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full),
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

from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=minn, vcenter=1, vmax=maxx)

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    ratio = ratios[i]
    for k, r in enumerate(R):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{ratio[k, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass

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
cbar = plt.colorbar(
    c,
    ax=axs,
    orientation="vertical",
    location="right",
    label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{rl})$",
)

cbar.ax.set_yscale("linear")

plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w28-accretion-2.pgf", format="pgf")
plt.show()
plt.close()


# %%


def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%

df = AbundanceTables()


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    # ab = Abundances(model=r, df=df)
    # return np.log10(ab.ba.envelope[-1])

    return np.max(r.R / rol(r))
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full),
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

from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=minn, vcenter=1, vmax=maxx)

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    ratio = ratios[i]
    for k, r in enumerate(R):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{ratio[k, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass

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
cbar = plt.colorbar(
    c,
    ax=axs,
    orientation="vertical",
    location="right",
    label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{rol})$",
)

cbar.ax.set_yscale("linear")

plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w28-accretion-3.pgf", format="pgf")
plt.show()
plt.close()


# %%
