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

from scipy.ndimage import uniform_filter1d

# %%


g1 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
g2 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
g3 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M2.2-2026-10-09/")
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3]

ls = []
for i_g, g in enumerate(gs):
    for m in g.models:
        if m.envelope_mass[-1] > 0.01:
            continue
        s = plt.scatter(
            m.period_days[-1],
            m.s,
            c=f"C{i_g}",
            label=f"$M = {m.params["m"]}\\; M_\\odot$",
            s=10,
        )
    ls.append(s)

fig.legend(loc="outside upper center", ncol=3, handles=ls)

plt.xscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-grid-thus-far.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3]
colors = ["C0", "C1", "C2"]

ls = []

cmaps = ["Blues", "Oranges", "Greens"]

for i_g, g in enumerate(gs):
    xs, ys, zs = [], [], []

    for m in g.models:
        if m.envelope_mass[-1] > 0.01:
            continue

        xs.append(m.period_days[-1])
        ys.append(m.params["q"])
        zs.append(m.s)

    contour = axs.tricontourf(
        xs, ys, zs, levels=10, linewidths=1, alpha=0.5, cmap=cmaps[i_g]
    )

    ls.append(
        plt.Line2D(
            [],
            [],
            color=colors[i_g],
            label=f"$M = {g.models[0].params['m']}\\; M_\\odot$",
        )
    )

fig.legend(loc="outside upper center", ncol=3, handles=ls)

axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
axs.set_xlabel("period (days)")
axs.set_ylabel("$q$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-grid-thus-far-tri.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(1, 1, figsize=set_size(column), constrained_layout=True)

gs = [g1, g2, g3]
colors = ["C0", "C1", "C2"]

handles = []

for i_g, g in enumerate(gs):
    xs, zs = [], []

    for m in g.models:
        if m.envelope_mass[-1] > 0.01:
            continue

        xs.append(m.period_days[-1])
        zs.append(m.s)

    axs.tricontour(
        xs,
        zs,
        levels=10,
        colors=colors[i_g],
        linewidths=1,
    )

    handles.append(
        plt.Line2D(
            [],
            [],
            color=colors[i_g],
            label=f"$M = {g.models[0].params['m']}\\; M_\\odot$",
        )
    )

fig.legend(handles=handles, loc="outside upper center", ncol=3)

axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
axs.set_xlabel("period (days)")
axs.set_ylabel("[s/Fe]")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-grid-thus-far-2.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fig, axs = plt.subplots(1, 1, figsize=set_size(column), constrained_layout=True)

gs = [g1, g2, g3]
colors = ["C0", "C1", "C2"]

handles = []
cmaps = ["Blues", "Oranges", "Greens"]

for i_g, g in enumerate(gs):
    points = np.array(
        [
            [np.log10(m.period_days[-1]), m.s]
            for m in g.models
            if m.envelope_mass[-1] <= 0.01
        ]
    ).T

    kde = gaussian_kde(points)
    density = kde(points)

    axs.tricontour(
        points[0],
        points[1],
        density,
        levels=np.quantile(density, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        cmap=cmaps[i_g],
        extend="both",
        vmin=-0.1,
    )

    handles.append(
        plt.Line2D(
            [],
            [],
            color=colors[i_g],
            label=f"$M = {g.models[0].params['m']}\\; M_\\odot$",
        )
    )

fig.legend(handles=handles, loc="outside upper center", ncol=3)

axs.set_xlabel("period (days)")
axs.set_ylabel("[s/Fe]")
axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)

# plt.savefig("/home/koen/LaTeX-setup/plots/w33-grid-thus-far.pgf", format="pgf")
plt.show()
plt.close()
# %%
from matplotlib.colors import TwoSlopeNorm

grid = g3
fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=0.75),
    constrained_layout=True,
)


def fes(r):
    if r.envelope_mass[-1] > 0.01:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    return r.s


ss = []
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        _, _, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        ss.append(s)

minn = 1e99
maxx = -1e99
for ratio in ss:
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx
print(maxx)

norm = TwoSlopeNorm(0.25, minn, maxx)
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="coolwarm", norm=norm)
        annotate_pcolormesh(
            axs[i, j],
            R,
            q,
            s.T,
            fmt=".2f",
            colors=["white", "black", "white"],
            norm=norm,
        )
        axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

cbar = plt.colorbar(
    c, ax=axs[0, 1], orientation="horizontal", location="top", label="[s/Fe]", aspect=20
)
cbar.ax.set_xscale("linear")
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w33-pcolor-1-m2.2.pgf", format="pgf")
plt.show()
plt.close()


# %%

from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=0.75),
    constrained_layout=True,
)


def fes(r):
    # if r.envelope_mass[-1] > 0.01:
    #     return np.nan
    # if r.period_days[-1] < 50:
    #     return np.nan

    smoothed = uniform_filter1d(r.R / r.rl_1, size=5, mode="nearest")

    mask = smoothed > 0.9

    # mask = r.R / r.rl_1 > 0.9
    inds = np.where(mask)[0]
    print(inds)

    if len(inds):
        starts = np.r_[0, np.where(np.diff(inds) > 1)[0] + 1]
        print(starts)
        if len(starts) > 2:
            ind = inds[starts[-len(starts)]]
        else:
            ind = inds[starts[-1]]
    else:
        ind = np.argmax(r.R / r.rl_1)
    return r.envelope_mass[ind]


ss = []
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        _, _, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        ss.append(s)

minn = 1e99
maxx = -1e99
for ratio in ss:
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="viridis", vmin=minn, vmax=maxx)

        annotate_pcolormesh(axs[i, j], R, q, s.T, fmt=".3f")
        axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

cbar = plt.colorbar(
    c,
    ax=axs[0, 1],
    orientation="horizontal",
    location="top",
    label="$M_\\textrm{env}|_{t=t(R_\\textrm{star}/R_\\textrm{RL} > 0.9)}$ ($M_\\odot$)",
    aspect=20,
)
cbar.ax.set_xscale("linear")
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5-m2.2.pgf", format="pgf")
plt.show()
plt.close()


# %%

from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=0.75),
    constrained_layout=True,
)


def fes(r):
    if r.envelope_mass[-1] > 0.01:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    return np.nanmax(r.R / r.rl_1)


ss = []
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        _, _, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        ss.append(s)

minn = 1e99
maxx = -1e99
for ratio in ss:
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx
print(maxx)

norm = TwoSlopeNorm(1, minn, 2.5)
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="coolwarm", norm=norm)
        annotate_pcolormesh(
            axs[i, j],
            R,
            q,
            s.T,
            fmt=".2f",
            colors=["white", "black", "white"],
            norm=norm,
        )
        axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

cbar = plt.colorbar(
    c,
    ax=axs[0, 1],
    orientation="horizontal",
    location="top",
    label="max$(R_\\textrm{star} / R_\\textrm{RL})$",
    aspect=20,
    extend="max",
)
cbar.ax.set_xscale("linear")
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w33-pcolor-8-m2.2.pgf", format="pgf")
plt.show()
plt.close()


# %%


def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%

from matplotlib.colors import TwoSlopeNorm

ms = [1.4, 1.8, 2.2]
for i_g, grid in enumerate([g1, g2, g3]):
    fig, axs = plt.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
        figsize=set_size(full, height=0.75),
        constrained_layout=True,
    )

    def fes(r):
        if r.envelope_mass[-1] > 0.01:
            return np.nan
        if r.period_days[-1] < 50:
            return np.nan
        return np.nanmax(r.R / rol(r))

    ss = []
    for i, eps in enumerate([0, 0.1]):
        for j, delta in enumerate([0.1, 0.2, 0.3]):
            _, _, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
            ss.append(s)

    minn = 1e99
    maxx = -1e99
    for ratio in ss:
        minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
        maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx
    print(maxx)

    print(minn, maxx)
    norm = TwoSlopeNorm(1, minn, 2.5)
    for i, eps in enumerate([0, 0.1]):
        for j, delta in enumerate([0.1, 0.2, 0.3]):
            R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
            c = axs[i, j].pcolormesh(R, q, s.T, cmap="coolwarm", norm=norm)
            annotate_pcolormesh(
                axs[i, j],
                R,
                q,
                s.T,
                fmt=".2f",
                colors=["white", "black", "white"],
                norm=norm,
            )
            axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

    cbar = plt.colorbar(
        c,
        ax=axs[0, 1],
        orientation="horizontal",
        location="top",
        label="max$(R_\\textrm{star} / R_\\textrm{ROL})$",
        aspect=20,
        extend="max",
    )
    fig.text(0.07, 0.95, f"$M= {ms[i_g]}\\;M_\\odot$")
    cbar.ax.set_xscale("linear")
    plt.xscale("log")
    fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
    fig.supylabel("Initial mass ratio $q$", fontsize=10)
    plt.savefig(f"/home/koen/LaTeX-setup/plots/w33-pcolor-9-{i_g}.pgf", format="pgf")
    plt.show()
    plt.close()

# %%
