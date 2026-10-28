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
g4 = MesaGrid("/home/koen/master-internship/mesa-models/grid-M2.6-2026-10-09/")
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3, g4]

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
plt.savefig("/home/koen/LaTeX-setup/plots/w34-grid-thus-far.pgf", format="pgf")
plt.show()
plt.close()
# %%
from matplotlib.colors import TwoSlopeNorm

grid = g4
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
plt.savefig("/home/koen/LaTeX-setup/plots/w34-pcolor-1-m2.6.pgf", format="pgf")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w34-pcolor-5-m2.2.pgf", format="pgf")
plt.show()
plt.close()

# %%


def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


from matplotlib.colors import TwoSlopeNorm

ms = [1.4, 1.8, 2.2, 2.6]
for i_g, grid in enumerate([g1, g2, g3, g4]):
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
        return r.period_days[-1]

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
    for i, eps in enumerate([0, 0.1]):
        for j, delta in enumerate([0.1, 0.2, 0.3]):
            R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
            c = axs[i, j].pcolormesh(R, q, s.T, cmap="viridis", vmin=minn, vmax=maxx)
            annotate_pcolormesh(
                axs[i, j],
                R,
                q,
                s.T,
                fmt=".0f",
                colors=["white", "black"],
                # norm=norm,
            )
            axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

    cbar = plt.colorbar(
        c,
        ax=axs[0, 1],
        orientation="horizontal",
        location="top",
        label="Period (days)",
        aspect=20,
        extend="max",
    )
    fig.text(0.07, 0.95, f"$M= {ms[i_g]}\\;M_\\odot$")
    cbar.ax.set_xscale("linear")
    plt.xscale("log")
    fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
    fig.supylabel("Initial mass ratio $q$", fontsize=10)
    plt.savefig(f"/home/koen/LaTeX-setup/plots/w34-pcolor-11-{i_g}.pgf", format="pgf")
    plt.show()
    plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3, g4]

ls = []
for i_g, g in enumerate(gs):
    for m in g.models:
        if m.envelope_mass[-1] > 0.01:
            continue
        s = plt.scatter(
            m.s,
            m.co,
            c=f"C{i_g}",
            label=f"$M = {m.params["m"]}\\; M_\\odot$",
            s=10,
        )
    ls.append(s)

fig.legend(loc="outside upper center", ncol=4, handles=ls)

# plt.xscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.ylabel("C/O number ratio")
plt.xlabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w34-s-co.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3, g4]

ls = []
for i_g, g in enumerate(gs):
    for m in g.models:
        if m.envelope_mass[-1] > 0.01:
            continue
        s = plt.scatter(
            m.s,
            m.ab[2],
            c=f"C{i_g}",
            label=f"$M = {m.params["m"]}\\; M_\\odot$",
            s=10,
        )
        print(m.elements_name[2])
    ls.append(s)

fig.legend(loc="outside upper center", ncol=4, handles=ls)

# plt.xscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.ylabel("C/O number ratio")
plt.xlabel("[s/Fe]")
# plt.savefig("/home/koen/LaTeX-setup/plots/w34-s-co.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

ppp = []
rrr = []
ms = [2.6, 2.2, 1.8, 1.4]
rs = []

for m in ms:
    grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M{m}-2026-10-09/")
    RS1 = grid.axes["R"]

    for i_R, R in enumerate(RS1):

        pp = []
        rr = []
        for m in grid.filter(R=R):
            if m.envelope_mass[-1] > 0.01:
                continue
            pp.append(m.period_days[-1])
            rr.append(np.nanmax(m.R / m.rl_1))
        ppp.append(pp)
        rrr.append(rr)

    rs += list(RS1)

print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
# color = cmap(norm(x))

print(len(ppp))

markers = ["s", "^", "v", "o"]
for i, (pp, rr) in enumerate(zip(ppp, rrr)):
    plt.scatter(
        pp,
        rr,
        label=f"$M={ms[i//5]}\\;M_\\odot,\\;R_\\textrm{{RL, i}} = {rs[i]:.0f},\\;R_\\odot$",
        s=10,
        marker=markers[i // 5],
    )


fig.legend(loc="outside upper center", ncols=2)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w34-final-period-rl-overflow.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

ppp = []
rrr = []
ms = [2.6, 2.2, 1.8, 1.4]
rs = []

for m in ms:
    grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M{m}-2026-10-09/")
    RS1 = grid.axes["R"]

    for i_R, R in enumerate(RS1):

        pp = []
        rr = []
        for m in grid.filter(R=R):
            if m.envelope_mass[-1] > 0.01:
                continue
            pp.append(m.period_days[-1])
            rr.append(np.nanmax(m.R / rol(m)))
        ppp.append(pp)
        rrr.append(rr)

    rs += list(RS1)

print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
# color = cmap(norm(x))

print(len(ppp))

markers = ["s", "^", "v", "o"]
for i, (pp, rr) in enumerate(zip(ppp, rrr)):
    plt.scatter(
        pp,
        rr,
        label=f"$M={ms[i//5]}\\;M_\\odot,\\;R_\\textrm{{RL, i}} = {rs[i]:.0f},\\;R_\\odot$",
        s=10,
        marker=markers[i // 5],
    )


fig.legend(loc="outside upper center", ncols=2)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w34-final-period-rl-overflow.pgf", format="pgf"
)
plt.show()
plt.close()
