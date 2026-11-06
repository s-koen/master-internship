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
plt.savefig("/home/koen/LaTeX-setup/plots/w35-pcolor-1-m3.pgf", format="pgf")
plt.show()
plt.close()


# %%

for m in grid.models:
    plt.plot(m.age, m.rl_1)
    plt.plot(m.age, m.R)
plt.show()
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
    return np.max(r.R / r.rl_1)


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
    label="max($R_\\textrm{star} / R_\\textrm{RL}$)",
    aspect=20,
)
cbar.ax.set_xscale("linear")
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w35-pcolor-2-m3.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

gs = [g1, g2, g3, g4, g5]

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

fig.legend(loc="outside upper center", ncol=2, handles=ls)

plt.xscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w35-grid-thus-far.pgf", format="pgf")
plt.show()
plt.close()
# %%
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
    "/home/koen/LaTeX-setup/plots/w35-final-period-rl-overflow.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

ppp = []
rrr = []
sss = []
ms = [3.0, 2.6, 2.2, 1.8, 1.4]
rs = []

for m in ms:
    grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M{m}-2026-10-09/")
    RS1 = grid.axes["R"]

    for R in RS1:
        pp = []
        rr = []
        ss = []

        for model in grid.filter(R=R):
            if model.envelope_mass[-1] > 0.01:
                continue

            pp.append(model.period_days[-1])
            rr.append(np.nanmax(model.R / rol(model)))
            ss.append(model.s)

        ppp.append(pp)
        rrr.append(rr)
        sss.append(ss)

    rs += list(RS1)

markers = ["s", "^", "v", "o", "+"]

for i, (pp, rr, ss) in enumerate(zip(ppp, rrr, sss)):
    colors = ["C9" if s < 0.25 else "C2" for r, s in zip(rr, ss)]

    plt.scatter(
        pp,
        rr,
        c=colors,
        s=10,
        marker=markers[i // len(RS1)],
    )

# fake points for the legend
for i, mass in enumerate(ms):
    axs.scatter(
        [],
        [],
        label=f"$M={mass}\\;M_\\odot$",
        marker=markers[i],
        s=10,
        color="C2",
    )

fig.legend(loc="outside upper center", ncols=4)

plt.axhline(1, linewidth=0.75, c="C9")

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{ROL}$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w35-final-period-rl-overflow.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%
