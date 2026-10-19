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

from scipy.ndimage import uniform_filter1d

# %%


def get_ls(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 3


def get_hs(ab):
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


def get_s(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


def get_ba(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba"]:
            fe += ab.iron_abundance[i]

    return fe


grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

from matplotlib.lines import Line2D

fs = [0.1, 0.2, 0.3]
markers = ["o", "s", "^"]
cmap = plt.get_cmap("viridis")

for f, marker in zip(fs, markers):
    x = []
    y = []
    z = []
    for m in grid.filter(f_delta=f):
        if m.envelope_mass[-1] > 0.01:
            continue
        x.append(m.period_days[-1])
        y.append(m.s)
        z.append(m.params["R"])

    plt.scatter(x, y, c=np.log10(z), marker=marker, cmap=cmap, s=15)

# gray markers for f_delta
marker_handles = [
    Line2D(
        [],
        [],
        marker=marker,
        linestyle="none",
        color="gray",
        markerfacecolor="gray",
        markersize=8,
        label=f"$f_\\delta = {f}$",
    )
    for f, marker in zip(fs, markers)
]

# colored dots for each R
Rs = sorted({m.params["R"] for m in grid.models})
norm = plt.Normalize(np.log10(min(Rs)), np.log10(max(Rs)))

color_handles = [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        color=cmap(norm(np.log10(R))),
        markersize=7,
        label=f"$R = {R:.2f}\\,R_\\odot$",
    )
    for R in Rs
]

fig.legend(handles=marker_handles, loc="outside upper center", ncols=3)

plt.legend(handles=color_handles, frameon=False)

plt.xscale("log")

# plt.xlim(200, 10000)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-period-sFe-period.pgf", format="pgf")
plt.show()
plt.close()


# %%
from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=0.75),
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
        axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

plt.colorbar(
    c, ax=axs[0, 1], orientation="horizontal", location="top", label="[s/Fe]", aspect=20
)
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w33-pcolor-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=0.75),
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
print(minn)

norm = TwoSlopeNorm(1, minn, maxx)
for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="coolwarm", norm=norm)
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
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5.pgf", format="pgf")
plt.show()
plt.close()

# %%

from matplotlib.colors import TwoSlopeNorm

fig, axs = plt.subplots(
    2,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=0.75),
    constrained_layout=True,
)


def fes(r):
    if r.envelope_mass[-1] > 0.01:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    try:
        ind = np.argwhere(r.R / r.rl_1 > 0.9)[0][0]
    except:
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

print(maxx)
print(minn)

for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="viridis", vmin=minn, vmax=maxx)
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
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5.pgf", format="pgf")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5-m1.8.pgf", format="pgf")
plt.show()
plt.close()

# %%

i = 0
for m in grid.models:
    if m.params["R"] != grid.axes["R"][-3]:
        continue
    if m.params["q"] != grid.axes["q"][-1]:
        continue
    plt.plot(
        m.age,
        m.rl_1,
        c=f"C{i}",
        label=f"$\\varepsilon = {m.params["eps"]:.1f}, \\delta = {m.params["delta"]:.1f}$",
    )
    plt.plot(m.age, m.R, c=f"C{i}", alpha=0.5, linewidth=3)
    i += 1
plt.legend()
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
    print(r.params["R"])
    if r.envelope_mass[-1] > 0.01:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan

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
# plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5-m1.8.pgf", format="pgf")
plt.show()
plt.close()


# %%

i = 0
for m in grid.models:
    if m.params["R"] != grid.axes["R"][1]:
        continue
    if m.params["q"] != grid.axes["q"][1]:
        continue
    plt.plot(
        m.age,
        m.rl_1,
        c=f"C{i}",
        label=f"$\\varepsilon = {m.params["eps"]:.1f}, \\delta = {m.params["delta"]:.1f}$",
    )
    plt.plot(m.age, m.R, c=f"C{i}", alpha=0.5, linewidth=3)
    i += 1
plt.legend()
plt.show()

# %%

i = 0
for m in grid.models:
    if m.params["R"] != grid.axes["R"][1]:
        continue
    if m.params["q"] != grid.axes["q"][1]:
        continue
    plt.plot(
        m.age,
        m.period_days,
        c=f"C{i}",
        label=f"$\\varepsilon = {m.params["eps"]:.1f}, \\delta = {m.params["delta"]:.1f}$",
    )
    i += 1
plt.legend()
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
    print(r.params["R"])
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


for i, eps in enumerate([0, 0.1]):
    for j, delta in enumerate([0.1, 0.2, 0.3]):
        R, q, s = grid.array(fes, "R", "q", eps=eps, f_delta=delta)
        c = axs[i, j].pcolormesh(R, q, s.T, cmap="viridis", vmin=minn, vmax=maxx)

        annotate_pcolormesh(axs[i, j], R, q, s.T, fmt=".0f")
        axs[i, j].set_title(f"$\\varepsilon = {eps}$, $f_\\delta = {delta}$")

cbar = plt.colorbar(
    c,
    ax=axs[0, 1],
    orientation="horizontal",
    location="top",
    label="Final period (days)",
    aspect=20,
)
cbar.ax.set_xscale("linear")
plt.xscale("log")
fig.supxlabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.supylabel("Initial mass ratio $q$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-6-m1.8.pgf", format="pgf")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w33-pcolor-1-m1.8.pgf", format="pgf")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w33-pcolor-8-m1.8.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
RS1 = grid.axes["R"]

ppp = []
rrr = []
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

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
RS2 = grid.axes["R"]
for i_R, R in enumerate(RS2):
    pp = []
    rr = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
    ppp.append(pp)
    rrr.append(rr)

rs = list(RS1) + list(RS2)
print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
# color = cmap(norm(x))

print(len(ppp))

ms = [1.8, 1.4]
markers = ["s", "^"]
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
    "/home/koen/LaTeX-setup/plots/w33-final-period-rl-overflow.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
RS1 = grid.axes["R"][-2:-1]

ppp = []
rrr = []
qqq = []
for i_R, R in enumerate(RS1):

    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["q"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
RS2 = grid.axes["R"][-2:-1]
for i_R, R in enumerate(RS2):
    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["q"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

rs = list(RS1) + list(RS2)
print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
cmaps = ["Purples", "Greens"]
# color = cmap(norm(x))

print(len(ppp))

ms = [1.8, 1.4]
markers = ["s", "^"]

scatters = []
for i, (pp, rr, qq) in enumerate(zip(ppp, rrr, qqq)):
    scatter = plt.scatter(
        pp,
        rr,
        label=f"$M={ms[i]}\\;M_\\odot,\\;R_\\textrm{{RL, i}} = {rs[i]:.0f},\\;R_\\odot$",
        s=10,
        marker=markers[i],
        c=qq,
        cmap=cmaps[i],
        vmin=0.31,
    )
    scatters.append(scatter)


plt.colorbar(scatters[0], label="$q$", pad=0.01)
cbar = plt.colorbar(pad=0.01)
cbar.set_ticks(ticks=[], labels=[])

fig.legend(loc="outside upper center", ncols=2)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-final-period-rl-overflow-2.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
RS1 = grid.axes["R"][-3:-2]

ppp = []
rrr = []
qqq = []
for i_R, R in enumerate(RS1):

    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["q"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
RS2 = grid.axes["R"][-3:-2]
for i_R, R in enumerate(RS2):
    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["q"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

rs = list(RS1) + list(RS2)
print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
cmaps = ["Purples", "Greens"]
# color = cmap(norm(x))

print(len(ppp))

ms = [1.8, 1.4]
markers = ["s", "^"]

scatters = []
for i, (pp, rr, qq) in enumerate(zip(ppp, rrr, qqq)):
    scatter = plt.scatter(
        pp,
        rr,
        label=f"$M={ms[i]}\\;M_\\odot,\\;R_\\textrm{{RL, i}} = {rs[i]:.0f},\\;R_\\odot$",
        s=10,
        marker=markers[i],
        c=qq,
        cmap=cmaps[i],
        vmin=0.31,
    )
    scatters.append(scatter)


plt.colorbar(scatters[0], label="$q$", pad=0.01)
cbar = plt.colorbar(pad=0.01)
cbar.set_ticks(ticks=[], labels=[])

fig.legend(loc="outside upper center", ncols=2)

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-final-period-rl-overflow-3.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column, height=1.5), constrained_layout=True
)


grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
ms = []
for m in grid.models:
    if m.params["R"] != grid.axes["R"][-2]:
        continue
    if m.params["delta"] != grid.axes["delta"][0]:
        continue
    ms.append(m)
    print(m.params["delta"])

ms = np.array(ms)
sortidx = np.argsort([m.params["q"] for m in ms])
for i_m, m in enumerate(ms[sortidx]):
    axs[0].plot(
        m.age,
        m.rl_1,
        label=f"$q={m.params["q"]:.2f}$",
        c=f"C{i_m}",
        alpha=0.5,
        linewidth=3,
    )
    axs[0].plot(m.age, m.R, c=f"C{i_m}")

ms = []
for m in grid.models:
    if m.params["R"] != grid.axes["R"][-2]:
        continue
    if m.params["delta"] != grid.axes["delta"][2]:
        continue
    ms.append(m)

ms = np.array(ms)
sortidx = np.argsort([m.params["q"] for m in ms])
print(sortidx)
for i_m, m in enumerate(ms[sortidx]):
    axs[1].plot(
        m.age,
        m.rl_1,
        alpha=0.5,
        linewidth=3,
    )
    axs[1].plot(m.age, m.R, c=f"C{i_m}")

fig.legend(loc="outside upper center", ncols=3)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

axs[0].text(0.05, 0.05, "$\\delta = 0.3$", transform=axs[0].transAxes, va="bottom")
axs[1].text(0.05, 0.05, "$\\delta = 0.1$", transform=axs[1].transAxes, va="bottom")

axs[0].set_xlim(2821637125.2771544, 2821656128.5742683)
axs[0].set_ylim(100, 500)
axs[1].set_xlim(2821648625.799295, 2821651159.645361)
axs[1].set_ylim(100, 600)

plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-look-at-col-m1.4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column, height=1.5), constrained_layout=True
)


grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
ms = []
for m in grid.models:
    if m.params["R"] != grid.axes["R"][-2]:
        continue
    if m.params["delta"] != grid.axes["delta"][0]:
        continue
    ms.append(m)
    print(m.params["delta"])

ms = np.array(ms)
sortidx = np.argsort([m.params["q"] for m in ms])
for i_m, m in enumerate(ms[sortidx]):
    axs[0].plot(
        m.age,
        m.rl_1,
        label=f"$q={m.params["q"]:.2f}$",
        c=f"C{i_m}",
        alpha=0.5,
        linewidth=3,
    )
    axs[0].plot(m.age, m.R, c=f"C{i_m}")

ms = []
for m in grid.models:
    if m.params["R"] != grid.axes["R"][-2]:
        continue
    if m.params["delta"] != grid.axes["delta"][2]:
        continue
    ms.append(m)

ms = np.array(ms)
sortidx = np.argsort([m.params["q"] for m in ms])
print(sortidx)
for i_m, m in enumerate(ms[sortidx]):
    axs[1].plot(
        m.age,
        m.rl_1,
        alpha=0.5,
        linewidth=3,
    )
    axs[1].plot(m.age, m.R, c=f"C{i_m}")

fig.legend(loc="outside upper center", ncols=3)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

axs[0].text(0.05, 0.05, "$\\delta = 0.3$", transform=axs[0].transAxes, va="bottom")
axs[1].text(0.05, 0.05, "$\\delta = 0.1$", transform=axs[1].transAxes, va="bottom")

# axs[0].set_xlim(2821637125.2771544, 2821656128.5742683)
# axs[0].set_ylim(100, 500)
# axs[1].set_xlim(2821648625.799295, 2821651159.645361)
# axs[1].set_ylim(100, 600)

plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w33-look-at-col-m1.4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.8-2026-10-09/")
RS1 = grid.axes["R"][-2:-1]

ppp = []
rrr = []
qqq = []
for i_R, R in enumerate(RS1):

    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["delta"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")
RS2 = grid.axes["R"][-2:-1]
for i_R, R in enumerate(RS2):
    pp = []
    rr = []
    qq = []
    for m in grid.filter(R=R):
        if m.envelope_mass[-1] > 0.01:
            continue
        pp.append(m.period_days[-1])
        rr.append(np.nanmax(m.R / m.rl_1))
        qq.append(m.params["delta"])
    ppp.append(pp)
    rrr.append(rr)
    qqq.append(qq)

rs = list(RS1) + list(RS2)
print(rs)

norm = plt.Normalize(min(rs), max(rs))
cmap = plt.cm.viridis
cmaps = ["Purples", "Greens"]
# color = cmap(norm(x))

print(len(ppp))

ms = [1.8, 1.4]
markers = ["s", "^"]

scatters = []
for i, (pp, rr, qq) in enumerate(zip(ppp, rrr, qqq)):
    scatter = plt.scatter(
        pp,
        rr,
        label=f"$M={ms[i]}\\;M_\\odot,\\;R_\\textrm{{RL, i}} = {rs[i]:.0f},\\;R_\\odot$",
        s=10,
        marker=markers[i],
        c=qq,
        cmap=cmaps[i],
        vmin=0.051,
    )
    scatters.append(scatter)


plt.colorbar(scatters[0], label="$\\delta$", pad=0.01)
cbar = plt.colorbar(pad=0.01)
cbar.set_ticks(ticks=[], labels=[])

fig.legend(loc="outside upper center", ncols=2)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Final period (days)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{RL}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-final-period-rl-overflow-4.pgf", format="pgf"
)
plt.show()
plt.close()
