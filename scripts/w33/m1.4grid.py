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


grid = MesaGrid(f"/home/koen/master-internship/mesa-models/grid-M1.4-2026-10-09/")

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

    # smoothed = uniform_filter1d(m.R, size=10, mode="nearest")

    # mask = smoothed / m.rl_1 > 0.9

    mask = r.R / r.rl_1 > 0.9
    inds = np.where(mask)[0]

    if len(inds):
        starts = np.r_[0, np.where(np.diff(inds) > 1)[0] + 1]
        if len(starts) > 2:
            ind = inds[starts[-2]]
        else:
            ind = inds[starts[-1]]
    else:
        id = np.argmax(r.R / r.rl_1)
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

        annotate_pcolormesh(axs[i, j], R, q, s.T)
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
plt.savefig("/home/koen/LaTeX-setup/plots/w32-pcolor-5-m1.4.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

counter = 0
for m in grid.models:
    print(m.bulk_names)
    if not m.params["R"] in grid.axes["R"][2:5]:
        continue
    if not np.isclose(m.params["q"], 1):
        continue
    if not np.isclose(m.params["eps"], 0.1):
        continue
    if not np.isclose(m.params["f_delta"], 0.1):
        continue
    plt.plot(m.age, m.log_R, c=f"C{counter}", linewidth=1)
    plt.plot(m.age, m.lg_mstar_dot_1, c=f"C{counter}", alpha=0.5, linewidth=2)
    counter += 1


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("$R$ ($R_\\odot$)")
plt.title("$q=1$, $\\varepsilon = 0.1$, $f_\\delta = 0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-no-RLOF.pgf", format="pgf")
plt.show()
plt.close()
# %%
from scipy.ndimage import uniform_filter1d

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

counter = 0
for m in grid.models:
    if not m.params["R"] in grid.axes["R"][2:3]:
        continue
    if not m.params["q"] in grid.axes["q"][1:5]:
        continue
    if not np.isclose(m.params["eps"], 0):
        continue
    if not np.isclose(m.params["f_delta"], 0.1):
        continue
    plt.plot(m.age, m.R, c=f"C{counter}", linewidth=1)
    if m.params["q"] == 0.55:
        plt.text(
            m.age[-1] - 2000,
            m.R[-1] + 120 - m.params["q"] * 250,
            f"$q={m.params["q"]:.2f}$",
            va="top",
            ha="center",
            fontsize=8,
            color=f"C{counter}",
        )
    else:
        plt.text(
            m.age[-1] - 2000,
            m.R[-1] + 120 - m.params["q"] * 250,
            f"${m.params["q"]:.2f}$",
            va="top",
            ha="center",
            fontsize=8,
            color=f"C{counter}",
        )
    plt.plot(m.age, m.rl_1, c=f"C{counter}", alpha=0.5, linewidth=2)

    mask = m.R / m.rl_1 > 0.9
    inds = np.where(mask)[0]

    if len(inds):
        starts = np.r_[0, np.where(np.diff(inds) > 1)[0] + 1]
        if len(starts) > 2:
            ind = inds[starts[-2]]
        else:
            ind = inds[starts[-1]]
    else:
        ind = np.argmax(m.R / m.rl_1)
    plt.scatter(m.age[ind], m.rl_1[ind], c=f"C{counter}", alpha=0.5, linewidth=2)
    counter += 1

    print(m.params["R"])
x_min = plt.gca().get_xlim()[0]
star = get_star(m=1.4)
plt.plot(star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9")

plt.xlim(2.821e9 + 530000, 2.821e9 + 700000)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("$R$ ($R_\\odot$)")
plt.title("$R_\\textrm{RL}= 521 \\;R_\\odot$, $\\varepsilon = 0.1$, $f_\\delta = 0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-no-RLOF-q.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.9), constrained_layout=True
)

ms = np.arange(1, 3.1, 0.1)

for m in ms:
    star = get_star(m=m)
    max_TP = int(np.max(star.TP_count))
    print(max_TP)
    mm = []
    m_envs = []
    for p in range(1, max_TP + 1):
        mm.append(m)
        ind = np.argwhere(star.TP_count == p)[-1][0]
        m_envs.append(star.m_env[ind])
    plt.plot(mm, m_envs, c="k", linewidth=1)
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="k",
        marker=".",
        s=[np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)][:-1],
        zorder=10,
    )
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="w",
        marker=".",
        s=2
        * np.array([np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)])[:-1],
        zorder=9,
    )


plt.yscale("log")
# plt.xscale("log")

plt.xticks(ms[::2])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-mass-at-TP.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.9), constrained_layout=True
)

ms = np.arange(1, 3.1, 0.1)

for m in ms:
    star = get_star(m=m)
    max_TP = int(np.max(star.TP_count))
    print(max_TP)
    mm = []
    m_envs = []
    for p in range(1, max_TP + 1):
        mm.append(m)
        ind = np.argwhere(star.TP_count == p)[-1][0]
        m_envs.append(star.m_env[ind])
    plt.plot(mm, m_envs, c="k", linewidth=1)
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="k",
        marker=".",
        s=[np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)][:-1],
        zorder=10,
    )
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="w",
        marker=".",
        s=2
        * np.array([np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)])[:-1],
        zorder=9,
    )


# plt.yscale("log")
# plt.xscale("log")

plt.xticks(ms[::2])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-mass-at-TP.pgf", format="pgf")
plt.show()
plt.close()
