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
    label=r"Log Difference of $\Delta M_\textrm{Ba}$ accreted ($M_\odot$)",
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
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=3, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle("$M=3\\;M_\\odot$, $q=0.6$, $R_\\textrm{RL}=1000\\;R_\\odot$", fontsize=10)

axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[0].legend(handles=[l1, l2], frameon=False)
axs[1].legend(handles=[m1, m2, m3], frameon=False)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-tp-ba-m3.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=2.6, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle(
    "$M=2.6\\;M_\\odot$, $q=0.6$, $R_\\textrm{RL}=1000\\;R_\\odot$", fontsize=10
)

axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-tp-ba-m2.6.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle(
    "$M=1.8\\;M_\\odot$, $q=0.4293$, $R_\\textrm{RL}=500\\;R_\\odot$", fontsize=10
)
axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlim(1.3262e9, 1.3279e9)

plt.xlabel("Star age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-tp-ba-m1.8.pgf", format="pgf")
plt.show()
plt.close()
# %%
for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=3)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


df = AbundanceTables()

ab = Abundances(model, df, method="tp")
ab.ba.intershell
# plt.plot(
#     ab.tp_count[star.ntpagb :],np.log10(ab.ba.intershell[star.ntpagb:]), c="k", linewidth=3
# )

fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal Pulse count")
plt.ylabel("$\log(X(\\textrm{Ba}_{138}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-thermal-pulse-naive.pgf", format="pgf")
plt.show()
plt.close()
# %%
for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=3)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


df = AbundanceTables()

ab = Abundances(model, df)
ab.ba.intershell
# plt.plot(
#     ab.tp_count[star.ntpagb :],np.log10(ab.ba.intershell[star.ntpagb:]), c="k", linewidth=3
# )

fig.legend(loc="outside upper center", ncols=3)

plt.xlim(-6)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.ylabel("$\log(X(\\textrm{Ba}_{138}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-m_dredge-naive.pgf", format="pgf")
plt.show()
plt.close()
# %%

for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=3)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


df = AbundanceTables()

ab = Abundances(model, df, method="tp offset")
ab.ba.intershell
# plt.plot(
#     ab.tp_count[star.ntpagb :],np.log10(ab.ba.intershell[star.ntpagb:]), c="k", linewidth=3
# )

fig.legend(loc="outside upper center", ncols=3)

# plt.xlim(-6)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("TP count + offset")
plt.ylabel("$\log(X(\\textrm{Ba}_{138}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-tp-count.pgf", format="pgf")
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
    ab = Abundances(model=r, df=df, method="tp offset")
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

norm = TwoSlopeNorm(0, minn, maxx)

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

plt.savefig("/home/koen/LaTeX-setup/plots/w27-grid-diff-tp-offset-2.pgf", format="pgf")
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
    ab = Abundances(model=r, df=df, method="tp offset")
    ac2 = ab.ba.m_accreted[-1]
    return ac2 - ac1
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

norm = TwoSlopeNorm(0, minn, maxx)

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
    label=r"Difference in $\Delta M_\textrm{Ba}$ accreted ($M_\odot$)",
)
car.ax.set_yscale("linear")

fig.suptitle("Parameterized by $M_\\textrm{DUP}$ vs TP count", fontsize=10)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w27-grid-diff-tp-offset-1.pgf", format="pgf")
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
    ac1 = ab.ba.envelope[-1]
    ab = Abundances(model=r, df=df, method="tp offset")
    ac2 = ab.ba.envelope[-1]
    return ac2 - ac1
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

norm = TwoSlopeNorm(0, minn, maxx)

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
    label=r"Difference in $X_\textrm{Ba}$ in the envelope",
)
car.ax.set_yscale("linear")

fig.suptitle("Parameterized by $M_\\textrm{DUP}$ vs TP count", fontsize=10)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-grid-diff-tp-offset-1-envelope.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=3, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp offset")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle("$M=3\\;M_\\odot$, $q=0.6$, $R_\\textrm{RL}=1000\\;R_\\odot$", fontsize=10)

axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP + offset",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[0].legend(handles=[l1, l2], frameon=False)
axs[1].legend(handles=[m1, m2, m3], frameon=False)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-tp-offset-ba-m3.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=2.6, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp offset")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle(
    "$M=2.6\\;M_\\odot$, $q=0.6$, $R_\\textrm{RL}=1000\\;R_\\odot$", fontsize=10
)

axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-tp-offset-ba-m2.6.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

df = AbundanceTables()

for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=3)

ab1 = Abundances(model, df)
ab2 = Abundances(model, df, method="tp offset")

axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.intershell[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C0",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.envelope[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C1",
)
axs[0].plot(
    ab1.time[star.ntpagb :],
    ab1.ba.m_accreted[star.ntpagb :],
    linewidth=3,
    alpha=0.5,
    c="C2",
)
(m1,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.intershell[star.ntpagb :],
    c="C0",
    label="$X(\\textrm{Ba})$ intershell",
)
(m2,) = axs[0].plot(
    ab2.time[star.ntpagb :],
    ab2.ba.envelope[star.ntpagb :],
    c="C1",
    label="$X(\\textrm{Ba})$ envelope",
)
(m3,) = axs[0].plot(
    ab1.time[star.ntpagb :],
    ab2.ba.m_accreted[star.ntpagb :],
    c="C2",
    label="$M_\\textrm{Ba,accr}$ ($M_\\odot$)",
)
axs[0].set_yscale("log")

axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.intershell[star.ntpagb :] - ab2.ba.intershell[star.ntpagb :])
    / ab2.ba.intershell[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.envelope[star.ntpagb :] - ab2.ba.envelope[star.ntpagb :])
    / ab2.ba.envelope[star.ntpagb :],
)
axs[1].plot(
    ab1.time[star.ntpagb :],
    np.abs(ab1.ba.m_accreted[star.ntpagb :] - ab2.ba.m_accreted[star.ntpagb :])
    / ab2.ba.m_accreted[star.ntpagb :],
)

fig.suptitle(
    "$M=1.8\\;M_\\odot$, $q=0.4293$, $R_\\textrm{RL}=500\\;R_\\odot$", fontsize=10
)
axs[1].set_ylabel("$|y_\\textrm{tp} - y_\\textrm{dredge}| / y_\\textrm{dredge}$")

axs[1].set_yscale("log")
axs[1].set_xlim(axs[1].get_xlim())
axs[1].set_ylim(axs[1].get_ylim())
(l1,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    label="method = TP",
)
(l2,) = axs[1].plot(
    [axs[1].get_xlim()[0], axs[1].get_xlim()[0]],
    [axs[1].get_ylim()[0], axs[1].get_ylim()[0]],
    c="C9",
    alpha=0.5,
    linewidth=3,
    label="method = $M_\\textrm{dup}$",
)
axs[1].set_ylim(1e-4)

for i in np.arange(-4, 4, 1):
    axs[1].axhline(10.0**i, c="C9", linewidth=0.75 / 2, zorder=-10)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlim(1.3262e9, 1.3279e9)

plt.xlabel("Star age (yr)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-tp-offset-ba-m1.8.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

df = AbundanceTables()

for model in grid.filter(m=1.8, R=500, q=0.4293):
    print(model)
    star = get_star(m=1.8)

ab = Abundances(model, df, method="tp offset")

for i, element in enumerate(list(ab.df.elements)[7:13]):

    print(element)
    plt.plot(
        ab.time[star.ntpagb :],
        ab.__getattr__(element).envelope[star.ntpagb :],
        c=f"C{i}",
    )
    plt.plot(
        ab.time[star.ntpagb :],
        ab.__getattr__(element).intershell[star.ntpagb :],
        c=f"C{i}",
        alpha=0.5,
        linewidth=3,
    )

plt.show()

# %%


fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.c.envelope[star.ntpagb :]
        o = ab.o.envelope[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c / o * 16 / 12, label=methods[j])
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_c12 / model.envelope_o16 * 16 / 12,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_c12[star.ntpagb : ab.simple_end_idx]
        / star.envelope_o16[star.ntpagb : ab.simple_end_idx]
        * 16
        / 12,
        c="C9",
    )
    ax.set_title(f"$M={m}\\;M_\\odot,\\;q=0.6,\\;R=1000\\;R_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("C/O (number ratio)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-co-ratio.pgf", format="pgf")
plt.show()
plt.close()

# %%
model.bulk_names
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

isos = [
    "envelope_he3",
    "envelope_he4",
    "envelope_c12",
    "envelope_c13",
    "envelope_n14",
    "envelope_o16",
    "envelope_ne20",
    "envelope_mg24",
]

init = []
for element in isos:
    if element in ["envelope_he4", "envelope_c13"]:
        init[-1] += star.__getattribute__(element)[star.ntpagb]
    else:
        init.append(star.__getattribute__(element)[star.ntpagb])


elements = ["He", "C", "N", "O", "Ne", "Mg"]
plt.plot(elements, init, label="MESA")


ab = Abundances(model, df, method="tp")

element_names = ["he", "c", "n", "o", "ne", "mg"]

init = []
for element in element_names:
    init.append(ab.__getattr__(element).envelope[star.ntpagb])

plt.plot(elements, init, label="Monash post-processing")

fig.legend(loc="outside upper center", ncols=2)
plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Element")
plt.ylabel("Mass fraction $X$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-initial-envelope-abundance.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

for m in np.arange(1, 3.1, 0.1):
    m = np.round(m, 1)
    star = get_star(m=m)
# %%


fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.c.envelope[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j])
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_c12,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_c12[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )
    ax.set_title(f"$M={m}\\;M_\\odot,\\;q=0.6,\\;R=1000\\;R_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("C/O (number ratio)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-c.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.n.envelope[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j])
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_n14,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_n14[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )
    ax.set_title(f"$M={m}\\;M_\\odot,\\;q=0.6,\\;R=1000\\;R_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("C/O (number ratio)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-c.pgf", format="pgf")
plt.show()
plt.close()


# %%

plt.plot(ab.time, np.cumsum(ab.m_dup))
plt.plot(star.age, star.m_DUP_time[0])
plt.plot(star.age, star.m_core)
plt.show()
# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.he.envelope[star.ntpagb :]
        c2 = ab.he.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_he3 + model.envelope_he4,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_he3[star.ntpagb : ab.simple_end_idx]
        + star.envelope_he4[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{He})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-he.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.c.envelope[star.ntpagb :]
        c2 = ab.c.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_c12 + model.envelope_c13,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_c12[star.ntpagb : ab.simple_end_idx]
        + star.envelope_c13[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{C})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-c.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.o.envelope[star.ntpagb :]
        c2 = ab.o.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_o16,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_o16[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{O})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-o.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.n.envelope[star.ntpagb :]
        c2 = ab.n.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_n14,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_n14[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{N})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-n.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.ne.envelope[star.ntpagb :]
        c2 = ab.ne.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_ne20,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_ne20[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{Ne})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-ne.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.mg.envelope[star.ntpagb :]
        c2 = ab.mg.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_mg24,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_mg24[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{Mg})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-mg.pgf", format="pgf")
plt.show()
plt.close()

# %%

model.header_names
# %%
