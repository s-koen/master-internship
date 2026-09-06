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

plt.xlabel("")
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

plt.xlabel("")
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

plt.xlabel("")
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
