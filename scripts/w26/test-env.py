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
from scripts.general_utils.m_dup import compute_m_DUP, AbundanceTables, Abundances

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
m = grid.models[30]

df = AbundanceTables()
ab = Abundances(model=m, df=df)

# %%

fig, axs = plt.subplots(
    2, 2, sharey=True, figsize=set_size(full), constrained_layout=True
)
axs = axs.flatten()

masses = [None]
mass = None

for ax in axs:

    while mass in masses:
        m = grid.models[np.random.randint(len(grid.models))]
        mass = m.params["m"]

    masses.append(mass)
    print(masses)
    ab = Abundances(model=m, df=df)

    star = get_star(m=m.params["m"])

    for i, name in enumerate([df.elements["ba"]]):
        print(name)
        if name in [
            "neutron",
            "proton",
            "deuterium",
            "li",
            "be",
            "b",
        ]:
            continue

        element = getattr(ab, str(name))
        (l1,) = ax.plot(ab.time, element.envelope, label="envelope Ba abundance")
        (l2,) = ax.plot(ab.time, element.intershell, label="intershell Ba abundance")
        (l3,) = ax.plot(ab.time, element.m_accreted, label="accreted mass Ba")
        (l4,) = ax.plot(ab.time, np.cumsum(ab.m_dup), label="total dredged up mass")

    ax.set_xlim(star.age[star.ntpagb], star.age[-1])
    ax.text(
        0.05,
        0.95,
        f"$M={m.params["m"]:.1f}\\;M_\\odot$, $R_\\textrm{{RL,i}} = {m.params["R"]:.0f}$, $q={m.params["q"]:.2f}$",
        transform=ax.transAxes,
    )

fig.legend(loc="outside upper center", ncols=4, handles=[l1, l2, l3, l4])
ax.set_yscale("log")
ax.set_ylim(1e-12)


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Time (yr)", fontsize=10)
fig.supylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-examples.pgf", format="pgf")
plt.show()
plt.close()

# %%
for i, name in enumerate(list(ab.df.elements)[:15]):
    print(name)
    if name in [
        "neutron",
        "proton",
        "deuterium",
        "li",
        "be",
        "b",
    ]:
        continue

    element = getattr(ab, name)
    plt.plot(ab.time, element.envelope, label=element.name, c=f"C{i}")
    plt.plot(ab.time, element.intershell, c=f"C{i}", linewidth=3, alpha=0.5)

plt.legend()
plt.show()

# %%


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    ab = Abundances(model=r, df=df)
    return ab.ba.m_accreted[-1]
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column),
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
    label=r"$\Delta M_\textrm{Ba}$ accreted ($M_\odot$)",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w26-grid-ba.pgf", format="pgf")
plt.show()
plt.close()


# %%


def z(r):
    print(r)
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    ab = Abundances(model=r, df=df)
    return np.log10(ab.ba.envelope[-1])
    # return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column),
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
    label=r"$\log(X_\textrm{Ba})$",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w26-grid-ba-env.pgf", format="pgf")
plt.show()
plt.close()


# %%
