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
ab1 = (ab.MS_abundances + 1e-99) / np.sum(ab.MS_abundances)
plt.plot(ab1 / ab1)
ab2 = (ab.initial_envelope_abundances["massfrac"] + 1e-99) / np.sum(
    ab.initial_envelope_abundances["massfrac"]
)
plt.plot(ab1 / ab2)
plt.show()
# %%
m = grid.models[30]

df = AbundanceTables()
ab = Abundances(model=m, df=df)

print(m.params)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

ab1 = ab.MS_abundances / np.sum(ab.MS_abundances)
z = ab.elements_mass

ab2 = ab.initial_envelope_abundances["massfrac"] / np.sum(
    ab.initial_envelope_abundances["massfrac"]
)
plt.plot(z, ab2, label="Initial MS surface", linewidth=1)
plt.plot(z, ab.accreted_abundances, label="Accreted material", linewidth=1)
plt.plot(z, ab.all_intershell, label="Intershell material", linewidth=1)
plt.plot(z, ab1, label="Final MS surface", linewidth=1)

plt.ylim(1e-12, 1e0)

element_labels(fig, ab.elements_mass, ab.elements_name)

plt.yscale("log")
print(ab.mixing_mass, ab.total_mass_accreted)

plt.ylabel("$X$")

for z in [38, 40, 56, 58]:
    plt.axvline(z, c="C9", linewidth=0.75 / 2, zorder=-10)

axs.axvspan(38, 40, alpha=0.2, color="C9")
axs.axvspan(56, 58, alpha=0.2, color="C9")

plt.legend(loc="upper right")

plt.text(
    4,
    1e-12,
    f"$M = {m.params["m"]}\\;M_\\odot$,\n"
    + f"$R_\\textrm{{RL}} = {m.params["R"]}\\;R_\\odot$,\n"
    + f"$q = {m.params["q"]}$,\n"
    + f"$\\epsilon = {m.params["eps"]:.2f}$,\n"
    + f"$\\delta = {m.params["delta"]:.2f}$,\n"
    + f"$\\beta = {m.params["beta"]:.2f}$\n",
)

plt.savefig("/home/koen/LaTeX-setup/plots/w29-1-model-test.pgf", format="pgf")
plt.show()
plt.close()

# %%
df = AbundanceTables()


def get_ba(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    print(".", end="", flush=True)
    ab = Abundances(model=r, df=df)
    return ab.MS_abundances[52]


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_ba, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)

fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(c, ax=axs, label="$X(\\textrm{Ba})_\\textrm{MS}$")

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-large-grid.pgf", format="pgf")
plt.show()
plt.close()

# %%
const_mass = MesaGrid(f"{MASTER}/const-ba-star-mass-grid")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


rs = []
for m in const_mass.models:
    rs.append(m.params["R"])

rs = np.unique(rs)

for r in rs:
    first = True
    for m in const_mass.filter(R=r):
        if first:
            first = False
            print(m.params["R"])
            ab = Abundances(model=m, df=df)
            z = ab.elements_mass
            c = np.argwhere(m.params["R"] == rs)[0][0]
            plt.plot(
                z,
                ab.MS_abundances,
                c=f"C{c}",
                label=f"$R={r:.0f}\\;R_\\odot$",
                linewidth=1,
            )
            plt.plot(
                z,
                ab.all_intershell,
                c=f"C{c}",
                zorder=-10,
                linestyle="-.",
                linewidth=0.5,
            )
            plt.plot(
                z,
                ab.accreted_abundances,
                c=f"C{c}",
                zorder=-10,
                linestyle=":",
                linewidth=0.5,
            )
            print(m.star_2_mass[-1])
        continue

plt.plot(z, ab.initial_envelope_abundances["massfrac"], c=f"C9", zorder=-10)
element_labels(fig, ab.elements_mass, ab.elements_name)

plt.yscale("log")
plt.ylim(1e-11, 1e-4)
plt.xlim(35, 61)
for z in [38, 40, 56, 58]:
    plt.axvline(z, c="C9", linewidth=0.75 / 2, zorder=-10)

axs.axvspan(38, 40, alpha=0.2, color="C9")
axs.axvspan(56, 58, alpha=0.2, color="C9")

fig.legend(loc="outside upper center", ncols=3)

plt.ylabel("$X$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-test-multiple.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, r in enumerate(rs):
    first = True
    for m in const_mass.filter(R=r):
        if first:
            first = False
            axs[0].plot(m.age, m.rl_1, c=f"C{i}")
            arg = np.argmin(m.period_days)
            axs[1].axvline(m.age[arg], c=f"C{i}", linewidth=0.75)
            axs[0].axvline(m.age[arg], c=f"C{i}", linewidth=0.75)

star = get_star(m=1.5)
axs[0].plot(
    star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", zorder=-10
)
axs[1].plot(star.age[star.ntpagb :], star.m_DUP_time[0][star.ntpagb :], c="C9")

axs[1].set_yscale("log")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-moment-of-interaction.pgf", format="pgf")
plt.show()
plt.close()

# %%
df = AbundanceTables()


def get_ba(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    print(".", end="", flush=True)
    ab = Abundances(model=r, df=df)
    return (
        ab.MS_abundances[52] / np.array(ab.initial_envelope_abundances["massfrac"])[52]
    )


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_ba, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)

# %%
fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(
    c, ax=axs, label="$X(\\textrm{Ba})_\\textrm{MS} / X(\\textrm{Ba})_\\textrm{MS, i}$"
)

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-large-grid-2.pgf", format="pgf")
plt.show()
plt.close()
# %%
df = AbundanceTables()


def get_ba(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df, full_mixing=True)
    ab2 = Abundances(model=r, df=df, full_mixing=False)
    return ab.MS_abundances[52] / ab2.MS_abundances[52]


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_ba, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)


# %%

fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(
    c,
    ax=axs,
    label="$X(\\textrm{Ba})_\\textrm{MS, full mixing} / X(\\textrm{Ba})_\\textrm{MS, informed mixing}$",
)

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-large-grid-3.pgf", format="pgf")
plt.show()
plt.close()


# %%
