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

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

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

for m in grid.models:
    print(m.period_days[-1], m.params["delta"], m.params["R"])
# %%
print(grid.axes["q"])

for m in grid.models:
    if m.params["q"] == 0.7:
        plt.scatter(m.params["R"], m.s)

plt.show()
# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
rs = np.logspace(np.log10(250), np.log10(1000), 500)

mi = 1.4
q = 0.85


def run_model(ri):
    a_init = inv_roche_lobe(ri, q)

    star = get_star(m=mi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, q, a_init, simple_only=True
    )

    bin = Bins[0]
    print(f"r = {ri:.2f}, m = {mi}, len = {len(bin.age)}")

    ab = Abundances(
        None,
        df,
        mass=mi,
        sb=bin,
        sampling=1,
        mass_transfer_efficiency=0.1,
    )

    return get_s(ab)


def run_model_2(ri):
    a_init = inv_roche_lobe(ri, q)

    star = get_star(m=mi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, q, a_init, simple_only=True
    )

    bin = Bins[0]
    print(f"r = {ri:.2f}, m = {mi}, len = {len(bin.age)}")

    ab = Abundances(
        None,
        df,
        mass=mi,
        sb=bin,
        sampling=1,
        mass_transfer_efficiency=0,
    )

    return get_s(ab)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abs1 = list(executor.map(run_model, rs))

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abs2 = list(executor.map(run_model_2, rs))
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
print(grid.axes["q"])

for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            marker = "o"
        case 0.20:
            marker = "s"
        case 0.30:
            marker = "^"
        case _:
            print(np.round(m.params["f_delta"], 2))
            marker = "red"

    if m.params["q"] == 0.85:
        if m.params["eps"] == 0:
            plt.scatter(m.params["R"], m.s, marker=marker, c="C0", s=15)
        else:
            plt.scatter(m.params["R"], m.s, marker=marker, c="C1", s=15)


plt.plot(rs, abs2, zorder=-10)
plt.plot(rs, abs1, zorder=-11)

from matplotlib.lines import Line2D

legend_elements = [
    # marker shapes: f_delta
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="gray",
        linestyle="none",
        label=r"$f_\delta = 0.10$",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="gray",
        linestyle="none",
        label=r"$f_\delta = 0.20$",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="gray",
        linestyle="none",
        label=r"$f_\delta = 0.30$",
    ),
    # colors: eps
    Line2D(
        [0, 0],
        [0, 0],
        color="C0",
        label=r"$\epsilon = 0$",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color="C1",
        label=r"$\epsilon = 0.1$",
    ),
]

axs.legend(handles=legend_elements)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-M1.4-compare-s.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][2]:
        continue
    if m.params["q"] != 0.85:
        continue
    plt.plot(m.age, m.binary_separation, c=color)

star = get_star(m=1.4)

a_init = inv_roche_lobe(grid.axes["R"][2], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)

bin = Bins[0]
plt.plot(bin.age[star.ntpagb :], bin.a[star.ntpagb :])

# plt.plot(star.age[star.ntpagb:], 10**star.log_R[star.ntpagb:])
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][2]:
        continue
    if m.params["q"] != 0.85:
        continue
    if m.params["eps"] == 0:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=2, alpha=0.5)
        axs[1].plot(m.age, m.star_2_mass - bin.m2[0], c=color, linewidth=2, alpha=0.5)
    else:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=0.75)
        axs[1].plot(m.age, m.star_2_mass - bin.m2[0], c=color, linewidth=0.75)

star = get_star(m=1.4)
axs[0].plot(star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9")
axs[0].set_ylim(287.51074231184987, 578.9699974276443)

a_init = inv_roche_lobe(584, 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)

bin = Bins[0]
rl_o = roche_lobe(bin.m1 / bin.m2) * bin.a
prev_len = len(bin.age)
bin_ages = bin.age[star.ntpagb :]

a_init = inv_roche_lobe(486, 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)
bin = Bins[0]
rl = roche_lobe(bin.m1 / bin.m2) * bin.a
ind = np.argwhere(star.age > bin.age[-1])[0][0]
ymin = np.append(
    rl[star.ntpagb :], (10**star.log_R)[ind : ind + prev_len - len(bin.age)]
)
print(len(bin_ages), len(ymin), len(rl_o[star.ntpagb :]))
axs[0].fill_between(
    bin_ages, ymin, rl_o[star.ntpagb :], color="C9", alpha=0.2, rasterized=True
)

a_init = inv_roche_lobe(grid.axes["R"][2], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)


bin = Bins[0]
rl = roche_lobe(bin.m1 / bin.m2) * bin.a
axs[0].plot(bin.age[star.ntpagb :], rl[star.ntpagb :], c="C9", zorder=-1)
bin.m2[-1] += (star.m_env[ind]) * 0.1
axs[1].plot(
    bin.age[star.ntpagb :], bin.m2[star.ntpagb :] - bin.m2[0], c="C9", zorder=-1
)

plt.xlim(2821534847.201768, 2821669242.3846593)

# axs[0].text(642000+2.821e9, 427, "$f_\\delta = 0.1$", c="C0", fontsize=8)
# axs[0].text(636000+2.821e9, 389, "$f_\\delta = 0.2$", c="C1", fontsize=8)
# axs[0].text(632000+2.821e9, 322, "$f_\\delta = 0.3$", c="C2", fontsize=8)

axs[0].text(657000 + 2.821e9, 427, "$f_\\delta = 0.1$", c="C0", fontsize=8)
axs[0].text(657000 + 2.821e9, 389, "$f_\\delta = 0.2$", c="C1", fontsize=8)
axs[0].text(657000 + 2.821e9, 322, "$f_\\delta = 0.3$", c="C2", fontsize=8)

axs[0].annotate(
    "Roche lobe",
    xy=(580000 + 2.821e9, 485),
    xycoords="data",
    xytext=(-5, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    color="C9",
    fontsize=8,
)

axs[0].annotate(
    "Star",
    xy=(600000 + 2.821e9, 330),
    xycoords="data",
    xytext=(-20, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    color="C9",
    fontsize=8,
)


axs[1].annotate(
    "Simple binary",
    xy=(632000 + 2.821e9, 0.115),
    xycoords="data",
    xytext=(25, -1.75),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    color="C9",
    fontsize=8,
)

axs[1].text(645000 + 2.821e9, 0.133, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[1].text(645000 + 2.821e9, 0.0975, "$\\varepsilon = 0.0$", c="C0", fontsize=8)


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].set_ylabel("$\\Delta M_\\textrm{acc}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-radius-3.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    7, 1, sharex=True, figsize=set_size(column, height=3.25), constrained_layout=True
)


def get_massfrac(element):
    try:
        return asplund.elements[element].massfrac
    except:
        return 1e-99


def compute_MS_spectroscopic(ab, ab_H, index, abs):
    N_i = ab / abs.atomic_weight[index]
    N_H = ab_H / abs.atomic_weight[0]
    eps = np.log10(N_i / N_H) + 12

    return eps


def compute_iron_abundance(spec, spec_iron, index, abs):

    asplund = Asplund(he_method="karakas")

    specs = spec
    fe_star = spec_iron
    fe_sun = asplund.elements[26].abundance

    fe_abundance_sun = asplund.elements[abs.elements_mass[index]].abundance - fe_sun
    fe_abundance_star = specs - fe_star

    fe_abundance_sun = np.array(fe_abundance_sun)
    fe_abundance_star = np.array(fe_abundance_star)

    return fe_abundance_star - fe_abundance_sun


for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][0]:
        continue
    if m.params["q"] != 0.85:
        continue

    abs = Abundances(m, df, save_full=True, sampling=1)
    ms = abs.initial_envelope_abundances.copy()
    asplund = Asplund(he_method="karakas")

    ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

    abbies_MS = 0
    abss = []
    epss = []
    fes = []

    for i in [52, 53, 54, 56, 34, 35, 36]:
        abbies_MS += ms.iloc[i]["massfrac"]

    abbies_MS *= abs.mixing_mass

    abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
    abss.append(
        (
            np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
            + ms.iloc[52]["massfrac"] * abs.mixing_mass
        )
        / (np.cumsum(abs.dm_acc) + abs.mixing_mass)
    )
    for i in [53, 54, 56, 34, 35, 36]:
        abss.append(
            (
                np.cumsum(abs.envelope[:, i] * abs.dm_acc)
                + ms.iloc[i]["massfrac"] * abs.mixing_mass
            )
            / (np.cumsum(abs.dm_acc) + abs.mixing_mass)
        )
        abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
    abss_H = (
        np.cumsum(abs.envelope[:, 0] * abs.dm_acc)
        + ms.iloc[0]["massfrac"] * abs.mixing_mass
    ) / (np.cumsum(abs.dm_acc) + abs.mixing_mass)

    abss_fe = (
        np.cumsum(abs.envelope[:, 22] * abs.dm_acc)
        + ms.iloc[22]["massfrac"] * abs.mixing_mass
    ) / (np.cumsum(abs.dm_acc) + abs.mixing_mass)
    print(abs.elements_name[22])

    for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
        epss.append(compute_MS_spectroscopic(abss[counter], abss_H, i, abs))

    epss_Fe = compute_MS_spectroscopic(abss_fe, abss_H, 22, abs)

    for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
        fes.append(compute_iron_abundance(epss[counter], epss_Fe, i, abs))

    s = np.mean(fes, axis=0)

    if m.params["eps"] == 0:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=2, alpha=0.5)
        axs[1].plot(m.age, m.star_2_mass - bin.m2[0], c=color, linewidth=2, alpha=0.5)
        axs[2].plot(abs.time, abbies, c=color, linewidth=2, alpha=0.5)
        axs[3].plot(
            abs.time, abbies / np.cumsum(abs.dm_acc), c=color, linewidth=2, alpha=0.5
        )
        axs[4].plot(
            abs.time,
            abbies / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
            c=color,
            linewidth=2,
            alpha=0.5,
        )
        axs[5].plot(
            abs.time,
            (abbies + abbies_MS) / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
            c=color,
            linewidth=2,
            alpha=0.5,
        )
        axs[6].plot(abs.time, s, c=color, linewidth=2, alpha=0.5)
        axs[6].scatter(abs.time[-1], m.s, c=color, alpha=0.5)
    else:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=0.75)
        axs[1].plot(m.age, m.star_2_mass - bin.m2[0], c=color, linewidth=0.75)
        axs[2].plot(
            abs.time,
            abbies,
            c=color,
            linewidth=0.75,
        )
        axs[3].plot(
            abs.time,
            abbies / (np.cumsum(abs.dm_acc)),
            c=color,
            linewidth=0.75,
        )
        axs[4].plot(
            abs.time,
            abbies / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
            c=color,
            linewidth=0.75,
        )
        axs[5].plot(
            abs.time,
            (abbies + abbies_MS) / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
            c=color,
            linewidth=0.75,
        )
        axs[6].plot(
            abs.time,
            s,
            c=color,
            linewidth=0.75,
        )
        axs[6].scatter(
            abs.time[-1],
            m.s,
            c=color,
        )

star = get_star(m=1.4)
axs[0].plot(
    star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", zorder=-1000
)
axs[0].set_ylim(287.51074231184987, 578.9699974276443)

a_init = inv_roche_lobe(386.8, 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)

bin = Bins[0]
rl_o = roche_lobe(bin.m1 / bin.m2) * bin.a
prev_len = len(bin.age)
bin_ages = bin.age[star.ntpagb :]

a_init = inv_roche_lobe(349.5, 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)
bin = Bins[0]
rl = roche_lobe(bin.m1 / bin.m2) * bin.a
ind = np.argwhere(star.age > bin.age[-1])[0][0]
ymin = np.append(
    rl[star.ntpagb :], (10**star.log_R)[ind : ind + prev_len - len(bin.age)]
)
print(len(bin_ages), len(ymin), len(rl_o[star.ntpagb :]))
axs[0].fill_between(
    bin_ages, ymin, rl_o[star.ntpagb :], color="C9", alpha=0.2, rasterized=True
)

a_init = inv_roche_lobe(grid.axes["R"][0], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)


bin = Bins[0]
rl = roche_lobe(bin.m1 / bin.m2) * bin.a
ind = np.argwhere(star.age > bin.age[-1])[0][0]
axs[0].plot(bin.age[star.ntpagb :], rl[star.ntpagb :], c="C9", zorder=-1)
bin.m2[-1] += (star.m_env[ind]) * 0.1
axs[1].plot(
    bin.age[star.ntpagb :], bin.m2[star.ntpagb :] - bin.m2[0], c="C9", zorder=-1000
)

abs = Abundances(
    None, df, sb=bin, sampling=1, mass=1.4, mass_transfer_efficiency=0.1, save_full=True
)

ms = abs.initial_envelope_abundances.copy()
asplund = Asplund(he_method="karakas")

ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

abbies_MS = 0
abss = []
epss = []
fes = []

for i in [52, 53, 54, 56, 34, 35, 36]:
    abbies_MS += ms.iloc[i]["massfrac"]

abbies_MS *= abs.mixing_mass

abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
abss.append(
    (
        np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
        + ms.iloc[52]["massfrac"] * abs.mixing_mass
    )
    / (np.cumsum(abs.dm_acc) + abs.mixing_mass)
)
for i in [53, 54, 56, 34, 35, 36]:
    abss.append(
        (
            np.cumsum(abs.envelope[:, i] * abs.dm_acc)
            + ms.iloc[i]["massfrac"] * abs.mixing_mass
        )
        / (np.cumsum(abs.dm_acc) + abs.mixing_mass)
    )
    abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
abss_H = (
    np.cumsum(abs.envelope[:, 0] * abs.dm_acc)
    + ms.iloc[0]["massfrac"] * abs.mixing_mass
) / (np.cumsum(abs.dm_acc) + abs.mixing_mass)

abss_fe = (
    np.cumsum(abs.envelope[:, 22] * abs.dm_acc)
    + ms.iloc[22]["massfrac"] * abs.mixing_mass
) / (np.cumsum(abs.dm_acc) + abs.mixing_mass)

for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
    epss.append(compute_MS_spectroscopic(abss[counter], abss_H, i, abs))

epss_Fe = compute_MS_spectroscopic(abss_fe, abss_H, 22, abs)

for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
    fes.append(compute_iron_abundance(epss[counter], epss_Fe, i, abs))

s = np.mean(fes, axis=0)

axs[2].plot(
    abs.time,
    abbies,
    c="C9",
    linewidth=0.75,
)
axs[3].plot(
    abs.time,
    abbies / (np.cumsum(abs.dm_acc)),
    c="C9",
    linewidth=0.75,
)
axs[4].plot(
    abs.time,
    (abbies) / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
    c="C9",
    linewidth=0.75,
)
axs[5].plot(
    abs.time,
    (abbies + abbies_MS) / (np.cumsum(abs.dm_acc) + abs.mixing_mass),
    c="C9",
    linewidth=0.75,
)
axs[6].plot(
    abs.time,
    s,
    c="C9",
    linewidth=0.75,
)
axs[6].scatter(
    abs.time[-1],
    get_s(abs),
    c="C9",
)

plt.xlim(2820936572.6464543, 2821440289.655047)
axs[0].set_ylim(150, 400)

# axs[0].text(400000+2.821e9, 427, "$f_\\delta = 0.1$", c="C0", fontsize=8)
# axs[0].text(400000+2.821e9, 389, "$f_\\delta = 0.2$", c="C1", fontsize=8)
# axs[0].text(400000+2.821e9, 322, "$f_\\delta = 0.3$", c="C2", fontsize=8)

axs[0].text(355000 + 2.821e9, 375, "$f_\\delta = 0.1$", c="C0", fontsize=8)
axs[0].text(255000 + 2.821e9, 295, "$f_\\delta = 0.2$", c="C1", fontsize=8)
axs[0].text(100000 + 2.821e9, 250, "$f_\\delta = 0.3$", c="C2", fontsize=8)

axs[0].annotate(
    "Roche lobe",
    xy=(2.821e9, 340),
    xycoords="data",
    xytext=(-20, -20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    color="C9",
    fontsize=8,
)

axs[0].annotate(
    "Star",
    xy=(2.821e9, 190),
    xycoords="data",
    xytext=(-8, 17),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    color="C9",
    fontsize=8,
)


axs[1].text(350000 + 2.821e9, 0.0935, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[1].text(350000 + 2.821e9, 0.055, "Simple binary", c="C9", fontsize=8)
axs[1].text(350000 + 2.821e9, 0.0175, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[2].text(350000 + 2.821e9, 8.9e-9, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[2].text(350000 + 2.821e9, 3e-9, "Simple binary", c="C9", fontsize=8)
axs[2].text(350000 + 2.821e9, 5.1e-9, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[3].text(350000 + 2.821e9, 2.9e-7, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[3].text(350000 + 2.821e9, 0.5e-7, "Simple binary", c="C9", fontsize=8)
axs[3].text(350000 + 2.821e9, 0.9e-7, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[4].text(350000 + 2.821e9, 1.6e-8, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[4].text(350000 + 2.821e9, 0.5e-8, "Simple binary", c="C9", fontsize=8)
axs[4].text(350000 + 2.821e9, 1.4e-8, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[5].text(350000 + 2.821e9, 5.65e-8, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[5].text(350000 + 2.821e9, 4.625e-8, "Simple binary", c="C9", fontsize=8)
axs[5].text(350000 + 2.821e9, 5.3e-8, "$\\varepsilon = 0.1$", c="C0", fontsize=8)
axs[6].text(350000 + 2.821e9, 0.17, "$\\varepsilon = 0.0$", c="C0", fontsize=8)
axs[6].text(350000 + 2.821e9, 0.07, "Simple binary", c="C9", fontsize=8)
axs[6].text(350000 + 2.821e9, 0.13, "$\\varepsilon = 0.1$", c="C0", fontsize=8)


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].set_ylabel("$\\Delta M_\\textrm{acc}$ ($M_\\odot$)")
axs[2].set_ylabel("$\\Delta M_\\textrm{s}$ ($M_\\odot$)")
axs[3].set_ylabel("$\\langle X\\rangle(s, \\textrm{accr. mat.})_{t=t}$")
axs[4].set_ylabel("$\\langle X\\rangle(s, \\textrm{MS})_{t=t}$, $X(s,i) = 0$")
axs[5].set_ylabel(
    "$\\langle X\\rangle(s, \\textrm{MS})_{t=t}$, $X(s,i) = X(s)_\\textrm{Asplund}$"
)
axs[6].set_ylabel("[s/Fe]$_{t=t}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-radius-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

df = AbundanceTables()
for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][0]:
        continue
    if m.params["q"] != 0.85:
        continue

    abs = Abundances(m, df, save_full=True, sampling=1)
    if m.params["eps"] == 0:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=2, alpha=0.5)
        abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
        for i in [53, 54, 56, 34, 35, 36]:
            abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)

        axs[1].plot(
            abs.time,
            abbies / (abs.total_mass_accreted + abs.mixing_mass),
            c=color,
            linewidth=2,
            alpha=0.5,
        )
    else:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=0.75)
        abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
        for i in [53, 54, 56, 34, 35, 36]:
            abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
        axs[1].plot(
            abs.time,
            abbies / (abs.total_mass_accreted + abs.mixing_mass),
            c=color,
            linewidth=0.75,
        )

a_init = inv_roche_lobe(grid.axes["R"][0], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)


bin = Bins[0]
abs = Abundances(
    None, df, sb=bin, sampling=1, mass=1.4, mass_transfer_efficiency=0.1, save_full=True
)

abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
for i in [53, 54, 56, 34, 35, 36]:
    abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
axs[1].plot(
    abs.time,
    abbies / (abs.total_mass_accreted + abs.mixing_mass),
    c="C9",
    linewidth=0.75,
)

plt.show()
# %%
["ba", "la", "ce", "nd", "sr", "y", "zr"]
for i, n in enumerate(abs.elements_name):
    print(i, n)
# %%

52, 53, 54, 56, 34, 35, 36


# %%
def get_massfrac(element):
    try:
        return asplund.elements[element].massfrac
    except:
        return 1e-99


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

df = AbundanceTables()
for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][0]:
        continue
    if m.params["q"] != 0.85:
        continue

    abs = Abundances(m, df, save_full=True, sampling=1)

    ms = abs.initial_envelope_abundances.copy()
    asplund = Asplund(he_method="karakas")

    ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

    abbies_MS = 0

    for i in [52, 53, 54, 56, 34, 35, 36]:
        abbies_MS += ms.iloc[i]["massfrac"]

    abbies_MS *= abs.mixing_mass

    if m.params["eps"] == 0:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=2, alpha=0.5)
        abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
        for i in [53, 54, 56, 34, 35, 36]:
            abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)

        axs[1].plot(
            abs.time,
            (abbies + abbies_MS) / (abs.total_mass_accreted + abs.mixing_mass),
            c=color,
            linewidth=2,
            alpha=0.5,
        )
    else:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=0.75)
        abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
        for i in [53, 54, 56, 34, 35, 36]:
            abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
        axs[1].plot(
            abs.time,
            (abbies + abbies_MS) / (abs.total_mass_accreted + abs.mixing_mass),
            c=color,
            linewidth=0.75,
        )

a_init = inv_roche_lobe(grid.axes["R"][0], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)


bin = Bins[0]
abs = Abundances(
    None, df, sb=bin, sampling=1, mass=1.4, mass_transfer_efficiency=0.1, save_full=True
)

abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
for i in [53, 54, 56, 34, 35, 36]:
    abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)


asplund = Asplund(he_method="karakas")

ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

abbies_MS = 0

for i in [52, 53, 54, 56, 34, 35, 36]:
    abbies_MS += ms.iloc[i]["massfrac"]

abbies_MS *= abs.mixing_mass


axs[1].plot(
    abs.time,
    (abbies + abbies_MS) / (abs.total_mass_accreted + abs.mixing_mass),
    c="C9",
    linewidth=0.75,
)


plt.show()
# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

df = AbundanceTables()
for m in grid.models:
    match np.round(m.params["f_delta"], 2):
        case 0.10:
            color = "C0"
        case 0.20:
            color = "C1"
        case 0.30:
            color = "C2"
        case _:
            print(np.round(m.params["f_delta"], 2))
            color = "red"

    if m.params["R"] != grid.axes["R"][0]:
        continue
    if m.params["q"] != 0.85:
        continue

    abs = Abundances(m, df, save_full=True, sampling=1)

    ms = abs.initial_envelope_abundances.copy()
    asplund = Asplund(he_method="karakas")

    ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

    abbies_MS = 0
    abss = []
    epss = []
    fes = []

    for i in [52, 53, 54, 56, 34, 35, 36]:
        abbies_MS += ms.iloc[i]["massfrac"]

    abbies_MS *= abs.mixing_mass

    abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
    abss.append(
        (
            np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
            + ms.iloc[52]["massfrac"] * abs.mixing_mass
        )
        / (abs.total_mass_accreted + abs.mixing_mass)
    )
    for i in [53, 54, 56, 34, 35, 36]:
        abss.append(
            (
                np.cumsum(abs.envelope[:, i] * abs.dm_acc)
                + ms.iloc[i]["massfrac"] * abs.mixing_mass
            )
            / (abs.total_mass_accreted + abs.mixing_mass)
        )
        abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)
    abss_H = (
        np.cumsum(abs.envelope[:, 0] * abs.dm_acc)
        + ms.iloc[0]["massfrac"] * abs.mixing_mass
    ) / (abs.total_mass_accreted + abs.mixing_mass)

    abss_fe = (
        np.cumsum(abs.envelope[:, 22] * abs.dm_acc)
        + ms.iloc[22]["massfrac"] * abs.mixing_mass
    ) / (abs.total_mass_accreted + abs.mixing_mass)
    print(abs.elements_name[22])

    for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
        epss.append(compute_MS_spectroscopic(abss[counter], abss_H, i, abs))

    epss_Fe = compute_MS_spectroscopic(abss_fe, abss_H, 22, abs)

    for counter, i in enumerate([52, 53, 54, 56, 34, 35, 36]):
        fes.append(compute_iron_abundance(epss[counter], epss_Fe, i, abs))

    s = np.mean(fes, axis=0)

    if m.params["eps"] == 0:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=2, alpha=0.5)

        axs[1].plot(
            abs.time,
            s,
            c=color,
            linewidth=2,
            alpha=0.5,
        )
        axs[1].scatter(abs.time[-1], m.s, c=color, alpha=0.5)
    else:
        axs[0].plot(m.age, m.rl_1, c=color, linewidth=0.75)
        axs[1].plot(
            abs.time,
            s,
            c=color,
            linewidth=0.75,
        )
        axs[1].scatter(abs.time[-1], m.s, c=color)

a_init = inv_roche_lobe(grid.axes["R"][0], 0.85)
star, Options, q_init, a_init, e_init, Bins = call_evolution(
    star, q, a_init, simple_only=True
)


bin = Bins[0]
abs = Abundances(
    None, df, sb=bin, sampling=1, mass=1.4, mass_transfer_efficiency=0.1, save_full=True
)

abbies = np.cumsum(abs.envelope[:, 52] * abs.dm_acc)
for i in [53, 54, 56, 34, 35, 36]:
    abbies += np.cumsum(abs.envelope[:, i] * abs.dm_acc)


asplund = Asplund(he_method="karakas")

ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

abbies_MS = 0

for i in [52, 53, 54, 56, 34, 35, 36]:
    abbies_MS += ms.iloc[i]["massfrac"]

abbies_MS *= abs.mixing_mass


axs[1].plot(
    abs.time,
    (abbies + abbies_MS) / (abs.total_mass_accreted + abs.mixing_mass),
    c="C9",
    linewidth=0.75,
)


plt.show()
# %%

for model in grid.models:
    plt.scatter(model.star_2_mass[-1], model.s)
plt.show()
# %%

fs = [0.1, 0.2, 0.3]
markers = ["^", "s", "*"]

for i, f in enumerate(fs):
    x = []
    y = []
    z = []
    for model in grid.filter(f_delta=f):
        x.append(model.star_2_mass[-1] - model.star_2_mass[0])
        y.append(model.s)
        z.append(model.params["R"])

    plt.scatter(x, y, c=np.log10(z), marker=markers[i], cmap="viridis")


plt.show()
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

plt.xlim(200, 10000)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/period-sFe-period.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

from matplotlib.lines import Line2D

fs = [0.0, 0.1]
markers = ["^", "s", "*"]
cmap = plt.get_cmap("viridis")

for f, marker in zip(fs, markers):
    x = []
    y = []
    z = []
    for m in grid.filter(eps=f):
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
        label=f"$R = {R:g}\\,R_\\odot$",
    )
    for R in Rs
]

fig.legend(handles=marker_handles, loc="outside upper center", ncols=3)

plt.legend(handles=color_handles, frameon=False)

plt.xscale("log")

plt.xlim(200, 10000)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
# plt.savefig("/home/koen/LaTeX-setup/plots/period-sFe-period.pgf", format="pgf")
plt.show()
plt.close()

# %%

for m in grid.models:
    if m.params["R"] != grid.axes["R"][1]:
        continue
    if m.envelope_mass[-1] > 0.02:
        continue
    if not np.isclose(m.params["f_delta"], 0.1):
        continue
    plt.plot(m.age, m.rl_1, label=m.params["q"])

plt.legend()
plt.show()
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
        z.append(m.params["q"])

    plt.scatter(x, y, c=z, marker=marker, cmap=cmap, s=15)

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
Rs = sorted({m.params["q"] for m in grid.models})
norm = plt.Normalize(min(Rs), max(Rs))

color_handles = [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        color=cmap(norm(R)),
        markersize=7,
        label=f"$q = {R:.2f}$",
    )
    for R in Rs
]

fig.legend(handles=marker_handles, loc="outside upper center", ncols=3)

plt.legend(handles=color_handles, frameon=False)

plt.xscale("log")

plt.xlim(200, 10000)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Period (days)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/period-sFe-period.pgf", format="pgf")
plt.show()
plt.close()
