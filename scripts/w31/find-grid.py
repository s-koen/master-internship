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
from concurrent.futures import ProcessPoolExecutor

# %%

df = AbundanceTables()


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


# %%


grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
df = AbundanceTables()

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue
    print(m.params["m"])
    mass = m.params["m"]
    ab = Abundances(model=m, df=df, sampling=1)
    m.ab = ab.iron_abundance
    m.elements_mass = ab.elements_mass
    m.elements_name = ab.elements_name
    m.t_av = ab.t_av
    m.m_dup_av = ab.m_dup_av


def get_ls_model(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 3


def get_hs_model(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += r.ab[i]

    return fe / 4


def get_s_model(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 7


# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
q = 0.6


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
        mass_transfer_efficiency=0.25,
    )

    return get_s(ab)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abs = list(executor.map(run_model, rs))
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x_m = []
y_m = []
for m in grid.models:

    x_m.append(m.params["R"])
    y_m.append(get_s_model(m))
x_m = np.array(x_m)
y_m = np.array(y_m)
idx = np.argsort(x_m)

x_m = x_m[idx]
y_m = y_m[idx]


plt.plot(rs, abs, label="Simple", c="C0", linewidth=1, zorder=100)
plt.plot(x_m, y_m, label="Detailed", c="C1", alpha=0.5, linewidth=3, zorder=-1)

fig.legend(loc="outside upper center", ncols=2)

plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)
plt.axvline(1193, linewidth=0.75, c="C9", zorder=-1)
plt.axvline(536, linewidth=0.75, c="C9", zorder=-1)
plt.annotate(
    "RLOF",
    xy=(536, 1.0),
    xycoords="data",
    xytext=(-80, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
),
plt.annotate(
    "RLOF + Wind",
    xy=(1193, 1.0),
    xycoords="data",
    xytext=(-64, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="w", zorder=-10, linewidth=0.75),
    c="C9",
)
plt.annotate(
    "Wind",
    xy=(1193, 1.0),
    xycoords="data",
    xytext=(40, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)


plt.annotate(
    "barium star",
    xy=(799.65, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(799.65, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-simple-vs-detailed.pgf", format="pgf")
plt.show()
plt.close()


# %%

10 ** ((np.log10(1193) + np.log10(536)) / 2)
# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
q = 0.6


def run_model(ri):
    a_init = inv_roche_lobe(ri, q)

    star = get_star(m=mi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, q, a_init, simple_only=True
    )

    bin = Bins[0]
    print(f"r = {ri:.2f}, m = {mi}, len = {len(bin.age)}")

    eps = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    ss = []
    for ep in eps:
        ab = Abundances(
            None,
            df,
            mass=mi,
            sb=bin,
            sampling=1,
            mass_transfer_efficiency=ep,
        )
        ss.append(get_s(ab))

    return ss


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abss = list(executor.map(run_model, rs))

with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances.pkl",
    "wb",
) as f:
    pickle.dump(abss, f, protocol=pickle.HIGHEST_PROTOCOL)
print(abss)
# %%
with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances.pkl",
    "rb",
) as f:
    abss = pickle.load(f)
print(abss)

# %%
#

abss = np.array(abss)
epss = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(min(np.log10(epss)), max(np.log10(epss)))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i in range(8)[::-1]:
    plt.plot(rs, abss[:, i], c=cmap(norm(np.log10(epss[i]))))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(
    sm, ax=plt.gca(), location="top", orientation="horizontal", aspect=50
)
cbar.set_ticks(ticks=np.log10(epss), labels=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
cbar.set_label("$\\varepsilon$")


plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)
plt.axvline(1193, linewidth=0.75, c="C9", zorder=-1)
plt.axvline(536, linewidth=0.75, c="C9", zorder=-1)
plt.annotate(
    "RLOF",
    xy=(536, 1.7),
    xycoords="data",
    xytext=(-80, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
),
plt.annotate(
    "RLOF + Wind",
    xy=(1193, 1.7),
    xycoords="data",
    xytext=(-64, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="w", zorder=-10, linewidth=0.75),
    c="C9",
)
plt.annotate(
    "Wind",
    xy=(1193, 1.7),
    xycoords="data",
    xytext=(40, -3),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)


plt.annotate(
    "barium star",
    xy=(799.65, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(799.65, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)


plt.ylim(plt.gca().get_ylim()[0], 1.75)

axs.spines[["right", "top"]].set_visible(False)
plt.xscale("log")
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-change-eps.pgf", format="pgf")
plt.show()
plt.close()

# %%

for mi in ms:
    star = get_star(m=mi)
    plt.plot(star.m_env, 10**star.log_R)

plt.show()
# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
qs = np.arange(0.1, 1.01, 0.1)


def run_model(ri):

    ss = []
    star = get_star(m=mi)
    for q in qs:

        a_init = inv_roche_lobe(ri, q)
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
            mass_transfer_efficiency=0.25,
        )
        ss.append(get_s(ab))

    return ss


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abss = list(executor.map(run_model, rs))

with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-q.pkl",
    "wb",
) as f:
    pickle.dump(abss, f, protocol=pickle.HIGHEST_PROTOCOL)
print(abss)
# %%
rs = np.logspace(np.log10(125), np.log10(12500), 101)
qs = np.arange(0.1, 1.01, 0.1)
with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-q.pkl",
    "rb",
) as f:
    abss = pickle.load(f)
print(abss)


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(0.2, 1)
cmap = plt.cm.viridis
# color = cmap(norm(x))


abss = np.array(abss)

for i in range(1, 10)[::-1]:
    plt.plot(rs, abss[:, i], c=cmap(norm(qs[i])))
axs.spines[["right", "top"]].set_visible(False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$q_\textrm{i} = M_\textrm{a,i} / M_\textrm{d,i}$")

plt.annotate(
    "barium star",
    xy=(600.65, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(600.65, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)

plt.xscale("log")
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-change-q.pgf", format="pgf")
plt.show()
plt.close()
# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.3)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
qs = np.arange(0.1, 1.01, 0.1)
q = 0.5


def run_model(ri):

    ss = []
    for mi in ms:
        star = get_star(m=mi)

        a_init = inv_roche_lobe(ri, q)
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
            mass_transfer_efficiency=0.25,
        )
        ss.append(get_s(ab))

    return ss


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abss = list(executor.map(run_model, rs))

with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m.pkl",
    "wb",
) as f:
    pickle.dump(abss, f, protocol=pickle.HIGHEST_PROTOCOL)
print(abss)
# %%
ms = np.arange(1, 3.01, 0.3)
rs = np.logspace(np.log10(125), np.log10(12500), 101)
with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m.pkl",
    "rb",
) as f:
    abss = pickle.load(f)
print(abss)


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 2.8)
cmap = plt.cm.viridis
cmap.levels = 7

abss = np.array(abss)

for i in range(0, 7)[::-1]:
    plt.plot(rs, abss[:, i], c=cmap(norm(ms[i])))
axs.spines[["right", "top"]].set_visible(False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca(), ticks=ms)
cbar.set_label(r"$M_\textrm{TPAGB,i}$ ($M_\odot$)")

plt.annotate(
    "barium star",
    xy=(170, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(170, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)

axs.spines[["right", "top"]].set_visible(False)
plt.xscale("log")
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-change-m.pgf", format="pgf")
plt.show()
plt.close()

# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.3)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
qs = np.arange(0.1, 1.01, 0.1)
q = 0.5


def run_model(ri):

    ss = []
    for mi in ms:
        star = get_star(m=mi)

        a_init = inv_roche_lobe(ri, q)
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
            sampling=33,
            mass_transfer_efficiency=0.25,
        )
        ss.append(get_s(ab))

    return ss


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abss = list(executor.map(run_model, rs))

with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m3.pkl",
    "wb",
) as f:
    pickle.dump(abss, f, protocol=pickle.HIGHEST_PROTOCOL)
print(abss)
# %%
ms = np.arange(1, 3.01, 0.3)
rs = np.logspace(np.log10(125), np.log10(12500), 101)
with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m3.pkl",
    "rb",
) as f:
    abss3 = pickle.load(f)

with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m2.pkl",
    "rb",
) as f:
    abss2 = pickle.load(f)

ms = np.arange(1, 3.01, 0.3)
rs = np.logspace(np.log10(125), np.log10(12500), 101)
with open(
    f"/home/koen/master-internship/scripts/w31/cache/abundances-m.pkl",
    "rb",
) as f:
    abss = pickle.load(f)
print(abss)

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 2.8)
cmap = plt.cm.viridis
cmap.levels = 7

abss = np.array(abss)
abss2 = np.array(abss2)
abss3 = np.array(abss3)

for i in range(0, 7)[::-1]:
    plt.plot(rs, abss[:, i], c=cmap(norm(ms[i])))
    plt.plot(rs, abss2[:, i], c=cmap(norm(ms[i])), linewidth=5, alpha=0.5)
    plt.plot(rs, abss3[:, i], c=cmap(norm(ms[i])), linewidth=5, alpha=0.5)
axs.spines[["right", "top"]].set_visible(False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca(), ticks=ms)
cbar.set_label(r"$M_\textrm{TPAGB,i}$ ($M_\odot$)")

plt.annotate(
    "barium star",
    xy=(170, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(170, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)

axs.spines[["right", "top"]].set_visible(False)
plt.xscale("log")
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-change-m.pgf", format="pgf")
plt.show()
plt.close()

# %%

from tqdm import tqdm
import pandas as pd

df = AbundanceTables()


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


class Result:
    def __init__(self):
        pass


from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 1)
rs = np.logspace(np.log10(125), np.log10(12500), 48)

qs = np.linspace(0.1, 1.0, 1)
epss = np.logspace(-3, 0, 20)


def run_model(ri):

    results = []

    for mi in ms:
        star = get_star(m=mi)

        for qi in qs:
            a_init = inv_roche_lobe(ri, qi)
            star, Options, q_init, a_init, e_init, Bins = call_evolution(
                star, qi, a_init, simple_only=True
            )
            bin = Bins[0]

            for eps in epss:

                ab = Abundances(
                    None,
                    df,
                    mass=mi,
                    sb=bin,
                    sampling=33,
                    mass_transfer_efficiency=eps,
                )
                if ab.m_env[-1] > 0.04:
                    m2f = bin.m2[-1] + ab.m_env[-1] * eps
                else:
                    m2f = bin.m2[-1]

                results.append(
                    {
                        "r_init": ri,
                        "a_init": a_init,
                        "m1i": mi,
                        "m1f": star.mass[-1],
                        "qi": qi,
                        "eps": eps,
                        "m2f": m2f,
                        "s": get_s(ab),
                        "hs": get_hs(ab),
                        "ls": get_ls(ab),
                        "ba": get_ba(ab),
                    }
                )
    return results


from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":

    np.random.shuffle(rs)
    print(rs)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_model, ri) for ri in rs]

        results = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
            )
        ]

    results = [result for r_results in results for result in r_results]

df_results = pd.DataFrame(results)
print(df_results)
# %%

with open(
    f"/home/koen/master-internship/scripts/w31/cache/df.pkl",
    "wb",
) as f:
    pickle.dump(df_results, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
from matplotlib.colors import TwoSlopeNorm

subset = df_results[(df_results["m1i"] == 2) & (df_results["qi"] == 0.1)]
subset

s_grid = subset.pivot(
    index="eps",
    columns="r_init",
    values="s",
)

norm = TwoSlopeNorm(
    vcenter=0.25,
    vmin=s_grid.min().min(),
    vmax=s_grid.max().max(),
)


fig, ax = plt.subplots()

im = ax.pcolormesh(
    s_grid.columns, s_grid.index, s_grid.values, norm=norm, cmap="coolwarm"
)

fig.colorbar(im, ax=ax, label=r"$[\mathrm{s}/\mathrm{Fe}]$")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$R_{\mathrm{L,init}}\;[R_\odot]$")
ax.set_ylabel(r"$\epsilon$")

plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 2.8)
cmap = plt.cm.viridis
# color = cmap(norm(x))

mm = [1.6, 1.9, 2.2, 2.5, 2.8]
for m in mm:
    star = get_star(m=m)
    plt.plot(star.m_env / star.m_env[star.ntpagb], 10**star.log_R, c=cmap(norm(m)))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label("$M_\\textrm{TPAGB,i}$ ($M_\\odot$)")


plt.xscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env} / M_\\textrm{env,i}$")
plt.ylabel("$R$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-env-mass-r.pgf", format="pgf")
plt.show()
plt.close()


# %%
def roche_lobe_over_a(q):
    return 0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log(1 + q ** (-1 / 3)))


qssss = np.logspace(-1, 0, 100)
plt.plot(qssss, roche_lobe_over_a(qssss))
plt.show()
# %%

from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
rs = np.logspace(np.log10(125), np.log10(12500), 101)

mi = 2.2
qs = np.arange(0.1, 1.01, 0.1)


def run_model(ri):

    ss = []
    star = get_star(m=mi)
    for q in qs:

        a_init = inv_roche_lobe(ri, q)
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
            sampling=33,
            mass_transfer_efficiency=0.25,
        )
        if ab.m_env[-1] > 0.04:
            m2f = bin.m2[-1] + ab.m_env[-1] * 0.25
        else:
            m2f = bin.m2[-1]

        m2i = mi * q

    return m2f - m2i


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        abss = list(executor.map(run_model, rs))

with open(
    f"/home/koen/master-internship/scripts/w31/cache/mass-q.pkl",
    "wb",
) as f:
    pickle.dump(abss, f, protocol=pickle.HIGHEST_PROTOCOL)
print(abss)
# %%
rs = np.logspace(np.log10(125), np.log10(12500), 101)
qs = np.arange(0.1, 1.01, 0.1)
with open(
    f"/home/koen/master-internship/scripts/w31/cache/mass-q.pkl",
    "rb",
) as f:
    abss = pickle.load(f)
print(abss)


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(0.2, 1)
cmap = plt.cm.viridis
# color = cmap(norm(x))


abss = np.array(abss)

for i in range(1, 10)[::-1]:
    plt.plot(rs, abss[:, i], c=cmap(norm(qs[i])))
axs.spines[["right", "top"]].set_visible(False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$q_\textrm{i} = M_\textrm{a,i} / M_\textrm{d,i}$")

plt.annotate(
    "barium star",
    xy=(600.65, 0.25),
    xycoords="data",
    xytext=(-26, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.annotate(
    "no barium star",
    xy=(600.65, 0.25),
    xycoords="data",
    xytext=(-32, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", color="C9", linewidth=0.75),
    c="C9",
)

plt.axhline(0.25, linewidth=0.75, c="C9", zorder=-1)

plt.xscale("log")
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("$\\Delta M_\\textrm{acc}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-change-q-macc.pgf", format="pgf")
plt.show()
plt.close()
