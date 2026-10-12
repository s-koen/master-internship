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

# %%

grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
df = AbundanceTables()

# %%

for model in grid.models:
    plt.plot(model.age, model.rl_1)
    # plt.plot(model.age, model.R)

star = get_star(m=2.2)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%

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


def get_ls(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 3


def get_hs(r):
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


def get_s(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 7


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:

    per.append(m.params["R"])
    s.append(get_s(m))

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25, c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-RL.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

xs = []
ys = []

xs_broken = []
ys_broken = []

for m in grid.models:
    if m.envelope_mass[-1] < 0.01:
        xs.append(m.params["R"])
        ys.append(m.period_days[-1])
    else:
        xs_broken.append(m.params["R"])
        ys_broken.append(m.period_days[-1])

xs = np.array(xs)
ys = np.array(ys)
idx = np.argsort(xs)

xs = xs[idx]
ys = ys[idx]

xs_broken = np.array(xs_broken)
ys_broken = np.array(ys_broken)
idx = np.argsort(xs_broken)

xs_broken = xs_broken[idx]
ys_broken = ys_broken[idx]


plt.plot(xs, ys, c="k", linewidth=1)
plt.scatter(xs, ys, s=25, c="k", marker=".", zorder=20)
plt.scatter(xs, ys, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Final period (days)")
plt.xscale("log")
plt.xlim(100)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-RL-vs-final-period.pgf", format="pgf")
plt.show()
plt.close()
# %%


def period_days(q, M1, R_L):
    """Compute the orbital period from the mass ratio and donor Roche lobe radius.

    Parameters
    ----------
    q : float or array-like
        Mass ratio M2 / M1.
    M1 : float
        Donor mass in solar masses.
    R_L : float or array-like
        Donor Roche lobe radius in solar radii.

    Returns
    -------
    float or ndarray
        Orbital period in days.
    """
    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    a = R_L / r_L * 695508000  # orbital separation in solar radii
    M_total = M1 * (1 + q) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

xs = []
ys_initial = []
ys_final = []

xs_broken = []
ys_broken_initial = []
ys_broken_final = []

for m in grid.models:
    print(m.params)
    M1 = m.params["m"]
    q = m.params["q"]
    R = m.params["R"]

    if m.envelope_mass[-1] < 0.01:
        xs.append(R)
        ys_initial.append(np.nanmin(m.period_days))
        ys_final.append(m.period_days[-1])
    else:
        xs_broken.append(R)
        ys_broken_initial.append(np.nanmin(m.period_days))
        ys_broken_final.append(m.period_days[-1])

Rs = np.logspace(2, np.log10(1250), 100)
plt.plot(Rs, period_days(0.6, 2.2, Rs), c="C9", linewidth=0.75)


xs = np.array(xs)
ys_initial = np.array(ys_initial)
ys_final = np.array(ys_final)
idx = np.argsort(xs)

xs = xs[idx]
ys_initial = ys_initial[idx]
ys_final = ys_final[idx]

xs_broken = np.array(xs_broken)
ys_broken_initial = np.array(ys_broken_initial)
ys_broken_final = np.array(ys_broken_final)
idx = np.argsort(xs_broken)

xs_broken = xs_broken[idx]
ys_broken_initial = ys_broken_initial[idx]
ys_broken_final = ys_broken_final[idx]


plt.text(
    xs_broken[-1] - 60,
    ys_broken_initial[-1] - 90,
    "Broken",
    color="C3",
    va="top",
    ha="left",
    fontsize=8,
)
plt.plot(xs, ys_initial, c="C0", linewidth=1)
plt.scatter(xs, ys_initial, s=25, c="C0", marker=".", zorder=20)
plt.scatter(xs, ys_initial, s=50, c="w", marker=".", zorder=10)
plt.text(
    1.01,
    ys_initial[-1] - 500,
    "Minimum",
    color="C0",
    va="top",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.text(
    1.01,
    10 ** ((np.log10(ys_final[-1] + 1000) + np.log10(ys_initial[-1] - 500)) / 2),
    "Initial",
    color="C9",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.text(
    1.01,
    ys_final[-1] + 1000,
    "Final",
    color="C2",
    va="bottom",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.xlim(100, xs[-1] + 100)


plt.plot(xs, ys_final, c="C2", linewidth=1)
plt.scatter(xs, ys_final, s=25, c="C2", marker=".", zorder=20)
plt.scatter(xs, ys_final, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_initial, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_initial, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_initial, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_final, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_final, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_final, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Period (days)")
plt.xscale("log")
plt.xlim(100)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-RL-vs-final-period.pgf", format="pgf")
plt.show()
plt.close()
# %%

for m in grid.models:
    plt.plot(m.age, m.period_days)
    plt.axhline(period_days(0.6, 2.2, m.params["R"]))
plt.show()
# %%


def period_days(q, M1, R_L):
    """Compute the orbital period from the mass ratio and donor Roche lobe radius.

    Parameters
    ----------
    q : float or array-like
        Mass ratio M2 / M1.
    M1 : float
        Donor mass in solar masses.
    R_L : float or array-like
        Donor Roche lobe radius in solar radii.

    Returns
    -------
    float or ndarray
        Orbital period in days.
    """
    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    a = R_L / r_L * 695508000  # orbital separation in solar radii
    M_total = M1 * (1 + q) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

xs = []
ys_initial = []
ys_final = []

xs_broken = []
ys_broken_initial = []
ys_broken_final = []

for m in grid.models:
    print(m.params)
    M1 = m.params["m"]
    q = m.params["q"]
    R = m.params["R"]

    if m.envelope_mass[-1] < 0.01:
        xs.append(R)
        ys_initial.append(np.nanmin(m.period_days) - period_days(0.6, 2.2, R))
        ys_final.append(m.period_days[-1] - period_days(0.6, 2.2, R))
    else:
        xs_broken.append(R)
        ys_broken_initial.append(np.nanmin(m.period_days) - period_days(0.6, 2.2, R))
        ys_broken_final.append(m.period_days[-1] - period_days(0.6, 2.2, R))

Rs = np.logspace(2, np.log10(1250), 100)


xs = np.array(xs)
ys_initial = np.array(ys_initial)
ys_final = np.array(ys_final)
idx = np.argsort(xs)

xs = xs[idx]
ys_initial = ys_initial[idx]
ys_final = ys_final[idx]

xs_broken = np.array(xs_broken)
ys_broken_initial = np.array(ys_broken_initial)
ys_broken_final = np.array(ys_broken_final)
idx = np.argsort(xs_broken)

xs_broken = xs_broken[idx]
ys_broken_initial = ys_broken_initial[idx]
ys_broken_final = ys_broken_final[idx]


plt.text(
    xs_broken[-1] - 60,
    ys_broken_initial[-1] - 90,
    "Broken",
    color="C3",
    va="top",
    ha="left",
    fontsize=8,
)
plt.plot(xs, ys_initial, c="C0", linewidth=1)
plt.scatter(xs, ys_initial, s=25, c="C0", marker=".", zorder=20)
plt.scatter(xs, ys_initial, s=50, c="w", marker=".", zorder=10)
plt.text(
    1.01,
    ys_initial[-1],
    "Minimum",
    color="C0",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.text(
    1.01,
    0,
    "Initial",
    color="C9",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)

plt.text(
    1.01,
    ys_final[-1],
    "Final",
    color="C2",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.xlim(100, xs[-1] + 100)


plt.plot(xs, ys_final, c="C2", linewidth=1)
plt.scatter(xs, ys_final, s=25, c="C2", marker=".", zorder=20)
plt.scatter(xs, ys_final, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_initial, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_initial, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_initial, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_final, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_final, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_final, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Period $-$ initial period (days)")
plt.xscale("log")
plt.xlim(100)
plt.axhline(0, color="C9", linewidth=0.75, zorder=-10)
# plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-RL-minus-initial.pgf", format="pgf")
plt.show()
plt.close()
# %%


def period_days(q, M1, R_L):
    """Compute the orbital period from the mass ratio and donor Roche lobe radius.

    Parameters
    ----------
    q : float or array-like
        Mass ratio M2 / M1.
    M1 : float
        Donor mass in solar masses.
    R_L : float or array-like
        Donor Roche lobe radius in solar radii.

    Returns
    -------
    float or ndarray
        Orbital period in days.
    """
    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    a = R_L / r_L * 695508000  # orbital separation in solar radii
    M_total = M1 * (1 + q) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

xs = []
ys_initial = []
ys_final = []

xs_broken = []
ys_broken_initial = []
ys_broken_final = []

for m in grid.models:
    print(m.params)
    M1 = m.params["m"]
    q = m.params["q"]
    R = m.params["R"]

    if m.envelope_mass[-1] < 0.01:
        xs.append(R)
        ys_initial.append(np.nanmin(m.period_days) - period_days(0.6, 2.2, R))
        ys_final.append(m.period_days[0] - period_days(0.6, 2.2, R))
    else:
        xs_broken.append(R)
        ys_broken_initial.append(np.nanmin(m.period_days) - period_days(0.6, 2.2, R))
        ys_broken_final.append(m.period_days[0] - period_days(0.6, 2.2, R))

Rs = np.logspace(2, np.log10(1250), 100)


xs = np.array(xs)
ys_initial = np.array(ys_initial)
ys_final = np.array(ys_final)
idx = np.argsort(xs)

xs = xs[idx]
ys_initial = ys_initial[idx]
ys_final = ys_final[idx]

xs_broken = np.array(xs_broken)
ys_broken_initial = np.array(ys_broken_initial)
ys_broken_final = np.array(ys_broken_final)
idx = np.argsort(xs_broken)

xs_broken = xs_broken[idx]
ys_broken_initial = ys_broken_initial[idx]
ys_broken_final = ys_broken_final[idx]


plt.text(
    xs_broken[-1] - 60,
    ys_broken_initial[-1] - 90,
    "Broken",
    color="C3",
    va="top",
    ha="left",
    fontsize=8,
)
plt.plot(xs, ys_initial, c="C0", linewidth=1)
plt.scatter(xs, ys_initial, s=25, c="C0", marker=".", zorder=20)
plt.scatter(xs, ys_initial, s=50, c="w", marker=".", zorder=10)
plt.text(
    1.01,
    ys_initial[-1],
    "Minimum",
    color="C0",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.text(
    1.01,
    0,
    "Initial",
    color="C9",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)

plt.text(
    1.01,
    ys_final[-1],
    "Final",
    color="C2",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.xlim(100, xs[-1] + 100)


plt.plot(xs, ys_final, c="C2", linewidth=1)
plt.scatter(xs, ys_final, s=25, c="C2", marker=".", zorder=20)
plt.scatter(xs, ys_final, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_initial, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_initial, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_initial, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_final, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_final, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_final, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Period $-$ initial period (days)")
plt.xscale("log")
plt.xlim(100)
plt.axhline(0, color="C9", linewidth=0.75, zorder=-10)
# plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w32-RL-minus-initial.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

xs = []
ys_initial = []
ys_final = []

xs_broken = []
ys_broken_initial = []
ys_broken_final = []

for m in grid.models:
    print(m.params)
    M1 = m.params["m"]
    q = m.params["q"]
    R = m.params["R"]

    if m.envelope_mass[-1] < 0.01:
        xs.append(R)
        ys_initial.append(np.nanmax(m.R / m.rl_1))
        ys_final.append(1)
    else:
        xs_broken.append(R)
        ys_broken_initial.append(np.nanmax(m.R / m.rl_1))
        ys_broken_final.append(1)

Rs = np.logspace(2, np.log10(1250), 100)


xs = np.array(xs)
ys_initial = np.array(ys_initial)
ys_final = np.array(ys_final)
idx = np.argsort(xs)

xs = xs[idx]
ys_initial = ys_initial[idx]
ys_final = ys_final[idx]

xs_broken = np.array(xs_broken)
ys_broken_initial = np.array(ys_broken_initial)
ys_broken_final = np.array(ys_broken_final)
idx = np.argsort(xs_broken)

xs_broken = xs_broken[idx]
ys_broken_initial = ys_broken_initial[idx]
ys_broken_final = ys_broken_final[idx]


plt.text(
    xs_broken[-1] - 60,
    ys_broken_initial[-1] - 90,
    "Broken",
    color="C3",
    va="top",
    ha="left",
    fontsize=8,
)

plt.text(
    1.01,
    ys_initial[-1],
    "Minimum",
    color="C0",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)
plt.text(
    1.01,
    0,
    "Initial",
    color="C9",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)

plt.text(
    1.01,
    ys_final[-1],
    "Final",
    color="C2",
    va="center",
    ha="left",
    fontsize=8,
    transform=axs.get_yaxis_transform(),
)

plt.plot(xs, ys_initial, c="C0", linewidth=1)
plt.scatter(xs, ys_initial, s=25, c="C0", marker=".", zorder=20)
plt.scatter(xs, ys_initial, s=50, c="w", marker=".", zorder=10)
plt.xlim(100, xs[-1] + 100)


# plt.plot(xs, ys_final, c="C2", linewidth=1)
# plt.scatter(xs, ys_final, s=25, c="C2", marker=".", zorder=20)
# plt.scatter(xs, ys_final, s=50, c="w", marker=".", zorder=10)

plt.plot(xs_broken, ys_broken_initial, c="C3", linewidth=1)
plt.scatter(xs_broken, ys_broken_initial, s=25, c="C3", marker=".", zorder=20)
plt.scatter(xs_broken, ys_broken_initial, s=50, c="w", marker=".", zorder=10)

# plt.plot(xs_broken, ys_broken_final, c="C3", linewidth=1)
# plt.scatter(xs_broken, ys_broken_final, s=25, c="C3", marker=".", zorder=20)
# plt.scatter(xs_broken, ys_broken_final, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Period $-$ initial period (days)")
plt.xscale("log")
plt.xlim(100)
plt.axhline(0, color="C9", linewidth=0.75, zorder=-10)
# plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w32-RL-minus-initial.pgf", format="pgf")
plt.show()
plt.close()
# %%
