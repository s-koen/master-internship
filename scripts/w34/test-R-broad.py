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
# %%

RL = []
P = []
for m in grid.models:
    RL.append(m.params["R"])
    P.append(m.period_days[-1])

RL = np.array(RL)
P = np.array(P)

args = np.argsort(RL)
RL = RL[args]
P = P[args]


plt.plot(RL, P)
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
        ys_initial.append(np.nanmin(m.period_days))
        ys_final.append(m.period_days[-1])
    else:
        xs_broken.append(R)
        ys_broken_initial.append(np.nanmin(m.period_days))
        ys_broken_final.append(m.period_days[-1])

Rs = np.logspace(2, np.log10(1250), 100)
plt.plot(Rs, period_days(0.6, 2.2, Rs), c="C9", linewidth=0.75)

plt.plot(Rs, period_days(0.1, 2.2, Rs), c="C9", linewidth=0.75, linestyle=":")
plt.plot(Rs, period_days(1, 2.2, Rs), c="C9", linewidth=0.75, linestyle=":")

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


# plt.text(
#     xs_broken[-1] - 60,
#     ys_broken_initial[-1] - 90,
#     "Broken",
#     color="C3",
#     va="top",
#     ha="left",
#     fontsize=8,
# )
# plt.plot(xs, ys_initial, c="C0", linewidth=1)
# plt.scatter(xs, ys_initial, s=25, c="C0", marker=".", zorder=20)
# plt.scatter(xs, ys_initial, s=50, c="w", marker=".", zorder=10)
# plt.text(
#     1.01,
#     ys_initial[-1] - 500,
#     "Minimum",
#     color="C0",
#     va="top",
#     ha="left",
#     fontsize=8,
#     transform=axs.get_yaxis_transform(),
# )
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

# plt.plot(xs_broken, ys_broken_initial, c="C3", linewidth=1)
# plt.scatter(xs_broken, ys_broken_initial, s=25, c="C3", marker=".", zorder=20)
# plt.scatter(xs_broken, ys_broken_initial, s=50, c="w", marker=".", zorder=10)
#
# plt.plot(xs_broken, ys_broken_final, c="C3", linewidth=1)
# plt.scatter(xs_broken, ys_broken_final, s=25, c="C3", marker=".", zorder=20)
# plt.scatter(xs_broken, ys_broken_final, s=50, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("Period (days)")
plt.xscale("log")
plt.xlim(100)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w34-RL-vs-final-period.pgf", format="pgf")
plt.show()
plt.close()
# %%

from scripts.general_utils.mesa_grid import MesaGrid

grid_fine = MesaGrid(f"{MASTER}/tides-grid-3")
# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid_fine.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid_fine.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid_fine.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
        continue

    # if model.age[0] < 0:
    #     ref_TP_count = 0
    # else:
    #     index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
    #     ref_TP_count = grid.ref_tpagb.TP_count[index]
    print(q, R)
    Z[i, j] = model.star.period_days[-1] / period_days(q, 2, R)


dR = np.diff(R_vals)

print(R_vals)
R_edges = np.concatenate(
    [[R_vals[0] - dR[0] / 2], R_vals[:-1] + dR / 2, [R_vals[-1] + dR[-1] / 2]]
)

dq = np.diff(q_vals)
q_edges = np.concatenate(
    [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
)

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

overlay = np.where(
    mask_bad,
    1,
    0,
)

print(R_edges)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="viridis",
    shading="auto",
)


plt.colorbar(mesh, label=r"Period (days)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

# plt.savefig("/home/koen/LaTeX-setup/plots/w34-grid-fine-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
