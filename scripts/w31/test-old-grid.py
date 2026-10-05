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
from scripts.general_utils.mesa_grid import MesaGrid

# %%
grid2 = MesaGrid(f"{MASTER}/tides-grid-7")

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid2.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z)

for R, q, model in grid2.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        if model.star.period_days[-1] < 50:
            mask_bad[i, j] = 0.9
        elif model.star.model_number[-1] < 500:
            mask_bad[i, j] = 0.5
        else:
            mask_bad[i, j] = 0.1
    else:
        Z[i, j] = model.star.period_days[-1]


logR = np.log10(R_vals)
dlogR = np.diff(logR)

logR_edges = np.concatenate(
    [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
)
R_edges = 10**logR_edges

dq = np.diff(q_vals)
q_edges = np.concatenate(
    [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
)

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

print(mask_bad)
ax.pcolormesh(
    R_edges,
    q_edges,
    mask_bad,
    shading="auto",
    cmap="PiYG",
    vmin=0,
    vmax=1,
    alpha=0.4,
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

ax.set_xscale("log")

# plt.savefig("/home/koen/LaTeX-setup/plots/w15-grid-1.pgf", format="pgf")
plt.show()
plt.close()


# %%

q_vals = np.array(sorted(set(q for R, q, _ in grid2.iter_models())))

i = 0
for q_ref in q_vals:
    first = True
    for model in grid2.iter_models():
        R, q, m = model
        if q == q_ref:
            if m.env_mass[-1] > 0.01 and m.star.period_days[-1] > 50:
                plt.plot(m.age, m.star.R, c=f"C3")
                plt.plot(m.age, m.star.rl_1, c=f"C3", linewidth=4, alpha=0.5)
            else:
                if first:
                    first = False
                    plt.plot(m.age, m.star.R, c=f"C2", zorder=-1)
                    plt.plot(
                        m.age, m.star.rl_1, c=f"C2", zorder=-1, linewidth=4, alpha=0.5
                    )

        i += 1

plt.show()


# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid2.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)

mask_bad = np.zeros_like(Z) * np.nan

for R, q, model in grid2.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        if model.star.period_days[-1] < 50:
            mask_bad[i, j] = 0.9
        elif model.star.model_number[-1] < 500:
            mask_bad[i, j] = 0.5
        else:
            mask_bad[i, j] = 0.1

    Z[i, j] = model.age[-1]


logR = np.log10(R_vals)
dlogR = np.diff(logR)

logR_edges = np.concatenate(
    [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
)
R_edges = 10**logR_edges

dq = np.diff(q_vals)
q_edges = np.concatenate(
    [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
)

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="viridis",
    shading="auto",
)

zm = np.ma.masked_less(mask_bad, 0.3)
print(mask_bad)
ax.pcolor(R_edges, q_edges, zm, hatch="//", alpha=0, rasterized=True)

zm = np.ma.masked_greater(mask_bad, 0.9)
ax.pcolor(R_edges, q_edges, zm, hatch="\\\\", alpha=0, rasterized=True)

plt.colorbar(mesh, label=r"TPAGB age (yr)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w31-grid-1.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))

i = 0
for model in grid2.iter_models():
    R, q, m = model
    if R == R_vals[2]:
        if m.env_mass[-1] > 0.1:
            axs[0].plot(m.age, m.star.R, c=f"C3", linewidth=1)
            axs[0].plot(m.age, m.star.rl_1, c=f"C3", linewidth=2, alpha=0.5)
            axs[1].plot(m.age, m.env_mass, c=f"C3", linewidth=1)
        else:
            axs[0].plot(m.age, m.star.R, c=f"C2", linewidth=1)
            axs[0].plot(m.age, m.star.rl_1, c=f"C2", linewidth=2, alpha=0.5)
            axs[1].plot(m.age, m.env_mass, c=f"C2", linewidth=1)

        i += 1

plt.xlim(216214.30304713378, 217492.63677769568)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("TPAGB age (yr)")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].set_ylabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-see-same-pulse.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))

i = 0
for model in grid2.iter_models():
    R, q, m = model
    if R == R_vals[2]:
        if m.env_mass[-1] > 0.1:
            axs[0].plot(m.star.R, m.star.Teff, c=f"C3", linewidth=1)
            axs[1].plot(m.age, m.env_mass, c=f"C3", linewidth=1)
        else:
            axs[0].plot(m.star.R, m.star.Teff, c=f"C2", linewidth=1)
            axs[1].plot(m.age, m.env_mass, c=f"C2", linewidth=1)

        i += 1

plt.xlim(216214.30304713378, 217492.63677769568)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("TPAGB age (yr)")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].set_ylabel("$M_\\textrm{env}$ ($M_\\odot$)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-see-same-pulse.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))

i = 0
for model in grid2.iter_models():
    R, q, m = model
    if R == R_vals[2]:
        if m.env_mass[-1] > 0.1:
            axs.plot(m.star.star_mass, m.star.lg_mstar_dot_1, c=f"C3", linewidth=1)
        else:
            axs.plot(m.star.star_mass, m.star.lg_mstar_dot_1, c=f"C2", linewidth=1)

        i += 1

plt.ylim(-3, 0)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
axs.set_ylabel("$\\textrm{log}(\\dot{M} / M_\\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-see-same-pulse-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

print(m.star.bulk_names)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))

i = 0
for model in grid2.iter_models():
    R, q, m = model
    if R == R_vals[2]:
        if m.env_mass[-1] > 0.1:
            axs.plot(m.star.log_Teff, m.star.log_L, c=f"C3", linewidth=1)
        else:
            axs.plot(m.star.log_Teff, m.star.log_L, c=f"C2", linewidth=1)

        i += 1


plt.gca().invert_xaxis()

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log($T_\\textrm{eff} / \\textrm{K}$)")
plt.ylabel("log($L / L_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-see-same-pulse-3.pgf", format="pgf")
plt.show()
plt.close()


# %%
