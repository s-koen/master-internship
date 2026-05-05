import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


def roche_lobe_separation(RL, q):
    """
    compute orbital separation from roche lobe radius and q = M_a / M_d

    parameters
    ----------
    RL : float or array
        roche lobe radius of the donor
    q : float or array
        mass ratio (M_accretor / M_donor)

    returns
    -------
    a : float or array
        orbital separation
    """

    q = np.asarray(q)

    q_inv_23 = q ** (-2 / 3)
    q_inv_13 = q ** (-1 / 3)

    f = 0.49 * q_inv_23 / (0.6 * q_inv_23 + np.log(1 + q_inv_13))

    return RL / f


def period_from_roche_lobe(RL, q, M_d):
    """
    compute orbital period (days) from roche lobe radius

    parameters
    ----------
    RL : float or array
        roche lobe radius of donor [R_sun]
    q : float or array
        mass ratio (M_a / M_d)
    M_d : float or array
        donor mass [M_sun]
    M_a : float or array
        accretor mass [M_sun]

    returns
    -------
    P : float or array
        orbital period [days]
    """

    # constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_sun = 1.98847e30  # kg
    R_sun = 6.957e8  # m

    M_a = M_d * q

    RL = np.asarray(RL)
    q = np.asarray(q)

    # roche lobe factor (your convention: q = M_a / M_d)
    q_inv_23 = q ** (-2 / 3)
    q_inv_13 = q ** (-1 / 3)

    f = 0.49 * q_inv_23 / (0.6 * q_inv_23 + np.log(1 + q_inv_13))

    # separation in meters
    a = (RL / f) * R_sun

    # total mass in kg
    M_tot = (M_d + M_a) * M_sun

    # kepler's law
    P_sec = 2 * np.pi * np.sqrt(a**3 / (G * M_tot))

    return P_sec / 86400.0  # seconds → days


# %%

grid = MesaGrid(f"{MASTER}/tides-grid-4")

# %%

for i, (q, R, model) in enumerate(grid.iter_models()):
    print(i)
model.star.bulk_names
# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=2), constrained_layout=True
)

count = 0
for i, (q, R, model) in enumerate(grid.iter_models()):
    # if i > 10:
    #     continue
    frac = model.star.I_eff * model.star.Omega_star / (model.star.J_orb)
    if np.max(frac) < 1 / 3:
        axs[0].plot(model.env_mass, frac, color="C9", zorder=-10, alpha=0.4)
        axs[1].plot(
            model.env_mass, -model.star.jdot_ls, color="C9", zorder=-10, alpha=0.4
        )

        axs[2].plot(
            model.env_mass, model.star.period_days, color="C9", zorder=-10, alpha=0.4
        )
        continue

    axs[0].plot(
        model.env_mass,
        frac,
        label=f"$R_\\textrm{{RL,i}} = {R}, q={q}$",
    )
    axs[1].plot(model.env_mass, -model.star.jdot_ls, linestyle=":", c=f"C{count}")
    axs[1].plot(model.env_mass, -model.star.jdot_ml, linestyle="--", c=f"C{count}")
    axs[1].plot(model.env_mass, -model.star.jdot_ml - model.star.jdot_ls, c=f"C{count}")
    axs[2].plot(model.env_mass, model.star.period_days)
    count += 1

fig.legend(loc="outside upper center", ncols=2)
axs[2].set_xlabel("Time (yr)")
axs[0].set_ylabel(r"$J_\textrm{star} / J_\textrm{orbit}$")
axs[1].set_ylabel(r"$\dot{J}_\textrm{orbit}$ (g cm$^{-2}$ s$^{-2}$)")
axs[2].set_ylabel(r"Period (days)")
axs[0].set_ylim(1e-3, 3)
axs[1].set_ylim(1e37, 1e49)
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[2].set_yscale("log")

# lock axes
axs[0].set_xlim(*axs[0].get_xlim())

# darwin instability line
axs[0].hlines(1 / 3, *axs[0].get_xlim(), color="C9", linewidth=0.75)

axs[0].invert_xaxis()
plt.savefig("/home/koen/LaTeX-setup/plots/w14-darwin.pgf", format="pgf")
plt.show()
plt.close()
# %%

for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):
    plt.plot(model.age, model.star.R)
    plt.plot(model.age, model.star.rl_1)

plt.show()
# %%

for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):
    plt.plot(model.age, model.star.R / model.star.rl_1)

plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


def delta(R, R_RL, R_OL):
    delta = 1 / (R_OL - R_RL) * R - R_RL / (R_OL - R_RL)
    return np.clip(delta, 0, 1)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        delta(model.star.R, model.star.rl_1, rol(model.star)),
    )
plt.xlabel("$q$")
plt.ylabel(r"$\delta$")
plt.savefig("/home/koen/LaTeX-setup/plots/w14-delta-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

grid = MesaGrid(f"{MASTER}/tides-grid-6")
# %%
grid_rad = MesaGrid(f"{MASTER}/tides-grid-2")

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.nanmax(model.star.R / rol(model.star))


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
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


plt.colorbar(mesh, label=r"$\max(R_\textrm{star} / R_\textrm{ROL})$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.nanmax(model.star.R / rol(model.star))

for R, q, model in grid_rad.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] /= np.nanmax(model.star.R / rol(model.star))


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmin = np.nanmin(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="coolwarm",
    shading="auto",
    vmin=Zmin,
    vmax=1 + (1 - Zmin),
)


plt.colorbar(
    mesh,
    label=r"$(\max(R_\textrm{star} / R_\textrm{ROL}))_\textrm{cons} / (\max(R_\textrm{star} / R_\textrm{ROL}))_\textrm{rad}$",
)

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-2.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
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

overlay = np.where(
    mask_bad,
    1,
    0,
)
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

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-3.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.period_days[-1]

for R, q, model in grid_rad.iter_models():
    print(R, q)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        print(model.star.period_days[-1])
        Z[i, j] /= model.star.period_days[-1]


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmin = np.nanmin(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="coolwarm",
    shading="auto",
    vmin=Zmin,
    vmax=1 + (1 - Zmin),
)


plt.colorbar(
    mesh, label=r"$(\textrm{Period})_\textrm{cons} / (\textrm{Period})_\textrm{rad}$"
)

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-4.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.star_2_mass[-1] / model.star.star_1_mass[-1]


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
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


plt.colorbar(mesh, label=r"$q_\textrm{final}$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-5.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.star_2_mass[-1] / model.star.star_1_mass[-1]


for R, q, model in grid_rad.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] /= model.star.star_2_mass[-1] / model.star.star_1_mass[-1]


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="coolwarm",
    shading="auto",
    vmin=1 - (Zmax - 1),
    vmax=Zmax,
)


plt.colorbar(
    mesh, label=r"$(q_\textrm{final})_\textrm{cons} / (q_\textrm{final})_\textrm{rad}$"
)

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-6.pgf", format="pgf")
plt.show()
plt.close()
# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.star_2_mass[-1]

for R, q, model in grid_rad.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] /= model.star.star_2_mass[-1]


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="coolwarm",
    shading="auto",
    vmin=1 - (Zmax - 1),
    vmax=Zmax,
)


plt.colorbar(
    mesh,
    label=r"$M_\textrm{a, cons} / M_\textrm{a, rad}$",
    extend="min",
)
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-8.pgf", format="pgf")
plt.show()
plt.close()
# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.star_2_mass[-1]

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

overlay = np.where(
    mask_bad,
    1,
    0,
)
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


plt.colorbar(
    mesh,
    label=r"$M_\textrm{a}$ ($M_\odot$)",
    extend="min",
)
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-7.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        m_initial = q * 2
        m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
        m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

        dt = np.diff(model.star.star_age)
        dm = np.diff(model.star.star_2_mass)

        Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
        Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
        M_c_transfer = np.cumsum(Xc * dm)
        M_o_transfer = np.cumsum(Xo * dm)
        M_c = m_C_initial + M_c_transfer
        M_o = m_O_initial + M_o_transfer
        Xc_final = M_c / model.star.star_2_mass[1:]
        Xo_final = M_o / model.star.star_2_mass[1:]

        Z[i, j] = Xc_final[-1] / Xo_final[-1] * 16 / 12


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


plt.colorbar(mesh, label=r"C/O-number ratio", extend="min")
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-9.pgf", format="pgf")
plt.show()
plt.close()


# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        m_initial = q * 2
        m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
        m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

        dt = np.diff(model.star.star_age)
        dm = np.diff(model.star.star_2_mass)

        Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
        Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
        M_c_transfer = np.cumsum(Xc * dm)
        M_o_transfer = np.cumsum(Xo * dm)
        M_c = m_C_initial + M_c_transfer
        M_o = m_O_initial + M_o_transfer
        Xc_final = M_c / model.star.star_2_mass[1:]
        Xo_final = M_o / model.star.star_2_mass[1:]

        Z[i, j] = Xc_final[-1] / Xo_final[-1] * 16 / 12

for R, q, model in grid_rad.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        m_initial = q * 2
        m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
        m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

        dt = np.diff(model.star.star_age)
        dm = np.diff(model.star.star_2_mass)

        Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
        Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
        M_c_transfer = np.cumsum(Xc * dm)
        M_o_transfer = np.cumsum(Xo * dm)
        M_c = m_C_initial + M_c_transfer
        M_o = m_O_initial + M_o_transfer
        Xc_final = M_c / model.star.star_2_mass[1:]
        Xo_final = M_o / model.star.star_2_mass[1:]

        Z[i, j] /= Xc_final[-1] / Xo_final[-1] * 16 / 12


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

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


plt.colorbar(
    mesh,
    label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
    extend="min",
)
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-10.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


for R, q, model in grid.get_R1_index(-1):
    if q != 0.6:
        continue
    plt.plot(model.age, model.star.R, c="C0")
    plt.plot(model.age, model.star.rl_1, c="C0")

for R, q, model in grid_rad.get_R1_index(-1):
    if q != 0.6:
        continue
    plt.plot(model.age, model.star.R, c="C1")
    plt.plot(model.age, model.star.rl_1, c="C1")

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w14-compare-1.pgf", format="pgf")
plt.show()
plt.close()
# %%
