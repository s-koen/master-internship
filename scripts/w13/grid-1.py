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

grid = MesaGrid(f"{MASTER}/tides-grid-2")

# %%

min = 1e99
max = -1
for R, q, model in grid.iter_models():
    if model.star.period_days[0] < min:
        min = model.star.period_days[0]
    if model.star.period_days[0] > max:
        max = model.star.period_days[0]

print(min)
print(max)
# %%

model.star.bulk_names
# %%

plt.plot(
    model.age[1:],
    np.cumsum(10 ** model.star.lg_mtransfer_rate[1:] * np.diff(model.star.star_age)),
)
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
ls = []
for i, (R, q, model) in enumerate(grid.iter_models()):

    ii = i % 7
    dt = np.diff(model.star.star_age)
    mdot = 0.5 * (
        10 ** model.star.lg_mtransfer_rate[:-1] + 10 ** model.star.lg_mtransfer_rate[1:]
    )
    Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])

    l = plt.scatter(
        model.age[-1],
        np.cumsum(mdot * Xc * dt)[-1],
        c=f"C{ii}",
        label=f"$q={q:.1f}$",
        s=7,
    )
    if i < 7:
        ls.append(l)

fig.legend(loc="outside upper center", handles=ls, ncols=4)
(l,) = plt.plot(
    grid.ref_tpagb.star_age,
    grid.ref_tpagb.surface_c12 * grid.ref_tpagb.star_mass,
    label="total C$_{12}$",
)
plt.legend(handles=[l], ncols=10, loc="upper left")
plt.ylim(0.001)
plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel(r"total $m(\textrm{C}_{12})$ transferred ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-transfer-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

(l,) = plt.plot(
    grid.ref_tpagb.star_age,
    grid.ref_tpagb.surface_c12 * grid.ref_tpagb.star_mass,
    label="total C$_{12}$",
    c="C9",
    linewidth=5,
)
plt.legend(handles=[l], ncols=10, loc="upper left")

ls = []
for i, (R, q, model) in enumerate(grid_cons.iter_models()):

    ii = i % 7
    dt = np.diff(model.star.star_age)
    mdot = 0.5 * (
        10 ** model.star.lg_mtransfer_rate[:-1] + 10 ** model.star.lg_mtransfer_rate[1:]
    )
    Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])

    # l = plt.scatter(model.age[-1], np.cumsum(mdot * Xc * dt)[-1], c = f"C{ii}", label=f"$q={q:.1f}$", s=7)
    (l,) = plt.plot(
        model.age,
        model.star.surface_c12 * model.star.star_1_mass,
        c=f"C{ii}",
        label=f"$q={q:.1f}$",
    )
    if i < 7:
        ls.append(l)

fig.legend(loc="outside upper center", handles=ls, ncols=4)
plt.ylim(0.0007)
plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel(r"total $m(\textrm{C}_{12})$ star ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-transfer-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

(l,) = plt.plot(
    grid.ref_tpagb.star_age,
    grid.ref_tpagb.surface_c12 / grid.ref_tpagb.surface_o16 * 16 / 12,
    label="Donor C$_{12}$",
    c="C9",
    linewidth=3,
)
plt.legend(handles=[l], ncols=10, loc="upper left")

ls = []
for i, (R, q, model) in enumerate(grid.iter_models()):
    if q not in [0.4, 1]:
        continue
    ii = i % 7

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

    (l,) = plt.plot(
        model.age[1:],
        Xc_final / Xo_final * 16 / 12,
        c=f"C{ii}",
        label=f"$q={q:.1f}$",
    )
    if i < 7:
        ls.append(l)

    plt.plot(
        model.age,
        model.star.surface_c12 / model.star.surface_o16 * 16 / 12,
        alpha=0.4,
        c=f"C{ii}",
        linewidth=3,
    )

fig.legend(loc="outside upper center", handles=ls, ncols=4)
plt.yscale("log")
plt.xlim(*plt.gca().get_xlim())
plt.hlines(1, *plt.gca().get_xlim(), color="C9", linewidth=0.75)
plt.xlabel("Time (yr)")
plt.ylabel(r"C/O-number ratio")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-transfer-4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Model number")
plt.ylabel("$X(\\textrm{C}_{12})$")

deltat = 0
plt.plot(grid.ref_ms.model_number, grid.ref_ms.surface_o16)
deltat += grid.ref_ms.model_number[-1]
plt.plot(grid.ref_gb.model_number + deltat, grid.ref_gb.surface_o16)
deltat += grid.ref_gb.model_number[-1]
plt.plot(grid.ref_cheb.model_number + deltat, grid.ref_cheb.surface_o16)
deltat += grid.ref_cheb.model_number[-1]
plt.plot(grid.ref_eagb.model_number + deltat, grid.ref_eagb.surface_o16)
deltat += grid.ref_eagb.model_number[-1]
plt.plot(grid.ref_tpagb.model_number + deltat, grid.ref_tpagb.surface_o16)

plt.savefig("/home/koen/LaTeX-setup/plots/w13-o16.pgf", format="pgf")
plt.show()
plt.close()
# %%

grid_cons = MesaGrid(f"{MASTER}/tides-grid-2")
# %%
grid = MesaGrid(f"{MASTER}/tides-grid-4")
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
        Z[i, j] = np.log10(np.nanmin(model.star.period_days))
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.log10(np.nanmin(model.star.period_days))


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


plt.colorbar(mesh, label=r"Minimum log orbital period (days)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-grid-1.pgf", format="pgf")
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


plt.colorbar(mesh, label=r"Orbital period (days)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-grid-2.pgf", format="pgf")
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

for R, q, model in grid_cons.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
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
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="viridis",
    shading="auto",
)


plt.colorbar(mesh, label=r"$P_\textrm{CCT} / P_\textrm{cons}$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-grid-3.pgf", format="pgf")
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
        Z[i, j] = np.nanmax(model.star.R / model.star.rl_1)

for R, q, model in grid_cons.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] /= np.nanmax(model.star.R / model.star.rl_1)


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
    label=r"$\max(R_\textrm{star} / R_\textrm{RL})_\textrm{CCT} / \max(R_\textrm{star} / R_\textrm{RL})_\textrm{cons}$",
)

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-grid-5.pgf", format="pgf")
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

plt.savefig("/home/koen/LaTeX-setup/plots/w13-grid-6.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(3)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=3,
        alpha=0.5,
    )
    plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.R, c=f"C{i}")

plt.xlabel("$q$")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-extreme1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(3)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.R / model.star.rl_1,
        c=f"C{i}",
    )

plt.xlabel("$q$")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-extreme2.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(4)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.R / model.star.rl_1,
        c=f"C{i}",
    )

plt.xlabel("$q$")
plt.ylim(0.2, 3.1)
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-no-extreme.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(7)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.R / model.star.rl_1,
        c=f"C{i}",
    )

plt.xlabel("$q$")
plt.yscale("log")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-extreme3.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(7)):
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=3,
        alpha=0.5,
    )
    plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.R, c=f"C{i}")

plt.xlabel("$q$")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w13-extreme4.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid_cons.iter_models():
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

Zmax = np.nanmax(Z)
print(Zmax)

ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


plt.colorbar(mesh, label=r"C/O-number ratio", extend="min")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-co-1.pgf", format="pgf")
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

plt.savefig("/home/koen/LaTeX-setup/plots/w13-co-2.pgf", format="pgf")
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

for R, q, model in grid_cons.iter_models():
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
Zmax = 1 - np.nanmax(Z)
Zmin = 1 - np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


plt.colorbar(
    mesh,
    label=r"$($C/O-number ratio$)_\textrm{CCT} / ($C/O-number ratio$)_\textrm{cons}$",
    extend="min",
)
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-co-3.pgf", format="pgf")
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

for R, q, model in grid_cons.iter_models():
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
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


plt.colorbar(
    mesh,
    label=r"$M_\textrm{a, CCT} / M_\textrm{a, cons}$",
    extend="min",
)
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w13-m2.pgf", format="pgf")
plt.show()
plt.close()
# %%
