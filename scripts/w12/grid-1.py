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

grid_fine = MesaGrid(f"{MASTER}/tides-grid-3")

# %%

grid.merge(grid_fine)

# %%

for i in range(5, 15):
    for R, q, model in grid.get_R1_index(i):
        if q != 0.5:
            continue
        plt.plot(model.env_mass, model.star.rl_1, alpha=0.8, c=f"C{i-5}")
        print(R)
        plt.plot(model.env_mass, model.star.R, alpha=0.3, c=f"C{i-5}")

plt.show()

# %%

for i, (R, q, model) in enumerate(grid.get_R1_index(4)):
    plt.plot(model.env_mass, model.star.rl_1, alpha=0.8, c=f"C{i}")
    print(R)
    plt.plot(model.env_mass, model.star.R, alpha=0.3, c=f"C{i}")

plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# collect unique sorted grid values
R_vals = sorted(set(np.log10(R) for R, q, _ in grid.iter_models()))
q_vals = sorted(set(q for R, q, _ in grid.iter_models()))

R_vals = np.array(R_vals)
q_vals = np.array(q_vals)

# create empty grid
Z = np.full((len(q_vals), len(R_vals)), np.nan)

# store invalid points for crosses
R_bad = []
q_bad = []

# fill grid
for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == np.log10(R))[0][0]

    if model.env_mass[-1] > 0.1:
        R_bad.append(np.log10(R))
        q_bad.append(q)
    else:
        Z[i, j] = np.max(model.star.R / model.star.rl_1)

# plot
fig, ax = plt.subplots(figsize=set_size(column))

im = ax.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[R_vals.min(), R_vals.max(), q_vals.min(), q_vals.max()],
    cmap="viridis",
    interpolation="none",
)

# overlay crosses

plt.colorbar(im, label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{RL})$")
ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

plt.show()  # %%

# %%
from matplotlib.colors import TwoSlopeNorm

import matplotlib as mpl

mpl.rcParams["hatch.linewidth"] = 0.75  # previous svg hatch linewidth
mpl.rcParams["hatch.color"] = "C9"  # previous svg hatch linewidth

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    Z[i, j] = np.max(model.star.R / rol(model.star))

norm = TwoSlopeNorm(vmin=np.nanmin(Z), vcenter=1.0, vmax=np.nanmax(Z))
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
    np.nan,
)

mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="bwr",
    norm=norm,
    shading="auto",  # important
)
print(overlay)
plt.pcolor(R_edges, q_edges, overlay, hatch="///", alpha=0.0, rasterized=True)


plt.colorbar(mesh, label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{ROL})$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-10.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R / model.star.rl_1)

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
    shading="auto",  # important
)


plt.colorbar(mesh, label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{RL})$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-1.pgf", format="pgf")
plt.show()
plt.close()
# %%


R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = (model.star.star_2_mass[-1] - model.star.star_2_mass[0]) / (
            model.star.star_1_mass[0] - model.star.star_1_mass[-1]
        )
        print(model.star.star_1_mass[0] - model.star.star_1_mass[-1])


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


plt.colorbar(mesh, label=r"$-\Delta M_\textrm{a} / \Delta M_\textrm{d}$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-2.pgf", format="pgf")
plt.show()
plt.close()
# %%


R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.binary_separation[-1] / roche_lobe_separation(R, q)

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


plt.colorbar(mesh, label=r"$a_f / a_i$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-3.pgf", format="pgf")
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

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-4.pgf", format="pgf")
plt.show()
plt.close()
# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
        continue

    if model.age[0] < 0:
        ref_TP_count = 0
    else:
        index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
        ref_TP_count = grid.ref_tpagb.TP_count[index]
    Z[i, j] = model.star.TP_count[-1] + ref_TP_count

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
    label=r"Final thermal pulse count",
)

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-5.pgf", format="pgf")
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
        Z[i, j] = model.star.star_1_mass[-1] + model.star.star_2_mass[-1]
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


plt.colorbar(mesh, label=r"Total mass ($M_\odot$)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-6.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        index = np.argmax(model.star.lg_mtransfer_rate)
        Z[i, j] = (
            model.star.surface_c12[index] / model.star.surface_o16[index] * 16 / 12
        )
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


plt.colorbar(mesh, label=r"C/O-number ratio at max($\dot{M}_\textrm{transfer}$)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-6.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("")
plt.ylabel("")
axs[0].plot(
    grid.ref_ms.star_age, grid.ref_ms.surface_c12 / grid.ref_ms.surface_o16 * 16 / 12
)
print(grid.ref_ms.bulk_names)
axs[1].plot(grid.ref_ms.star_age, grid.ref_ms.c12_core_mass)
plt.show()
plt.close()

# %%

print(model.star.bulk_names)

plt.plot(model.age, model.star.lg_mtransfer_rate)
plt.show()
# %%

for R, q, model in grid.get_R1_index(5):
    print(R)
    plt.plot(model.env_mass, model.star.R / model.star.rl_1, label=f"$q = {q}$")
plt.legend(ncols=7)
plt.show()
# %%

for i, (R, q, model) in enumerate(grid.get_R1_index(5)):
    print(R)
    plt.plot(model.age, model.star.R, label=f"$q = {q}$", c=f"C{i}")
    plt.plot(model.age, model.star.rl_1, label=f"$q = {q}$", c=f"C{i}")
plt.legend(ncols=7)
plt.show()

# %%

for i, (R, q, model) in enumerate(grid.get_R1_index(7)):
    print(R)
    plt.plot(
        model.star.star_1_mass, model.star.star_2_mass, label=f"$q = {q}$", c=f"C{i}"
    )

plt.show()
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

    if model.age[0] < 0:
        ref_TP_count = 0
    else:
        index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
        ref_TP_count = grid.ref_tpagb.TP_count[index]
    Z[i, j] = model.star.TP_count[-1] + ref_TP_count


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


plt.colorbar(mesh, label=r"Final TP count")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    3,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()
axs[-1].axis("off")
axs[-2].axis("off")

axs[-3].set_xlabel("Star age (yr)")
axs[-4].set_xlabel("Star age (yr)")
axs[-5].set_xlabel("Star age (yr)")
axs[0].set_ylabel("Roche lobe radius ($R_\\odot$)")
axs[3].set_ylabel("Roche lobe radius ($R_\\odot$)")
axs[6].set_ylabel("Roche lobe radius ($R_\\odot$)")

for c, q_ref in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.age,
                model.star.rl_1,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[-2].legend(ncols=1, handles=ls, loc="upper left")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-3.pgf", format="pgf")
plt.show()
plt.close()


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

    if model.age[0] < 0:
        ref_TP_count = 0
    else:
        index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
        ref_TP_count = grid.ref_tpagb.TP_count[index]
    Z[i, j] = model.star.period_days[-1]


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

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()
axs[-1].axis("off")
axs[-2].axis("off")

# for axis in axs:
#     axis.invert_xaxis()

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.star.star_2_mass / model.star.star_1_mass,
                model.star.binary_separation,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[-3].set_xlabel("$q$")
axs[-4].set_xlabel("$q$")
axs[-5].set_xlabel("$q$")
axs[0].set_ylabel("Orbital separation ($R_\\odot$)")
axs[3].set_ylabel("Orbital separation ($R_\\odot$)")
axs[6].set_ylabel("Orbital separation ($R_\\odot$)")


axs[-2].legend(ncols=1, handles=ls, loc="upper left")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-4.pgf", format="pgf")
plt.show()
plt.close()

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

    if model.age[0] < 0:
        ref_TP_count = 0
    else:
        index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
        ref_TP_count = grid.ref_tpagb.TP_count[index]
    Z[i, j] = model.star.TP_count[-1] + ref_TP_count


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


plt.colorbar(mesh, label=r"Final TP count")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    3,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()
axs[-1].axis("off")
axs[-2].axis("off")

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.age,
                model.star.rl_1,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[-2].legend(ncols=1, handles=ls, loc="upper left")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-3.pgf", format="pgf")
plt.show()
plt.close()


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

    if model.age[0] < 0:
        ref_TP_count = 0
    else:
        index = np.argwhere(np.abs(grid.ref_tpagb.star_age - model.age[0]) < 1000)[0][0]
        ref_TP_count = grid.ref_tpagb.TP_count[index]
    Z[i, j] = model.star.period_days[-1]


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

plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()
axs[-1].axis("off")
axs[-2].axis("off")

# for axis in axs:
#     axis.invert_xaxis()

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.star.star_2_mass / model.star.star_1_mass,
                model.star.binary_separation,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[-2].legend(ncols=1, handles=ls, loc="upper left")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-4.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    3,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()
axs[-1].axis("off")
axs[-2].axis("off")

# for axis in axs:
#     axis.invert_xaxis()

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.star.star_2_mass / model.star.star_1_mass,
                model.star.R / model.star.rl_1,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.9,
        0.9,
        f"$q = {q_ref}$",
        ha="right",
        va="top",
        transform=axs[c].transAxes,
    )

axs[-2].legend(ncols=1, handles=ls, loc="upper left")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-5.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=0.45),
    constrained_layout=True,
)

axs = axs.flatten()

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.6, 0.7, 0.9]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.age,
                model.star.rl_1,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R:.0f}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[0].set_xlim(859700, 860600)
axs[0].set_ylim(240, 450)

axs[1].set_xlim(859700, 860600)
axs[1].set_ylim(265, 420)

axs[2].set_xlim(977500, 978450)
axs[2].set_ylim(295, 445)

fig.legend(ncols=5, handles=ls, loc="outside upper center")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-6.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1,
    3,
    # sharex=True,
    # sharey=True,
    figsize=set_size(full, height=0.45),
    constrained_layout=True,
)

axs = axs.flatten()

plt.xlabel("")
plt.ylabel("")

for c, q_ref in enumerate([0.6, 0.7, 0.9]):
    ls = []
    for i in range(5, 15):
        for R, q, model in grid.get_R1_index(i):
            if q != q_ref:
                continue
            (l,) = axs[c].plot(
                model.age,
                model.star.R / model.star.rl_1,
                c=f"C{i-5}",
                label=f"$R_\\textrm{{RL}} = {R:.0f}\\;R_\\odot$",
                linewidth=0.5,
            )
            ls.append(l)
            # axs[c].plot(model.age, model.star.R, alpha=0.3, c=f"C{i-5}")
    axs[c].text(
        0.1,
        0.9,
        f"$q = {q_ref}$",
        ha="left",
        va="top",
        transform=axs[c].transAxes,
    )

axs[0].set_xlim(859700, 860600)
# axs[0].set_ylim(240,450)

axs[1].set_xlim(859700, 860600)
# axs[1].set_ylim(265,420)

axs[2].set_xlim(977500, 978450)
# axs[2].set_ylim(295,445)

for ax in axs:
    ax.hlines(
        y=1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color="C9", linewidth=0.75
    )


fig.legend(ncols=5, handles=ls, loc="outside upper center")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-fine-7.pgf", format="pgf")
plt.show()
plt.close()


# %%
