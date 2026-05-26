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

grid = MesaGrid(f"{MASTER}/tides-M3.5")

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
# plt.savefig("/home/koen/LaTeX-setup/plots/w14-darwin.pgf", format="pgf")
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
# plt.savefig("/home/koen/LaTeX-setup/plots/w14-delta-2.pgf", format="pgf")
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

# plt.savefig("/home/koen/LaTeX-setup/plots/w14-grid-9.pgf", format="pgf")
plt.show()
plt.close()


# %%


R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        if model.star.period_days[-1] < 50:
            mask_bad[i, j] = 0.9
        elif -model.star.lg_mstar_dot_1[-1] > model.star.quasi_adiabatic_Mdot[-1]:
            mask_bad[i, j] = 0.5
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

plt.savefig("/home/koen/LaTeX-setup/plots/w15-grid-10000.pgf", format="pgf")
plt.show()
plt.close()


# %%

model.star.bulk_names
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):
    print(R)
    plt.plot(model.age, model.star.R, c=f"C{i}", label=f"$q={q}$")
    plt.plot(model.age, model.star.rl_1, c=f"C{i}", linewidth=3, alpha=0.5)

plt.plot(
    grid.ref_tpagb.star_age,
    grid.ref_tpagb.R,
    c="C9",
    linewidth=3,
    alpha=0.5,
    zorder=-10,
)

fig.legend(loc="outside upper center", ncols=3)
plt.xlim(0.35e6, 0.7e6)
plt.ylim(100, 900)
plt.xlabel("Age (yr)")
plt.ylabel("Radius ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w16-grid2-10.pgf", format="pgf")
plt.show()
plt.close()


# %%
