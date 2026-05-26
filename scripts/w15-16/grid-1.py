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

grid2 = MesaGrid(f"{MASTER}/tides-grid-7")

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
plt.savefig("/home/koen/LaTeX-setup/plots/w15-darwin.pgf", format="pgf")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w15-delta-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

grid = MesaGrid(f"{MASTER}/tides-grid-6")
# %%
grid_rad = MesaGrid(f"{MASTER}/tides-grid-2")

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

plt.savefig("/home/koen/LaTeX-setup/plots/w15-grid-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(6)):
    print(R)
    if model.star.period_days[-1] < 50:
        plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.R, c=f"C9")
        plt.plot(
            model.star.star_2_mass / model.star.star_1_mass, model.star.rl_1, c=f"C9"
        )
        continue

    plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.R, c=f"C{i}")
    plt.plot(
        model.star.star_2_mass / model.star.star_1_mass, model.star.rl_1, c=f"C{i}"
    )

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-grid-10.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(-3)):
    plt.plot(model.env_mass, model.star.period_days, c=f"C{i}", label=f"$q={q:.3f}$")

fig.legend(loc="outside upper center", ncols=4)
plt.gca().invert_xaxis()
plt.xlabel("Envelope mass ($M_\\odot$)")
plt.ylabel("Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-grid-11.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, (R, q, model) in enumerate(grid.get_R1_index(3)):
    print(q, R)
    if i != 3:
        continue
    plt.plot(model.age, model.star.R, label="Star radius")
    plt.plot(model.age, model.star.rl_1, label="Roche lobe")


plt.title(r"$R_\textrm{RL,i} = 244.55, q_\textrm{i} = 0.65$")
fig.legend(loc="outside lower center", ncols=2)
plt.xlim(344220, 344320)
plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-16-zoom-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
for i, (R, q, model) in enumerate(grid.get_R1_index(3)):
    plt.plot(model.env_mass, model.star.period_days, label="Star radius")


plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w16-check-period.pgf", format="pgf")
plt.show()
plt.close()
# %%
