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

G = 6.67430e-8


# %%

custom = mr.MesaData(
    f"{MASTER}/mesa-rotation-test/early-custom/LOGS/TPAGB/history.data"
)


standard = mr.MesaData(
    f"{MASTER}/mesa-rotation-test/early-mesa/LOGS/TPAGB/history.data"
)

BSE = mr.MesaData(f"{MASTER}/mesa-rotation-test/early-BSE/LOGS/TPAGB/history.data")


# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(full, height=1.25), constrained_layout=True
)

axs[0].plot(standard.age, standard.R, label="Star MESA")
axs[0].plot(custom.age, custom.R, c="C0", alpha=0.4, linewidth=4, label="Star custom")

axs[0].plot(standard.age, standard.rl_1, label="Roche lobe MESA")
axs[0].plot(
    custom.age, custom.rl_1, c="C1", alpha=0.4, linewidth=4, label="Roche lobe custom"
)

axs[0].legend(framealpha=0, ncols=2, loc="upper left")

interp_rl_1 = np.interp(standard.age, custom.age, custom.rl_1)
interp_r = np.interp(standard.age, custom.age, custom.R)

rdiff = standard.R - interp_r
t = standard.age


def plot_segments(ax, x, y, mask, label, **kwargs):

    ls = []
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return

    splits = np.where(np.diff(idx) > 1)[0] + 1

    for segment in np.split(idx, splits):
        if len(segment) == 1:
            ax.scatter(x[segment], np.abs(y[segment]), **kwargs)
        else:
            if len(ls) != 0:
                print("true")
                ax.plot(x[segment], np.abs(y[segment]), **kwargs)
            else:
                l = ax.plot(x[segment], np.abs(y[segment]), label=label, **kwargs)
                ls.append(l)


plot_segments(axs[1], t, rdiff, rdiff > 0, r"Star ($\Delta > 0$)", color="C0")
plot_segments(axs[1], t, rdiff, rdiff < 0, r"Star ($\Delta < 0$)", color="C2")


t = standard.age
rl = standard.rl_1 - interp_rl_1

plot_segments(axs[1], t, rl, rl > 0, r"Roche lobe ($\Delta > 0$)", color="C1")
plot_segments(axs[1], t, rl, rl < 0, r"Roche lobe ($\Delta < 0$)", color="C3")

axs[1].legend(framealpha=0)

axs[0].set_ylim(80, 380)
axs[1].set_ylim(1e-5, 400)
axs[1].set_yscale("log")
axs[1].grid(axis="y")
axs[1].set_ylabel(r"$|$residual$|$ (MESA - custom)")
axs[0].set_ylabel(r"Radius ($R_\odot$)")

plt.xlabel("Time (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w18-tides-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=1.25), constrained_layout=True
)

axs.plot(standard.age, standard.R, label="Star MESA")
axs.plot(custom.age, custom.R, c="C0", alpha=0.4, linewidth=4, label="Star custom")
axs.plot(BSE.age, BSE.R, c="C2", alpha=0.4, linewidth=4, label="Star custom")

axs.plot(standard.age, standard.rl_1, label="Roche lobe MESA")
axs.plot(
    custom.age, custom.rl_1, c="C1", alpha=0.4, linewidth=4, label="Roche lobe custom"
)
axs.plot(BSE.age, BSE.rl_1, c="C2", alpha=0.4, linewidth=4, label="Roche lobe custom")

plt.show()

# %%

plt.plot(BSE.age, BSE.t_conv)
plt.plot(standard.age, standard.t_conv)

plt.show()

# %%


plt.plot(BSE.age, BSE.jdot_ls)
plt.plot(standard.age, standard.jdot_ls)
plt.plot(standard.age, standard.extra_jdot)
plt.plot(custom.age, custom.jdot_ls)

plt.ylim(-3e40, 5e40)
plt.show()


# %%

rl_1BSE = np.interp(BSE.age, standard.age, standard.rl_1)

plt.plot(BSE.age, BSE.rl_1 / rl_1BSE, label="Roche lobe MESA")

plt.show()

# %%

standard.bulk_names
# %%

plt.plot(BSE.age, BSE.Jdot)
plt.plot(standard.age, standard.Jdot)
plt.plot(custom.age, custom.Jdot)

plt.ylim(-3e40, 5e40)
plt.show()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

dOmegadt = BSE.jdot_ls / BSE.I_eff
interp_R = np.interp(BSE.age, standard.age, standard.R)
dOmegadt_mod = (
    dOmegadt
    / (BSE.R / BSE.binary_separation) ** 6
    * (interp_R / BSE.binary_separation) ** 6
)

G = 6.67430e-8
R_sun = 6.957e10  # cm


def reconstruct_separation(a0_rsun, m1_arr, m2_arr, jdot, dt, return_rsun=True):

    # convert initial separation to cgs
    a = a0_rsun * R_sun

    Msun = 1.989e33
    dt = dt * 365.25 * 24 * 3600

    m1_arr = m1_arr * Msun
    m2_arr = m2_arr * Msun

    m1 = m1_arr[0]
    m2 = m2_arr[0]

    M = m1 + m2
    mu = (m1 * m2) / M

    # initial orbital angular momentum (cgs)
    J = mu * np.sqrt(G * M * a)

    a_list = []
    J_list = []

    for i, (dj, dti) in enumerate(zip(jdot, dt)):

        m1 = m1_arr[i]
        m2 = m2_arr[i]

        M = m1 + m2
        mu = (m1 * m2) / M
        J += dj * dti * 10
        a = (J**2) / (mu**2 * G * M)

        a_list.append(a)
        J_list.append(J)

    a_list = np.array(a_list)

    if return_rsun:
        a_list = a_list / R_sun

    return a_list, np.array(J_list)


Jdot_MESA = standard.Jdot
Jdot_custom = custom.Jdot
Jdot_mod = dOmegadt_mod * BSE.I_eff + BSE.jdot_ml + BSE.jdot_gr
Jdot_BSE = dOmegadt * BSE.I_eff + BSE.jdot_ml + BSE.jdot_gr

plt.plot(
    custom.age,
    reconstruct_separation(
        custom.binary_separation[0],
        custom.star_1_mass,
        custom.star_2_mass,
        Jdot_custom,
        custom.dt,
    )[0],
    label="custom",
)
plt.plot(
    standard.age,
    reconstruct_separation(
        standard.binary_separation[0],
        standard.star_1_mass,
        standard.star_2_mass,
        Jdot_MESA,
        standard.dt,
    )[0],
    label="MESA",
)
plt.plot(
    BSE.age,
    reconstruct_separation(
        BSE.binary_separation[0], BSE.star_1_mass, BSE.star_2_mass, Jdot_BSE, BSE.dt
    )[0],
    label="BSE",
)
plt.plot(
    BSE.age,
    reconstruct_separation(
        BSE.binary_separation[0], BSE.star_1_mass, BSE.star_2_mass, Jdot_mod, BSE.dt
    )[0],
    label="BSE + radius post-processing",
)
fig.legend(loc="outside upper center", ncols=4)
plt.ylim(765, 815)


plt.xlabel("Time (yr)")
plt.ylabel(r"Separation ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w18-compare-separation.pgf", format="pgf")
plt.show()
plt.close()
# %%

plt.plot(standard.age, standard.R)
plt.plot(BSE.age, BSE.R)
plt.show()
# %%
