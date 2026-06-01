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

grid = MesaGrid(f"{MASTER}/tides-grid-4")

# %%

mesa_default = mr.MesaData(
    f"{MASTER}/mesa-rotation-test/R398.69_q0.400/LOGS/TPAGB/history.data"
)

# %%


for R, q, model in grid.get_R1_index(6):
    if q != 0.4:
        continue
    own = model

# %%
own.star.bulk_names
# %%
mesa_default.bulk_names
# %%

plt.plot(own.star.model_number, own.star.Omega_star)
plt.plot(mesa_default.model_number, mesa_default.surf_avg_omega)

plt.show()

# %%

plt.plot(own.star.age, own.star.rl_1)
plt.plot(own.star.age, own.star.R)
plt.plot(mesa_default.age, mesa_default.rl_1)
plt.plot(mesa_default.age, mesa_default.R)
plt.show()
# %%

plt.plot(own.star.envelope_mass, np.abs(own.star.jdot_ls))
plt.plot(own.star.envelope_mass, np.abs(own.star.jdot_ml))

plt.plot(mesa_default.envelope_mass, np.abs(mesa_default.jdot_ls))
plt.plot(mesa_default.envelope_mass, np.abs(mesa_default.jdot_ml))


plt.gca().invert_xaxis()
plt.yscale("log")
plt.show()
# %%

plt.plot(own.star.envelope_mass, own.star.Omega_star * own.star.I_eff / own.star.J_orb)


plt.plot(
    mesa_default.envelope_mass,
    mesa_default.Omega_star * mesa_default.I_eff / mesa_default.J_orb,
)

plt.gca().invert_xaxis()
plt.yscale("log")
plt.show()

# %%

plt.plot(own.star.envelope_mass, own.star.Omega_star)
plt.plot(own.star.envelope_mass, own.star.Omega_orb)

plt.plot(mesa_default.envelope_mass, mesa_default.surf_avg_omega)
plt.plot(mesa_default.envelope_mass, mesa_default.surf_avg_omega_crit)
plt.plot(mesa_default.envelope_mass, mesa_default.Omega_orb)


plt.gca().invert_xaxis()
plt.yscale("log")
plt.show()


# %%

plt.plot(own.star.envelope_mass, own.star.period_days)

plt.plot(mesa_default.envelope_mass, mesa_default.period_days)


plt.gca().invert_xaxis()
plt.yscale("log")
plt.show()


# %%

plt.plot(mesa_default.envelope_mass, mesa_default.surf_avg_omega_div_omega_crit)
plt.gca().invert_xaxis()

plt.show()
# %%

plt.plot(own.star.L)
plt.plot(mesa_default.L)
plt.show()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(own.star.age, own.star.elapsed_time / 3600 * 8)
axs[0].plot(mesa_default.age, mesa_default.elapsed_time / 3600 * 10)

axs[1].plot(own.star.age, own.star.R)
axs[1].plot(own.star.age, own.star.rl_1)
axs[1].plot(mesa_default.age, mesa_default.R)
axs[1].plot(mesa_default.age, mesa_default.rl_1)

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w18-compare-wall-time.pgf", format="pgf")
plt.show()
plt.close()

plt.show()
# %%


def crit(R, M):
    return (2 / 3) ** (3 / 2) * (G * M / (R**3)) ** (1 / 2)


# %%
fig, axs = plt.subplots(
    4, 2, sharex="col", figsize=set_size(full, height=1.333), constrained_layout=True
)

axs = axs.flatten()

(h1,) = axs[0].plot(own.star.envelope_mass, own.star.period_days, label="custom")
(h2,) = axs[0].plot(
    mesa_default.envelope_mass, mesa_default.period_days, label="MESA default"
)
axs[0].set_ylabel(r"Period (days)")

axs[2].plot(own.star.envelope_mass, own.star.R)
axs[2].plot(mesa_default.envelope_mass, mesa_default.R)
axs[2].set_ylabel(r"Star radius ($R_\odot$)")

# axs[2].plot(own.star.envelope_mass,own.star.Omega_star)
# axs[2].plot(mesa_default.envelope_mass,mesa_default.surf_avg_omega_crit)
axs[4].plot(mesa_default.envelope_mass, mesa_default.surf_avg_omega_crit, alpha=0)
axs[4].plot(mesa_default.envelope_mass, mesa_default.surf_avg_omega_div_omega_crit)
axs[4].set_xlim(axs[2].get_xlim())
axs[4].hlines(1, 2, 0, color="C9", linewidth=0.75)
axs[4].set_ylabel(r"$\Omega_\textrm{star} / \Omega_\textrm{crit}$")

axs[5].plot(own.star.age, own.star.rl_1)
axs[5].plot(mesa_default.age, mesa_default.rl_1)
axs[5].set_ylabel(r"Roche lobe radius ($R_\odot$)")
axs[5].set_ylim(365, 385)

axs[7].plot(own.star.age, own.star.R)
axs[7].plot(mesa_default.age, mesa_default.R)
axs[7].set_ylabel(r"Star radius ($R_\odot$)")
axs[7].set_ylim(150, 280)

axs[1].axis("off")
axs[3].axis("off")

axs[6].plot(own.star.envelope_mass, np.abs(own.star.jdot_ls))
axs[6].plot(mesa_default.envelope_mass, np.abs(mesa_default.jdot_ls))
axs[6].set_xlim(axs[2].get_xlim())
axs[6].set_yscale("log")
axs[6].set_ylim(1e35)
axs[6].set_ylabel(r"$\left|\dot{J}_\textrm{ls-coupling}\right|$ (g cm$^{-2}$ s$^{-2}$)")


axs[0].invert_xaxis()

axs[3].legend(loc="lower left", handles=[h1, h2])

axs[6].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[7].set_xlabel(r"Time (yr)")


plt.savefig("/home/koen/LaTeX-setup/plots/w18-tides-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

plt.xlabel("")
plt.ylabel("")

plt.plot(own.star.envelope_mass, own.star.jdot_ls, label="custom")
plt.plot(mesa_default.envelope_mass, mesa_default.jdot_ls, label="MESA default")
plt.gca().invert_xaxis()
ax = plt.gca()
plt.xlim(1.5, 0.5)
plt.ylim(-3e43, 3e43)

# inset Axes....
x1, x2, y1, y2 = 1.428, 1.412, -5e41, 1e41  # subregion of the original image
axins = ax.inset_axes([0.5, 0.1, 0.47, 0.3], xlim=(x1, x2), ylim=(y1, y2))

axins.plot(own.star.envelope_mass, own.star.jdot_ls)
axins.plot(mesa_default.envelope_mass, mesa_default.jdot_ls)
axins.set_facecolor("None")

inset_indicator = ax.indicate_inset_zoom(axins, edgecolor="C9")
inset_indicator.connectors[0].set_visible(True)
inset_indicator.connectors[1].set_visible(True)
inset_indicator.connectors[2].set_visible(False)
inset_indicator.connectors[3].set_visible(False)

axins.set_ylim(-5e41, 1e41)


x1, x2, y1, y2 = 1.426, 1.418, -1e40, 2e40  # subregion of the original image
axins2 = ax.inset_axes([0.5, 0.6, 0.47, 0.3], xlim=(x2, x1), ylim=(y1, y2))
axins2.plot(own.star.envelope_mass, own.star.jdot_ls)
axins2.plot(mesa_default.envelope_mass, mesa_default.jdot_ls)

inset_indicator2 = axins.indicate_inset_zoom(axins2, edgecolor="C9")
inset_indicator2.connectors[0].set_visible(True)
inset_indicator2.connectors[1].set_visible(False)
inset_indicator2.connectors[2].set_visible(True)
inset_indicator2.connectors[3].set_visible(False)


axins2.invert_xaxis()

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Envelope mass ($M_\odot$)")
plt.ylabel(r"$\dot{J}_\textrm{ls-coupling}$ (g cm$^{-2}$ s$^{-2}$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w18-ls-zoom.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

plt.xlabel("")
plt.ylabel("")

plt.plot(own.star.age, own.star.jdot_ls, label="custom")
plt.plot(mesa_default.age, mesa_default.jdot_ls, label="MESA default")
ax = plt.gca()
plt.ylim(-3e43, 3e43)

# inset Axes....
x1, x2, y1, y2 = 0, 50000, -1e42, 1e42  # subregion of the original image
axins = ax.inset_axes([0.2, 0.1, 0.47, 0.3], xlim=(x1, x2), ylim=(y1, y2))

axins.plot(own.star.age, own.star.jdot_ls)
axins.plot(mesa_default.age, mesa_default.jdot_ls)
axins.set_facecolor("None")


inset_indicator = ax.indicate_inset_zoom(axins, edgecolor="C9")
inset_indicator.connectors[0].set_visible(True)
inset_indicator.connectors[1].set_visible(False)
inset_indicator.connectors[2].set_visible(False)
inset_indicator.connectors[3].set_visible(True)


axins.set_ylim(-1e40, 1e40)


x1, x2, y1, y2 = 87750, 88500, -8e41, 2e41  # subregion of the original image
axins2 = ax.inset_axes([0.4, 0.6, 0.47, 0.3], xlim=(x1, x2), ylim=(y1, y2))

axins2.plot(own.star.age, own.star.jdot_ls)
axins2.plot(mesa_default.age, mesa_default.jdot_ls)
axins2.set_facecolor("None")


inset_indicator = ax.indicate_inset_zoom(axins2, edgecolor="C9")
inset_indicator.connectors[0].set_visible(True)
inset_indicator.connectors[1].set_visible(False)
inset_indicator.connectors[2].set_visible(False)
inset_indicator.connectors[3].set_visible(True)


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"$\dot{J}_\textrm{ls-coupling}$ (g cm$^{-2}$ s$^{-2}$)")

plt.savefig("/home/koen/LaTeX-setup/plots/w18-ls-zoom-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(own.star.envelope_mass, own.star.Omega_star / own.star.Omega_orb)
axs[0].plot(
    mesa_default.envelope_mass, mesa_default.surf_avg_omega / mesa_default.Omega_orb
)

axs[1].plot(own.star.envelope_mass, own.star.jdot_ls)
axs[1].plot(mesa_default.envelope_mass, mesa_default.jdot_ls)

plt.gca().invert_xaxis()


plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w18-spins.pgf", format="pgf")
plt.show()
plt.close()
# %%
