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

hist = mr.MesaData(f"{MASTER}/binary-tides/LOGS/TPAGB/history.data")

# %%
plt.plot(hist.age, hist.Omega_star)
plt.plot(hist.age, hist.Omega_orb)
# plt.plot(hist.age,hist.binary_separation)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.plot(
    hist.age, hist.Omega_orb, c="C0", label=r"$\Omega_\textrm{orbit}$", linestyle="--"
)
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.rl_1, c="C2", label="Roche Lobe radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.R, c="C2", label="Star radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

hist = mr.MesaData(f"{MASTER}/binary-tides-2/LOGS/TPAGB/history.data")


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.R, c="C2", label="Star radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-3.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.plot(
    hist.age, hist.Omega_orb, c="C0", label=r"$\Omega_\textrm{orbit}$", linestyle="--"
)
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.rl_1, c="C2", label="Roche Lobe radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-4.pgf", format="pgf")
plt.show()
plt.close()


# %%

hist = mr.MesaData(f"{MASTER}/binary-tides-4/LOGS/TPAGB/history.data")


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.R, c="C2", label="Star radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-5.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.plot(
    hist.age, hist.Omega_orb, c="C0", label=r"$\Omega_\textrm{orbit}$", linestyle="--"
)
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.rl_1, c="C2", label="Roche Lobe radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-6.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

plt.plot(hist.age, hist.R, label="Star radius")
plt.plot(hist.age, hist.rl_1, label="Roche lobe radius")

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-7.pgf", format="pgf")
plt.show()
plt.close()
# %%

plt.plot(hist.age, np.abs(hist.jdot_ls))
plt.plot(hist.age, np.abs(hist.jdot_ml))

plt.show()
# %%
