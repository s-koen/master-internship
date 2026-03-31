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
default_wind = mr.MesaData(f"{MASTER}/tides/6/LOGS/TPAGB/history.data")
# %%
no_wind = mr.MesaData(f"{MASTER}/wind/1/LOGS/TPAGB/history.data")
# %%
wind = mr.MesaData(f"{MASTER}/wind/2/LOGS/TPAGB/history.data")
# %%
wind_sal = mr.MesaData(f"{MASTER}/wind/4/LOGS/TPAGB/history.data")
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
# plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-5.pgf", format="pgf")
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

# plt.plot(hist1.age, hist1.R, c="C9")
plt.plot(
    default_wind.age,
    default_wind.rl_1,
    label="MESA default",
)
plt.plot(
    wind.age,
    wind.rl_1,
    label="Fast isotropic wind momentum loss",
    alpha=0.4,
    linewidth=6,
    zorder=-1,
    c="C2",
)
plt.plot(
    no_wind.age,
    no_wind.rl_1,
    label="No wind angular momentum loss",
    linestyle=":",
    linewidth=3,
    c="C3",
)
plt.plot(no_wind.age, no_wind.R, c="C9", label="Star Radius")

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.ylim(450, 750)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-wind-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

# plt.plot(no_wind.age, no_wind.jdot_ml)
plt.plot(wind.age, -wind.jdot_ml)
plt.plot(wind_sal.age, -wind_sal.jdot_ml)

plt.yscale("log")
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

# plt.plot(hist1.age, hist1.R, c="C9")
plt.plot(
    wind.age,
    wind.rl_1,
    label="Fast isotropic wind momentum loss",
    zorder=-1,
    c="C2",
)
plt.plot(
    no_wind.age,
    no_wind.rl_1,
    label="No wind angular momentum loss",
    c="C3",
)
plt.plot(
    wind_sal.age,
    wind_sal.rl_1,
    label="Saladino (no accretion)",
    linestyle="-",
    c="C0",
)
plt.plot(no_wind.age, no_wind.R, c="C2", alpha=0.4, linewidth=6)
plt.plot(wind_sal.age, wind_sal.R, c="C0", alpha=0.5, linewidth=6)

fig.legend(loc="outside upper center", ncols=3)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.ylim(350, 750)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-wind-2.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(wind_sal.age, wind_sal.Omega_star)
plt.plot(wind_sal.age, wind_sal.Omega_orb)
plt.show()
# %%
