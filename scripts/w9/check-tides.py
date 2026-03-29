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

hist1 = mr.MesaData(f"{MASTER}/tides/4/LOGS/TPAGB/history.data")
# %%
hist2 = mr.MesaData(f"{MASTER}/tides/5/LOGS/TPAGB/history.data")
# %%
hist3 = mr.MesaData(f"{MASTER}/tides/6/LOGS/TPAGB/history.data")
# %%
hist4 = mr.MesaData(f"{MASTER}/tides/7/LOGS/TPAGB/history.data")
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

# plt.plot(hist1.age, hist1.R, c="C9")
plt.plot(hist1.age, hist1.rl_1, label="Incorrect convective turnover")
plt.plot(hist2.age, hist2.rl_1, label="Incorrect convective envelope")
plt.plot(hist3.age, hist3.rl_1, label='"Correct"')
plt.plot(hist4.age[-9000:], hist4.rl_1[-9000:], label="Actually corect")

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w9-tides-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

plt.plot(hist.age, np.abs(hist.jdot_ls))
plt.plot(hist.age, np.abs(hist.jdot_ml))

plt.show()
# %%
