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
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%
grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16-clean")
# %%

grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16", loc="LOGS/TPAGB/")

# %%

rng = np.random.default_rng(seed=9)
models = rng.permutation(grid.models)
for model in models:

    if model.period_days[-1] < 50:
        print(model.params)
        break

plt.plot(model.age, model.R)
plt.plot(model.age, model.rl_1)
plt.show()


# %%

fix_model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/grid-masses-2026-08-14-clean/R800.00_q0.400_eps0.250_delta0.200_M3.0/LOGS/history.data"
)
# %%

plt.plot(fix_model.star_age, fix_model.R)
plt.plot(fix_model.star_age, fix_model.rl_1)
plt.plot(model.star_age, model.R)
plt.show()

# %%

grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16-clean")

# %%

grid_old = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16", loc="LOGS/TPAGB/")
grid_old.models
# %%
fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(full), constrained_layout=True
)

axs = axs.flatten()

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0.01, 3)
    ax.set_yscale("log")

for ax, m in zip(axs, [1.8, 2.2, 2.6, 3.0]):
    m = np.round(m, 1)
    ax.text(
        0.95,
        0.05,
        f"$M = {m:.1f}\\;M_\\odot$",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
    )
    for model in grid.filter(m=m):
        if m != 1.8:
            if model.env_mass[-1] < 0.01:
                ax.plot(model.rl_1, model.env_mass, c="C2", linewidth=2, zorder=1000)
                ax.plot(model.rl_1, model.env_mass, c="white", linewidth=5, zorder=100)
            else:
                (n,) = ax.plot(
                    model.rl_1, model.env_mass, c="C3", label="Unsuccessful model"
                )
        else:
            if model.env_mass[-1] < 0.01:
                (l,) = ax.plot(
                    model.rl_1, model.env_mass, c="C2", label="Successful model"
                )
            else:
                ax.plot(model.rl_1, model.env_mass, c="C3", linewidth=2, zorder=1000)
                ax.plot(model.rl_1, model.env_mass, c="white", linewidth=5, zorder=100)
    for model in grid_old.filter(m=m):
        if m != 1.8:
            if model.env_mass[-1] < 0.01:
                ax.plot(model.rl_1, model.env_mass, c="C2", linewidth=2, zorder=10000)
                ax.plot(model.rl_1, model.env_mass, c="white", linewidth=5, zorder=1000)
            else:
                (n,) = ax.plot(
                    model.rl_1,
                    model.env_mass,
                    c="C3",
                    label="Unsuccessful model",
                    zorder=100000,
                )
        else:
            if model.env_mass[-1] < 0.01:
                (l,) = ax.plot(
                    model.rl_1, model.env_mass, c="C2", label="Successful model"
                )
            else:
                ax.plot(model.rl_1, model.env_mass, c="C3", linewidth=2, zorder=10000)
                ax.plot(model.rl_1, model.env_mass, c="white", linewidth=5, zorder=1000)


# fig.legend(handles=[n, l], loc="outside upper center", ncols=2)

fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel(r"Envelope mass ($M_\odot$)", fontsize=10)
# plt.savefig("/home/koen/LaTeX-setup/plots/w24-grid-1-success.pgf", format="pgf")
plt.show()
plt.close()

# %%
