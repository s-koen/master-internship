import os
import numpy as np
import mesa_reader as mr
import sys

# WARNING: CHECK / MODIFY THESE PATHS
proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/new-abundances/M2.0/"

# WARNING: SETTINGS FOR THE GRID

qs = np.linspace(0.4, 1, 7)

sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

models_dir = f"{single_star_dir}/models/"
Star = read_stellar_models(single_star_dir)[0]
# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

cmap = plt.get_cmap("viridis_r")  # choose any colormap you like
norm = mcolors.Normalize(vmin=0.32, vmax=1)

for j, R in enumerate(np.logspace(np.log10(300), np.log10(400), 7)):
    print(R)
    for i, q in enumerate(qs):
        color = cmap(norm(q))
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star, q, a_init, simple_only=True
        )
        R_star = 10**Star.log_R
        ages_star = Star.age
        bin = Bins[0]

        q_evolve = bin.m2 / bin.m1
        RL = roche_lobe(1 / q_evolve) * bin.a

        plt.plot(
            [1.008e9, 1.0085e9, bin.age[-1]],
            [RL[0], RL[0], RL[-1]],
            c=color,
            linewidth=0.8,
            zorder=1 + 2 * j,
        )
        plt.plot(
            [1.008e9, 1.0085e9, bin.age[-1]],
            [RL[0], RL[0], RL[-1]],
            c="w",
            linewidth=3,
            zorder=0 + 2 * j,
        )
        plt.scatter(
            [1.008e9, bin.age[-1]],
            [RL[0], RL[-1]],
            color=color,
            zorder=101 + 10 * j,
            s=10,
        )
        plt.scatter(
            [1.008e9, bin.age[-1]],
            [RL[0], RL[-1]],
            color="w",
            zorder=100 + 10 * j,
            s=40,
        )
    plt.text(
        1.00825e9, RL[0] + 2, f"$R_\\textrm{{RL}} = {R:.0f}$", ha="center", va="bottom"
    )


plt.plot(Star.age, 10**Star.log_R, c="C9")
plt.xlim(1.0079e9, 1.0105e9)
plt.ylim(250, 425)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, label="$q$", ax=axs, aspect=50)

plt.xlabel("Star Age (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-spacing.pgf", format="pgf")
plt.show()
plt.close()


# %%
