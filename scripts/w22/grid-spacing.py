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

for R in np.logspace(2, np.log10(1250), 50):
    for i, q in enumerate([0.4, 1]):
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star, q, a_init, simple_only=True
        )
        R_star = 10**Star.log_R
        ages_star = Star.age
        bin = Bins[0]

        q_evolve = bin.m2 / bin.m1
        RL = roche_lobe(1 / q_evolve) * bin.a

        plt.plot(bin.age, RL, c="k" if i == 0 else "r")
plt.xlim(1.006e9, 1.014e9)
plt.show()


# %%
