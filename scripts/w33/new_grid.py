import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.plot_helpers import *
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.accretor import *
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

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

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)


Star = get_star(m=2.4)
R = 1100
qs = [0.4, 0.55, 0.7, 0.85, 1]
for q in qs:
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
        Star, q, a_init, simple_only=True
    )

    R_star = 10**Star.log_R
    ages_star = Star.age
    bin = Bins[0]

    q_evolve = bin.m2 / bin.m1
    RL = roche_lobe(1 / q_evolve) * bin.a

    plt.plot(bin.age, RL, label=f"$q = {q:.2f}$", linewidth=1)
plt.plot(Star.age, 10**Star.log_R, c="C9", zorder=-1, linewidth=1)

from scipy.interpolate import interp1d

# envelope mass as a function of age
age = Star.age
m_env = Star.mass - Star.m_core

# ensure the interpolation data are sorted by age
idx = np.argsort(age)
age = age[idx]
m_env = m_env[idx]

# interpolate in both directions
age_to_m_env = interp1d(age, m_env, bounds_error=False, fill_value="extrapolate")
m_env_to_age = interp1d(
    m_env[::-1], age[::-1], bounds_error=False, fill_value="extrapolate"
)

# secondary axis
secax = axs.secondary_xaxis(
    "top",
    functions=(age_to_m_env, m_env_to_age),
)
secax.set_xlabel(r"Envelope mass ($M_\odot$)")
secax.set_xticks([1.75, 1.7, 1.6, 1.5, 1, 0.8, 0.5, 0.2, 0])
secax.set_xticklabels(["1.75", "1.7", "1.6", "1.5", "1", "0.8", "0.5", "0.2", "0"])


plt.xlabel("Age (yr)")
plt.ylabel("Radius ($R_\\odot$)")

fig.legend(loc="outside upper center", ncols=3)

plt.xlim(726213946.5060341, 726512869.7479917)
plt.ylim(100, 1200)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w33-test-higher-mass-low-env.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
