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
from scripts.general_utils.mesa_grid import MesaGrid

# %%
grid2 = MesaGrid(f"{MASTER}/tides-grid-7")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

R_vals = np.array(sorted(set(R for R, q, _ in grid2.iter_models())))

i = 0
for model in grid2.iter_models():
    R, q, m = model
    if R == R_vals[3]:
        if m.env_mass[-1] > 0.1:
            axs.plot(
                m.star.star_2_mass / m.star.star_1_mass,
                m.star.log_R,
                c=f"C3",
                linewidth=1,
            )
        else:
            axs.plot(
                m.star.star_2_mass / m.star.star_1_mass,
                m.star.log_R,
                c=f"C2",
                linewidth=1,
            )

        i += 1


plt.gca().invert_xaxis()

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log($T_\\textrm{eff} / \\textrm{K}$)")
plt.ylabel("log($L / L_\\odot$)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-see-same-pulse-3.pgf", format="pgf")
plt.show()
plt.close()
