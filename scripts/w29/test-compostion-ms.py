import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)

plt.cplot = cplot
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

data = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M1.0/LOGS/MS/profile1.data"
)
data.bulk_names
# %%

print(np.logspace(-4, np.log10(0.7), 50))
# %%

profiles = []
for i in range(1, 41):
    profiles.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-ms-stars/M1.9/LOGS/MS/profile{i}.data"
        )
    )
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


for i, profile in enumerate(profiles):
    alpha = i / len(profiles)
    alpha = alpha * 0.85 + 0.15
    (l1,) = plt.plot(profile.mass, profile.h1, c="C1", alpha=alpha, label="X")
    (l2,) = plt.plot(profile.mass, profile.he4, c="C2", alpha=alpha, label="Y")
    (l3,) = plt.plot(
        profile.mass, profile.z_mass_fraction_metals, c="C3", alpha=alpha, label="Z"
    )

fig.legend(loc="outside upper center", ncols=3, handles=[l1, l2, l3])
plt.ylim(1e-3)
plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$")
plt.ylabel("Mass fraction")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-change-in-Z-envelope.pgf", format="pgf")
plt.show()
plt.close()
# %%
