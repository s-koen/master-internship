import numpy as np
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
from scripts.general_utils.m_dup import compute_m_DUP, AbundanceTables, Abundances

plt.cplot = cplot
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid3 = MesaGrid(f"{MASTER}/grid-masses-3-2026-08-24")
grid4 = MesaGrid(f"{MASTER}/grid-masses-4-2026-08-25")
grid.merge(grid2)
grid.merge(grid3, overwrite=True)
grid.merge(grid4, overwrite=True)

# %%

df = AbundanceTables()

model = grid.models[50]

ab = Abundances(model=model, df=df)

ab.o.envelope
# %%
for model in grid.filter(m=3, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab = Abundances(model, df, method="tp")

element_names = ["he", "c", "n", "o", "ne", "mg"]

init_old = []
for element in element_names:
    init_old.append(ab.__getattr__(element).envelope[star.ntpagb])

# %%
ab = Abundances(model=model, df=df)

init_new = []
for element in element_names:
    init_new.append(ab.__getattr__(element).envelope[star.ntpagb])

# %%


fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

isos = [
    "envelope_he3",
    "envelope_he4",
    "envelope_c12",
    "envelope_c13",
    "envelope_n14",
    "envelope_o16",
    "envelope_ne20",
    "envelope_mg24",
]

init_MESA = []
for element in isos:
    if element in ["envelope_he4", "envelope_c13"]:
        init_MESA[-1] += star.__getattribute__(element)[star.ntpagb]
    else:
        init_MESA.append(star.__getattribute__(element)[star.ntpagb])


elements = ["He", "C", "N", "O", "Ne", "Mg"]
axs[0].plot(elements, init_MESA, label="MESA")
axs[0].plot(elements, init_old, label="Monash post-processing old")
axs[0].plot(elements, init_new, label="Monash post-processing new")
axs[0].set_yscale("log")

axs[1].plot(
    elements, np.abs(np.array(init_old) - np.array(init_MESA)) / np.array(init_MESA)
)
axs[1].plot(
    elements, np.abs(np.array(init_new) - np.array(init_MESA)) / np.array(init_MESA)
)
axs[1].plot(
    elements, np.abs(np.array(init_MESA) - np.array(init_MESA)) / np.array(init_MESA)
)

fig.legend(loc="outside upper center", ncols=3)
plt.yscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Element")
axs[0].set_ylabel("Mass fraction $X$")
# plt.savefig(
#     "/home/koen/LaTeX-setup/plots/w27-initial-envelope-abundance.pgf", format="pgf"
# )
plt.show()
plt.close()
# %%
