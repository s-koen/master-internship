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
from scripts.general_utils.m_dup import compute_m_DUP

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")

# %%

for m in grid.models:
    star = get_star(m=m.params["m"])
    plt.plot(star.age, m.sb.a)
    plt.plot(m.age, m.binary_separation)
    break
plt.show()
# %%

for m in grid.models:
    star = get_star(m=m.params["m"])
    plt.plot(star.age, m.sb.m2)
    plt.plot(m.age, m.star_2_mass)
    break
plt.show()
# %%
