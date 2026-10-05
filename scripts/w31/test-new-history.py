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

# %%

h = mr.MesaData(
    "/home/koen/master-internship/mesa-models/test-new-history/LOGS/history.data"
)


# plt.plot(h.age, h.sad_m_min)
plt.plot(h.age, h.sad_s_m_max)
plt.plot(h.age, h.min_S)
plt.plot(h.age, h.surface_S)
# plt.plot(h.age, h.R / 5)
plt.show()
# %%

plt.plot(h.radius_min_S, h.min_S)
plt.show()
# %%

plt.plot(h.age, h.thermal_time_min_S / 3600 / 24 / 365)
plt.show()
# %%

plt.plot(h.age, h.R)
plt.plot(h.age, h.rl_1)
plt.show()
# %%

plt.plot(h.star_age, h.star_mass - h.mass_min_S / 1.989e33)
plt.show()
# %%

h.bulk_names
# %%

plt.plot(h.age, h.sad_r_m_max - h.sad_r_m_min)
plt.show()
# %%
