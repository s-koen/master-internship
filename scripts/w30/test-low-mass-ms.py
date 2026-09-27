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

low = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-ms-stars-3/M0.6/LOGS/MS/history.data"
)
# %%

low.bulk_names
# %%

plt.cplot(np.log10(low.min_kapR), np.log10(low.min_T), low.star_age)
plt.show()
# %%

low_AESO = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-ms-stars-3/M0.6-AESO/LOGS/MS/history.data"
)
# %%

plt.cplot(np.log10(low_AESO.min_kapR), np.log10(low_AESO.min_T), low_AESO.star_age)
plt.cplot(np.log10(low.min_kapR), np.log10(low.min_T), low.star_age)
plt.show()

# %%

plt.cplot(low_AESO.log_Teff, low_AESO.log_L, low_AESO.star_age)
plt.cplot(low.log_Teff, low.log_L, low.star_age)
plt.gca().invert_xaxis()
plt.show()


# %%
