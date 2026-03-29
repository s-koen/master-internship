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
MASTER = "/home/koen/master-internship/mesa-models/"

sys.path.insert(1, "/home/koen/master-internship/scripts/evolve/")
import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

from scripts.evolve.constants import *
from scripts.evolve.bin_input import *
from scripts.evolve.read_mist_models import *
from scripts.evolve.mrenv import *
from scripts.evolve.orbit_evol import *
from scripts.evolve.rgbf import *
from scripts.evolve.star_model import *

import pickle as pkl

# %%

with open("/home/koen/master-internship/scripts/evolve/test.pkl", "rb") as file:
    file = pkl.load(file)


hist = mr.MesaData(f"{MASTER}/tides/5/LOGS/TPAGB/history.data")


# %%

print(file)
bin = file[-1][0]
star = file[0]

RL_0 = roche_lobe(0.8) * bin.a
TPAGB = np.where(star.phase >= 5)[0][0]
TPAGB = np.where(bin.age >= star.age[TPAGB])[0][0]
print(TPAGB)
TPAGB_age = bin.age[TPAGB]
age = bin.age - TPAGB_age
print(age)

plt.plot(age[TPAGB:], RL_0[TPAGB:])
plt.plot(hist.age, hist.rl_1)

plt.show()
# %%
