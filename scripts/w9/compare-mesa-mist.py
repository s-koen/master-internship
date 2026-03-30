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

with open("/home/koen/master-internship/scripts/evolve/no-accretion.pkl", "rb") as file:
    no_accretion = pkl.load(file)

with open("/home/koen/master-internship/scripts/evolve/saladino.pkl", "rb") as file:
    saladino = pkl.load(file)

with open("/home/koen/master-internship/scripts/evolve/fast.pkl", "rb") as file:
    fast = pkl.load(file)

with open("/home/koen/master-internship/scripts/evolve/BHL.pkl", "rb") as file:
    BHL = pkl.load(file)

with open("/home/koen/master-internship/scripts/evolve/scaled-vw.pkl", "rb") as file:
    scaled_vw = pkl.load(file)


# %%
hist = mr.MesaData(f"{MASTER}/wind/2/LOGS/TPAGB/history.data")
# %%

model_names = ["no accretion", "saladino", "fast", "BHL", "scaled vw"]
for i, model in enumerate([no_accretion, saladino, fast, BHL, scaled - vw]):

    bin = model[-1][0]
    star = model[0]

    q = bin.m1 / bin.m2
    RL_0 = roche_lobe(q) * bin.a
    TPAGB = np.where(star.phase >= 5)[0][0]
    TPAGB = np.where(bin.age >= star.age[TPAGB])[0][0]

    TPAGB_age = bin.age[TPAGB]
    age = bin.age - TPAGB_age

    plt.plot(age[TPAGB:], RL_0[TPAGB:], label=model_names[i])

plt.plot(hist.age + 2.5e6, hist.rl_1)
plt.plot(hist.age + 2.5e6, hist.R, c="C9")
plt.plot(star.age - TPAGB_age, star.radius, c="C9")
plt.xlim(2e6, 3.2e6)
plt.ylim(300)

plt.legend()
plt.show()
# %%
