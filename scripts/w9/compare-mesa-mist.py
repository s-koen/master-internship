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
hist2 = mr.MesaData(f"{MASTER}/wind/4/LOGS/TPAGB/history.data")
# %%
hist3 = mr.MesaData(f"{MASTER}/standard-2msun/LOGS/TPAGB/history.data")
# %%

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(full, height=1), constrained_layout=True
)

model_names = [
    r"Pols \texttt{evolve} - Saladino (no accretion)",
    r"Pols \texttt{evolve} - Jeans mode",
]
ls = []
for i, model in enumerate([no_accretion, fast]):

    bin = model[-1][0]
    star = model[0]

    q = bin.m1 / bin.m2
    RL_0 = roche_lobe(q) * bin.a
    TPAGB = np.where(star.phase >= 5)[0][0]
    TPAGB = np.where(bin.age >= star.age[TPAGB])[0][0]

    TPAGB_age = bin.age[TPAGB]
    age = bin.age - TPAGB_age

    (l,) = axs[0].plot(age[TPAGB:], RL_0[TPAGB:], label=model_names[i], c=f"C{2*i}")
    ls.append(l)

axs[1].plot(star.age - TPAGB_age - 6e5, star.radius, label="MIST")
axs[1].plot(hist3.star_age, hist3.R, label="Rees (2023)")
(l,) = axs[0].plot(
    hist2.age + 2.5e6,
    hist2.rl_1,
    label=r"\texttt{MESA} - Saladino (no accretion)",
    c="C1",
)
ls.append(l)
(l,) = axs[0].plot(
    hist.age + 2.5e6, hist.rl_1, label=r"\texttt{MESA} - Jeans mode", c="C3"
)
ls.append(l)
axs[0].set_xlim(2.3e6, 3e6)
axs[0].set_xlabel(r"Age, shifted to match (yr)")
axs[0].set_ylabel(r"Roche lobe radius ($R_\odot$)")
axs[1].set_ylabel(r"Star radius ($R_\odot$)")
axs[1].set_xlabel(r"TPAGB age (yr)")
axs[1].set_xlim(0, 3e6)
axs[0].set_ylim(400)

axs[1].legend(loc="upper center", ncols=2, frameon=False)
fig.legend(loc="outside upper center", ncols=2, handles=ls)

plt.savefig("/home/koen/LaTeX-setup/plots/w9-compare-mesa.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)
model_names = [
    r"Pols \texttt{evolve} - Saladino (no accretion)",
    r"Pols \texttt{evolve} - Saladino (accretion)",
]
ages = []
RL_0s = []
TPAGBs = []
for i, model in enumerate([no_accretion, saladino]):

    bin = model[-1][0]
    star = model[0]

    q = bin.m1 / bin.m2
    RL_0 = roche_lobe(q) * bin.a
    TPAGB = np.where(star.phase >= 5)[0][0]
    TPAGB = np.where(bin.age >= star.age[TPAGB])[0][0]

    TPAGB_age = bin.age[TPAGB]
    age = bin.age - TPAGB_age
    ages.append(age)
    RL_0s.append(RL_0)
    TPAGBs.append(TPAGB)

    (l,) = axs[0].plot(age[TPAGB:], RL_0[TPAGB:], label=model_names[i], c=f"C{2*i}")
    ls.append(l)

# --- difference plot ---
# use first model as reference grid
age_ref = ages[0][TPAGBs[0] :]
RL_ref = RL_0s[0][TPAGBs[0] :]

# second model
age_other = ages[1][TPAGBs[1] :]
RL_other = RL_0s[1][TPAGBs[1] :]

# interpolate second onto first
RL_other_interp = np.interp(age_ref, age_other, RL_other)

# difference
diff = RL_other_interp - RL_ref

axs[1].plot(age_ref, diff, c="k")
axs[1].axhline(0, linestyle="-", linewidth=0.75, c="C9")
axs[1].set_ylabel(r"$\Delta$(Roche lobe radius) ($R_\odot$)")
fig.legend(loc="outside upper center", ncols=2)
axs[1].set_xlabel(r"Age (yr)")
axs[0].set_ylabel(r"Roche lobe radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w9-compare-accretion.pgf", format="pgf")
plt.show()
plt.close()
# %%
