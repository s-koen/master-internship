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

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.cache import get_star
from scripts.general_utils.mesa_grid import MesaGrid

# %%

w = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M2.5/LOGS/TPAGB/history.data"
)
# %%

r = mr.MesaData(f"{MASTER}/single-stars/z0.00557/fixes/M2.5/LOGS/TPAGB/history.data")
# %%

plt.plot(r.star_age, r.log_R)
plt.plot(w.star_age, w.log_R)
plt.plot(w.star_age, w.surface_c12 / w.surface_o16)
plt.plot(w.star_age, w.envelope_c12 / w.envelope_o16)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16)
plt.plot(r.star_age, r.surface_c12 / r.surface_o16)
plt.show()
# %%
plt.plot(r.model_number, r.log_R)
plt.show()
# %%

with open(
    f"/home/koen/master-internship/mesa-models/single-stars/new-abundances/M2.0/combined_star.pkl",
    "rb",
) as f:
    Star = pickle.load(f)
# %%
plt.plot(Star.age, Star.log_R)
plt.show()

# %%

low = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M1.4/LOGS/TPAGB/history.data"
)
# %%

plt.plot(low.star_age, low.log_R)
plt.plot(r.star_age, r.log_R)
plt.plot(low.star_age, low.envelope_c12 / low.envelope_o16 * 16 / 12)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16 * 16 / 12)
plt.show()

# %%

med = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M1.7/LOGS/TPAGB/history.data"
)
# %%

plt.plot(low.star_age, low.log_R)
plt.plot(med.star_age, med.log_R)
plt.plot(r.star_age, r.log_R)
plt.plot(low.star_age, low.envelope_c12 / low.envelope_o16 * 16 / 12)
plt.plot(med.star_age, med.envelope_c12 / med.envelope_o16 * 16 / 12)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16 * 16 / 12)
plt.show()


# %%
plt.plot(low.star_age, low.lambda_DUP)
plt.plot(med.star_age, med.lambda_DUP)
plt.plot(r.star_age, r.lambda_DUP)
plt.show()


# %%
last = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M1.9/LOGS/TPAGB/history.data"
)
# %%

plt.plot(low.star_age, low.log_R)
plt.plot(med.star_age, med.log_R)
plt.plot(last.star_age, last.log_R)
plt.plot(r.star_age, r.log_R)
plt.plot(low.star_age, low.envelope_c12 / low.envelope_o16 * 16 / 12)
plt.plot(med.star_age, med.envelope_c12 / med.envelope_o16 * 16 / 12)
plt.plot(last.star_age, last.envelope_c12 / last.envelope_o16 * 16 / 12)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16 * 16 / 12)
plt.show()


# %%

w = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M2.1/LOGS/TPAGB/history.data"
)
# %%

r = mr.MesaData(f"{MASTER}/single-stars/z0.00557/fixes/M2.1/LOGS/TPAGB/history.data")

# %%

plt.plot(r.star_age, r.log_R)
plt.plot(w.star_age, w.log_R)
plt.plot(w.star_age, w.surface_c12 / w.surface_o16)
plt.plot(w.star_age, w.envelope_c12 / w.envelope_o16)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16)
plt.plot(r.star_age, r.surface_c12 / r.surface_o16)
plt.show()

# %%

w = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M2.2/LOGS/TPAGB/history.data"
)
# %%

r = mr.MesaData(f"{MASTER}/single-stars/z0.00557/fixes/M2.2/LOGS/TPAGB/history.data")

# %%

plt.plot(r.star_age, r.log_R)
plt.plot(w.star_age, w.log_R)
plt.plot(w.star_age, w.surface_c12 / w.surface_o16)
plt.plot(w.star_age, w.envelope_c12 / w.envelope_o16)
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16)
plt.plot(r.star_age, r.surface_c12 / r.surface_o16)
plt.show()

# %%

w = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/completed/M1.0/LOGS/TPAGB/history.data"
)
# %%

plt.plot(w.star_age, w.log_R)
plt.plot(w.star_age, w.star_mass)

plt.show()
# %%

models = []
for m in range(0, 10):
    models.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00557/completed/M1.{m}/LOGS/TPAGB/history.data"
        )
    )


# %%

colors = plt.cm.viridis(np.linspace(0, 1, 10))

for i, model in enumerate(models):
    plt.plot(
        model.star_age,
        model.envelope_c12 / model.envelope_o16 * 16 / 12,
        color=colors[i],
    )
plt.show()
# %%

bad = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M2.4/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(2, 2, figsize=set_size(full), constrained_layout=True)

start = -23000

for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)

axs[0, 0].plot(bad.star_age[start:], bad.R[start:], c="k", linewidth=0.8)
axs[0, 0].scatter(bad.star_age[-1], bad.R[-1], c="k")
axs[0, 0].set_xlabel("star age (yr)")
axs[0, 0].set_ylabel(r"radius ($R_\odot$)")

axs[0, 1].plot(bad.envelope_mass[start:], bad.R[start:], c="k", linewidth=0.8)
axs[0, 1].scatter(bad.envelope_mass[-1], bad.R[-1], c="k")
axs[0, 1].invert_xaxis()
axs[0, 1].set_xlabel("envelope mass ($M_\odot$)")
axs[0, 1].set_ylabel(r"radius ($R_\odot$)")

axs[1, 0].plot(bad.log_Teff[start:], bad.log_L[start:], c="k", linewidth=0.8)
axs[1, 0].scatter(bad.log_Teff[-1], bad.log_L[-1], c="k")
axs[1, 0].invert_xaxis()
axs[1, 0].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[1, 0].set_ylabel(r"$\log(L / L_\odot)$")

axs[1, 1].plot(bad.model_number[start:], bad.log_dt[start:], c="k", linewidth=0.8)
axs[1, 1].scatter(bad.model_number[-1], bad.log_dt[-1], c="k")
axs[1, 1].set_xlabel(r"model number")
axs[1, 1].set_ylabel(r"$\log(\Delta t / \textrm{yr})$")


plt.savefig("/home/koen/LaTeX-setup/plots/w22-bad-M2.4.pgf", format="pgf")
plt.show()
plt.close()

# %%
w = mr.MesaData(
    f"{MASTER}/single-stars/z0.00557/first-try/M2.3/LOGS/TPAGB/history.data"
)
# %%

r = mr.MesaData(f"{MASTER}/single-stars/z0.00557/fixes/M2.4/LOGS/TPAGB/history.data")
# %%


# plt.plot(r.star_age, r.envelope_mass)
plt.plot(r.star_age, r.R)
plt.plot(star3.age - star3.age[star3.ntpagb], 10**star3.log_R)
# plt.plot(w.star_age, w.R)
# plt.plot(w.star_age, w.surface_c12 / w.surface_o16)
# plt.plot(w.star_age, w.envelope_c12 / w.envelope_o16)
# plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16)
# plt.plot(r.star_age, r.surface_c12 / r.surface_o16)

plt.show()

# %%

star = get_star(m=2.2)
star2 = get_star(m=2.1)
star3 = get_star(m=2.5)
star4 = get_star(m=2.0)
star5 = get_star(m=2.3)
star6 = get_star(m=2.4)
# %%
plt.plot(star.age, star.log_R)
plt.plot(star2.age, star2.log_R)
plt.plot(star3.age, star3.log_R)
plt.plot(star4.age, star4.log_R)
plt.plot(star5.age, star5.log_R)
plt.plot(star6.age, star6.log_R)
plt.show()

# %%

plt.plot(np.cumsum(star.m_DUP))
plt.plot(np.cumsum(star2.m_DUP))
plt.plot(np.cumsum(star3.m_DUP))
plt.show()
# %%

for s in [star, star2, star3, star4]:
    plt.plot(
        s.age[s.ntpagb :] - s.age[s.ntpagb],
        s.envelope_c12[s.ntpagb :] / s.envelope_o16[s.ntpagb :] * 16 / 12,
    )
plt.plot(r.star_age, r.envelope_c12 / r.envelope_o16 * 16 / 12)
plt.show()
# %%

for s in [star, star2, star3, star4]:
    plt.plot(
        10 ** s.log_R[s.ntpagb :],
        s.envelope_c12[s.ntpagb :] / s.envelope_o16[s.ntpagb :] * 16 / 12,
    )
plt.plot(r.R, r.envelope_c12 / r.envelope_o16 * 16 / 12)
plt.show()

# %%

r = mr.MesaData(f"{MASTER}/single-stars/z0.00557/fixes/M2.6/LOGS/TPAGB/history.data")
# %%

plt.plot(r.star_age, r.R)
plt.show()
# %%

for s in [star, star2, star3, star4]:
    plt.plot(s.age[s.ntpagb :] - s.age[s.ntpagb], 10 ** s.log_R[s.ntpagb :])
plt.plot(r.star_age, 10**r.log_R)
plt.show()

# %%
