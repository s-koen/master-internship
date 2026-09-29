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

broken = mr.MesaData(
    "/home/koen/master-internship/mesa-models/R150.85_q0.600_eps0.250_delta0.200_M2.2/LOGS/history.data"
)
broken2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/R230.26_q0.600_eps0.250_delta0.200_M2.2/LOGS/history.data"
)

good = mr.MesaData(
    "/home/koen/master-internship/mesa-models/R241.34_q0.600_eps0.250_delta0.200_M2.2/LOGS/history.data"
)


# %%

plt.plot(broken.age, broken.R)
plt.plot(broken2.age, broken2.R)
plt.plot(good.age, good.R)
plt.plot(broken.age, broken.rl_1)
plt.plot(broken2.age, broken2.rl_1)
plt.plot(good.age, good.rl_1)
star = get_star(m=2.2)
plt.plot(star.age - star.age[star.ntpagb], 10**star.log_R)
plt.show()
# %%
broken.bulk_names

# %%
plt.plot(broken.Teff, broken.min_kapR)
plt.scatter(broken.Teff[0], broken.min_kapR[0])
plt.plot(broken2.Teff, broken2.min_kapR)
plt.plot(good.Teff, good.min_kapR)
plt.show()

# %%

plt.plot(broken2.age, broken2.log_Teff)
plt.plot(good.age, good.log_Teff)
plt.plot(broken2.age, broken2.log_R + 1)
plt.plot(good.age, good.log_R + 1)
plt.plot(star.age - star.age[star.ntpagb], star.log_Teff)
plt.plot(star.age - star.age[star.ntpagb], star.log_R + 1)
plt.show()
# %%

for m in grid.filter(m=2.2, R=600):
    continue

plt.plot(m.age, m.log_Teff)
plt.plot(m.age, m.log_R + 1)
plt.plot(star.age, star.log_Teff)
plt.plot(star.age, star.log_R + 1)
plt.show()
# %%

plt.plot(m.age, np.log10(m.quasi_adiabatic_Mdot))
plt.plot(m.age, m.lg_mstar_dot_1)
plt.show()
# %%

plt.plot(broken2.envelope_mass, np.log10(broken2.quasi_adiabatic_Mdot))
plt.plot(broken2.envelope_mass, broken2.lg_mstar_dot_1)
plt.show()
# %%

plt.plot(star.age, star.log_Mdot_crit)
plt.plot(star.age, star.Mdot)
plt.show()
# %%

profiles = []
for i in range(1, 9):
    profiles.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/R230.26_q0.600_eps0.250_delta0.200_M2.2/LOGS/profile{i}.data"
        )
    )
# %%

profiles[0].bulk_names

# %%

for profile in profiles:
    plt.plot(profile.mass, profile.thermal_time_to_surface)
plt.show()
# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for p in profiles:

    axs[0, 0].plot(p.mass[0] - p.mass, p.logT)
    axs[0, 1].plot(p.mass[0] - p.mass, p.entropy)

    axs[1, 0].plot(p.mass[0] - p.mass, p.eps_grav)
    axs[1, 1].plot(
        p.mass[0] - p.mass, 10**p.log_thermal_time_to_surface / 365 / 24 / 3600
    )
plt.show()
# %%

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for p in profiles:
    axs[0].plot(p.mass[0] - p.mass, 10**p.log_thermal_time_to_surface / 365 / 24 / 3600)
    axs[1].plot(
        p.mass[0] - p.mass,
        (p.mass[0] - p.mass) / (10**p.log_thermal_time_to_surface / 365 / 24 / 3600),
    )

axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.show()
# %%

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for p in profiles:
    axs[0].plot(p.mass[0] - p.mass, 10**p.logT)
    axs[1].plot(
        p.mass[0] - p.mass,
        (p.mass[0] - p.mass) / (10**p.log_thermal_time_to_surface / 365 / 24 / 3600),
    )

axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.show()
# %%

profiles_good = []
for i in range(1, 50):
    profiles_good.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/R241.34_q0.600_eps0.250_delta0.200_M2.2/LOGS/profile{i}.data"
        )
    )
# %%

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for p in profiles_good:
    axs[0].plot(p.mass[0] - p.mass, 10**p.logT)
    axs[1].plot(
        p.mass[0] - p.mass,
        (p.mass[0] - p.mass) / (10**p.log_thermal_time_to_surface / 365 / 24 / 3600),
    )

axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.show()
# %%

plt.plot(broken2.envelope_mass, broken2.log_Teff)
plt.plot(broken2.envelope_mass, broken2.log_R)
plt.plot(good.envelope_mass, good.log_Teff)
plt.plot(good.envelope_mass, good.log_R)
plt.plot(star.m_env, star.log_Teff)
plt.plot(star.m_env, star.log_R)
plt.show()
# %%
