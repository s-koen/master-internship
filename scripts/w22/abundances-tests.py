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

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

# %%

p = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/new-abundances/M2.0/LOGS/TPAGB/profile173.data"
)
# %%

plt.plot(p.mass, p.c12 / p.o16 * 16 / 12)
plt.plot(p.mass, p.gradT)
plt.plot(p.mass, p.grada)

plt.show()
# %%

p.bulk_names
# %%
h = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/new-abundances/M2.0/LOGS/TPAGB/history.data"
)
# %%
h2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00453/M2.0/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.9), constrained_layout=True, sharex=True
)

plt.xlabel("Time (yr)")
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel("C/O-ratio")
axs[0].plot(h.star_age, h.R, c="k", linewidth=0.8)
axs[1].plot(
    h.star_age,
    h.envelope_c12 / h.envelope_o16 * 16 / 12,
    label="Envelope average",
    linewidth=0.8,
)
axs[1].plot(
    h.star_age,
    h.surface_c12 / h.surface_o16 * 16 / 12,
    label="Surface average",
    linewidth=0.8,
)

axs[0].spines[["right", "top"]].set_visible(False)
axs[1].spines[["right", "top"]].set_visible(False)
axs[1].legend()
plt.savefig("/home/koen/LaTeX-setup/plots/w22-hr+abundance.pgf", format="pgf")
plt.show()
plt.close()

# %%
p = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/new-abundances/M2.0/LOGS/TPAGB/profile231.data"
)
plt.plot(p.mass, p.c12 / p.o16 * 16 / 12)
plt.plot(p.mass, np.log10(p.c13))
plt.plot(p.mass, p.gradT)
plt.plot(p.mass, p.grada)

plt.show()

# %%
h.bulk_names
# %%
plt.plot(h2.star_age, h2.c13_eff_pocket_mass)
plt.plot(h.star_age, h.c13_eff_pocket_mass)
plt.show()
# %%
plt.plot(h2.star_age, h2.lambda_DUP)
plt.plot(h.star_age, h.lambda_DUP)
plt.show()
# %%
plt.plot(h2.star_age, h2.he_core_mass)
plt.plot(h2.star_age, h2.lambda_DUP)
plt.show()
# %%

max_he_core = []
for i in range(len(h2.he_core_mass)):
    if i > 0:
        max_he_core.append(h2.he_core_mass[:i].max())
    else:
        max_he_core.append(h2.he_core_mass[0])
# %%

m_DUP = []
pulses = []
current_pulse = 0
previous_pulse = 0
max = -1e99
for i in range(len(h2.he_core_mass)):
    if i < 0:
        continue
    current_pulse = h2.TP_count[i]
    if current_pulse != previous_pulse:
        print(current_pulse)
        pulses.append(current_pulse)
        m_DUP.append(max)
        previous_pulse = current_pulse
        max = -1e99
        continue

    current_max = np.max(max_he_core[i] - h2.he_core_mass[i])
    if current_max > max:
        max = current_max

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
# plt.plot(h2.star_age, max_he_core - h2.he_core_mass)
# plt.plot(h2.TP_count[1:], np.cumsum(np.diff(max_he_core - h2.he_core_mass)))
plt.plot(pulses[1:], np.cumsum(m_DUP[1:]), c="k", linewidth=0.8)
plt.scatter(pulses[1:], np.cumsum(m_DUP[1:]), c="w", marker=".", s=200, zorder=10)
plt.scatter(pulses[1:], np.cumsum(m_DUP[1:]), c="k", marker=".", zorder=11)


axs.spines[["right", "top"]].set_visible(False)

plt.xlim(1, 21)
plt.ylim(-0.005, 0.07)
plt.xticks([2, 5, 10, 15, 20])
plt.xlabel("TP-count")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-M_DUP.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.9), constrained_layout=True, sharex=True
)

plt.xlabel("Time (yr)")
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel("C/O-ratio")
axs[0].plot(h.star_age, h.R, c="k", linewidth=0.8)
axs[1].plot(
    h.star_age,
    h.envelope_c12 / h.envelope_c13 * 13 / 12,
    label="Envelope average",
    linewidth=0.8,
)

axs[0].spines[["right", "top"]].set_visible(False)
axs[1].spines[["right", "top"]].set_visible(False)
axs[1].legend()
plt.savefig("/home/koen/LaTeX-setup/plots/w22-hr+abundance-c12-c13.pgf", format="pgf")
plt.show()
plt.close()


# %%
