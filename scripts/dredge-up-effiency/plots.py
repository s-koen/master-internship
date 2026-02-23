import mesa_reader as mr

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
# %%

model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/TPAGB/history.data"
)
model_simple = mr.MesaData(
    "/home/koen/master-internship/mesa-models/mesa-default-other-mass-loss/LOGS/TPAGB/history.data"
)
# %%

print(model_simple.bulk_names)
fig, axs = plt.subplots(
    4, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

condition = np.where(model.star_age > 1e6)

axs[0].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.R[condition],
    label="detailed from Rees (2024)",
)
axs[0].plot(model_simple.star_age, model_simple.R, label="simple")
axs[1].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.he_core_mass[condition],
    label="detailed from Rees (2024)",
)
axs[1].plot(model_simple.star_age, model_simple.he_core_mass, label="simple")
axs[2].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.lambda_DUP[condition],
)
axs[3].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.surface_c12[condition] / model.surface_o16[condition] * 16 / 12,
)
axs[3].plot(
    model_simple.star_age,
    model_simple.surface_c12 / model_simple.surface_o16 * 16 / 12,
)
axs[0].legend()

axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$M_\textrm{He-core}$ ($M_\odot$)")
axs[2].set_ylabel(r"$\lambda_\textrm{DUP}$")
axs[3].set_ylabel(r"C/O number ratio")
axs[3].set_xlabel(r"Star age (yr)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/dredge-up-efficiency-2-msun.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/TPAGB/history.data"
)
model_simple = mr.MesaData(
    "/home/koen/master-internship/mesa-models/mesa-default-other-mass-loss-5M/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    4, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

condition = np.where(model.star_age >= 0)

axs[0].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.R[condition],
    label="detailed from Rees (2024)",
)
axs[0].plot(model_simple.star_age, model_simple.R, label="simple")
axs[1].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.he_core_mass[condition],
    label="detailed from Rees (2024)",
)
axs[1].plot(model_simple.star_age, model_simple.he_core_mass, label="simple")
axs[2].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.lambda_DUP[condition],
)
axs[3].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.surface_c12[condition] / model.surface_o16[condition] * 16 / 12,
)
axs[3].plot(
    model_simple.star_age,
    model_simple.surface_c12 / model_simple.surface_o16 * 16 / 12,
)
axs[0].legend()

axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$M_\textrm{He-core}$ ($M_\odot$)")
axs[2].set_ylabel(r"$\lambda_\textrm{DUP}$")
axs[3].set_ylabel(r"C/O number ratio")
axs[3].set_xlabel(r"Star age (yr)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/dredge-up-efficiency-5-msun.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    4, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
)

condition = np.where(model.star_age >= 0)

axs[0].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.R[condition],
    label="detailed from Rees (2024)",
)
axs[0].plot(model_simple.star_age, model_simple.R, label="simple")
axs[1].plot(
    model.star_age[condition] - model.star_age[condition][0],
    model.star_mass[condition],
    label="detailed from Rees (2024)",
)
axs[1].plot(model_simple.star_age, model_simple.star_mass, label="simple")
axs[2].plot(
    model.star_age[condition] - model.star_age[condition][0],
    np.log10(5 - model.star_mass[condition]),
)
axs[2].plot(model_simple.star_age, np.log10(5 - model_simple.star_mass), label="simple")
axs[3].plot(model_simple.star_age, model_simple.elapsed_time / 3600, c="C1")
axs[0].legend()

axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"Star mass ($M_\odot$)")
axs[2].set_ylabel(r"$\log(5 - M/ M_\odot)$ ")
axs[3].set_ylabel(r"Wall time (hr)")
axs[3].set_xlabel(r"Star age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/mass-loss-5-msun.pgf", format="pgf")
plt.show()
plt.close()
# %%
