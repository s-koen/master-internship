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

MS = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/MS/history.data"
)
GB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/GB/history.data"
)
CHeB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/CHeB/history.data"
)
EAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/EAGB/history.data"
)
TPAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/TPAGB/history.data"
)
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(MS.log_Teff, MS.log_L)
ax.plot(GB.log_Teff, GB.log_L)
ax.plot(CHeB.log_Teff, CHeB.log_L)
ax.plot(EAGB.log_Teff, EAGB.log_L)
ax.plot(TPAGB.log_Teff, TPAGB.log_L)
ax.invert_xaxis()
plt.show()
# %%

fig, ax = plt.subplots(1, 1)
delta = 0
phase = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]
for i, model in enumerate([MS, GB, CHeB, EAGB, TPAGB]):
    ax.plot(delta + model.star_age, model.R, label=f"{phase[i]}")
    delta += model.star_age[-1]

ax.set_yscale("log")
# ax.set_xscale("log")
plt.legend()
plt.show()
# %%

fig, ax = plt.subplots(1, 1)
delta = 0
phase = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]
for i, model in enumerate([MS, GB, CHeB, EAGB, TPAGB]):
    ax.plot(delta + model.model_number, model.R, label=f"{phase[i]}")
    delta += model.model_number[-1]

ax.set_yscale("log")
# ax.set_xscale("log")
plt.legend()
plt.show()
# %%

fig, ax = plt.subplots(1, 1)
delta = 0
for model in [MS, GB, CHeB, EAGB, TPAGB]:
    ax.plot(model.log_cntr_Rho, model.log_cntr_T)

# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%

TAMS_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024/LOGS/TPAGB/profile5.data"
)
TAMS_mod.bulk_names
# %%
plt.plot(TAMS_mod.mass, TAMS_mod.y_mass_fraction_He)
plt.plot(TAMS_mod.mass, TAMS_mod.x_mass_fraction_H)
plt.plot(TAMS_mod.mass, TAMS_mod.pp / np.max(TAMS_mod.pp))
plt.plot(TAMS_mod.mass, TAMS_mod.tri_alpha / np.max(TAMS_mod.tri_alpha))
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True)
delta = 0
for model in [TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.lambda_DUP)
    axs[1].set_ylabel(r"$\lambda_\text{DUP}$")
    axs[2].plot(delta + model.star_age, model.surface_c12 / model.surface_o16)
    axs[2].set_ylabel(r"$\text{C}_{12} / \text{O}_{16}$")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True)
delta = 0
for model in [TPAGB]:
    axs[0].plot(delta + model.model_number, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.model_number, model.lambda_DUP)
    axs[1].set_ylabel(r"$\lambda_\text{DUP}$")
    axs[2].plot(delta + model.model_number, model.surface_c12 / model.surface_o16)
    axs[2].set_ylabel(r"$\text{C}_{12} / \text{O}_{16}$")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"model number")
# ax.set_yscale("log")
# xax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(full, height=2))


delta = 0
print(TPAGB.bulk_names)
for model in [TPAGB]:
    axs[0].plot(delta + model.star_age, model.log_R)
    axs[0].set_ylabel("$\log R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.surface_c12 / model.surface_o16)
    axs[1].set_ylabel(r"$\textrm{C}_{12} / \textrm{O}_{16}$")
    axs[2].plot(delta + model.star_age, model.star_mass - model.he_core_mass)
    axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("4msuntpagb.pdf", format="pdf")
plt.savefig("/home/koen/LaTeX-setup/plots/4msuntpagb-full.pgf", format="pgf")

# %%
fig, axs = plt.subplots(3, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
for model in [TPAGB]:
    axs[0].plot(delta + model.model_number, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.model_number, model.he_core_mass)
    axs[1].set_ylabel(r"$M_\text{core}$ ($M_\odot$)")
    axs[2].plot(delta + model.model_number, model.star_mass - model.he_core_mass)
    axs[2].set_ylabel(r"$M_\text{env}$ ($M_\odot$)")
    delta += model.model_number[-1]

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%

fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.Teff)
    axs[1].set_ylabel(r"$T_\text{eff}$ (K)")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.model_number)
    axs[1].set_ylabel(r"model number")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.log_dt)
    axs[1].set_ylabel(r"$\Delta t$ (yr)")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.num_zones)
    axs[1].set_ylabel(r"number of zones")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.tri_alpha)
    axs[1].set_ylabel(r"$3-\alpha$")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.log_abs_mdot)
    axs[1].set_ylabel(r"$\dot{M}$ ($M_\odot/$yr)")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
CHeB_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024/LOGS/CHeB/profile1.data"
)
CHeB_mod.bulk_names
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(np.log10(1 - (CHeB_mod.mass / CHeB_mod.star_mass)), CHeB_mod.logT)
axs[1].plot(np.log10(1 - (CHeB_mod.mass / CHeB_mod.star_mass)), CHeB_mod.L)
axs[0].set_xlim(-9, -3)
axs[0].invert_xaxis()
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.model_number, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.model_number, model.log_dt)
    axs[1].set_ylabel(r"$\Delta t$ (yr)")
    delta += model.model_number[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
delta = 0
print(TPAGB.bulk_names)
delta = 0
for model in [EAGB, TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.min_beta)
    axs[1].set_ylabel(r"$\Delta t$ (yr)")
    delta += model.star_age[-1]

axs[1].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0
for model in [TPAGB]:
    axs[0].plot(delta + model.model_number / 1000, model.log_R)
    axs[0].set_ylabel("$\log R$ ($R_\\odot$)")
    axs[1].plot(delta + model.model_number / 1000, model.log_dt)
    axs[1].set_ylabel(r"$\log(\Delta t)$ (yr)")
    axs[2].plot(delta + model.model_number / 1000, model.min_beta)
    axs[2].set_ylabel(r"$\beta_\textrm{min} = \min(P_\textrm{gas}/P)$")
    delta += model.model_number[-1]

axs[2].set_xlabel(r"model number $/ 1000$")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/4msuntpagbmodelnumber.pgf", format="pgf")
# %%

mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024/LOGS/TPAGB/profile30.data"
)
mod2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024/LOGS/TPAGB/profile27.data"
)
mod3 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024/LOGS/TPAGB/profile26.data"
)
mod.bulk_names
# %%
plt.plot(mod.mass, mod.beta)
plt.plot(mod.mass, mod.x_mass_fraction_H)
plt.plot(mod2.mass, mod2.beta)
plt.plot(mod2.mass, mod2.x_mass_fraction_H)
plt.plot(mod3.mass, mod3.beta)
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
range = 10090
for model in [TPAGB]:
    axs[0].plot(model.star_age[range:] - model.star_age[range], model.log_R[range:])
    axs[0].set_ylabel("$\log R$ ($R_\\odot$)")
    axs[1].plot(model.star_age[range:] - model.star_age[range], model.log_dt[range:])
    axs[1].set_ylabel(r"$\log(\Delta t)$ (yr)")
    axs[2].plot(model.star_age[range:] - model.star_age[range], model.min_beta[range:])
    axs[2].set_ylabel(r"$\beta_\textrm{min} = \min(P_\textrm{gas}/P)$")

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/4msuntpagbzoom.pgf", format="pgf")
# %%

fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column))

plt.xlabel("")
plt.ylabel("")
plt.savefig("dredge-up-effiency-2-solar-mass.pdf", format="pdf")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/dredge-up-effiency-2-solar-mass.pgf", format="pgf"
)
