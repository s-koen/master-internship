import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import mesa_reader as mr

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

width = 255.22122
plt.style.use("default")
plt.style.use("tex rm")

# %%

MS = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/MS/history.data"
)
GB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/GB/history.data"
)
CHeB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/CHeB/history.data"
)
EAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/EAGB/history.data"
)
TPAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/TPAGB/history.data"
)
# %%

MS = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/MS/history.data"
)
GB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/GB/history.data"
)
CHeB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/CHeB/history.data"
)
EAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/EAGB/history.data"
)
TPAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/TPAGB/history.data"
)

# %%

figure, axs = plt.subplots(1, 2)
for phase in [MS, GB, CHeB, EAGB, TPAGB]:
    axs[0].plot(phase.log_cntr_Rho, phase.log_cntr_T)
    axs[1].plot(phase.log_Teff, phase.log_L)
axs[1].invert_xaxis()
axs[0].set_xlabel(r"$\rho_\text{c}$ (g cm$^{-3}$)")
axs[0].set_ylabel(r"$T_\text{c}$ (K)")
axs[1].set_xlabel(r"$T_\text{eff}$ (K)")
axs[1].set_ylabel(r"$L$ ($L_\odot$)")
plt.show()
# %%

figure, axs = plt.subplots(1, 2)
delta = 0
for phase in [MS, GB, CHeB, EAGB, TPAGB]:
    axs[0].plot(delta + phase.star_age, phase.R)
    delta += phase.star_age[-1]
axs[1].invert_xaxis()
axs[0].set_xlabel(r"$\rho_\text{c}$ (g cm$^{-3}$)")
axs[0].set_ylabel(r"$T_\text{c}$ (K)")
axs[1].set_xlabel(r"$T_\text{eff}$ (K)")
axs[1].set_ylabel(r"$L$ ($L_\odot$)")
plt.show()

# %%
MS_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/MS/profile1.data"
)

GB_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/GB/profile1.data"
)

CHeB_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/CHeB/profile1.data"
)

EAGB_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/EAGB/profile1.data"
)

TPAGB_mod = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/TPAGB/profile1.data"
)

TPAGB_mod_2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/tpagb-test/LOGS/TPAGB/profile12.data"
)

# %%
figure, axs = plt.subplots(2, 1, sharex=True)

for model in [CHeB_mod, EAGB_mod, TPAGB_mod, TPAGB_mod_2]:
    print(model.bulk_names)
    dm = np.diff(model.mass)
    superadiabatic = np.argwhere(model.grada_sub_gradT < -0.1)
    axs[0].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.thermal_time_to_surface[superadiabatic],
        linewidth=10,
        color="white",
        alpha=0.2,
    )
    axs[0].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface,
    )
    axs[1].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        (model.star_mass - model.mass[superadiabatic])
        / model.thermal_time_to_surface[superadiabatic]
        * 31556926,
        linewidth=10,
        color="white",
        alpha=0.2,
    )
    axs[1].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
    )

axs[0].invert_xaxis()
axs[0].set_xlim(-9, -1)
axs[0].set_yscale("log")
axs[0].set_ylim(10**-5, 10**10)
axs[0].set_ylabel(r"$\tau_\text{th}$ (yr)")

axs[1].invert_xaxis()
axs[1].set_yscale("log")
axs[1].set_xlabel(r"$\log(1- m / M)$")
axs[1].set_ylabel(r"$\dot{M}_\text{th}$ ($M_\odot$ yr$^{-1}$)")


plt.show()
# %%
figure, axs = plt.subplots(2, 1, sharex=True)
for phase in [TPAGB]:
    axs[0].plot(phase.star_age, phase.R)
    axs[1].plot(phase.star_age, phase.surface_c12 / phase.surface_o16)
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_xlabel(r"$t$ (yr)")
axs[1].set_ylabel(r"$\text{C}_{12} / \text{O}_{16}$")
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

axs[2].set_xlabel(r"model number")
# ax.set_yscale("log")
# xax.set_xscale("log")
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0
print(TPAGB.bulk_names)
for model in [TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.he_core_mass)
    axs[1].set_ylabel(r"$M_\text{core}$ ($M_\odot$)")
    axs[2].plot(delta + model.star_age, model.star_mass - model.he_core_mass)
    axs[2].set_ylabel(r"$M_\text{env}$ ($M_\odot$)")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("test.pdf")
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))

delta = 0
print(TPAGB.bulk_names)
for model in [TPAGB]:
    axs[0].plot(delta + model.star_age, model.R)
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.he_core_mass)
    axs[1].set_ylabel(r"$M_\textrm{core}$ ($M_\odot$)")
    axs[2].plot(delta + model.star_age, model.star_mass - model.he_core_mass)
    axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"$t$ (yr)")
plt.tight_layout()
plt.savefig("test.pdf", format="pdf")
plt.savefig("/home/koen/LaTeX-setup/plots/test.pgf", format="pgf")
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))

delta = 0
for model in [TPAGB]:
    axs[0].plot(delta + model.star_age, model.log_R)
    axs[0].set_ylabel("$\\log R$ ($R_\\odot$)")
    axs[1].plot(delta + model.star_age, model.surface_c12 / model.surface_o16)
    axs[1].set_ylabel(r"$\textrm{C}_{12} / \textrm{O}_{16}$")
    axs[2].plot(delta + model.star_age, model.star_mass - model.he_core_mass)
    axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
    delta += model.star_age[-1]

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("5msuntpagb.pdf", format="pdf")
plt.savefig("/home/koen/LaTeX-setup/plots/5msuntpagb.pgf", format="pgf")
# %%
models_simple = {}

for j, phase in enumerate(phases):
    models_simple[phase] = []
    for i in range(1, 100):
        try:
            models_simple[phase].append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/{phase}/profile{i}.data"
                )
            )
        except:
            break

# %%

import matplotlib as mpl
import matplotlib.cm as cm

fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))

model_numbers = []

norm = mpl.colors.Normalize(vmin=347.98267018114024, vmax=808.1474163104338)
cmap = cm.viridis  # or plasma, magma, etc.

for model in models_simple["TPAGB"]:

    if model.model_number % 500 == 0 or model.model_number == 38639:
        continue
    model_numbers.append(model.model_number)
    superadiabatic = np.argwhere(model.grada_sub_gradT < -0.1)

    color = cmap(norm(model.photosphere_r))
    axs[0].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.entropy[superadiabatic],
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[0].plot(np.log10(1 - model.mass / model.mass[0]), model.entropy, color=color)

    axs[1].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.thermal_time_to_surface[superadiabatic] / 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[1].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface / 31556926,
        color=color,
    )
    axs[2].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        (model.star_mass - model.mass[superadiabatic])
        / model.thermal_time_to_surface[superadiabatic]
        * 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[2].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
        color=color,
    )

axs[0].set_ylim(0, 85)
axs[0].set_ylabel(r"$s / (N_\textrm{a} k_\textrm{B})$")

axs[1].invert_xaxis()
axs[1].set_xlim(-4, 0)
axs[1].set_yscale("log")
axs[1].set_ylim(10**-3, 10**2)
axs[1].set_ylabel(r"$\tau_\textrm{th}$ (yr)")

axs[2].invert_xaxis()
axs[2].set_yscale("log")
axs[2].set_ylim(10**-2, 10**1)
axs[2].set_xlabel(r"$\log(1- m / M)$")
axs[2].set_ylabel(r"$\dot{M}_\textrm{th}$ ($M_\odot$ yr$^{-1}$)")

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(sm, ax=axs[0], pad=0.02, location="top")
cbar.set_label(r"$R$ ($R_\odot$)")

plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/5msuntpagbprofiles.pgf", format="pgf")
plt.close()
plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0
for model in [TPAGB]:
    axs[0].plot(delta + model.model_number / 1000, model.log_R)
    axs[0].set_ylabel("$\\log R$ ($R_\\odot$)")
    axs[1].plot(delta + model.model_number / 1000, model.log_dt)
    axs[1].set_ylabel(r"$\log(\Delta t)$ (yr)")
    axs[2].plot(delta + model.model_number / 1000, model.min_beta)
    axs[2].set_ylabel(r"$\beta_\textrm{min} = \min(P_\textrm{gas}/P)$")
    delta += model.model_number[-1]

axs[2].set_xlabel(r"model number $/ 1000$")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/5msuntpagbmodelnumber.pgf", format="pgf")
# %%
