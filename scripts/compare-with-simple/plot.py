import numpy as np
import matplotlib.pyplot as plt
import mesa_reader as mr

# %%
phases = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]

complex = []
for phase in phases:
    try:
        complex.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/{phase}/history.data"
            )
        )
    except:
        ...


simple = []
for phase in phases:
    try:
        simple.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/mesa-default-other-mass-loss/LOGS/{phase}/history.data"
            )
        )
    except:
        ...

simple_no_wind = []
for phase in phases:
    try:
        simple_no_wind.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/mesa-default/LOGS/{phase}/history.data"
            )
        )
    except:
        ...
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

width = 255.22122
plt.style.use("default")
plt.style.use("tex rm")
# %%
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(width))

plt.ylabel("Star age (yr)")
plt.xlabel("Wall time (hr)")

plt.plot(
    (
        complex[-1].elapsed_time[np.where(complex[-1].star_age > 1e6)]
        - complex[-1].elapsed_time[np.where(complex[-1].star_age > 1e6)][0]
    )
    / 3600,
    complex[-1].star_age[np.where(complex[-1].star_age > 1e6)]
    - complex[-1].star_age[np.where(complex[-1].star_age > 1e6)][0],
    label="detailed",
)
plt.plot((simple[-1].elapsed_time) / 3600, simple[-1].star_age, label="simple")
plt.plot(
    (simple_no_wind[-1].elapsed_time) / 3600,
    simple_no_wind[-1].star_age,
    label="simple no wind",
)

plt.legend()
plt.tight_layout()
plt.savefig("compare-times-1.pdf", format="pdf")
plt.savefig("/home/koen/latex-setup/plots/compare-times-2.pgf", format="pgf")
plt.close()
# plt.show()
# %%

figure, axs = plt.subplots(1, 2)
for i in range(len(simple)):
    comp = complex[i]
    simp = simple[i]

    axs[0].plot(comp.log_Teff, comp.log_L, c="C6")
    axs[0].plot(simp.log_Teff, simp.log_L, c="C1", linestyle="dotted")
    axs[1].plot(comp.log_cntr_Rho, comp.log_cntr_T, c="C6")
    axs[1].plot(simp.log_cntr_Rho, simp.log_cntr_T, c="C1", linestyle="dotted")
axs[0].invert_xaxis()
axs[1].set_xlabel(r"$\rho_\text{c}$ (g cm$^{-3}$)")
axs[1].set_ylabel(r"$T_\text{c}$ (K)")
axs[0].set_xlabel(r"$T_\text{eff}$ (K)")
axs[0].set_ylabel(r"$L$ ($L_\odot$)")
plt.show()


# %%

models_simple = {}

for j, phase in enumerate(phases):
    models_simple[phase] = []
    for i in range(1, 100):
        try:
            models_simple[phase].append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/mesa-default-other-mass-loss/LOGS/{phase}/profile{i}.data"
                )
            )
        except:
            break


models_complex = {}

for j, phase in enumerate(phases):
    models_complex[phase] = []
    for i in range(1, 100):
        try:
            models_complex[phase].append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/{phase}/profile{i}.data"
                )
            )
        except:
            break

# %%
import matplotlib as mpl
import matplotlib.cm as cm

fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))

model_numbers = []

norm = mpl.colors.Normalize(vmin=137.12434703550161, vmax=314.0160941640382)
cmap = cm.viridis  # or plasma, magma, etc.

i = 0
for model in models_simple["TPAGB"]:
    if model.model_number % 500 == 0 or model.model_number == 38639:
        continue
    if i > 16 or i % 2 == 0:
        i += 1
        continue

    model_numbers.append(model.model_number)
    superadiabatic = np.argwhere(model.grada_sub_gradT < -0.1)

    print(model.photosphere_r)
    color = cmap(norm(model.photosphere_r))
    if i == 9:
        axs[0].plot(
            np.log10(1 - model.mass / model.mass[0]),
            model.entropy,
            color=color,
            label="simple",
        )
    else:
        axs[0].plot(
            np.log10(1 - model.mass / model.mass[0]), model.entropy, color=color
        )

    axs[1].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface / 31556926,
        color=color,
    )
    axs[2].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
        color=color,
    )
    i += 1

i = 1
for model in models_complex["TPAGB"]:
    if model.model_number % 500 == 0 or model.model_number == 38639:
        continue
    if i > 23 or i % 2 == 0 or i == 1:
        i += 1
        continue

    model_numbers.append(model.model_number)
    superadiabatic = np.argwhere(model.grada_sub_gradT < -0.1)

    print(model.photosphere_r)
    color = cmap(norm(model.photosphere_r))

    if i == 3:
        axs[0].plot(
            np.log10(1 - model.mass / model.mass[0]),
            model.entropy,
            color="k",
            linestyle="dotted",
            label="detailed",
        )
    else:
        axs[0].plot(
            np.log10(1 - model.mass / model.mass[0]),
            model.entropy,
            color="k",
            linestyle="dotted",
        )

    axs[1].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface / 31556926,
        color="k",
        linestyle="dotted",
    )
    axs[2].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
        color="k",
        linestyle="dotted",
    )
    i += 1


axs[0].set_ylim(20, 34)
axs[0].set_ylabel(r"$s / (N_\textrm{a} k_\textrm{B})$")
axs[0].legend()

axs[1].invert_xaxis()
axs[1].set_xlim(-6, 0)
axs[1].set_yscale("log")
axs[1].set_ylim(10**-5, 10**3)
axs[1].set_ylabel(r"$\tau_\textrm{th}$ (yr)")

axs[2].invert_xaxis()
axs[2].set_yscale("log")
axs[2].set_ylim(10**-3, 10**0)
axs[2].set_xlabel(r"$\log(1- m / M)$")
axs[2].set_ylabel(r"$\dot{M}_\textrm{th}$ ($M_\odot$ yr$^{-1}$)")

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(sm, ax=axs[0], pad=0.02, location="top")
cbar.set_label(r"$R$ ($R_\odot$)")

plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/compare-profiles-1.pgf", format="pgf")
plt.close()
# plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0
for i in range(len(simple)):
    comp = complex[-1]
    simp = simple[-1]
    condition = np.where(complex[-1].star_age > 1e6)

    axs[0].plot(
        comp.star_age[condition] - comp.star_age[condition][0],
        comp.R[condition],
        c="C6",
    )

    axs[0].plot(simp.star_age, simp.R, c="C1")
    axs[0].set_ylabel("$R$ ($R_\\odot$)")
    axs[1].plot(
        comp.star_age[condition] - comp.star_age[condition][0],
        comp.log_dt[condition],
        c="C6",
    )
    axs[1].plot(simp.star_age, simp.log_dt, c="C1")
    axs[1].set_ylabel(r"$\log (\Delta t)$ (yr)")
    axs[2].plot(
        comp.star_age[condition] - comp.star_age[condition][0],
        comp.star_mass[condition],
        c="C6",
    )
    axs[2].plot(simp.star_age, simp.star_mass, c="C1")
    axs[2].set_ylabel(r"$\textrm{C}_{12} / \textrm{O}_{16}$")

axs[2].set_xlabel(r"model number")
# ax.set_yscale("log")
# xax.set_xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/2msuntpagbprofiles.pgf", format="pgf")
plt.close()
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
plt.savefig("/home/koen/LaTeX-setup/plots/2msuntpagbmodelnumber.pgf", format="pgf")
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
plt.savefig("2msuntpagb.pdf", format="pdf")
plt.savefig("/home/koen/LaTeX-setup/plots/2msuntpagb.pgf", format="pgf")
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0

comp = complex[-1]
simp = simple[-1]
condition = np.where(complex[-1].star_age > 1e6)

axs[0].plot(
    comp.star_age[condition] - comp.star_age[condition][0],
    comp.R[condition],
    c="C0",
    label="detailed",
)
axs[0].plot(simp.star_age, simp.R, c="C1", label="simple")
axs[0].set_ylabel("$R$ ($R_\\odot$)")
axs[1].plot(
    comp.star_age[condition] - comp.star_age[condition][0],
    comp.log_dt[condition],
    c="C0",
)
axs[1].plot(simp.star_age, simp.log_dt, c="C1")
axs[1].set_ylabel(r"$\log (\Delta t)$ (yr)")
axs[2].plot(
    comp.star_age[condition] - comp.star_age[condition][0],
    comp.star_mass[condition] - comp.he_core_mass[condition],
    c="C0",
)
P = 10 ** (-2.07 + 1.94 * np.log10(simp.R) - 0.9 * np.log10(simp.star_mass))

mdot = 10.0 ** (-11.4 + 0.0123 * P)
axs[2].plot(simp.star_age, simp.star_mass - simp.he_core_mass, c="C1")
axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")

axs[2].set_xlabel(r"$t$ (yr)")
# ax.set_yscale("log")
# xax.set_xscale("log")

axs[0].legend()
plt.tight_layout()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/compare-evolution-complex-simple-mass-loss.pgf",
    format="pgf",
)

plt.close()
# plt.show()
# %%
fig, axs = plt.subplots(3, 1, sharex=True, figsize=set_size(width, height=2))
delta = 0
simp = simple[-1]
comp = complex[-1]

axs[0].plot(delta + comp.model_number / 1000, comp.log_R)
axs[0].plot(delta + simp.model_number / 1000, simp.log_R)
axs[0].set_ylabel("$\\log R$ ($R_\\odot$)")
axs[1].plot(delta + comp.model_number / 1000, comp.log_dt)
axs[1].plot(delta + simp.model_number / 1000, simp.log_dt)
axs[1].set_ylabel(r"$\log(\Delta t)$ (yr)")
axs[2].plot(delta + comp.model_number / 1000, comp.min_beta)
axs[2].plot(delta + simp.model_number / 1000, simp.min_beta)
axs[2].set_ylabel(r"$\beta_\textrm{min} = \min(P_\textrm{gas}/P)$")

axs[2].set_xlabel(r"model number $/ 1000$")
# ax.set_yscale("log")
# ax.set_xscale("log")
plt.tight_layout()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/2msuntpagbmodelnumbersimple.pgf", format="pgf"
)
#

# %%
