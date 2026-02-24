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
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/MS/history.data"
)
GB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/GB/history.data"
)
CHeB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/CHeB/history.data"
)
EAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/EAGB/history.data"
)
TPAGB = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/TPAGB/history.data"
)
phases = [MS, GB, CHeB, EAGB]
phases_names = ["MS", "RGB", "CHeB", "EAGB"]
# %%
models = {}

for j, phase in enumerate(phases_names):
    models[phase] = []
    for i in range(1, 100):
        try:
            models[phase].append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/rees2024-2M/LOGS/{phase}/profile{i}.data"
                )
            )
        except:
            break

# %%
fig, axs = plt.subplots(
    1, 2, figsize=set_size(column, height=0.5), constrained_layout=True
)

delta = 0

model_numbers = [
    [1000, 2000],
    [1000, 2000, 3000, 4000, 5000],
    [1000, 1500],
    [100, 200, 300, 400, 500, 600, 700],
]

PMS_index = np.where(MS.center_h1 / MS.center_h1[0] < 0.997)[0][0]

indices = [[], [], [], []]
for i, phase in enumerate(model_numbers):
    for model_number in phase:
        if i == 0:
            if np.argwhere(phases[i].model_number == model_number) > PMS_index:
                indices[i].append(np.argwhere(phases[i].model_number == model_number))
        else:
            indices[i].append(np.argwhere(phases[i].model_number == model_number))


for i, phase in enumerate(phases):
    if phase == MS:
        axs[0].plot(
            phase.log_Teff[PMS_index:],
            phase.log_L[PMS_index:],
            label=f"{phases_names[i]}",
        )
        axs[1].plot(phase.model_number[PMS_index:] + delta, phase.R[PMS_index:])
        # for index in indices[i]:
        #     axs[0].scatter(phase.log_Teff[index], phase.log_L[index], c=f"C{i}", s=15)
        #     axs[1].scatter(phase.model_number[index], phase.R[index], c=f"C{i}", s=15)

    else:
        axs[0].plot(phase.log_Teff, phase.log_L, label=f"{phases_names[i]}")
        # for index in indices[i]:
        #     axs[0].scatter(phase.log_Teff[index], phase.log_L[index], c=f"C{i}", s=15)
        #     axs[1].scatter(
        #         phase.model_number[index] + delta, phase.R[index], c=f"C{i}", s=15
        #     )
        axs[1].plot(phase.model_number + delta, phase.R)
    delta += phase.model_number[-1]
axs[0].invert_xaxis()
axs[0].legend()

axs[0].set_ylabel(r"Luminosity $(L_\odot)$")
axs[0].set_xlabel(r"Effective temperature (K)")
axs[1].set_ylabel(r"Radius $(R_\odot)$")
axs[1].set_xlabel(r"Model number ")
plt.savefig("/home/koen/LaTeX-setup/plots/model-selection-test.pgf", format="pgf")
plt.show()
plt.close()

# %%

MS = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_MS/history.data"
)

TO25R = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_TO25R/history.data"
)

TO50R = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_TO50R/history.data"
)

TO75R = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_TO75R/history.data"
)

phases = [MS, TO25R, TO50R, TO75R]
phases_names = [
    "MS",
    r"GB ($R < 25 R_\odot$)",
    r"GB ($25 R_\odot < R < 50 R_\odot$)",
    r"GB ($50 R_\odot < R < 75 R_\odot$)",
]
# %%

fig, axs = plt.subplots(
    1, 2, figsize=set_size(full, height=0.5), constrained_layout=True
)

delta = 0


PMS_index = np.where(MS.center_h1 / MS.center_h1[0] < 0.997)[0][0]


for i, phase in enumerate(phases):
    if phase == MS:
        axs[0].plot(
            phase.log_Teff[PMS_index:],
            phase.log_L[PMS_index:],
            label=f"{phases_names[i]}",
        )
        axs[0].scatter(phase.log_Teff[-1], phase.log_L[-1], zorder=1000)
        axs[1].plot(phase.model_number[PMS_index:], phase.R[PMS_index:])
        axs[1].scatter(phase.model_number[-1], phase.R[-1], zorder=1000)
    else:
        axs[0].plot(phase.log_Teff, phase.log_L, label=f"{phases_names[i]}")
        axs[0].scatter(phase.log_Teff[-1], phase.log_L[-1], zorder=1000)
        axs[1].plot(phase.model_number, phase.R)
        axs[1].scatter(phase.model_number[-1], phase.R[-1], zorder=1000)
axs[0].invert_xaxis()
axs[0].legend()

axs[0].set_ylabel(r"Luminosity $(L_\odot)$")
axs[0].set_xlabel(r"Effective temperature (K)")
axs[1].set_ylabel(r"Radius $(R_\odot)$")
axs[1].set_xlabel(r"Model number ")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/model-selection-test-simple.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

B_MS = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_MS/history.data"
)

B_TO25R = mr.MesaData(
    "/home/koen/master-internship/mesa-models/simple-hb/LOGS_TO25R/history.data"
)

B_TO75R = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-load-model/LOGS/history.data"
)

phases = [B_MS, B_TO25R, B_TO75R]
old_phases = [MS, TO25R, TO50R, TO75R]
phases_names = [
    "MS",
    r"GB ($R < 25 R_\odot$)",
    r"GB ($50 R_\odot < R < 75 R_\odot$)",
]

binary_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-load-model/binary_history.data"
)
# %%


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes["x"].join(target, ax)
        if sharey:
            target._shared_axes["y"].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which="both", labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which="both", labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


# %%

fig, axs = plt.subplots(
    1,
    5,
    figsize=set_size(full, height=0.5),
    constrained_layout=True,
    width_ratios=[
        2,
        6 / 72,
        1 - 3 / 72,
        1 - 3 / 72,
        1 / 72,
    ],
)

delta = 0


PMS_index = np.where(B_MS.center_h1 / B_MS.center_h1[0] < 0.997)[0][0]


for i, phase in enumerate(phases):
    if phase == B_MS:
        axs[0].plot(
            phase.log_Teff[PMS_index:],
            phase.log_L[PMS_index:],
            label=f"{phases_names[i]}",
        )
        axs[0].scatter(phase.log_Teff[-1], phase.log_L[-1], zorder=1000)

    elif phase == B_TO75R:

        index = np.argwhere(phase.R > 50)[0][0]

        axs[0].plot(
            phase.log_Teff[:index],
            phase.log_L[:index],
            label=r"GB $(25R_\odot< R < 50R_\odot)$",
        )
        axs[0].scatter(phase.log_Teff[index], phase.log_L[index], zorder=1000)

        axs[0].plot(
            phase.log_Teff[index:],
            phase.log_L[index:],
            label=f"{phases_names[i]}",
        )
        axs[0].scatter(phase.log_Teff[-1], phase.log_L[-1], zorder=1000)
    else:
        axs[0].plot(phase.log_Teff, phase.log_L, label=f"{phases_names[i]}")
        axs[0].scatter(phase.log_Teff[-1], phase.log_L[-1], zorder=1000)

PMS_index = np.where(MS.center_h1 / MS.center_h1[0] < 0.997)[0][0]

for i, phase in enumerate(old_phases):
    if phase == MS:
        axs[0].plot(
            phase.log_Teff[PMS_index:], phase.log_L[PMS_index:], c="C0", alpha=0.35
        )
        axs[0].scatter(
            phase.log_Teff[-1], phase.log_L[-1], zorder=1000, c="C0", alpha=0.35
        )
    else:
        axs[0].plot(phase.log_Teff, phase.log_L, c=f"C{i}", alpha=0.35)
        axs[0].scatter(
            phase.log_Teff[-1], phase.log_L[-1], zorder=1000, c=f"C{i}", alpha=0.35
        )

index = np.argwhere(
    binary_history.rl_1 * (1 + binary_history.rl_relative_overflow_1) > 50
)[0][0]
axs[2].plot(
    binary_history.age[:index] / 1e6, binary_history.lg_mstar_dot_1[:index], c="C2"
)
axs[2].plot(
    binary_history.age[index:] / 1e6, binary_history.lg_mstar_dot_1[index:], c="C3"
)
axs[2].set_ylim(-10, 0)
axs[2].set_xlim(1.7, 2.099)
axs[3].plot(
    binary_history.age[:index] / 1e6, binary_history.lg_mstar_dot_1[:index], c="C2"
)
axs[3].plot(
    binary_history.age[index:] / 1e6, binary_history.lg_mstar_dot_1[index:], c="C3"
)
axs[3].set_ylim(-10, 0)
axs[3].set_xlim(2.099)

axs[2].spines["right"].set_visible(False)
axs[3].spines["left"].set_visible(False)
axs[2].yaxis.tick_left()
set_share_axes(axs[2:3], sharey=True)
axs[3].yaxis.set_ticklabels([])
axs[3].yaxis.set_ticks([])

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axs[2].transAxes, color="C8", clip_on=False, linewidth=0.75)
axs[2].plot((1 - d, 1 + d), (-d, +d), **kwargs)
axs[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=axs[3].transAxes)  # switch to the bottom axes
axs[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)
axs[3].plot((-d, +d), (-d, +d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

axs[0].invert_xaxis()
axs[0].legend()

axs[0].set_ylabel(r"Luminosity $(L_\odot)$")
axs[0].set_xlabel(r"Effective temperature (K)")
axs[2].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
fig.text(0.79, 0.045, "Time (Myr)", ha="center", va="center")

axs[1].axis("off")  # Turn off axis lines and labels for this column
axs[4].axis("off")  # Turn off axis lines and labels for this column

fig.get_layout_engine().set(w_pad=-2, wspace=0)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/model-selection-test-simple-binary.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

folders = [
    "rees2024-2M/LOGS/",
    "binary-tests/rees2024-2.5M/LOGS/",
    "binary-tests/rees2024-3M/LOGS/",
]

phase_folders = ["MS", "GB", "CHeB", "EAGB"]
mass = ["2", "2.5", "3"]

histories = {}
for i, folder in enumerate(folders):
    histories[mass[i]] = {}
    for j, phase_folder in enumerate(phase_folders):
        histories[mass[i]][phases_names[j]] = mr.MesaData(
            f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}/history.data"
        )
# %%

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(full, height=0.5),
    subplot_kw={"projection": "3d"},
    constrained_layout=True,
)


for m, model in enumerate(histories):
    delta = 0
    phases = histories[model]
    for i, phase_name in enumerate(phases):
        phase = phases[phase_name]

        if phase_name == "MS":
            PMS_index = np.where(phase.center_h1 / phase.center_h1[0] < 0.997)[0][0]
            axs[0].plot(
                phase.log_Teff[PMS_index:],
                phase.log_L[PMS_index:],
                phase.star_age[PMS_index:] / 1e9,
                c=f"C{m}",
            )
            axs[1].plot(
                phase.log_cntr_Rho[PMS_index:],
                phase.log_cntr_T[PMS_index:],
                phase.star_age[PMS_index:] / 1e9,
                label=f"{float(mass[m]):.1f} $R_\\odot$",
                c=f"C{m}",
            )
        else:
            axs[0].plot(
                phase.log_Teff,
                phase.log_L,
                delta + phase.star_age / 1e9,
                c=f"C{m}",
            )
            axs[1].plot(
                phase.log_cntr_Rho,
                phase.log_cntr_T,
                delta + phase.star_age / 1e9,
                c=f"C{m}",
            )
        delta += phase.star_age[-1] / 1e9
axs[0].invert_xaxis()

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")

axs[0].set_proj_type("ortho")
axs[0].set_ylabel(r"$L$ $(L_\odot)$")
axs[0].set_xlabel(r"$T_\textrm{eff}$ (K)")
axs[0].set_zlabel(r" ")
axs[1].set_proj_type("ortho")
axs[1].set_ylabel(r"$T_\textrm{c}$ (K)")
axs[1].set_xlabel(r"$\rho_\textrm{c}$ (g cm$^{-3}$)")

ax = axs[1]
tmp_planes = ax.zaxis._PLANES
ax.zaxis._PLANES = (
    tmp_planes[2],
    tmp_planes[3],
    tmp_planes[0],
    tmp_planes[1],
    tmp_planes[4],
    tmp_planes[5],
)
# rotate label
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel(
    r"$\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,$   Star age (Gyr)", rotation=90
)
ax.grid(False)
axs[0].grid(False)

fig.get_layout_engine().set(w_pad=0, wspace=0.25)
plt.savefig("/home/koen/LaTeX-setup/plots/model-selection-compare.pgf", format="pgf")
plt.show()
plt.close()
# %%

scatter_folders = [
    "binary-tests/rees2024-2.5M/LOGS/",
    "binary-tests/rees2024-3M/LOGS/",
]

phases_names = ["MS", "RGB", "CHeB", "EAGB"]
phase_folders = ["MS", "GB", "CHeB", "EAGB"]
mass = ["2.5", "3"]

scatter_models = {}
for i, folder in enumerate(scatter_folders):
    scatter_models[mass[i]] = {}
    for j, phase_folder in enumerate(phase_folders):
        try:
            models = mr.MesaLogDir(
                f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}"
            ).model_numbers
        except:
            models = [
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}/profile1.data"
                ).model_number
            ]

        scatter_models[mass[i]][phases_names[j]] = [
            histories[mass[i]][phases_names[j]].index_of_model_number(m) for m in models
        ][:-1]


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m, model in enumerate(histories):
    delta = 0
    phases = histories[model]
    for i, phase_name in enumerate(phases):
        phase = phases[phase_name]

        if phase_name == "MS":
            PMS_index = np.where(phase.center_h1 / phase.center_h1[0] < 0.997)[0][0]
            plt.plot(
                phase.star_age[PMS_index:] / 1e9,
                phase.R[PMS_index:],
                c=f"C{m}",
            )
        else:
            plt.plot(
                delta + phase.star_age / 1e9,
                phase.R,
                c=f"C{m}",
            )
        if m == 0:
            delta += phase.star_age[-1] / 1e9
            continue
        indices = scatter_models[model][phase_name]
        print(phase, indices)

        plt.scatter(
            delta + phase.star_age[indices] / 1e9, phase.R[indices], c=f"C{m}", s=5
        )
        delta += phase.star_age[-1] / 1e9


plt.xlabel("Star age (Gyr)")
plt.ylabel(r"$\log R$ ($R_\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-radius.pgf", format="pgf")
plt.show()
plt.close()
# %%
folders = [
    "rees2024-2M/LOGS/",
    "binary-tests/rees2024-2.5M-fix/LOGS/",
    "binary-tests/rees2024-3M-fix/LOGS/",
]

phase_folders = ["MS", "GB", "CHeB", "EAGB"]
mass = ["2", "2.5", "3"]

histories = {}
for i, folder in enumerate(folders):
    histories[mass[i]] = {}
    for j, phase_folder in enumerate(phase_folders):
        histories[mass[i]][phases_names[j]] = mr.MesaData(
            f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}/history.data"
        )


scatter_folders = [
    "binary-tests/rees2024-2.5M-fix/LOGS/",
    "binary-tests/rees2024-3M-fix/LOGS/",
]

phases_names = ["MS", "RGB", "CHeB", "EAGB"]
phase_folders = ["MS", "GB", "CHeB", "EAGB"]
mass = ["2.5", "3"]

scatter_models = {}
for i, folder in enumerate(scatter_folders):
    scatter_models[mass[i]] = {}
    for j, phase_folder in enumerate(phase_folders):
        try:
            models = mr.MesaLogDir(
                f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}"
            ).model_numbers
        except:
            models = [
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/{folder}{phase_folder}/profile1.data"
                ).model_number
            ]

        scatter_models[mass[i]][phases_names[j]] = [
            histories[mass[i]][phases_names[j]].index_of_model_number(m) for m in models
        ][:-1]
# %%
print(scatter_models)
# %%
for history in histories["3"]:
    f = histories["3"][history]
    plt.plot(f.star_age, f.last_saved_R)

plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

masses = ["2.0", "2.5", "3.0"]
for m, model in enumerate(histories):
    delta = 0
    phases = histories[model]
    for i, phase_name in enumerate(phases):
        phase = phases[phase_name]

        if phase_name == "MS":
            PMS_index = np.where(phase.center_h1 / phase.center_h1[0] < 0.997)[0][0]
            plt.plot(
                phase.star_age[PMS_index:] / 1e9,
                phase.R[PMS_index:],
                c=f"C{m}",
                label=f"${float(masses[m]):.1f}R_\odot$",
            )
        else:
            plt.plot(
                delta + phase.star_age / 1e9,
                phase.R,
                c=f"C{m}",
            )
        if m == 0:
            delta += phase.star_age[-1] / 1e9
            continue
        indices = scatter_models[model][phase_name]
        print(indices)

        if phase_name == "EAGB" and masses[m] == "3.0":
            plt.scatter(
                delta + phase.star_age[indices[-3]] / 1e9,
                phase.R[indices[-3]],
                c="k",
                s=60,
                zorder=1000,
                marker="*",
            )
            plt.scatter(
                delta + phase.star_age[indices[-6]] / 1e9,
                phase.R[indices[-6]],
                c="k",
                s=60,
                zorder=1000,
                marker="*",
            )
        plt.scatter(
            delta + phase.star_age[indices] / 1e9, phase.R[indices], c=f"C{m}", s=15
        )
        delta += phase.star_age[-1] / 1e9


plt.legend()
plt.xlabel("Star age (Gyr)")
plt.ylabel(r"$\log R$ ($R_\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-radius-fix.pgf", format="pgf")
plt.show()
plt.close()
# %%
