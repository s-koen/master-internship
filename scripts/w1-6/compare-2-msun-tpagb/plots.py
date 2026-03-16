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

histories = {}

mass_loss_models_folders = [
    "trabucchi",
    "vassiliadis-woods",
    "reimer+blocker-cinquegrana",
    "reimer+blocker-default",
]
mass_loss_models_labels = [
    "Trabucchi",
    "Vassiliadis + Woods",
    r"Reimers + Blöcker $(\eta_\textrm{Blöcker}=0.01)$",
    r"Reimers + Blöcker $(\eta_\textrm{Blöcker}=0.1)$",
]
phases = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]

for folder in mass_loss_models_folders:
    histories[folder] = {}
    for phase in phases:
        try:
            histories[folder][phase] = mr.MesaData(
                f"/home/koen/master-internship/mesa-models/compare-2-msun/{folder}/LOGS/{phase}/history.data"
            )
        except:
            print(f"failed to retrieve {phase} of {folder}")
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    plt.plot(
        history.star_age[move:] - history.star_age[move],
        history.R[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )

    if i == 0:
        continue
    for j, r in enumerate(history.R[move:]):
        if np.abs(history.last_saved_R[move:][j] - r) < 1e-5:
            plt.scatter(
                history.star_age[move:][j] - history.star_age[move], r, color=f"C{i}"
            )


plt.legend()
plt.xlabel("TPAGB age (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-mass-loss.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    print(history.bulk_names)
    plt.plot(
        history.star_age[move:] - history.star_age[move],
        history.log_abs_mdot[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )
plt.xlabel("TPAGB age (yr)")
plt.ylabel(r"$\log|\dot{M} / (M_\odot \textrm{ yr}^{-1})|$")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-mdot.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    print(history.bulk_names)
    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    plt.plot(
        history.log_Teff[move:],
        history.log_L[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )


plt.legend()
axs.invert_xaxis()
plt.xlabel(r"$\log T_\textrm{eff}$ (K)")
plt.ylabel(r"$\log L$ ($L_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-mass-loss-HR.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    print(history.bulk_names)
    plt.plot(
        history.star_age[move:] - history.star_age[move],
        history.star_mass[move:] - history.he_core_mass[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )
plt.legend()
plt.xlabel("TPAGB age (yr)")
plt.ylabel(r"Envelope mass ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-env-mass.pgf", format="pgf")
plt.show()
plt.close()
# %%


def find_lambda_DUP(history, move):

    age = []
    dup = []

    for count in range(history.TP_count[0], history.TP_count[-1] + 1):
        indices = np.argwhere(history.TP_count[move:] == count)
        max = np.argmax(history.lambda_DUP[move:][indices])
        dup.append(history.lambda_DUP[move:][indices][max][0])
        age.append(history.star_age[move:][indices][max][0] - history.star_age[move])

    return age, dup


# %%

fig, axs = plt.subplots(
    5, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)
order = [100, 10, 1, 1000]
max_age = 0

co_list = []
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0

    axs[0].plot(
        history.star_age[move:] - history.star_age[move],
        history.R[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )
    axs[1].plot(
        history.star_age[move:] - history.star_age[move],
        history.star_mass[move:] - history.he_core_mass[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )
    axs[2].plot(
        history.star_age[move:] - history.star_age[move],
        history.he_core_mass[move:],
    )
    age, dup = find_lambda_DUP(history, move)
    axs[3].plot(age, dup, ".-")
    axs[4].plot(
        history.star_age[move:] - history.star_age[move],
        history.surface_c12[move:] / history.surface_o16[move:] * 16 / 12,
    )
    if history.star_age[-1] - history.star_age[move] > max_age:
        max_age = history.star_age[-1] - history.star_age[move]

    co_index = np.argwhere(
        history.surface_c12[move:] / history.surface_o16[move:] * 16 / 12 > 1
    )[0][0]
    co_list.append(history.star_age[move:][co_index] - history.star_age[move])

for ax in axs:
    ylims = ax.get_ylim()
    for i, line in enumerate(co_list):
        ax.vlines(line, *ylims, linewidth=0.75, color=f"C{i}")
    ax.set_ylim(*ylims)

axs[4].set_xlim(-0.05 * max_age, 1.05 * max_age)
axs[4].plot([-1e9, 1e9], [1, 1], c="C8", linewidth=0.75)
axs[1].legend(frameon=False)
axs[4].set_xlabel(r"TPAGB age (yr)")
axs[0].set_ylabel(r"$R$ $(R_\odot)$")
axs[1].set_ylabel(r"$M_\textrm{env}$ $(M_\odot)$")
axs[2].set_ylabel(r"$M_\textrm{He-core}$ $(M_\odot)$")
axs[3].set_ylabel(r"$\lambda_\textrm{DUP}$")
axs[4].set_ylabel(r"C/O-number ratio")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-dredge-up.pgf", format="pgf")
plt.show()
plt.close()
# %%

histories = {}

mass_loss_models_folders = [
    "trabucchi",
    "ferguson-opacity",
]
mass_loss_models_labels = [
    r"\AE SOPUS",
    "Ferguson",
]
phases = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]

for folder in mass_loss_models_folders:
    histories[folder] = {}
    for phase in phases:
        try:
            histories[folder][phase] = mr.MesaData(
                f"/home/koen/master-internship/mesa-models/compare-2-msun/{folder}/LOGS/{phase}/history.data"
            )
        except:
            print(f"failed to retrieve {phase} of {folder}")
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    print(history.bulk_names)
    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    plt.plot(
        history.star_age[move:] - history.star_age[move],
        history.R[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )

    if i == 0:
        continue
    for j, r in enumerate(history.R[move:]):
        if np.abs(history.last_saved_R[move:][j] - r) < 1e-5:
            plt.scatter(
                history.star_age[move:][j] - history.star_age[move], r, color=f"C{i}"
            )


plt.legend()
plt.xlabel("TPAGB age (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-opacity.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

order = [100, 10, 1, 1000]
for i, mass_loss in enumerate(histories):
    try:
        history = histories[mass_loss]["TPAGB"]
    except:
        continue

    print(history.bulk_names)
    if i == 0:
        move = np.argwhere(history.star_age > 1e5)[0][0]

    else:
        move = 0
    plt.plot(
        history.log_Teff[move:],
        history.log_L[move:],
        label=mass_loss_models_labels[i],
        zorder=order[i],
    )


plt.legend()
axs.invert_xaxis()
plt.xlabel(r"$\log T_\textrm{eff}$ (K)")
plt.ylabel(r"$\log L$ ($L_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/compare-opacity-HR.pgf", format="pgf")
plt.show()
plt.close()
# %%
standard = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/standard-2msun/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

axs[0].plot(standard.star_age, np.log10(standard.min_kapR))
axs[0].set_ylim(-9, 2)
axs[0].set_xlim(*axs[0].get_xlim())
axs[0].fill_between([*axs[0].get_xlim()], [-8, -8], [1, 1], alpha=0.4, color="C8")
axs[1].plot(standard.star_age, np.log10(standard.min_T))
axs[1].set_ylim(3.1, 4.1)
axs[1].set_xlim(*axs[1].get_xlim())
axs[1].fill_between([*axs[1].get_xlim()], [3.2, 3.2], [4, 4], alpha=0.4, color="C8")

axs[0].set_ylabel(r"$\min(\log R)$")
axs[1].set_ylabel(r"$\min(\log T)$")
axs[1].set_xlabel(r"TPAGB age (yr)")

plt.savefig("/home/koen/LaTeX-setup/plots/standard_minima.pgf", format="pgf")
plt.show()
plt.close()

# %%

q125 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.125/LOGS/TPAGB/history.data"
)

q250 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.250/LOGS/TPAGB/history.data"
)

q250_2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/history.data"
)

q375 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R150.00_q0.375/LOGS/TPAGB/history.data"
)

q500 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R150.00_q0.500/LOGS/TPAGB/history.data"
)

succes = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.375/LOGS/TPAGB/history.data"
)

# %%
fig, axss = plt.subplots(
    2,
    2,
    sharex=True,
    figsize=set_size(
        full,
    ),
    constrained_layout=True,
    width_ratios=[1.6, 1],
)

axs = axss[:, 0]
laxs = axss[:, 1]

for lax in laxs:
    lax.axis("off")

style = ["-", ":"]
labels = [
    r"$R_\textrm{RL} = 225\,R_\odot$, $q=0.125$",
    r"$R_\textrm{RL} = 225\,R_\odot$, $q=0.250$",
    r"$R_\textrm{RL} = 225\,R_\odot$, $q=0.375$",
    r"$R_\textrm{RL} = 126\,R_\odot$, $q=0.250$",
    r"$R_\textrm{RL} = 150\,R_\odot$, $q=0.375$",
    r"$R_\textrm{RL} = 150\,R_\odot$, $q=0.500$",
]
ls = []
for i, model in enumerate([q125, q250, succes, q250_2, q375, q500]):
    (l,) = axs[0].plot(
        model.star_mass - model.he_core_mass,
        np.log10(model.min_kapR),
        zorder=1000 - i,
        label=labels[i],
    )
    ls.append(l)
    axs[1].plot(
        model.star_mass - model.he_core_mass, np.log10(model.min_T), zorder=1000 - i
    )
laxs[0].legend(loc="center left", handles=ls)
axs[0].set_ylim(-4.5, -3.5)
axs[1].set_xlim(1.5, 0.9)
axs[0].fill_between([*axs[0].get_xlim()], [-8, -8], [1, 1], alpha=0.4, color="C8")
axs[1].set_ylim(3.5, 3.65)
axs[1].fill_between([*axs[1].get_xlim()], [3.2, 3.2], [4, 4], alpha=0.4, color="C8")

axs[0].set_ylabel(r"$\min(\log R)$")
axs[1].set_ylabel(r"$\min(\log T)$")
axs[1].set_xlabel(r"Envelope mass ($M_\odot$)")

plt.savefig("/home/koen/LaTeX-setup/plots/binary_minima.pgf", format="pgf")
plt.show()
plt.close()
# %%
