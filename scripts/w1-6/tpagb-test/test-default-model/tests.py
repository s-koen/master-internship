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
history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/history.data"
)

history_2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test/model-3/LOGS/TPAGB/history.data"
)

# %%

start = 0
plt.plot(history.min_kapR[start:], history.min_T[start:])
plt.plot(history.max_kapR[start:], history.max_T[start:])
plt.xscale("log")
plt.yscale("log")
plt.show()
# %%

plt.plot(history.model_number, history.quasi_adiabatic_Mdot)
plt.yscale("log")
plt.show()
# %%

start = 0
plt.plot(history.model_number, history.min_T[start:])
plt.plot(history.model_number, history.Teff)
plt.yscale("log")
plt.show()
# %%
plt.plot(history.model_number, history.R)
plt.plot(history.model_number, history.rl_1)
plt.show()
# %%
plt.plot(history.star_age, history.R)
plt.plot(history.star_age, history.rl_1)
plt.show()
# %%
plt.plot(history.star_age, history.star_mass - history.he_core_mass)
plt.plot(history_2.star_age, history_2.star_mass - history_2.he_core_mass)
plt.ylim(0, 1.5)
plt.show()
# %%

start = 0
plt.plot(history.model_number, history.dt)
plt.yscale("log")
plt.show()
# %%
fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{14}.data"
)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 2.5], [5, 4.65]], [[0.75, 1], [0.81, 0.85]], [[-1, -6], [-2.08, -2.4]]]
ylims = [[-1.5, 4], [-1.5, 1]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
        ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
        ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")


plt.ylabel(r"$\nabla$")
axss[0, 1].legend()
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/problem-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for id in range(14, 15):
    print(model.bulk_names)
    model = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{id}.data"
    )

    plt.plot(model.logT, model.gradr, zorder=1000)

plt.xlabel(r"$r$ ($R_\odot$)")
plt.ylabel(r"$\nabla$")
plt.xlim(2.5, 15)
plt.gca().invert_xaxis()
plt.ylim(-1.5, 4)
# plt.savefig("/home/koen/LaTeX-setup/plots/problem-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{12}.data"
)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 2.5], [5, 4.65]], [[0.75, 1], [0.81, 0.85]], [[-1, -6], [-2.08, -2.4]]]
ylims = [[-1.5, 4], [-1.5, 1]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
        ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
        ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")


plt.ylabel(r"$\nabla$")
axss[0, 1].legend()
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/success-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{14}.data"
)
model2 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{12}.data"
)
print(model.bulk_names)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis2 = [
    model2.logP,
    model2.R / model2.R[0],
    np.log10(1 - model2.mass / model2.mass[0]),
]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 2.5], [5, 4.65]], [[0.75, 1], [0.81, 0.85]], [[0, -6], [-2.08, -2.4]]]
ylims = [[20, 32], [29.5, 31.25]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$S / (N_\textrm{a}k_\textrm{B})$")
        ax.plot(xaxis[i], model.entropy, zorder=1000, label=r"failing model")
        ax.plot(xaxis2[i], model2.entropy, zorder=1000, label=r"successful model")
        # ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
        # ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")


axss[0, 0].legend()
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/succes-entropy.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axsss = plt.subplots(
    3,
    3,
    figsize=set_size(full, height=0.8),
    constrained_layout=True,
    width_ratios=[0.65, 0.65, 1],
)

axss = axsss[:, 0:2]
laxs = axsss[:, 2]

for lax in laxs:
    lax.axis("off")

model1 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-3/model-3/LOGS/TPAGB/profile{14}.data"
)
model2 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R300.00_q0.125/LOGS/TPAGB/profile17.data"
)
model3 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.250/LOGS/TPAGB/profile4.data"
)

model4 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.125/LOGS/TPAGB/profile28.data"
)

model5 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R150.00_q0.375/LOGS/TPAGB/profile7.data"
)

models = [model2, model3, model4, model5, model1]
model_names = [
    r"$R_\textrm{RL} = 300$, $q=0.125$",
    r"$R_\textrm{RL} = 225$, $q=0.225$",
    r"$R_\textrm{RL} = 225$, $q=0.125$",
    r"$R_\textrm{RL} = 150$, $q=0.375$",
    r"$R_\textrm{RL} = 126$, $q=0.50$",
]

# for i in range(1, 31):
#     mod = mr.MesaData(
#         f"/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R150.00_q0.375/LOGS/TPAGB/profile{i}.data"
#     )
#     models.append(mod)

xaxis = []
for model in models:
    xaxis.append(
        [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
    )

print(model3.bulk_names)
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 2.5], [5, 2.8]], [[0.75, 1], [0.82, 0.92]], [[-1, -6], [-2.25, -2.65]]]
ylims = [[-1.5, 4], [-1.5, 1]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        handles = []
        for m, model in enumerate(models):
            (l,) = ax.plot(xaxis[m][i], model.gradT, zorder=1000, label=model_names[m])
            handles.append(l)
        # ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
        # ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")

axsss[0, 2].legend(loc="center left", handles=handles)
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/all-fails.pgf", format="pgf")
plt.show()
plt.close()
# %%
