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

data = mr.MesaData(f"{MASTER}/tides/5/LOGS/TPAGB/profile5.data")
data.bulk_names
# %%

plt.plot(10**data.logR, data.gradr)
plt.plot(10**data.logR, data.grada)
plt.plot(10**data.logR, data.gradT)

plt.show()
# %%
plt.plot(data.mass, data.gradr)
plt.plot(data.mass, data.grada)
plt.plot(data.mass, data.gradT)

plt.show()

# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(f"{MASTER}/tides/5/LOGS/TPAGB/profile5.data")

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 1.5], [3.1, 2.5]], [[0.75, 1], [0.91, 0.955]], [[-1, -4], [-1.8, -2.4]]]
ylims = [[-0.5, 3], [0.37, 0.41]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        if i == 0 and j == 0:
            ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
            ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
            ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")
        else:
            ax.plot(xaxis[i], model.gradT, zorder=1000)
            ax.plot(xaxis[i], model.gradr)
            ax.plot(xaxis[i], model.grada)
plt.ylabel(r"$\nabla$")
fig.legend(loc="outside upper center", ncols=3)
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-gradT.pgf", format="pgf")
plt.show()
plt.close()


# %%

plt.plot(10**model.logR / 10 ** model.logR[0], model.entropy)
plt.plot(10**model.logR / 10 ** model.logR[0], model.gradT)
plt.show()
# %%


data = mr.MesaData(f"{MASTER}/tides/6/LOGS/TPAGB/profile15.data")
data.bulk_names
# %%

plt.plot(10**data.logR, data.gradr)
plt.plot(10**data.logR, data.grada)
plt.plot(10**data.logR, data.gradT)

plt.show()
# %%
plt.plot(1 - data.mass / data.mass[0], data.gradr)
plt.plot(1 - data.mass / data.mass[0], data.grada)
plt.plot(1 - data.mass / data.mass[0], data.gradT)

plt.show()

# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(f"{MASTER}/tides/5/LOGS/TPAGB/profile5.data")

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 1.5], [3.1, 2.5]], [[0.75, 1], [0.91, 0.955]], [[-1, -4], [-1.8, -2.4]]]
ylims = [[-0.5, 3], [0.37, 0.41]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        if i == 0 and j == 0:
            ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
            ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
            ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")
        else:
            ax.plot(xaxis[i], model.gradT, zorder=1000)
            ax.plot(xaxis[i], model.gradr)
            ax.plot(xaxis[i], model.grada)
plt.ylabel(r"$\nabla$")
fig.legend(loc="outside upper center", ncols=3)
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-gradT.pgf", format="pgf")
plt.show()
plt.close()
