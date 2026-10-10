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

with open("/home/koen/master-internship/scripts/w31/profiles.pkl", "rb") as file:
    profiledict = pickle.load(file)


# %%

import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# plt.style.use('/home/koen/mesa-r23.05.1/eos/plotter/mesa_eos_regions.mplstyle')


def parse(fname):
    nY, nX = np.loadtxt(fname, max_rows=1, skiprows=3, unpack=True, dtype=int)
    data = np.loadtxt(fname, skiprows=4)
    data = np.reshape(data, ((nX, nY, -1)))
    Yran = data[0, :, 0]
    Xran = data[:, 0, 1]
    data = np.swapaxes(data, 0, 1)
    return data, Yran, Xran


with open("/home/koen/mesa-r23.05.1/eos/plotter/eos_plotter.dat") as f:
    title = f.readline().strip()
    xlabel = f.readline().strip()
    ylabel = f.readline().strip()

# overwrite with fancier labels
xlabel = r"$\log(\rho/{\rm g\,cm^{-3}})$"
ylabel = r"$\log(T/{\rm K})$"
title = r"MESA EOS Regions ($X=0.7$, $Z=0.02$)"

eosDT, Yran, Xran = parse("/home/koen/mesa-r23.05.1/eos/plotter/eos_plotter.dat")

apjcolwidth = 3.38
# set up plot and labels
# fig, ax = plt.subplots(figsize=(apjcolwidth,apjcolwidth*4./5.)) # for paper figures

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=3 / 4), constrained_layout=True
)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(Xran.min(), Xran.max())
ax.set_ylim(Yran.min(), Yran.max())


# set up color map (slightly customized to make Skye blue)
my_colors = np.array(mpl.cm.Set2.colors)  # array so that entries are editable
# tmp = my_colors[4].copy()
# my_colors[4] = my_colors[3]
# my_colors[5] = tmp
cmap = colors.ListedColormap(my_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 7.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

pcol = ax.pcolormesh(
    Xran, Yran, eosDT[..., 2], shading="nearest", cmap=cmap, norm=norm, rasterized=True
)
pcol.set_edgecolor("face")
cax = fig.colorbar(
    pcol,
    ticks=[0, 1, 2, 3, 4.5, 6.5],
    orientation="horizontal",
    location="top",
    aspect=30,
)
cax.set_label("")
cax.ax.minorticks_off()
cax.ax.set_xticklabels(["blend", "HELM", "OPAL/SCVH", "FreeEOS", "Skye", "ideal"])

# save figure
# fig.savefig('eos_regions.pdf')

# for i, profile in enumerate(profiles[-1:]):
#     plt.plot(profile.logRho, profile.logT, c="k", linewidth=2)
#     plt.scatter(
#         profile.logRho[2303], profile.logT[2303], color="r", marker="x", zorder=10000
#     )
#     plt.scatter(
#         profile.logRho[1191], profile.logT[1191], color="r", marker="x", zorder=10000
#     )


for i in range(1, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )
    (l1,) = plt.cplot(
        prof_heavy.logRho, prof_heavy.logT, prof_heavy.mu, label="MESA", linewidth=1
    )


plt.savefig("/home/koen/LaTeX-setup/plots/w32-eos.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
