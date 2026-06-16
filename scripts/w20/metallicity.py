import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

import pandas as pd

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("presentation")

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

# %%

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")

# %%

data["Porb"] = data["Porb"][1:].astype("float")
data["ecc"] = data["ecc"][1:].astype("float")
data["[Fe/H]"] = data["[Fe/H]"][1:].astype("float")
# %%
fig = plt.figure()
plt.scatter(data["Porb"], data["ecc"], s=50)
plt.xscale("log")
plt.xlim(1e1, 1e5)
plt.ylim(0, 1)
ax = plt.gca()
ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))
plt.title("Forming barium stars", size=50)
plt.xlabel("Period (days)")
plt.show()

# %%

plt.plot(np.sort(data["[Fe/H]"]))
plt.show()
# %%

print(f"mean:   {np.nanmean(data["[Fe/H]"]):.3f}")
print(f"median: {np.nanmedian(np.sort(data["[Fe/H]"])):.3f}")
# %%


#!/usr/bin/env python

import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def parse(fname):
    nY, nX = np.loadtxt(fname, max_rows=1, skiprows=3, unpack=True, dtype=int)
    data = np.loadtxt(fname, skiprows=4)
    data = np.reshape(data, ((nX, nY, -1)))
    Yran = np.array(data[0, :, 0])
    Xran = np.array(data[:, 0, 1])
    data = np.swapaxes(data, 0, 1)
    return data, Yran, Xran


with open("/home/koen/mesa-r23.05.1/eos/plotter/eos_plotter.dat") as f:
    title = f.readline().strip()
    xlabel = f.readline().strip()
    ylabel = f.readline().strip()

eosDT, Yran, Xran = parse("/home/koen/mesa-r23.05.1/eos/plotter/eos_plotter.dat")

# set up plot and labels

print(xlabel)
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlim(Xran.min(), Xran.max())
ax.set_ylim(Yran.min(), Yran.max())


# set color bar limits
# None will auto-set limits
cbar_min = None
cbar_max = None

pcol = ax.pcolormesh(
    Xran,
    Yran,
    eosDT[..., 2],
    shading="nearest",
    cmap="viridis",
    vmin=cbar_min,
    vmax=cbar_max,
)
pcol.set_edgecolor("face")
cax = fig.colorbar(pcol, extend="both")

plt.show()
# %%

data = pd.read_csv("/home/koen/master-internship/scripts/w20/decastro.dat")


print(data["[Fe I/H]"])
# %%

print(f"mean:   {np.nanmean(data["[Fe II/H]"]):.3f}")
print(f"whei:   {np.average(data["[Fe II/H]"], weights=1/data["e[Fe II/H]"]**2):.3f}")
print(f"median: {np.nanmedian(np.sort(data["[Fe II/H]"])):.3f}")

# %%
print(f"mean:   {np.nanmean(data["[Fe I/H]"]):.3f}")
print(f"whei:   {np.average(data["[Fe I/H]"], weights=1/data["e[Fe I/H]"]**2):.3f}")
print(f"median: {np.nanmedian(np.sort(data["[Fe I/H]"])):.3f}")


# %%

for dat in data["[Fe I/H]"]:
    print(dat)
# %%
