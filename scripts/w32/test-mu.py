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

from matplotlib.colors import TwoSlopeNorm

for i in range(13, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(prof_heavy.mu),
        vmax=0.64,
    )

    c = plt.cplot(
        prof_heavy.logRho,
        prof_heavy.logT,
        prof_heavy.mu,
        label="MESA",
        linewidth=2,
        norm=norm,
        zorder=11,
        cmap="bwr",
    )

    plt.plot(
        prof_heavy.logRho, prof_heavy.logT, "k", label="MESA", linewidth=3, zorder=10
    )

cbar = plt.colorbar(c, ax=ax, extend="max", label="$\mu$")
cbar.ax.set_yscale("linear")


plt.savefig("/home/koen/LaTeX-setup/plots/w32-eos.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

begin_blend = 1017
begin_Skye = 1057

for i in range(13, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(prof_heavy.mu),
        vmax=0.64,
    )

    plt.plot(
        prof_heavy.logR, prof_heavy.mu, "k", label="MESA EOS", linewidth=1.5, zorder=10
    )
    plt.plot(
        prof_heavy.logR,
        compute_mu_profile(prof_heavy),
        "k",
        label="Full ionization",
        linewidth=0.75,
        zorder=10,
        alpha=0.5,
    )

plt.legend(frameon=False)
plt.axvspan(-2, prof_heavy.logR[1057], color=my_colors[5], alpha=0.75, edgecolor=None)
plt.axvspan(
    prof_heavy.logR[1057],
    prof_heavy.logR[1017],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(prof_heavy.logR[1017], 2, color=my_colors[4], alpha=0.75, edgecolor=None)
plt.xlim(-2, 1.75)
plt.ylim(0.60, 0.64)

for x, label in [
    (-1.9, "Skye"),
    (-1.4, "blend"),
    (-0.8, "FreeEOS"),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="left",
        clip_on=False,
    )

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log$(R / R_\\odot)$")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-EOS-profile-RGB.pgf", format="pgf")
plt.show()
plt.close()


# %%
def compute_mu_profile(profile, which="float"):
    names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    z_s = [1, 2, 2, 6, 7, 8, 10, 12]
    if which == "integer":
        a_s = [1.0, 3.0, 4.00, 12.0, 14.00, 16.0, 20.0, 24.0]
    else:
        a_s = [
            1.00782503224,
            3.0160293201,
            4.00260325413,
            12,
            14.00307400443,
            15.99491461957,
            19.9924401762,
            23.985041697,
        ]

    mu_inv = 0
    X_tot = np.zeros(len(profile.mass))
    for n, Z_i, A_i in zip(names, z_s, a_s):
        X_i = profile.data(n)
        mu_inv += X_i * (1 + Z_i) / A_i
        X_tot += X_i
        print(n, X_i[0], (X_i * (1 + Z_i) / A_i)[0])
    mu = 1 / mu_inv
    return mu


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

begin_blend = 866
begin_Skye = 925

for profile in profiledict[4.4][2:3]:
    plt.plot(profile.logR, profile.mu, "k", label="MESA EOS", linewidth=1.5, zorder=11)
    plt.plot(
        profile.logR,
        compute_mu_profile(profile, "integer"),
        "k",
        label="Full ionization",
        linewidth=0.75,
        zorder=10,
        alpha=0.5,
    )
plt.xlim(-2)

plt.axhline(mu, c="k", linestyle=":", label="Asplund (2009)", linewidth=0.75)

plt.legend(frameon=False)
plt.axvspan(
    -2, profile.logR[begin_Skye], color=my_colors[5], alpha=0.75, edgecolor=None
)
plt.axvspan(
    profile.logR[begin_blend],
    profile.logR[begin_Skye],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(
    profile.logR[begin_blend], 2, color=my_colors[4], alpha=0.75, edgecolor=None
)
plt.xlim(-1.5, 0.5)

for x, label in [
    (-1.2, "Skye"),
    (-0.6, "blend"),
    (-0, "FreeEOS"),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="left",
        clip_on=False,
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log$(R/R_\\odot)$")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-mu-2.pgf", format="pgf")
plt.show()
plt.close()
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

from matplotlib.colors import TwoSlopeNorm

for profile in profiledict[0.4][2:3]:

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(profile.mu),
        vmax=0.64,
    )

    plt.plot(profile.logRho, profile.logT, "k", linewidth=1, zorder=10)

    ind = np.argwhere(profile.h1 < 0.68)[0]
    plt.scatter(profile.logRho[ind], profile.logT[ind], c="k", s=3)

for profile in profiledict[1][2:3]:

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(profile.mu),
        vmax=0.64,
    )

    plt.plot(profile.logRho, profile.logT, "k", linewidth=1, zorder=10)

    ind = np.argwhere(profile.h1 < 0.68)[0]
    plt.scatter(profile.logRho[ind], profile.logT[ind], c="k", s=3)

for profile in profiledict[4.4][2:3]:

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(profile.mu),
        vmax=0.64,
    )

    plt.plot(profile.logRho, profile.logT, "k", linewidth=1, zorder=10)

    ind = np.argwhere(profile.h1 < 0.68)[0]
    plt.scatter(
        profile.logRho[ind], profile.logT[ind], c="k", s=3, label="Core boundary"
    )

plt.legend(frameon=False)

plt.text(-9, 4.45, "$4.4\\;M_\\odot$", fontsize=8, ha="right", va="top")
plt.text(-7, 3.95, "$1\\;M_\\odot$", fontsize=8, ha="right", va="top")
plt.text(-5.4, 3.55, "$0.4\\;M_\\odot$", fontsize=8, ha="right", va="top")

plt.savefig("/home/koen/LaTeX-setup/plots/w32-eos-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

asplund = Asplund(z=0.005573, he_method="karakas")

mu_inv = 0
for i, (key, ab) in enumerate(asplund.elements.items()):
    X_i = ab.massfrac
    Z_i = ab.Z
    A_i = ab.atomic_mass
    mu_inv += X_i * (1 + Z_i) / A_i
mu = 1 / mu_inv
print(mu)
# %%


def compute_mu_asplund(asplund):

    mu_inv = 0
    for i, element in enumerate(asplund.elements):
        X_i = asplund.elements[element].massfrac
        Z_i = asplund.elements[element].Z
        A_i = asplund.elements[element].atomic_mass
        mu_inv += X_i * (1 + Z_i) / A_i
        print(asplund.elements[element].name, X_i, (X_i * (1 + Z_i) / A_i))
    mu = 1 / mu_inv
    return mu


# %%

compute_mu_profile(profile)
# asplund = Asplund()
asplund = Asplund(z=0.005573, he_method="karakas")
compute_mu_asplund(asplund)
# %%


plt.plot(profile.mass, profile.he4)
plt.show()


# %%
def compute_mu_profile(profile, which="float"):
    names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    z_s = [1, 2, 2, 6, 7, 8, 10, 12]
    if which == "integer":
        a_s = [1.0, 3.0, 4.00, 12.0, 14.00, 16.0, 20.0, 24.0]
    else:
        a_s = [
            1.00782503224,
            3.0160293201,
            4.00260325413,
            12,
            14.00307400443,
            15.99491461957,
            19.9924401762,
            23.985041697,
        ]

    mu_inv = 0
    X_tot = np.zeros(len(profile.mass))
    for n, Z_i, A_i in zip(names, z_s, a_s):
        X_i = profile.data(n)
        mu_inv += X_i * (1 + Z_i) / A_i
        X_tot += X_i
        print(n, X_i[0], (X_i * (1 + Z_i) / A_i)[0])
    mu = 1 / mu_inv
    return mu


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

begin_blend = 866
begin_Skye = 925

for profile in profiledict[4.4][2:3]:
    plt.plot(profile.logR, profile.mu, "k", label="MESA EOS", linewidth=1, zorder=11)
    plt.plot(
        profile.logR,
        compute_mu_profile(profile, "floats"),
        "C0",
        label="Full ionization\n(atomic mass)",
        linewidth=3,
        zorder=10,
        alpha=0.5,
    )
    plt.plot(
        profile.logR,
        compute_mu_profile(profile, "integer"),
        "C3",
        label="Full ionization\n(mass number)",
        linewidth=3,
        zorder=10,
        alpha=0.5,
    )
plt.xlim(-2)


plt.legend(frameon=False, loc="lower left")
plt.axvspan(
    -2, profile.logR[begin_Skye], color=my_colors[5], alpha=0.75, edgecolor=None
)
plt.axvspan(
    profile.logR[begin_blend],
    profile.logR[begin_Skye],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(
    profile.logR[begin_blend], 2, color=my_colors[4], alpha=0.75, edgecolor=None
)
plt.xlim(-1.5, 0.5)

for x, label in [
    (-1.2, "Skye"),
    (-0.6, "blend"),
    (-0, "FreeEOS"),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="left",
        clip_on=False,
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log$(R/R_\\odot)$")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-mu-3.pgf", format="pgf")
plt.show()
plt.close()


# %%
def compute_mu_profile(profile, which="float"):
    names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    z_s = [1, 2, 2, 6, 7, 8, 10, 12]
    if which == "integer":
        a_s = [1.0, 3.0, 4.00, 12.0, 14.00, 16.0, 20.0, 24.0]
    else:
        a_s = [
            1.00782503224,
            3.0160293201,
            4.00260325413,
            12,
            14.00307400443,
            15.99491461957,
            19.9924401762,
            23.985041697,
        ]

    mu_inv = 0
    X_tot = np.zeros(len(profile.mass))
    for n, Z_i, A_i in zip(names, z_s, a_s):
        X_i = profile.data(n)
        mu_inv += X_i * (1 + Z_i) / A_i
        X_tot += X_i
        print(n, X_i[0], (X_i * (1 + Z_i) / A_i)[0])
    mu = 1 / mu_inv
    return mu


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

begin_blend = 1017
begin_Skye = 1057

for i in range(13, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )

    norm = TwoSlopeNorm(
        vcenter=0.609,
        vmin=np.nanmin(prof_heavy.mu),
        vmax=0.64,
    )

    plt.plot(
        prof_heavy.logR, prof_heavy.mu, "k", label="MESA EOS", linewidth=1, zorder=11
    )
    plt.plot(
        prof_heavy.logR,
        compute_mu_profile(prof_heavy, "float"),
        "C0",
        label="Full ionization\n(atomic mass)",
        linewidth=3,
        zorder=10,
        alpha=0.5,
    )

    plt.plot(
        prof_heavy.logR,
        compute_mu_profile(prof_heavy, "integer"),
        "C3",
        label="Full ionization\n(mass number)",
        linewidth=3,
        zorder=10,
        alpha=0.5,
    )

plt.legend(frameon=False)
plt.axvspan(-2, prof_heavy.logR[1057], color=my_colors[5], alpha=0.75, edgecolor=None)
plt.axvspan(
    prof_heavy.logR[1057],
    prof_heavy.logR[1017],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(prof_heavy.logR[1017], 2, color=my_colors[4], alpha=0.75, edgecolor=None)
plt.xlim(-2, 1.75)
plt.ylim(0.60, 0.64)

for x, label in [
    (-1.9, "Skye"),
    (-1.4, "blend"),
    (-0.8, "FreeEOS"),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="left",
        clip_on=False,
    )

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("log$(R / R_\\odot)$")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w32-EOS-profile-RGB-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
