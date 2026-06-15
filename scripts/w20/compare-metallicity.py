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

model = mr.MesaData(
    f"{MASTER}/single-stars/z0.00735/M2.0-debug/LOGS/TPAGB/history.data"
)

model_mesh = mr.MesaData(
    f"{MASTER}/single-stars/z0.00735/M2.0-mesh-refine/LOGS/TPAGB/history.data"
)
# %%

model_solar = mr.MesaData(f"{MASTER}/standard-2msun-v3/LOGS/TPAGB/history.data")

# %%


profiles = []

for i in range(110, 133):
    profiles.append(
        mr.MesaData(f"{MASTER}/single-stars/z0.00735/M2.0/LOGS/TPAGB/profile{i}.data")
    )
# %%

profiles[0].bulk_names
# %%
for profile in profiles:
    plt.plot(profile.R, profile.gradT)

plt.show()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
plt.plot(model_solar.star_age, model_solar.R, c="C9", label="$z=0.014$")
plt.plot(model.star_age, model.R, c="C3", label="$z=0.00735$")
plt.scatter(model.star_age[-1], model.R[-1], marker="x", c="C3")


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-check-radius.pgf", format="pgf")
plt.show()
plt.close()

# %%
model.envelope_mass[-1]
# %%
plt.plot(model.R)
plt.show()

# %%

for i, profile in enumerate(profiles[-2:]):
    plt.plot(profile.logRho, profile.logT)

plt.show()


# %%
# DEBUG

profiles = []
profiles_mesh = []

for i in range(185, 190):
    profiles.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00735/M2.0-debug/LOGS/TPAGB/profile{i}.data"
        )
    )
    profiles_mesh.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00735/M2.0-mesh-refine/LOGS/TPAGB/profile{i}.data"
        )
    )


# %%
colormap = plt.cm.inferno(np.linspace(0, 1, int(1e6)))

for i, profile in enumerate(profiles):
    frac = model.envelope_mass[profile.model_number - 1] / model.envelope_mass[0]
    print(frac)
    plt.plot(profile.logRho, profile.logT, color=colormap[int(frac * 1e6)])

plt.show()


# %%
for profile in profiles:
    plt.plot(profile.mass, profile.gradT)
    plt.plot(profile.mass, profile.logeps_nuc)
    plt.plot(profile.mass, profile.eps_grav_ad)

plt.show()


# %%
colormap = plt.cm.inferno(np.linspace(0, 1, int(1e6)))

for i, profile in enumerate(profiles):
    frac = model.envelope_mass[profile.model_number - 1] / model.envelope_mass[0]
    print(frac)
    plt.scatter(
        profile.logRho, profile.logT, c=profile.gradT, cmap="viridis", vmin=-100, vmax=1
    )

plt.show()


# %%

profiles[0].bulk_names
# %%
for profile in profiles[-2:]:
    plt.plot(profile.zone, profile.gradT)
    # plt.plot(profile.mass, profile.logh1)
    # plt.plot(profile.mass, profile.loghe4)
    plt.plot(profile.zone, profile.logeps_nuc)
    # plt.plot(profile.mass, profile.logc13)
    # plt.plot(profile.mass, profile.logn14)

plt.show()


# %%
for profile in profiles[-2:]:
    plt.plot(profile.zone[1:], np.log10(-np.diff(profile.mass)))
    # plt.plot(profile.mass, profile.logh1)
    # plt.plot(profile.mass, profile.loghe4)
    plt.plot(profile.zone, profile.logeps_nuc)
    # plt.plot(profile.mass, profile.logc13)
    # plt.plot(profile.mass, profile.logn14)

plt.show()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for profile in profiles_mesh[-1:]:
    plt.plot(
        profile.mass,
        profile.gradT,
        c="C0",
        label="$0.5\\times$ mesh spacing",
        linewidth=0.8,
    )
for profile in profiles[4:5]:
    plt.plot(
        profile.mass,
        profile.gradT,
        c="C9",
        label="original mesh spacing",
        linewidth=0.8,
    )

plt.ylim(-1, 1)

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("internal mass $m$ ($m_\odot$)")
plt.ylabel("$\\nabla$")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-mesh-refine.pgf", format="pgf")
plt.show()
plt.close()
# %%
plt.plot(model.model_number, model.num_retries)
plt.plot(model_mesh.model_number, model_mesh.num_retries)
plt.show()

# %%
fig, axs = plt.subplots(
    3, 1, sharex=False, figsize=set_size(column, height=2), constrained_layout=True
)


for profile in profiles[-1:]:
    axs[0].plot(profile.logR, profile.gradT, label=r"$\nabla$")
    axs[0].plot(profile.logR, profile.gradr, label=r"$\nabla_\textrm{rad}$")
    axs[0].plot(profile.logR, profile.grada, label=r"$\nabla_\textrm{ad}$")
    axs[1].plot(profile.mass, profile.gradT)
    axs[1].plot(profile.mass, profile.gradr)
    axs[1].plot(profile.mass, profile.grada)
    axs[2].plot(profile.logP, profile.gradT)
    axs[2].plot(profile.logP, profile.gradr)
    axs[2].plot(profile.logP, profile.grada)

axs[2].invert_xaxis()
axs[0].set_xlabel("$\log(r/R_\odot)$")
axs[0].set_ylabel(r"$\nabla$")
axs[1].set_ylabel(r"$\nabla$")
axs[2].set_ylabel(r"$\nabla$")
axs[1].set_xlabel("internal mass $m$ ($m_\odot$)")
axs[2].set_xlabel("$\log P$ (dyn / cm$^2$)")
axs[0].set_ylim(-2, 2)
axs[1].set_ylim(-2, 2)
axs[2].set_ylim(-2, 2)
fig.legend(loc="outside upper center", ncols=3)
plt.savefig("/home/koen/LaTeX-setup/plots/w20-diagnose-gradient.pgf", format="pgf")
plt.show()
plt.close()
plt.show()
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

for i, profile in enumerate(profiles[-1:]):
    plt.plot(profile.logRho, profile.logT, c="k", linewidth=2)
    plt.scatter(
        profile.logRho[2303], profile.logT[2303], color="r", marker="x", zorder=10000
    )
    plt.scatter(
        profile.logRho[1191], profile.logT[1191], color="r", marker="x", zorder=10000
    )


plt.savefig("/home/koen/LaTeX-setup/plots/w20-eos.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

for profile in profiles[-1:]:
    plt.plot(profile.gradT, label=r"$\nabla$")

plt.show()
# %%


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


with open("/home/koen/mesa-r23.05.1/kap/plotter/kap_plotter.dat") as f:
    title = f.readline().strip()
    xlabel = f.readline().strip()
    ylabel = f.readline().strip()

kapDT, Yran, Xran = parse("/home/koen/mesa-r23.05.1/kap/plotter/kap_plotter.dat")

# set up plot and labels

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column, height=6 / 4), constrained_layout=True
)


ax = axs[0]
ax.set_xlabel(r"$\log(\rho/ \textrm{g cm}^{-3})$")
ax.set_ylabel(r"$\log(T/ \textrm{K})$")
ax.set_xlim(Xran.min(), Xran.max())
ax.set_ylim(Yran.min(), Yran.max())

# set up color map
cmap = copy.copy(mpl.cm.inferno)
cmap.set_over("white")
cmap.set_under("black")

# set color bar limits
# None will auto-set limits
cbar_min = None
cbar_max = None

pcol = ax.pcolormesh(
    Xran,
    Yran,
    kapDT[..., 2],
    shading="nearest",
    cmap=cmap,
    vmin=cbar_min,
    vmax=cbar_max,
    rasterized=True,
)
pcol.set_edgecolor("face")
cax = fig.colorbar(
    pcol, extend="neither", orientation="horizontal", location="top", aspect=30
)
cax.set_label(r"$\log(\kappa / \textrm{cm}^2 \textrm{ g})$")


for i, profile in enumerate(profiles[-1:]):
    ax.plot(profile.logRho, profile.logT, c="k", linewidth=2)
    ax.scatter(
        profile.logRho[2303], profile.logT[2303], color="b", marker="x", zorder=10000
    )
    ax.scatter(
        profile.logRho[1191], profile.logT[1191], color="b", marker="x", zorder=10000
    )

ax.set_facecolor("C9")

ax = axs[1]
ax.set_xlabel(r"$\log(\rho/ \textrm{g cm}^{-3})$")
ax.set_ylabel(r"$\log(T/ \textrm{K})$")
ax.set_xlim(Xran.min(), Xran.max())
ax.set_ylim(Yran.min(), Yran.max())

# set up color map
cmap = copy.copy(mpl.cm.inferno)
cmap.set_over("white")
cmap.set_under("black")

# set color bar limits
# None will auto-set limits
cbar_min = -15
cbar_max = 5

pcol = ax.pcolormesh(
    Xran,
    Yran,
    kapDT[..., 2],
    shading="nearest",
    cmap=cmap,
    vmin=cbar_min,
    vmax=cbar_max,
    rasterized=True,
)
pcol.set_edgecolor("face")
cax = fig.colorbar(
    pcol, extend="both", orientation="horizontal", location="top", aspect=30
)
cax.set_label(r"$\log(\kappa / \textrm{cm}^2 \textrm{ g})$")


for i, profile in enumerate(profiles[-1:]):
    ax.plot(profile.logRho, profile.logT, c="k", linewidth=2)
    ax.scatter(
        profile.logRho[2303], profile.logT[2303], color="b", marker="x", zorder=10000
    )
    ax.scatter(
        profile.logRho[1191], profile.logT[1191], color="b", marker="x", zorder=10000
    )

ax.set_facecolor("C9")


axs[0].set_xlim(-15, 10)
axs[1].set_xlim(-15, 10)
axs[0].set_ylim(2, 10)
axs[1].set_ylim(2, 10)
plt.savefig("/home/koen/LaTeX-setup/plots/w20-kap.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

profile.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

plt.xlabel(r"$\log(r / R_\odot)$")
plt.ylabel("$X$")


for profile in profiles[-1:]:
    plt.text(
        profile.logR[0] - 3.2,
        profile.logh1[0],
        r"$\textrm{H}_1$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.loghe3[0],
        r"$\textrm{He}_3$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.loghe4[0],
        r"$\textrm{He}_4$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.logc12[0],
        r"$\textrm{C}_{12}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.logc13[0],
        r"$\textrm{C}_{13}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.logn14[0],
        r"$\textrm{N}_{14}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.logo16[0],
        r"$\textrm{O}_{16}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.5,
        profile.logne20[0],
        r"$\textrm{Ne}_{20}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 3.2,
        profile.logmg24[0],
        r"$\textrm{Mg}_{24}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.plot(profile.logR, profile.logh1)
    plt.plot(profile.logR, profile.loghe3)
    plt.plot(profile.logR, profile.loghe4)
    plt.plot(profile.logR, profile.logc12)
    plt.plot(profile.logR, profile.logc13)
    plt.plot(profile.logR, profile.logn14)
    plt.plot(profile.logR, profile.logo16)
    plt.plot(profile.logR, profile.logne20)
    plt.plot(profile.logR, profile.logmg24)

plt.ylim(-5.5, 0.5)
plt.xlim(-3, 0)
plt.vlines(
    profile.logR[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
plt.vlines(
    profile.logR[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)

plt.savefig("/home/koen/LaTeX-setup/plots/w20-abundances.pgf", format="pgf")
plt.show()
plt.close()
# %%
