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

# %%
model_mesh = mr.MesaData(
    f"{MASTER}/single-stars/z0.00735/M2.0-mesh-refine/LOGS/TPAGB/history.data"
)

# %%
model_mesh_2 = mr.MesaData(
    f"{MASTER}/single-stars/z0.00735/M2.0-mesh-refine-2/LOGS/TPAGB/history.data"
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
profiles_mesh_2 = []
for i in range(180, 183):
    profiles_mesh_2.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00735/M2.0-mesh-refine-2/LOGS/TPAGB/profile{i}.data"
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
    1, 1, sharex=True, figsize=set_size(column, height=1.20), constrained_layout=True
)

plt.xlabel(r"$\log(r / R_\odot)$")
plt.ylabel("$\log(X)$")


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

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1.20), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(X)$")


for profile in profiles[-1:]:
    plt.text(
        profile.logR[0] - 2.0,
        profile.logh1[0],
        r"$\textrm{H}_1$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.loghe3[0],
        r"$\textrm{He}_3$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.loghe4[0],
        r"$\textrm{He}_4$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.logc12[0],
        r"$\textrm{C}_{12}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.logc13[0],
        r"$\textrm{C}_{13}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.logn14[0],
        r"$\textrm{N}_{14}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.logo16[0],
        r"$\textrm{O}_{16}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.1,
        profile.logne20[0],
        r"$\textrm{Ne}_{20}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.text(
        profile.logR[0] - 2.0,
        profile.logmg24[0],
        r"$\textrm{Mg}_{24}$",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="white"),
    )
    plt.plot(profile.mass, profile.logh1)
    plt.plot(profile.mass, profile.loghe3)
    plt.plot(profile.mass, profile.loghe4)
    plt.plot(profile.mass, profile.logc12)
    plt.plot(profile.mass, profile.logc13)
    plt.plot(profile.mass, profile.logn14)
    plt.plot(profile.mass, profile.logo16)
    plt.plot(profile.mass, profile.logne20)
    plt.plot(profile.mass, profile.logmg24)

plt.ylim(-5.5, 0.5)
plt.vlines(
    profile.mass[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
plt.vlines(
    profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)

plt.savefig("/home/koen/LaTeX-setup/plots/w20-abundances-mass.pgf", format="pgf")
plt.show()
plt.close()

# %%
profiles_all = []

for i in range(5, 10, 5):
    profiles_all.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00735/M2.0-start/LOGS/TPAGB/profile{i}.data"
        )
    )

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(X)$")


for profile in profiles[-1:]:

    plt.plot(profile.mass, profile.c12)
    plt.plot(profile.mass, profile.o16)

plt.gca().set_ylim(plt.gca().get_ylim())
plt.vlines(
    profile.mass[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
# plt.vlines(
#     profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
# )

plt.savefig("/home/koen/LaTeX-setup/plots/w20-abundances-mass-zoom.pgf", format="pgf")
plt.show()
plt.close()

print(profile.mass[2303])

# %%

profiles_all = []

for i in range(5, 10, 5):
    profiles_all.append(
        mr.MesaData(
            f"{MASTER}/single-stars/z0.00735/M2.0-start/LOGS/CHeB/profile{i}.data"
        )
    )

# %%
import mesa_reader as mr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
sys.path.append("/home/koen/astro-codes/mkipp")
sys.path.append("/home/koen/astro-codes/read_mist/")

import read_mist_models
import mkipp
import kipp_data
import mesa_data
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

mkipp.kipp_plot(
    mkipp.Kipp_Args(
        logs_dirs=[
            "/home/koen/master-internship/mesa-models/compare-overshooting/rees/3msun/LOGS/MS/"
        ],
        core_masses=["He", "CO"],
        levels=[],
        log_levels=True,
        num_levels=10,
        xaxis="star_age",
    )
)
plt.show()

# %%
import matplotlib as mpl
from matplotlib.patches import PathPatch

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]

fig, axs = plt.subplots(
    1, 1, sharex="col", figsize=set_size(column), constrained_layout=True
)

mass = 2

axis = axs

method = methods[0]
phase = phases[2]
kipp_args = mkipp.Kipp_Args(
    logs_dirs=[
        f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/"
    ],
    xaxis="star_age",
    save_file=False,
    decorate_plot=False,
    # contour_colormap=plt.get_cmap("Greens"),
    levels=np.linspace(-1, 5, 50),
)
mkipp.kipp_plot(kipp_args, axis=axis)
axis.set_rasterization_zorder(-10)

cmap = mpl.cm.Blues
norm = mpl.colors.Normalize(vmin=-1, vmax=5)

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    label=r"$\log(\epsilon_\textrm{nuc})$",
    fraction=0.05,
    extend="both",
)

plt.gca().set_xlim(plt.gca().get_xlim())
plt.hlines(profile.mass[2303], *plt.gca().get_xlim(), color="C9", linewidth=0.8)
plt.xlabel("Time (Myr)")
plt.ylabel("internal mass $m$ ($M_\odot$)")
plt.ylim(0, 0.61)
plt.savefig(f"/home/koen/LaTeX-setup/plots/w20-kipp.pgf", format="pgf", dpi=600)
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(\epsilon_\\textrm{nuc})$")


for profile in profiles[-1:]:

    plt.plot(profile.mass, profile.logeps_nuc)

plt.gca().set_ylim(plt.gca().get_ylim())
plt.vlines(
    profile.mass[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
plt.vlines(
    profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)

plt.savefig("/home/koen/LaTeX-setup/plots/w20-eps_nuc.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(\epsilon_\\textrm{nuc})$")


for profile in profiles[-1:]:

    plt.plot(profile.mass, profile.eps_grav_ad)

plt.gca().set_ylim(plt.gca().get_ylim())
plt.vlines(
    profile.mass[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
plt.vlines(
    profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)

# plt.savefig("/home/koen/LaTeX-setup/plots/w20-eps_nuc.pgf", format="pgf")
plt.show()
plt.close()


# %%
profile.bulk_names

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(X)$")


for profile in profiles[-1:]:

    plt.plot(profile.logP[1:], np.diff(profile.logc12) / np.diff(profile.logP))

plt.gca().set_ylim(plt.gca().get_ylim())
plt.vlines(
    profile.logP[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
# plt.vlines(
#     profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
# )

# plt.savefig("/home/koen/LaTeX-setup/plots/w20-abundances-mass-zoom.pgf", format="pgf")
plt.show()
plt.close()

print(profile.mass[2303])


# %%
# plt.plot(model_mesh.star_age, model_mesh.log_dt)
# plt.plot(model_mesh_2.star_age, model_mesh_2.log_dt)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("model number")
plt.ylabel("$\log(\\Delta t / \\textrm{yr})$")

plt.plot(model.model_number, model.log_dt)

plt.savefig("/home/koen/LaTeX-setup/plots/w12-dt.pgf", format="pgf")
plt.show()
plt.close()


# %%
plt.plot(model_mesh.star_age, model_mesh.num_zones)
plt.plot(model_mesh_2.star_age, model_mesh_2.num_zones)
plt.plot(model.star_age, model.num_zones)
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
for profile in profiles_mesh_2[-1:]:
    plt.plot(
        profile.mass,
        profile.gradT,
        c="C2",
        label="$0.5\\times$ mesh spacing at C$_{12}$ changes",
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
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for profile in profiles_mesh[-1:]:
    plt.plot(
        profile.mass,
        profile.zone,
        c="C0",
        label="$0.5\\times$ mesh spacing",
        linewidth=0.8,
    )
for profile in profiles[4:5]:
    plt.plot(
        profile.mass,
        profile.zone,
        c="C9",
        label="original mesh spacing",
        linewidth=0.8,
    )
for profile in profiles_mesh_2[-1:]:
    plt.plot(
        profile.mass,
        profile.zone,
        c="C2",
        label="$0.5\\times$ mesh spacing at C$_{12}$ changes",
        linewidth=0.8,
        zorder=-1,
    )

# plt.ylim(-1, 1)

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("internal mass $m$ ($m_\odot$)")
plt.ylabel("$\\nabla$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w20-mesh-refine.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("model number")
plt.ylabel("$\log(\\Delta t / \\textrm{yr})$")

plt.plot(model.star_age, model.log_dt)
plt.plot(model_mesh.star_age, model_mesh.log_dt)
plt.plot(model_mesh_2.star_age, model_mesh_2.log_dt)

# plt.savefig("/home/koen/LaTeX-setup/plots/w12-dt.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"internal mass $m$ ($M_\odot$)")
plt.ylabel("$\log(X)$")


for profile in profiles[-1:]:

    plt.plot(profile.logP[1:], np.diff(profile.logc12) / np.diff(profile.logP))

plt.gca().set_ylim(plt.gca().get_ylim())
plt.vlines(
    profile.logP[2303], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
)
# plt.vlines(
#     profile.mass[1191], *plt.gca().get_ylim(), color="C9", linewidth=0.8, zorder=-10
# )

# plt.savefig("/home/koen/LaTeX-setup/plots/w20-abundances-mass-zoom.pgf", format="pgf")
plt.show()
plt.close()

print(profile.c12[2300:2310])


# %%

model2 = mr.MesaData(f"{MASTER}/single-stars/z0.00883/M2.0/LOGS/TPAGB/history.data")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
plt.plot(model_solar.star_age, model_solar.R, c="C9", label="$z=0.014$")
plt.plot(model.star_age, model.R, c="C3", label="$z=0.00735$", alpha=0.5)
plt.plot(model2.star_age, model2.R, c="C0", label="$z=0.00883$")
plt.scatter(model.star_age[-1], model.R[-1], marker="x", c="C3")


fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-check-radius-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
full_model_solar = []
full_model = []
full_model2 = []

phases = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]
for phase in phases:
    full_model_solar.append(
        mr.MesaData(f"{MASTER}/standard-2msun-v3/LOGS/{phase}/history.data")
    )
    full_model.append(
        mr.MesaData(f"{MASTER}/single-stars/z0.00735/M2.0/LOGS/{phase}/history.data")
    )
    full_model2.append(
        mr.MesaData(f"{MASTER}/single-stars/z0.00883/M2.0/LOGS/{phase}/history.data")
    )

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

full_models = [full_model_solar, full_model, full_model2]
names = ["$z=0.014$", "$z=0.00735$", "$z=0.00883$"]
colors = ["C9", "C3", "C0"]
ls = []
for i, model in enumerate(full_models):
    for phase in model:
        (l,) = plt.plot(phase.log_Teff, phase.log_L, c=colors[i], label=names[i])
    ls.append(l)


plt.gca().invert_xaxis()
fig.legend(loc="outside upper center", ncols=3, handles=ls)
plt.xlabel("$\log(T_\\textrm{eff} / \\textrm{K})$")
plt.ylabel("$\log(L / L_\\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-HR.pgf", format="pgf")
plt.show()
plt.close()
# %%

l = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00883/M2.0/LOGS/TPAGB")

# %%
for profile in l.profile_numbers:
    profile = l.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)

# %%

import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# settings
# --------------------------------------------------

n_grid = 300

# --------------------------------------------------
# common radius grid
# --------------------------------------------------
profiles = list(l.profile_dict.values())[60:]

logR_min = min(np.min(p.logR) for p in profiles)
logR_max = max(np.max(p.logR) for p in profiles)
print(10**logR_max)

logR_grid = np.linspace(logR_min, logR_max, n_grid)

# --------------------------------------------------
# precompute all interpolated abundances
# --------------------------------------------------

interp_data = {}


cube = np.empty((len(profiles), n_grid))

for i, profile in enumerate(profiles):

    x = profile.logR[::-1]
    y = profile.logS_per_baryon[::-1]

    cube[i] = np.interp(
        logR_grid,
        x,
        y,
        left=np.nan,
        right=np.nan,
    )

interp_data = cube

# --------------------------------------------------
# initial frame
# --------------------------------------------------

fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=interp_data[0],
        mode="lines",
        name="entropy",
        line=dict(width=2),
    )
)

# --------------------------------------------------
# animation frames
# --------------------------------------------------

frames = []

for i, profile in enumerate(profiles):

    frame_data = []

    frame_data.append(
        go.Scatter(
            x=logR_grid,
            y=interp_data[i],
        )
    )

    age = getattr(profile, "star_age", None)

    if age is not None:
        title = f"Age = {age:.3e} yr"
    else:
        title = f"Profile {i}"

    frames.append(
        go.Frame(
            name=str(i),
            data=frame_data,
            layout=go.Layout(title=title),
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
# --------------------------------------------------

slider_steps = []

for i in range(len(profiles)):
    slider_steps.append(
        {
            "method": "animate",
            "label": str(i),
            "args": [
                [str(i)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0},
                    "transition": {"duration": 0},
                },
            ],
        }
    )

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
    width=1200,
    height=700,
    xaxis_title=r"log(R/R☉)",
    yaxis_title="Mass fraction",
    hovermode="x unified",
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 20,
                                "redraw": False,
                            },
                            "transition": {
                                "duration": 0,
                            },
                            "fromcurrent": True,
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            "active": 0,
            "steps": slider_steps,
            "x": 0.1,
            "y": -0.08,
            "len": 0.8,
        }
    ],
)

fig.show()

# %%


import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# settings
# --------------------------------------------------

n_grid = 300

# --------------------------------------------------
# common radius grid
# --------------------------------------------------
profiles = list(l.profile_dict.values())[60:]

logR_min = 0.58
logR_max = max(np.max(p.mass) for p in profiles)
print(10**logR_max)

logR_grid = np.linspace(logR_min, logR_max, n_grid)

# --------------------------------------------------
# precompute all interpolated abundances
# --------------------------------------------------

interp_data = {}


cube = np.empty((len(profiles), n_grid + 1))
cube2 = np.empty((len(profiles), n_grid + 1))

for i, profile in enumerate(profiles):

    x = profile.mass[::-1]
    y = profile.logS_per_baryon[::-1]

    max_R = profile.mass[::-1][np.argmax(y)]
    data = np.append(logR_grid, max_R)
    data.sort()
    cube2[i] = data

    cube[i] = np.interp(
        cube2[i],
        x,
        y,
        left=np.nan,
        right=np.nan,
    )

interp_Rs = cube2
interp_data = cube

# --------------------------------------------------
# initial frame
# --------------------------------------------------

fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=interp_Rs[0],
        y=interp_data[0],
        mode="lines",
        name="entropy",
        line=dict(width=2),
    )
)

# --------------------------------------------------
# animation frames
# --------------------------------------------------

frames = []

for i, profile in enumerate(profiles):

    frame_data = []

    frame_data.append(
        go.Scatter(
            x=interp_Rs[i],
            y=interp_data[i],
        )
    )

    age = getattr(profile, "star_age", None)

    if age is not None:
        title = f"Age = {age:.3e} yr"
    else:
        title = f"Profile {i}"

    frames.append(
        go.Frame(
            name=str(i),
            data=frame_data,
            layout=go.Layout(title=title),
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
# --------------------------------------------------

slider_steps = []

for i in range(len(profiles)):
    slider_steps.append(
        {
            "method": "animate",
            "label": str(i),
            "args": [
                [str(i)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0},
                    "transition": {"duration": 0},
                },
            ],
        }
    )

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
    width=1200,
    height=700,
    xaxis_title=r"log(R/R☉)",
    yaxis_title="Mass fraction",
    hovermode="x unified",
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 20,
                                "redraw": False,
                            },
                            "transition": {
                                "duration": 0,
                            },
                            "fromcurrent": True,
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ],
)

fig.show()
# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())[60:]
n = len(profiles)

n_grid = 300

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = min(np.min(p.logR) for p in profiles)
logR_max = max(np.max(p.logR) for p in profiles)
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = 0.58
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.linspace(m_min, m_max, n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid + 1))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.logR[::-1]
    yR = p.logS_per_baryon[::-1]

    xM = p.mass[::-1]
    yM = p.logS_per_baryon[::-1]

    m_grid = np.linspace(m_min, m_max, n_grid)
    max_S = p.mass[::-1][np.argmax(yM)]
    m_grid = np.append(m_grid, max_S)
    m_grid.sort()

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    entropy_M[i] = np.interp(m_grid, xM, yM, left=np.nan, right=np.nan)

# radius evolution
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
)

# use index instead of age for stability
idx = np.arange(n)

print(entropy_R[0])

# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
    subplot_titles=[
        "Entropy vs log R",
        "Entropy vs mass",
        "Radius evolution",
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
    ),
    row=1,
    col=1,
)

# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=m_grid,
        y=entropy_M[0],
        mode="lines",
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2, col=1
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=m_grid, y=entropy_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
# --------------------------------------------------

steps = [
    dict(
        method="animate",
        args=[
            [str(i)],
            {
                "mode": "immediate",
                "frame": {"duration": 20, "redraw": False},
                "transition": {"duration": 0},
            },
        ],
        label=str(i),
    )
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 30, "redraw": False},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"mode": "immediate"}],
                ),
            ],
        )
    ],
    sliders=[dict(active=0, steps=steps)],
)

# axis labels
fig.update_xaxes(title_text="log R", row=1, col=1)
fig.update_xaxes(title_text="mass coordinate", row=1, col=2)
fig.update_xaxes(title_text="profile index", row=2, col=1)

fig.update_yaxes(title_text="entropy", row=1, col=1)
fig.update_yaxes(title_text="entropy", row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

fig.write_html(
    "/home/koen/figures/plots/master-internship/w20/post-AGB-instabilities.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)
# %%

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())[60:]
n = len(profiles)

n_grid = 300

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = min(np.min(p.logR) for p in profiles)
logR_max = max(np.max(p.logR) for p in profiles)
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = 0.58
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.linspace(m_min, m_max, n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid + 1))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.logR[::-1]
    yR = p.gradT[::-1]

    xM = p.mass[::-1]
    yM = p.gradT[::-1]

    m_grid = np.linspace(m_min, m_max, n_grid)
    max_S = p.mass[::-1][np.argmax(yM)]
    m_grid = np.append(m_grid, max_S)
    m_grid.sort()

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    entropy_M[i] = np.interp(m_grid, xM, yM, left=np.nan, right=np.nan)

# radius evolution
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
)

# use index instead of age for stability
idx = np.arange(n)

print(entropy_R[0])

# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
    subplot_titles=[
        "Entropy vs log R",
        "Entropy vs mass",
        "Radius evolution",
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
    ),
    row=1,
    col=1,
)

# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=m_grid,
        y=entropy_M[0],
        mode="lines",
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2, col=1
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=m_grid, y=entropy_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
# --------------------------------------------------

steps = [
    dict(
        method="animate",
        args=[
            [str(i)],
            {
                "mode": "immediate",
                "frame": {"duration": 20, "redraw": False},
                "transition": {"duration": 0},
            },
        ],
        label=str(i),
    )
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 30, "redraw": False},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"mode": "immediate"}],
                ),
            ],
        )
    ],
    sliders=[dict(active=0, steps=steps)],
)

# axis labels
fig.update_xaxes(title_text="log R", row=1, col=1)
fig.update_xaxes(title_text="mass coordinate", row=1, col=2)
fig.update_xaxes(title_text="profile index", row=2, col=1)

fig.update_yaxes(title_text="entropy", row=1, col=1)
fig.update_yaxes(title_text="entropy", row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w20/post-AGB-instabilities.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )
# %%
p.bulk_names

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(
    figsize=set_size(full, height=0.5),
    constrained_layout=True
)

# --------------------------------------------------
# main plot
# --------------------------------------------------

ax.plot(
    model2.star_age[-20000:],
    model2.R[-20000:],
    c="k",
    lw=0.8
)

ax.set_xlim(2.02e6,2.2e6)
# --------------------------------------------------
# inset
# --------------------------------------------------

axins = inset_axes(
    ax,
    width="45%",
    height="80%",
    loc="upper right",
)

axins.plot(
    model2.star_age[-10000:],
    model2.R[-10000:],
    c="k",
    lw=0.8
)

# zoom limits
x1 = model2.star_age[-10000]
x2 = model2.star_age[-1]

mask = model2.star_age >= x1
y1 = np.min(model2.R[mask])
y2 = np.max(model2.R[mask])

pad = 0.05 * (y2 - y1)

axins.set_xlim(x1, x2)
axins.set_ylim(y1 - pad, y2 + pad)

# remove tick labels if desired
axins.tick_params(labelsize=7)

# draw rectangle and connecting lines
pp, p1, p2 = mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none",
    ec="0.3",
    lw=0.8,
    color="C9"
)

p1.set_linewidth(0.8)
p2.set_linewidth(0.8)

pp.set_edgecolor("C9")
p1.set_color("C9")
p2.set_color("C9")

# --------------------------------------------------

ax.set_xlabel("Star age (yr)")
ax.set_ylabel(r"Radius ($R_\odot$)")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w20-radius-end.pgf",
    format="pgf"
)

plt.show()
plt.close()

# %%
l2 = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00453/M2.0/LOGS/TPAGB")

# %%
for profile in l2.profile_numbers:
    profile = l2.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)


# %%

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l2.profile_dict.values())
n = len(profiles)

n_grid = 300

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = min(np.min(p.logR) for p in profiles)
logR_max = max(np.max(p.logR) for p in profiles)
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = min(np.min(p.mass) for p in profiles)
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.linspace(m_min, m_max, n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid + 1))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.logR[::-1]
    yR = p.logeps_nuc[::-1]

    xM = p.mass[::-1]
    yM = p.logeps_nuc[::-1]

    m_grid = np.linspace(m_min, m_max, n_grid)
    max_S = p.mass[::-1][np.argmax(yM)]
    m_grid = np.append(m_grid, max_S)
    m_grid.sort()

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    entropy_M[i] = np.interp(m_grid, xM, yM, left=np.nan, right=np.nan)

# radius evolution
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
)

# use index instead of age for stability
idx = np.arange(n)


# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
    subplot_titles=[
        "Entropy vs log R",
        "Entropy vs mass",
        "Radius evolution",
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
    ),
    row=1,
    col=1,
)

# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=m_grid,
        y=entropy_M[0],
        mode="lines",
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2, col=1
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=m_grid, y=entropy_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
# --------------------------------------------------

steps = [
    dict(
        method="animate",
        args=[
            [str(i)],
            {
                "mode": "immediate",
                "frame": {"duration": 20, "redraw": False},
                "transition": {"duration": 0},
            },
        ],
        label=str(i),
    )
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 30, "redraw": False},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"mode": "immediate"}],
                ),
            ],
        )
    ],
    sliders=[dict(active=0, steps=steps)],
)

# axis labels
fig.update_xaxes(title_text="log R", row=1, col=1)
fig.update_xaxes(title_text="mass coordinate", row=1, col=2)
fig.update_xaxes(title_text="profile index", row=2, col=1)

fig.update_yaxes(title_text="entropy", row=1, col=1)
fig.update_yaxes(title_text="entropy", row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()


# %%

