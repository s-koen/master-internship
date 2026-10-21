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
plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

# %%

profs_no = []
i = 1
while True:
    try:
        prof = mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
        )
        profs_no.append(prof)
        i += 1
    except:
        break

profs_th = []
i = 1
while True:
    try:
        prof = mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline/LOGS/GB/profile{i}.data"
        )
        profs_th.append(prof)
        i += 1
    except:
        break

profs_th_rot = []
i = 1
while True:
    try:
        prof = mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline-rot-test/LOGS/GB/profile{i}.data"
        )
        profs_th_rot.append(prof)
        i += 1
    except:
        break
# %%
import numpy as np
import matplotlib.pyplot as plt


def add_ratio_bar(ax, logR, ratio, y, height, cmap="viridis", vmin=0, vmax=1, alpha=1):
    logR = np.asarray(logR)
    ratio = np.asarray(ratio)

    edges = np.empty(len(logR) + 1)
    edges[1:-1] = (logR[:-1] + logR[1:]) / 2
    edges[0] = logR[0] - (edges[1] - logR[0])
    edges[-1] = logR[-1] + (logR[-1] - edges[-2])

    ax.pcolormesh(
        edges,
        [y, y + height],
        ratio[None, :],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        alpha=alpha,
        rasterized=True,
    )


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for p in profs_th[-1:]:
    plt.plot(p.logR, p.mu, c="C0")


for p in profs_th_rot[-1:]:
    plt.plot(p.logR, p.mu, c="C1")
    # plt.plot(p.logR, p.eos_frac_Skye, c="C2")


plt.ylim(0.604, 0.616)
plt.xlim(-2, p.logR[0])

add_ratio_bar(axs, p.logR, p.eos_frac_Skye, y=0.604, height=0.001, cmap="coolwarm")
add_ratio_bar(
    axs,
    p.logR,
    p.mixing_type,
    y=0.605,
    height=0.1,
    cmap="tab10",
    vmin=0,
    vmax=10,
    alpha=0.3,
)

plt.text(-1.75, 0.60445, "Skye", fontsize=8, c="w", ha="left", va="center")
plt.text(0, 0.60445, "FreeEOS", fontsize=8, c="w", ha="left", va="center")
axs.annotate(
    "$\\Omega_\\textrm{ZAMS} = 0.4\\;\\Omega_\\textrm{crit}$",
    xy=(0.25, 0.6108),
    xycoords="data",
    xytext=(0, -30),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs.annotate(
    "$\\Omega_\\textrm{ZAMS} = 0.0\\;\\Omega_\\textrm{crit}$",
    xy=(0.25, 0.6115),
    xycoords="data",
    xytext=(0, 30),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

for x, label in [
    (-1.025, "TH"),
    (-0.5, "no mixing"),
    (-0.025, "OV"),
    (0.8, "conv."),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="center",
        clip_on=False,
        fontsize=8,
    )

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(R/R_\\odot)$")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-th-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

for p in profs_th[-1:]:
    plt.plot(p.logR, p.mixing_type, c="C1")

for p in profs_th_rot[-1:]:
    plt.plot(p.logR, p.mixing_type, c="C2")

for p in profs_no[-1:]:
    plt.plot(p.logR, p.mixing_type, c="C0")

for p in profs_th[-1:]:
    pass
# plt.ylim(0.602, 0.616)
plt.xlim(-2, 2)
plt.show()
# %%
p.bulk_names
# %%


for p in profs_th[-1:]:
    plt.plot(p.logR, p.eos_frac_Skye, c="C0")
    plt.plot(p.logR, p.mixing_type, c="C0")

for p in profs_th_rot[-1:]:
    plt.plot(p.logR, p.eos_frac_Skye, c="C1")
    plt.plot(p.logR, p.mixing_type, c="C1")

plt.show()
# %%

rot = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline-rot-test/LOGS/GB/history.data"
)
norot = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline/LOGS/GB/history.data"
)

plt.plot(rot.star_age, rot.R, linewidth=1)
plt.plot(norot.star_age, norot.R, linewidth=1)
plt.show()
# %%

rot.bulk_names
# %%

rot = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline-rot-test/LOGS/MS/history.data"
)
norot = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0-thermohaline/LOGS/MS/history.data"
)

plt.plot(rot.log_Teff, rot.log_L, linewidth=1)
plt.plot(norot.log_Teff, norot.log_L, linewidth=1)
plt.show()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

for p in profs_no[-1:]:
    axs[1].plot(p.logR, p.h1, c="C0", linewidth=2, zorder=1000, alpha=0.5)
    axs[1].plot(p.logR, p.he4, c="C1", linewidth=2, zorder=1000, alpha=0.5)
    axs[1].plot(p.logR, p.o16, c="C2", linewidth=2, zorder=1000, alpha=0.5)
    axs[1].plot(p.logR, p.c12, c="C3", linewidth=2, zorder=1000, alpha=0.5)
    axs[1].plot(p.logR, p.n14, c="C4", linewidth=2, zorder=1000, alpha=0.5)


for p in profs_th_rot[-1:]:
    axs[1].plot(p.logR, p.h1, c="C0", linewidth=1, zorder=1000)
    axs[1].plot(p.logR, p.he4, c="C1", linewidth=1, zorder=1000)
    axs[1].plot(p.logR, p.o16, c="C2", linewidth=1, zorder=1000)
    axs[1].plot(p.logR, p.c12, c="C3", linewidth=1, zorder=1000)
    axs[1].plot(p.logR, p.n14, c="C4", linewidth=1, zorder=1000)


# axs[1].ylim(0.604, 0.616)
axs[1].set_xlim(-2, p.logR[0])
axs[1].set_yscale("log")
axs[1].set_ylim(5e-6, 1.1)

add_ratio_bar(axs[1], p.logR, p.eos_frac_Skye, y=0.0, height=0.000015, cmap="coolwarm")
add_ratio_bar(
    axs[1],
    p.logR,
    p.mixing_type,
    y=0.000015,
    height=0.9999,
    cmap="tab10",
    vmin=0,
    vmax=10,
    alpha=0.3,
)

axs[1].text(-1.75, 0.000008, "Skye", fontsize=8, c="w", ha="left", va="center")
axs[1].text(0, 0.000008, "FreeEOS", fontsize=8, c="w", ha="left", va="center")
axs[1].annotate(
    "$\\textrm{O}_{16}$",
    xy=(0.25, 0.003),
    xycoords="data",
    xytext=(-7, 20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs[1].annotate(
    "$\\textrm{C}_{12}$",
    xy=(0.25, 0.0005),
    xycoords="data",
    xytext=(-7, -20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs[1].annotate(
    "$\\textrm{He}_4$",
    xy=(0.25, 0.2),
    xycoords="data",
    xytext=(-7, -20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs[1].annotate(
    "$\\textrm{N}_{14}$",
    xy=(-1.4, 0.004),
    xycoords="data",
    xytext=(20, 10),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs[1].annotate(
    "$\\textrm{H}_1$",
    xy=(-1.4, 0.05),
    xycoords="data",
    xytext=(20, 0),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

for x, label in [
    (-1.025, "TH"),
    (-0.5, "no mixing"),
    (-0.025, "OV"),
    (0.8, "conv."),
]:
    axs[1].text(
        x,
        1.04,
        label,
        transform=axs[1].get_xaxis_transform(),
        ha="center",
        clip_on=False,
        fontsize=8,
    )


for p in profs_th[-1:]:
    axs[0].plot(p.logR, p.mu, c="C0")


for p in profs_th_rot[-1:]:
    axs[0].plot(p.logR, p.mu, c="C1")
    # axs[0].plot(p.logR, p.eos_frac_Skye, c="C2")


axs[0].set_ylim(0.604, 0.616)
axs[0].set_xlim(-2, p.logR[0])

add_ratio_bar(axs[0], p.logR, p.eos_frac_Skye, y=0.604, height=0.001, cmap="coolwarm")
add_ratio_bar(
    axs[0],
    p.logR,
    p.mixing_type,
    y=0.605,
    height=0.1,
    cmap="tab10",
    vmin=0,
    vmax=10,
    alpha=0.3,
)

axs[0].text(-1.75, 0.60445, "Skye", fontsize=8, c="w", ha="left", va="center")
axs[0].text(0, 0.60445, "FreeEOS", fontsize=8, c="w", ha="left", va="center")
axs[0].annotate(
    "$\\Omega_\\textrm{ZAMS} = 0.4\\;\\Omega_\\textrm{crit}$",
    xy=(0.25, 0.6108),
    xycoords="data",
    xytext=(0, -30),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs[0].annotate(
    "$\\Omega_\\textrm{ZAMS} = 0.0\\;\\Omega_\\textrm{crit}$",
    xy=(0.25, 0.6115),
    xycoords="data",
    xytext=(0, 30),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

for x, label in [
    (-1.025, "TH"),
    (-0.5, "no mixing"),
    (-0.025, "OV"),
    (0.8, "conv."),
]:
    axs[0].text(
        x,
        1.04,
        label,
        transform=axs[0].get_xaxis_transform(),
        ha="center",
        clip_on=False,
        fontsize=8,
    )


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
axs[0].set_ylabel("$\\mu$")
plt.xlabel("$\\log(R/R_\\odot)$")
plt.ylabel("$X$")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-th-2.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # always use scientific notation

axs.yaxis.set_major_formatter(formatter)
for p in profs_no[-1:]:
    axs.plot(p.logR, p.n14, c="C4", linewidth=2, zorder=1000, alpha=0.5)
    axs.plot(p.logR, p.c12, c="C3", linewidth=2, zorder=1000, alpha=0.5)


for p in profs_th_rot[-1:]:
    axs.plot(p.logR, p.n14, c="C4", linewidth=1, zorder=1000)
    axs.plot(p.logR, p.c12, c="C3", linewidth=1, zorder=1000)


add_ratio_bar(
    axs,
    p.logR,
    p.mixing_type,
    y=0.0,
    height=1,
    cmap="tab10",
    alpha=0.3,
    vmin=0,
    vmax=10,
)


for x, label in [
    (-1.025, "TH"),
    (-0.5, "no mixing"),
    (-0.025, "OV"),
    (0.15, "conv."),
]:
    axs.text(
        x,
        1.04,
        label,
        transform=axs.get_xaxis_transform(),
        ha="center",
        clip_on=False,
        fontsize=8,
    )
axs.annotate(
    "TH \& rot.",
    xy=(-1.1, 0.00058),
    xycoords="data",
    xytext=(0, -20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs.annotate(
    "No TH \& rot.",
    xy=(-1.15, 0.0006),
    xycoords="data",
    xytext=(-45, 20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs.annotate(
    "$\\textrm{N}_{14}$",
    xy=(-0.4, 0.00078),
    xycoords="data",
    xytext=(-7, -20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)

axs.annotate(
    "$\\textrm{C}_{12}$",
    xy=(-0.4, 0.0006),
    xycoords="data",
    xytext=(-7, -20),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
)


plt.ylim(0.0005, 0.0009)
plt.xlim(-1.5, 0.3)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(R/R_\\odot)$")
plt.ylabel("$X$")
plt.savefig("/home/koen/LaTeX-setup/plots/w33-TH-3.pgf", format="pgf")
plt.show()
plt.close()
# %%
