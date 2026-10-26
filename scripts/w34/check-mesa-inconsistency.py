import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
column = 362.98032
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


def add_ratio_bar(ax, logR, ratio, y, height, cmap="viridis", vmin=0, vmax=1, alpha=1):
    logR = np.asarray(logR)
    ratio = np.asarray(ratio)

    edges = np.empty(len(logR) + 1)
    edges[1:-1] = (logR[:-1] + logR[1:]) / 2
    edges[0] = logR[0] - (edges[1] - logR[0])
    edges[-1] = logR[-1] + (logR[-1] - edges[-2])

    c = ax.pcolormesh(
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
    return c


from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "WhiteReds",
    ["white", *plt.get_cmap("Greens")(np.linspace(0.1, 1, 255))],
)


mass_excess = np.array(
    [
        7.288970500,
        13.135721580,
        14.931214750,
        2.424915650,
        14.908141000,
        15.770034000,
        22.921490000,
        0,
        3.125011290,
        5.345481000,
        2.863417040,
        0.101438050,
        8.007356000,
        2.855605000,
        -4.737001410,
        -0.808813000,
        -0.781522000,
        1.951701000,
        0.873701000,
        -1.487386000,
        5.317166000,
        1.751440000,
        -7.041931310,
        -0.396963000,
        -13.933567000,
    ]
)

isos = [
    "h1",
    "h2",
    "he3",
    "he4",
    "li7",
    "be7",
    "b8",
    "c12",
    "c13",
    "n13",
    "n14",
    "n15",
    "o14",
    "o15",
    "o16",
    "o17",
    "o18",
    "f17",
    "f18",
    "f19",
    "ne18",
    "ne19",
    "ne20",
    "mg22",
    "mg24",
]


def average_weight(profile):
    A = np.array([int("".join(filter(str.isdigit, iso))) for iso in isos])

    X = np.array([profile.data(iso) for iso in isos]).squeeze()

    W = A + mass_excess / 931.49410242
    Y = X / A[:, None]

    return np.sum(Y * W[:, None], axis=0) / np.sum(Y, axis=0)


# Nuclear charge Z for each isotope, in the same order as `isos`.
z_s = np.array(
    [
        1,
        1,
        2,
        2,
        3,
        4,
        5,
        6,
        6,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        10,
        10,
        10,
        12,
        12,
    ]
)

A = np.array([int("".join(filter(str.isdigit, iso))) for iso in isos])

W = A + mass_excess / 931.49410242


def compute_mu_profile(profile, ints=False):
    """Mean molecular weight for a fully ionized gas, using actual isotope masses."""
    X = np.array([profile.data(iso) for iso in isos]).squeeze()

    if ints:
        mu_inv = np.sum(X * ((1 + z_s) / np.round(W, 0))[:, None], axis=0)
        print(np.round(W, 0))
    else:
        print(W)
        mu_inv = np.sum(X * ((1 + z_s) / W)[:, None], axis=0)
    return 1 / mu_inv


# %%
profiles = []

for i in range(1, 100):
    p = mr.MesaData(
        f"/home/koen/master-internship/mesa-inconsistency/hb_2M/LOGS_POSTMS/profile{i}.data"
    )
    profiles.append(p)
print(p)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)
my_colors = np.array(mpl.cm.Set2.colors)  # array so that entries are editable

plt.plot(p.logR, p.mu, c="k", zorder=100, linewidth=1)
# plt.plot(p.logR, p.log_D_thrm)
plt.plot(p.logR, p.abar / (1 + p.zbar), c="w", alpha=1, linewidth=3)
plt.plot(p.logR, p.abar / (1 + p.zbar), c="C0", alpha=0.6, linewidth=3)
plt.plot(p.logR, compute_mu_profile(p), c="w", alpha=0.6, linewidth=3)
plt.plot(p.logR, compute_mu_profile(p), c="C1", alpha=0.6, linewidth=3)
plt.ylim(0.619125, 0.628)


print(np.argwhere(p.eos_frac_Skye == 1)[0][0])
print(np.argwhere(p.eos_frac_Skye > 0)[0][0])


plt.axvspan(-2, p.logR[975], color=my_colors[5], alpha=0.75, edgecolor=None)
plt.axvspan(
    p.logR[975],
    p.logR[923],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(p.logR[923], 2, color=my_colors[4], alpha=0.75, edgecolor=None)
plt.axhspan(0.62, 0.63, color="white")

c = add_ratio_bar(
    axs, p.logR, p.log_D_thrm, y=0.62, height=1, cmap=cmap, alpha=0.8, vmin=3, vmax=6.5
)

plt.colorbar(
    c,
    ax=axs,
    extend="min",
    label="$\\log\\left(D_\\textrm{TH} / (\\textrm{cm}^{2}\,\\textrm{s}^{-1})\\right)$",
)

plt.text(-1.55, 0.6195, "Skye EOS", fontsize=8, c="k", ha="left", va="center")
plt.text(-1.27, 0.6195, "blend", fontsize=8, c="k", ha="left", va="center")
plt.text(-1.07, 0.6195, "FreeEOS", fontsize=8, c="k", ha="left", va="center")
plt.text(-1.585, 0.628, "He core", fontsize=8, c="k", ha="center", va="top")
plt.text(-0.77, 0.628, "Envelope", fontsize=8, c="k", ha="center", va="top")
# plt.text(0, 0.60445, "FreeEOS", fontsize=8, c="w", ha="left", va="center")
axs.annotate(
    # "$\\displaystyle\\mu^{-1} = \\sum_{i}X_i \\frac{1+Z_i}{A_i}$",
    "$\\mu$ computed using\n integer mass number",
    xy=(-0.77, 0.6212),
    xycoords="data",
    xytext=(0, 15),
    textcoords="offset points",
    ha="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="#0f446a"),
    fontsize=8,
    color="#0f446a",
)

axs.annotate(
    # "$\\displaystyle\\mu^{-1} = \\sum_{i}X_i \\frac{1+Z_i}{W_i}$",
    "$\\mu$ computed using\n isotope masses",
    xy=(-0.77, 0.6252),
    xycoords="data",
    xytext=(0, -27),
    textcoords="offset points",
    ha="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="#db6100"),
    fontsize=8,
    color="#db6100",
)

axs.annotate(
    "$\\mu$ reported by MESA",
    xy=(-0.77, 0.6255),
    xycoords="data",
    xytext=(0, 15),
    textcoords="offset points",
    ha="center",
    arrowprops=dict(arrowstyle="->", linewidth=0.75),
    fontsize=8,
)

plt.xlim(-1.7, -0.6)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(\\textrm{Radius} / R_\\odot)$")
plt.ylabel("Mean Molecular Weight $\\mu$")
plt.savefig("Skye-vs-FreeEOS.pdf", format="pdf")
plt.show()
plt.close()

# %%
p.bulk_names
# %%
plt.plot(p.logR, p.eos_frac_Skye)
plt.ylim(0.62, 0.64)
plt.show()
# %%

h = mr.MesaData(
    "/home/koen/master-internship/mesa-inconsistency/hb_2M/LOGS_POSTMS/history.data"
)
plt.plot(h.data("log_Teff"), h.log_L)
plt.gca().invert_xaxis()
plt.show()
# %%
