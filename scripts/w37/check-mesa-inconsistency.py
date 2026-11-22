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
    try:
        p = mr.MesaData(
            f"/home/koen/master-internship/mesa-inconsistency/hb_2M/LOGS_POSTMS/profile{i}.data"
        )
        profiles.append(p)
    except:
        continue
print(p)

# %%
p = profiles[35]
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
# plt.savefig("Skye-vs-FreeEOS.pdf", format="pdf")
plt.savefig("/home/koen/LaTeX-setup/plots/Skye-vs-FreeEOS.pgf", format="pgf")
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
print(p.bulk_names)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)
my_colors = np.array(mpl.cm.Set2.colors)  # array so that entries are editable

plt.plot(p.logR, p.pp, c="k", zorder=100, linewidth=1)
# plt.plot(p.logR, p.log_D_thrm)
# plt.ylim(0.619125, 0.628)


max = np.argwhere(p.eos_frac_Skye == 1)[0][0]
min = np.argwhere(p.eos_frac_Skye > 0)[0][0]


plt.axvspan(-2, p.logR[max], color=my_colors[5], alpha=0.75, edgecolor=None)
plt.axvspan(
    p.logR[max],
    p.logR[min],
    color=my_colors[0],
    alpha=0.75,
    edgecolor=None,
)
plt.axvspan(p.logR[min], 2, color=my_colors[4], alpha=0.75, edgecolor=None)

# c = add_ratio_bar(
#     axs, p.logR, p.log_D_thrm, y=0.62, height=1, cmap=cmap, alpha=0.8, vmin=3, vmax=6.5
# )
plt.ylim(-3e-5, 1e-5)

plt.colorbar(
    c,
    ax=axs,
    extend="min",
    label="$\\log\\left(D_\\textrm{TH} / (\\textrm{cm}^{2}\,\\textrm{s}^{-1})\\right)$",
)

plt.xlim(-1.7, -0.6)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(\\textrm{Radius} / R_\\odot)$")
plt.ylabel("Mean Molecular Weight $\\mu$")
# plt.savefig("Skye-vs-FreeEOS.pdf", format="pdf")
# plt.savefig("/home/koen/LaTeX-setup/plots/Skye-vs-FreeEOS.pgf", format="pgf")
plt.show()
plt.close()


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

# sort profiles by age
profiles_sorted = sorted(profiles, key=lambda p: p.star_age)

fig, ax = plt.subplots(
    1,
    1,
    figsize=set_size(column, height=0.7),
    constrained_layout=True,
)

my_colors = np.array(mpl.cm.Set2.colors)

# initialise plot
(line,) = ax.plot([], [], c="k", zorder=100, linewidth=1)
(line2,) = ax.plot([], [], c="k", zorder=100, linewidth=1)

# shaded EOS regions
spans = [
    ax.axvspan(-2, -1, color=my_colors[5], alpha=0.75, edgecolor=None),
    ax.axvspan(-1, -0.5, color=my_colors[0], alpha=0.75, edgecolor=None),
    ax.axvspan(-0.5, 2, color=my_colors[4], alpha=0.75, edgecolor=None),
]

ax.set_xlim(-1.7, -0.6)
ax.set_ylim(-3e-5, 1e-5)

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel(r"$\log(\textrm{Radius} / R_\odot)$")
ax.set_ylabel(r"$\nabla_{\rm L, composition}$")

title = ax.set_title("")


def update(i):
    p = profiles_sorted[i]

    line.set_data(p.logR, p.gradL_composition_term)
    line2.set_data(p.logR, p.log_D_conv)

    # determine EOS transition boundaries
    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):
        x_max = p.logR[full_skye[0]]
        x_min = p.logR[any_skye[0]]

        # update shaded regions
        spans[0].set_x(-2)
        spans[0].set_width(x_max + 2)

        spans[1].set_x(x_max)
        spans[1].set_width(x_min - x_max)

        spans[2].set_x(x_min)
        spans[2].set_width(2 - x_min)

    title.set_text(f"Age = {p.star_age:.4e} yr")

    return line, *spans, title


ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=150,
    blit=False,
)

plt.show()
# %%
print(p.bulk_names)
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

# sort profiles by age
profiles_sorted = sorted(profiles, key=lambda p: p.star_age)[2:]

fig, ax = plt.subplots(
    1,
    1,
    figsize=set_size(column, height=0.7),
    constrained_layout=True,
)

my_colors = np.array(mpl.cm.Set2.colors)

# initialise plot
(line,) = ax.plot([], [], c="royalblue", zorder=100, linewidth=1, label="Ledoux B")
(line2,) = ax.plot([], [], c="firebrick", zorder=100, linewidth=1, label="$^{3}$He")

# shaded EOS regions
spans = [
    ax.axvspan(-2, -1, color=my_colors[5], alpha=0.5, edgecolor=None),
    ax.axvspan(-1, -0.5, color=my_colors[0], alpha=0.5, edgecolor=None),
    ax.axvspan(-0.5, 2, color=my_colors[4], alpha=0.5, edgecolor=None),
]

ax.set_ylim(-10e-5, 50e-5)

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("$m$ ($M_\\odot$)")
ax.set_ylabel(r"")

title = ax.set_title("")
plt.legend()


def update(i):
    p = profiles_sorted[i]

    line.set_data(p.mass, p.gradL_composition_term)
    line2.set_data(p.mass, p.he3)

    # determine EOS transition boundaries
    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):
        x_max = p.mass[full_skye[0]]
        x_min = p.mass[any_skye[0]]

        # update shaded regions
        spans[0].set_x(-2)
        spans[0].set_width(x_max + 2)

        spans[1].set_x(x_max)
        spans[1].set_width(x_min - x_max)

        spans[2].set_x(x_min)
        spans[2].set_width(2 - x_min)

    title.set_text(f"Age = {p.star_age:.4e} yr")
    ax.set_xlim(x_max - 5 * (x_min - x_max), x_min + 15 * (x_min - x_max))

    return line, *spans, title


ax.set_xticks(np.arange(0.2, 0.5, 0.001))

ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=1000 / 60,
    blit=False,
)

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

# sort profiles by age
profiles_sorted = sorted(profiles, key=lambda p: p.star_age)[2:]

fig, ax = plt.subplots(
    1,
    1,
    figsize=set_size(column, height=0.7),
    constrained_layout=True,
)

my_colors = np.array(mpl.cm.Set2.colors)

# initialise plot
(line,) = ax.plot([], [], c="royalblue", zorder=100, linewidth=1, label="Ledoux B")
(line2,) = ax.plot([], [], c="firebrick", zorder=100, linewidth=1, label="$^{3}$He")

# shaded EOS regions
spans = [
    ax.axvspan(-2, -1, color=my_colors[5], alpha=0.5, edgecolor=None, zorder=-10),
    ax.axvspan(-1, -0.5, color=my_colors[0], alpha=0.5, edgecolor=None, zorder=-10),
    ax.axvspan(-0.5, 2, color=my_colors[4], alpha=0.5, edgecolor=None, zorder=-10),
]


ax.set_ylim(-10e-5, 50e-5)

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel(r"$m$ ($M_\odot$)")
ax.set_ylabel(r"$\nabla_{\rm L, composition}$")

title = ax.set_title("")


def update(i):
    p = profiles_sorted[i]

    line.set_data(p.mass, p.gradL_composition_term)
    line2.set_data(p.mass, p.he3)

    # determine EOS transition boundaries
    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):
        x_max = p.mass[full_skye[0]]
        x_min = p.mass[any_skye[0]]

        # update shaded regions
        spans[0].set_x(-2)
        spans[0].set_width(x_max + 2)

        spans[1].set_x(x_max)
        spans[1].set_width(x_min - x_max)

        spans[2].set_x(x_min)
        spans[2].set_width(2 - x_min)

    title.set_text(f"Age = {p.star_age:.4e} yr")

    return line, *spans, title


ax.set_xlim(0.3, 0.425)
ax.set_xticks(np.arange(0.25, 0.425, 0.01))

ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=1000 / 60,
    blit=False,
)

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

# sort profiles by age
profiles_sorted = sorted(profiles, key=lambda p: p.star_age)[2:]

fig, ax = plt.subplots(
    1,
    1,
    figsize=set_size(column, height=0.7),
    constrained_layout=True,
)

my_colors = np.array(mpl.cm.Set2.colors)

# initialise plot
(line,) = ax.plot([], [], c="k", zorder=100, linewidth=1)

# shaded EOS regions
spans = [
    ax.axvspan(-2, -1, color=my_colors[5], alpha=0.75, edgecolor=None, zorder=-10),
    ax.axvspan(-1, -0.5, color=my_colors[0], alpha=0.75, edgecolor=None, zorder=-10),
    ax.axvspan(-0.5, 2, color=my_colors[4], alpha=0.75, edgecolor=None, zorder=-10),
]

ax.axhspan(0, 1, color="w", alpha=1, edgecolor=None, zorder=-5)


ax.set_ylim(-10e-5, 10e-5)

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel(r"$\log(\textrm{Radius} / R_\odot)$")
ax.set_ylabel(r"$\nabla_{\rm L, composition}$")

title = ax.set_title("")


def update(i):
    p = profiles_sorted[i]

    line.set_data(p.mass, p.pp)

    # determine EOS transition boundaries
    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):
        x_max = p.mass[full_skye[0]]
        x_min = p.mass[any_skye[0]]

        # update shaded regions
        spans[0].set_x(-2)
        spans[0].set_width(x_max + 2)

        spans[1].set_x(x_max)
        spans[1].set_width(x_min - x_max)

        spans[2].set_x(x_min)
        spans[2].set_width(2 - x_min)

    title.set_text(f"Age = {p.star_age:.4e} yr")

    return line, *spans, title


ax.set_xlim(0.3, 0.425)
ax.set_xticks(np.arange(0.25, 0.425, 0.01))

ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=1000 / 60,
    blit=False,
)

plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

profiles_sorted = sorted(profiles, key=lambda p: p.star_age)[2:]

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(column, height=0.8),
    sharey=True,
    constrained_layout=True,
)

ax_fixed, ax_tracking = axs

my_colors = np.array(mpl.cm.Set2.colors)

# initialise plots
lines = []
for ax in axs:
    (line,) = ax.plot([], [], c="royalblue", zorder=100, linewidth=1, label="Ledoux B")
    (line2,) = ax.plot([], [], c="firebrick", zorder=100, linewidth=1, label=r"$^3$He")
    lines.append((line, line2))

# EOS region colours
eos_colors = [my_colors[5], my_colors[0], my_colors[4]]

# Create split shading:
# upper half: EOS regions
# lower half: hydrogen-burning shell
spans = []
for ax in axs:
    ax_spans = []

    for color in eos_colors:
        upper = ax.axvspan(
            -2,
            -1,
            ymin=0.5,
            ymax=1,
            color=color,
            alpha=0.5,
            edgecolor=None,
            zorder=-10,
        )
        lower = ax.axvspan(
            -2,
            -1,
            ymin=0,
            ymax=0.5,
            color="white",
            edgecolor=None,
            zorder=-10,
        )
        ax_spans.append((upper, lower))

    spans.append(ax_spans)

# axis formatting
for ax in axs:
    ax.set_ylim(-10e-5, 50e-5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(r"$m$ ($M_\odot$)")

ax_fixed.set_ylabel(r"$\nabla_{\rm L, composition}$")
ax_fixed.set_title("Fixed mass range")
ax_tracking.set_title("Mixing region")

ax_fixed.set_xlim(0.3, 0.425)
ax_fixed.set_xticks(np.arange(0.25, 0.425, 0.01))

title = fig.suptitle("")

# Hydrogen-burning shell criterion
pp_fraction = 0.1


def update(i):
    p = profiles_sorted[i]

    # Update lines
    for line, line2 in lines:
        line.set_data(p.mass, p.gradL_composition_term)
        line2.set_data(p.mass, p.he3)

    # Determine EOS transition boundaries
    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):
        x_max = p.mass[full_skye[0]]
        x_min = p.mass[any_skye[0]]

        boundaries = [-2, x_max, x_min, 2]

        # Identify pp-burning shell
        pp = np.asarray(p.pp)

        if np.nanmax(pp) <= 0:
            # logarithmic values
            burning = pp > np.nanmax(pp) - 1
        else:
            # linear values
            burning = pp > 0.01 * np.nanmax(pp)

        # Determine contiguous burning intervals
        indices = np.flatnonzero(burning)

        # Update EOS shading
        for ax_idx, ax in enumerate(axs):
            for j, (upper, lower) in enumerate(spans[ax_idx]):
                upper.set_x(boundaries[j])
                upper.set_width(boundaries[j + 1] - boundaries[j])

                lower.set_x(boundaries[j])
                lower.set_width(boundaries[j + 1] - boundaries[j])

                # White by default
                lower.set_facecolor("white")

            # Overlay hydrogen-burning shell on lower half
            # only where pp burning is active
            for j in range(3):
                left = boundaries[j]
                right = boundaries[j + 1]

                for k in range(len(indices)):
                    if not burning[indices[k]]:
                        continue

                    mass_left = p.mass[indices[k]]
                    mass_right = (
                        p.mass[indices[k] + 1]
                        if indices[k] + 1 < len(p.mass)
                        else mass_left
                    )

                    overlap_left = np.maximum(left, mass_left)
                    overlap_right = np.minimum(right, mass_right)

                    if overlap_right > overlap_left:
                        ax.axvspan(
                            overlap_left,
                            overlap_right,
                            ymin=0,
                            ymax=0.5,
                            color="tomato",
                            alpha=0.6,
                            edgecolor=None,
                            zorder=-9,
                        )

        # Tracking panel follows mixing region
        dx = x_min - x_max
        ax_tracking.set_xlim(
            x_max - 5 * dx,
            x_min + 15 * dx,
        )

    title.set_text(f"Age = {p.star_age:.4e} yr")

    return (
        [line for pair in lines for line in pair]
        + [span for group in spans for pair in group for span in pair]
        + [title]
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=1000 / 60,
    blit=False,
)

plt.show()
# %%


plt.plot(p.mass, p.pp > 0.05 * np.max(p.pp))
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

# sort profiles by age
profiles_sorted = sorted(profiles, key=lambda p: p.star_age)[2:]

# --------------------------------------------------
# Figure
# --------------------------------------------------

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(full, height=0.85),
    sharey=True,
    constrained_layout=True,
)

ax_fixed, ax_tracking = axs

my_colors = np.array(mpl.cm.Set2.colors)
eos_colors = [my_colors[5], my_colors[0], my_colors[4]]

# --------------------------------------------------
# Initialise lines
# --------------------------------------------------

lines = []

for ax in axs:
    (line1,) = ax.plot(
        [],
        [],
        c="royalblue",
        zorder=100,
        linewidth=1,
        label="Ledoux B",
    )

    (line2,) = ax.plot(
        [],
        [],
        c="firebrick",
        zorder=100,
        linewidth=1,
        label=r"$^3$He",
    )

    lines.append((line1, line2))

# --------------------------------------------------
# EOS shading (upper half)
# --------------------------------------------------

eos_spans = []

for ax in axs:
    spans = []

    for color in eos_colors:
        span = ax.axvspan(
            -2,
            -1,
            ymin=0.5,
            ymax=1,
            color=color,
            alpha=0.5,
            edgecolor="none",
            zorder=-10,
        )
        spans.append(span)

    eos_spans.append(spans)

# --------------------------------------------------
# Hydrogen-burning shading (lower half)
# --------------------------------------------------

burning_spans = []

for ax in axs:
    spans = [
        ax.axvspan(
            0,
            0,
            ymin=0,
            ymax=0.5,
            color="tomato",
            alpha=0.6,
            edgecolor="none",
            zorder=-9,
            visible=False,
        )
    ]

    burning_spans.append(spans)

# --------------------------------------------------
# Axis formatting
# --------------------------------------------------

for ax in axs:
    ax.set_ylim(-10e-5, 50e-5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(r"$m$ ($M_\odot$)")

ax_fixed.set_ylabel(r"$\nabla_{\rm L, composition}$")

ax_fixed.set_title("Fixed mass range")
ax_tracking.set_title("Mixing region")

ax_fixed.set_xlim(0.3, 0.425)
ax_fixed.set_xticks(np.arange(0.25, 0.425, 0.01))

ax_tracking.set_xticks(np.arange(0.25, 0.425, 0.01))

# --------------------------------------------------
# Animation title
# --------------------------------------------------

title = fig.suptitle("")

# --------------------------------------------------
# Update function
# --------------------------------------------------


def update(i):

    p = profiles_sorted[i]

    # Update lines
    for line1, line2 in lines:
        line1.set_data(p.mass, p.gradL_composition_term)
        line2.set_data(p.mass, p.he3)

    # --------------------------------------------------
    # Determine EOS transition boundaries
    # --------------------------------------------------

    full_skye = np.flatnonzero(p.eos_frac_Skye == 1)
    any_skye = np.flatnonzero(p.eos_frac_Skye > 0)

    if len(full_skye) and len(any_skye):

        x_max = p.mass[full_skye[0]]
        x_min = p.mass[any_skye[0]]

        boundaries = [-2, x_max, x_min, 2]

        # Update EOS shading
        for spans in eos_spans:

            for j, span in enumerate(spans):
                left = boundaries[j]
                right = boundaries[j + 1]

                span.set_xy(
                    [
                        [left, 0],
                        [left, 1],
                        [right, 1],
                        [right, 0],
                        [left, 0],
                    ]
                )

        # Update tracking panel
        dx = x_min - x_max

        ax_tracking.set_xlim(
            x_max - 5 * dx,
            x_min + 15 * dx,
        )

    # --------------------------------------------------
    # Identify hydrogen-burning shell
    # --------------------------------------------------

    pp = np.asarray(p.pp)

    burning = pp > 0.1 * np.nanmax(pp)

    # Find contiguous burning regions
    edges = np.diff(np.r_[False, burning, False].astype(int))

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    # --------------------------------------------------
    # Update hydrogen-burning shading
    # --------------------------------------------------

    for ax, spans in zip(axs, burning_spans):

        # Hide previous patches
        for span in spans:
            span.set_visible(False)

        # Create additional patches if necessary
        while len(spans) < len(starts):

            spans.append(
                ax.axvspan(
                    0,
                    0,
                    ymin=0,
                    ymax=0.5,
                    color="tomato",
                    alpha=0.6,
                    edgecolor="none",
                    zorder=-9,
                    visible=False,
                )
            )

        # Update patches
        for j, (start, end) in enumerate(zip(starts, ends)):

            x0 = p.mass[start]
            x1 = p.mass[min(end, len(p.mass) - 1)]

            spans[j].set_xy(
                [
                    [x0, 0],
                    [x0, 0.5],
                    [x1, 0.5],
                    [x1, 0],
                    [x0, 0],
                ]
            )

            spans[j].set_visible(True)

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    title.set_text(f"Age = {p.star_age:.4e} yr")

    return (
        [line for pair in lines for line in pair]
        + [span for group in eos_spans for span in group]
        + [span for group in burning_spans for span in group]
        + [title]
    )


# --------------------------------------------------
# Animate
# --------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(profiles_sorted),
    interval=1000 / 60,
    blit=False,
)

plt.show()

# %%
