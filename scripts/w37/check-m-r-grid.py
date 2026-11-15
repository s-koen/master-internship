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

plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary
from matplotlib.colors import TwoSlopeNorm

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *
import matplotlib.pyplot as plt
from cmap import Colormap

# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/grid-M-R-2026-11-02/")
# %%

from scipy.optimize import brentq


def period_days(q, M1, R_L):
    """Compute the orbital period from the mass ratio and donor Roche lobe radius.

    Parameters
    ----------
    q : float or array-like
        Mass ratio M2 / M1.
    M1 : float
        Donor mass in solar masses.
    R_L : float or array-like
        Donor Roche lobe radius in solar radii.

    Returns
    -------
    float or ndarray
        Orbital period in days.
    """
    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    a = R_L / r_L * 695508000  # orbital separation in solar radii
    M_total = M1 * (1 + q) * 1.989e30
    G = 6.67430e-11
    T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

    return T


# %%


from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.s)

print(min(z), max(z))
norm = TwoSlopeNorm(0.25, 0.0, max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=20,
    cmap="coolwarm",
    zorder=-20,
    norm=norm,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=20, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%
cmap = Colormap("seaborn:vlag").to_mpl()

df = AbundanceTables()

rr = [
    200,
    250,
    300,
    350,
    400,
    450,
]

ms = np.arange(1.0, 3.1, 0.1)

fig, axs = plt.subplots(
    7, 3, sharey="row", figsize=set_size(full, height=2.2), constrained_layout=True
)

axs = axs.flatten()


def plot(ax, ri, mi, c, zorder, alpha, linewidth):

    star = get_star(m=mi)
    a_init = inv_roche_lobe(ri, qi)

    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    q = bin.m2 / bin.m1

    rl = bin.a * (
        0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log1p(q ** (-1 / 3)))
    )

    ax.plot(
        bin.age,
        rl,
        c=c,
        zorder=zorder,
        alpha=alpha,
        linewidth=linewidth,
    )


for i, m in enumerate(ms):
    mi = m
    print(f"\ncomputing {mi}")
    ax = axs[i]

    for r in rr[::-1]:
        print(".", end="", flush=True)
        qi = 0.9
        plot(ax, r, mi, c="k", zorder=-1, alpha=1, linewidth=0.5)

    star = get_star(m=mi)

    norm = TwoSlopeNorm(1e-7, 0.99e-7, 1.01e-7)

    abs = Abundances(None, df, mass=mi, m_acc=1.5, sb=None, save_full=True)

    X = None

    for i, name in enumerate(abs.elements_name):
        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            if type(X) == type(None):
                X = abs.envelope[:, i]
            else:
                X += abs.envelope[:, i]

    cplot(
        star.age[::50],
        10 ** star.log_R[::50],
        X[::5],
        linewidth=1,
        cmap=cmap,
        ax=ax,
        norm=norm,
    )

    ax.text(
        0.95,
        0.05,
        f"${mi:.1f}\\; M_\\odot$",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlim(
        star.age[star.ntpagb] - (star.age[star.ntpagb] - star.age[star.ntacheb]) / 10,
        star.age[-1],
    )
    ax.set_ylim(0, np.max([ax.get_ylim()[1], np.max(1.2 * 10**star.log_R)]))

axs[-2].set_xlabel("Star age (yr)")
# fig.supylabel(r"$R_\textrm{RL}$ $(R_\odot)$", fontsize=10)

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    if i % 3 == 0:
        ax.set_ylabel(r"$R_\textrm{RL}$ $(R_\odot)$")

plt.savefig("/home/koen/LaTeX-setup/plots/w37-roche-radii-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

plt.plot(star.age[star.ntpagb :], star.m_DUP_time[0][star.ntpagb :])
plt.show()
# %%

df = AbundanceTables()
plt.plot(abs.time, X)
plt.show()
# %%


from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(np.nanmax(r.R / r.rl_1))

print(min(z), max(z))
norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=10,
    cmap="coolwarm",
    zorder=-20,
    norm=norm,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=10, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

levels = 15

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(np.nanmax(r.R / r.rl_1))

print(min(z), max(z))
norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="coolwarm",
    zorder=-20,
    norm=norm,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

df = AbundanceTables()


def get_ls(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 3


def get_hs(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += ab.iron_abundance[i]

    return fe / 4


def get_s(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


def get_ba(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba"]:
            fe += ab.iron_abundance[i]

    return fe


def period_to_RL(P, q, m1):

    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a * r_L / R_sun


def period_to_a(P, q, m1):

    q_donor = 1 / q

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a / R_sun


def get_rlof_properties(star, bin):
    """
    Get stellar properties at the moment of RLOF.

    Returns
    -------
    m_env_rlof : float
        Envelope mass at RLOF, or nan if no RLOF occurred.
    m_star_rlof : float
        Stellar mass at RLOF, or nan if no RLOF occurred.
    rlof_tpagb : bool
        Whether RLOF occurred during the TP-AGB.
    """

    t_rlof = bin.age[-1]

    # Check whether the binary evolved to the end without RLOF
    if np.isclose(t_rlof, star.age[-1]):
        return np.nan, np.nan, False

    # Interpolate stellar properties onto the binary time
    m_env_rlof = np.interp(t_rlof, star.age, star.m_env)
    m_star_rlof = np.interp(t_rlof, star.age, star.mass)

    # Check whether RLOF happened during the TP-AGB
    t_tpagb = star.age[star.ntpagb]
    rlof_tpagb = t_rlof >= t_tpagb

    return m_env_rlof, m_star_rlof, rlof_tpagb


class Result:
    def __init__(self):
        pass


rr_large = [
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
]

ms = np.arange(1.0, 3.01, 0.1)

qi = 0.9

eps = 0.1


class SimpleGrid:

    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)


class SimpleModel:

    def __init__(
        self,
        params,
        bin,
    ):
        self.params = params
        self.bin = bin
        self.period_days = self.get_final_period()
        self.rl_1 = self.get_rl()

    def __getattr__(self, name):
        try:
            return getattr(self.bin, name)
        except:
            return getattr(get_star(m=self.params["m"]), name)

    def get_final_period(self):
        star = get_star(m=self.params["m"])

        bin = self.bin

        q = bin.m2 / bin.m1
        a = bin.a * 695508000

        M_total = (bin.m1 + bin.m2) * 1.989e30
        G = 6.67430e-11
        T = np.sqrt(a**3 * 4 * np.pi**2 / (G * (M_total))) / (3600 * 24)

        return T

    def get_rl(self):
        rl = self.bin.a * (
            0.49
            * self.params["q"] ** (-2 / 3)
            / (
                0.6 * self.params["q"] ** (-2 / 3)
                + np.log1p(self.params["q"] ** (-1 / 3))
            )
        )

        rl = np.interp(
            get_star(m=self.params["m"]).age,
            self.bin.age,
            rl,
        )
        return rl


sgrid = SimpleGrid()

for mi in ms:
    max_R = -1
    for m in grid.filter(m=mi):
        if m.params["R"] > max_R:
            max_R = m.params["R"]
    for r in rr_large:
        if r > max_R:
            print(f"{mi:.1f}", r)

            star = get_star(m=mi)
            a_init = inv_roche_lobe(r, qi)

            star, Options, q_init, a_init, e_init, Bins = call_evolution(
                star, qi, a_init, simple_only=True
            )
            bin = Bins[0]

            m_env_rlof, m_star_rlof, rlof_tpagb = get_rlof_properties(star, bin)

            ab = Abundances(
                None,
                df,
                mass=mi,
                sb=bin,
                sampling=10,
                mass_transfer_efficiency=eps,
            )
            if ab.m_env[-1] > 0.04:
                m2f = bin.m2[-1] + ab.m_env[-1] * eps
            else:
                m2f = bin.m2[-1]

            params = {"m": mi, "R": r, "q": qi}
            model = SimpleModel(params, bin)

            model.r_init = r
            model.a_init = a_init
            model.m1i = mi
            model.m1f = star.mass[-1]
            model.m_env_rlof = m_env_rlof
            model.m_star_rlof = m_star_rlof
            model.rlof_tpagb = rlof_tpagb
            model.qi = qi
            model.eps = eps
            model.m2f = m2f
            model.s = get_s(ab)
            model.hs = get_hs(ab)
            model.ls = get_ls(ab)
            model.ba = get_ba(ab)

            sgrid.add_model(model)

# %%

levels = 15

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(np.nanmax(r.R / r.rl_1))


for r in sgrid.models:

    print(r.age[-1], get_star(m=r.params["m"]).age[-1])
    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(np.nanmax(10**r.log_R / r.rl_1))

print(min(z), max(z))
norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="coolwarm",
    zorder=-20,
    norm=norm,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

levels = 15

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

x = []
y = []
z = []
for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.s)


for r in sgrid.models:

    print(r.age[-1], get_star(m=r.params["m"]).age[-1])
    if r.bin.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    x.append(r.params["R"])
    y.append(r.params["m"])
    z.append(r.s)

print(min(z), max(z))
norm = TwoSlopeNorm(0.25, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="coolwarm",
    zorder=-20,
    norm=norm,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

levels = 25

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.7), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    x_mesa.append(r.params["R"])
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(np.log10(r.period_days[-1]))
    z_mesa.append(np.log10(r.period_days[-1]))


for r in sgrid.models:
    if r.params["R"] > 10000:
        continue
    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
        continue
    x.append(r.params["R"])
    x_simple.append(r.params["R"])
    y.append(r.params["m"])
    y_simple.append(r.params["m"])
    z.append(np.log10(r.period_days[-1]))
    z_simple.append(np.log10(r.period_days[-1]))


xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=20,
)

plt.scatter(
    x_simple,
    y_simple,
    c=z_simple,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=15,
    marker="s",
)
c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\log(P/\\textrm{days})$")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

levels = 13

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(r.params["R"])
    x_mesa.append(r.params["R"])
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(r.period_days[-1])
    z_mesa.append(r.period_days[-1])


# for r in sgrid.models:
#
#     if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
#         print(r)
#         continue
#     if np.isclose(r.params["m"],3) and r.params["R"] < 20000:
#         continue
#     x.append(r.params["R"])
#     x_simple.append(r.params["R"])
#     y.append(r.params["m"])
#     y_simple.append(r.params["m"])
#     z.append(np.log10(r.period_days[-1]))
#     z_simple.append(np.log10(r.period_days[-1]))
#

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=20,
)

# plt.scatter(
#     x_simple,
#     y_simple,
#     c=z_simple,
#     cmap="viridis",
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     vmin=np.nanmin(Z),
#     vmax=np.nanmax(Z),
#     s=15,
#     marker="s"
# )

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\log(P/\\textrm{days})$")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(1.25, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-3.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

levels = 13

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(r.period_days[-1])
    z_mesa.append(r.period_days[-1])


# for r in sgrid.models:
#
#     if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
#         print(r)
#         continue
#     if np.isclose(r.params["m"],3) and r.params["R"] < 20000:
#         continue
#     x.append(r.params["R"])
#     x_simple.append(r.params["R"])
#     y.append(r.params["m"])
#     y_simple.append(r.params["m"])
#     z.append(np.log10(r.period_days[-1]))
#     z_simple.append(np.log10(r.period_days[-1]))
#

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=20,
)

# plt.scatter(
#     x_simple,
#     y_simple,
#     c=z_simple,
#     cmap="viridis",
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     vmin=np.nanmin(Z),
#     vmax=np.nanmax(Z),
#     s=15,
#     marker="s"
# )

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\log(P/\\textrm{days})$")

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(1.25, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

levels = 20

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(np.log10(r.period_days[-1]))
    z_mesa.append(np.log10(r.period_days[-1]))


for r in sgrid.models:

    if r.params["R"] > 10000:
        continue
    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
        continue
    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_simple.append(r.params["m"])
    z.append(np.log10(r.period_days[-1]))
    z_simple.append(np.log10(r.period_days[-1]))


xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=20,
)

plt.scatter(
    x_simple,
    y_simple,
    c=z_simple,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    vmin=np.nanmin(Z),
    vmax=np.nanmax(Z),
    s=15,
    marker="s",
)

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\log(P/\\textrm{days})$")

plt.xlim(np.min(x) - 20, 2e5)
# plt.ylim(1.25, np.max(y) + 0.05)
plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-5.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 17

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))
    z_mesa.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))


for r in sgrid.models:

    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
        continue
    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_simple.append(r.params["m"])
    z.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))
    z_simple.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))

norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap=cmap,
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
    s=20,
)

plt.scatter(
    x_simple,
    y_simple,
    c=z_simple,
    cmap=cmap,
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
    s=15,
    marker="s",
)

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap=cmap,
    norm=norm,
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$P_\\textrm{f}/P_\\textrm{i}$")

plt.xlim(np.min(x) - 20, 2e5)
# plt.ylim(1.25, np.max(y) + 0.05)
plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-6.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


def q_f(q_i, M_i, M_f, alpha, beta, delta):
    eps = 1 - alpha - beta - delta
    deltaM_d = M_i - M_f
    deltaM_a = deltaM_d * eps

    M_a_i = q_i * M_i
    M_a_f = M_a_i + deltaM_a

    return M_a_f / M_f


class CombineEvolve:
    def __init__(self, alpha, beta, delta, gamma, qs, A=1, M_1_i=2):
        self.alpha = alpha  # fraction lost from donor as wind
        self.beta = beta  # fraction lost from accretor as wind
        self.delta = delta  # fraction transferred to CCT
        self.gamma = gamma  # measure for CCT distance
        self.epsilon = 1 - self.alpha - self.beta - self.delta
        self.qs = 1 / qs
        self.q0 = 1 / qs[0]

        self.A = A  # enhancement factor
        self.M_1_i = M_1_i  # initial mass of donor

        self.curA = self._comp_curA()
        self.curB = self._comp_curB()
        self.curC = self._comp_curC()

    def _comp_curA(self):
        # equation B8 soberman1997
        return self.A * self.alpha + self.gamma * self.delta

    def _comp_curB(self):
        # equation B9 soberman1997
        return (self.A * self.alpha + self.beta) / (1 - self.epsilon)

    def _comp_curC(self):
        # equation B10 soberman1997
        term_1 = (self.gamma * self.delta * (1 - self.epsilon)) / self.epsilon
        term_2 = (self.A * self.alpha * self.epsilon) / (1 - self.epsilon)
        term_3 = (self.beta) / (self.epsilon * (1 - self.epsilon))
        return term_1 + term_2 + term_3

    @property
    def L_over_L0(self):
        # table 5 soberman1997
        term_1 = (self.qs / self.q0) ** (self.curA)
        term_2 = ((1 + self.qs) / (1 + self.q0)) ** (-1 * self.curB)
        term_3 = ((1 + self.epsilon * self.qs) / (1 + self.epsilon * self.q0)) ** (
            self.curC
        )
        return term_1 * term_2 * term_3

    @property
    def a_over_a0(self):
        # table 5 soberman1997
        term_1 = (self.qs / self.q0) ** (2 * self.curA - 2)
        term_2 = ((1 + self.qs) / (1 + self.q0)) ** (1 - 2 * self.curB)
        term_3 = ((1 + self.epsilon * self.qs) / (1 + self.epsilon * self.q0)) ** (
            2 * self.curC + 3
        )
        return term_1 * term_2 * term_3

    @property
    def M_tot_over_M_tot0(self):
        # table 5 soberman1997
        term_1 = (1 + self.qs) / (1 + self.q0)
        term_2 = ((1 + self.epsilon * self.qs) / (1 + self.epsilon * self.q0)) ** (-1)
        return term_1 * term_2

    @property
    def P_over_P0(self):
        # table 5 soberman1997
        term_1 = (self.qs / self.q0) ** (3 * self.curA - 3)
        term_2 = ((1 + self.qs) / (1 + self.q0)) ** (1 - 3 * self.curB)
        term_3 = ((1 + self.epsilon * self.qs) / (1 + self.epsilon * self.q0)) ** (
            3 * self.curC + 5
        )
        return term_1 * term_2 * term_3

    @property
    def M_tot(self):
        return self.M_1_i * (1 + 1 / self.q0) * self.M_tot_over_M_tot0

    @property
    def M_d(self):
        return self.M_tot / (1 + 1 / self.qs)

    @property
    def M_a(self):
        return 1 / self.qs * self.M_d


class WindEvolve:
    def __init__(self, alpha, beta, qs, A=1, M_1_i=2):
        self.alpha = alpha  # fraction lost from donor as wind
        self.beta = beta  # fraction lost from accretor as wind
        self.epsilon = 1 - self.alpha - self.beta
        self.qs = qs
        self.q0 = qs[0]

        self.A = A  # enhancement factor
        self.M_1_i = M_1_i  # initial mass of donor

        self.curA = self._comp_curA()
        self.curB = self._comp_curB()
        self.curC = self._comp_curC()

    def _comp_curA(self):
        # equation B8 soberman1997
        return self.A * self.alpha

    def _comp_curB(self):
        # equation B9 soberman1997
        return (self.A * self.alpha + self.beta) / (1 - self.epsilon)

    def _comp_curC(self):
        # equation B10 soberman1997
        term_1 = (self.A * self.alpha * self.epsilon) / (1 - self.epsilon)
        term_2 = (self.beta) / (self.epsilon * (1 - self.epsilon))
        return term_1 + term_2

    def a_over_a0(self, q):
        # table 5 soberman1997
        term_1 = (q / self.q0) ** (2 * self.curA - 2)
        term_2 = ((1 + q) / (1 + self.q0)) ** (1 - 2 * self.curB)
        term_3 = ((1 + self.epsilon * q) / (1 + self.epsilon * self.q0)) ** (
            2 * self.curC + 3
        )
        return term_1 * term_2 * term_3

    def M_tot_over_M_tot0(self, q):
        # table 5 soberman1997
        term_1 = (1 + q) / (1 + self.q0)
        term_2 = (1 + self.epsilon * self.q0) / (1 + self.epsilon * q)
        return term_1 * term_2

    def P_over_P_0(self, q):
        # table 5 soberman1997
        term_1 = (q / self.q0) ** (3 * self.curA - 3)
        term_2 = ((1 + q) / (1 + self.q0)) ** (1 - 3 * self.curB)
        term_3 = ((1 + self.epsilon * q) / (1 + self.epsilon * self.q0)) ** (
            3 * self.curC + 5
        )
        return term_1 * term_2 * term_3

    def M_tot(self, q):
        return self.M_1_i * (1 + self.q0) * self.M_tot_over_M_tot0(q)

    def M_d(self, q):
        return self.M_tot(q) / (1 + q)

    def M_a(self, q):
        return q * self.M_d(q)


def a_final(a_initial, M_initial, M_final, q):
    M_tot = M_initial + q * M_initial
    return (
        a_initial
        * ((M_initial * (M_tot - M_initial)) / (M_final * (M_tot - M_final))) ** 2
    )


def rl(a, q):
    return a * 0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log(1 + q ** (-1 / 3)))


def get_separation(R, q):
    return (
        (0.6 * (q) ** (-2 / 3) + np.log(1 + (q) ** (-1 / 3)))
        / (0.49 * (q) ** (-2 / 3))
        * R
    )


def evolve_q(q_initial, M1_initial, M1):
    return (q_initial * M1_initial + (M1_initial - M1)) / M1


def get_period(a, M, q):
    return (
        np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * (1 + q) * M * (Msun))) / 3600 / 24
    )


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

plt.xlabel("Star mass $M_\\odot$")
plt.ylabel("$P / P_i$")

qs = np.linspace(0.5, 1.5, 100)
wind = CombineEvolve(
    alpha=0.15,
    beta=0.0,
    delta=0,
    gamma=0,
    qs=qs,
)
plt.plot(wind.M_d, wind.P_over_P0, c="C0", label="Isotropic wind, $q_i=0.5$")

qs = np.linspace(0.5, 1.5, 100)
wind = CombineEvolve(alpha=0.15, beta=0.0, delta=0, gamma=0, qs=qs, M_1_i=3)

plt.plot(wind.M_d, wind.P_over_P0, c="C0", label="Isotropic wind, $q_i=0.5$")
axs.tick_params(axis="y", which="minor", length=0)
plt.savefig("/home/koen/LaTeX-setup/plots/w37-analytic-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 17

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))
    z_mesa.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))


for r in sgrid.models:

    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
        continue
    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_simple.append(r.params["m"])
    z.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))
    z_simple.append(r.period_days[-1] / period_days(0.9, r.params["m"], r.params["R"]))

norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

# plt.scatter(
#     x_mesa,
#     y_mesa,
#     c=z_mesa,
#     cmap=cmap,
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     norm=norm,
#     s=20,
# )
#
# plt.scatter(
#     x_simple,
#     y_simple,
#     c=z_simple,
#     cmap=cmap,
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     norm=norm,
#     s=15,
#     marker="s",
# )

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap=cmap,
    norm=norm,
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="major")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$P_\\textrm{f}/P_\\textrm{i}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-7.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

plt.xlabel("Star mass $M_\\odot$")
plt.ylabel("$P / P_i$")

mi = 2
qs = np.linspace(0.9, q_f(0.9, mi, 0.7, 0, 0.8, 0.1), 100)
wind = CombineEvolve(alpha=0, beta=0.8, delta=0.1, gamma=1.44, qs=qs, M_1_i=mi)
plt.plot(wind.M_d, wind.P_over_P0, c="C0", label="Isotropic wind, $q_i=0.5$")

mi = 3
qs = np.linspace(0.9, q_f(0.9, mi, 0.7, 0, 0.8, 0.1), 100)
wind = CombineEvolve(
    alpha=0.2, beta=0.8 * 0.8, delta=0.1 * 0.8, gamma=1.44, qs=qs, M_1_i=mi
)

plt.plot(wind.M_d, wind.P_over_P0, c="C0", label="Isotropic wind, $q_i=0.5$")
axs.tick_params(axis="y", which="minor", length=0)
plt.savefig("/home/koen/LaTeX-setup/plots/w37-analytic-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

for m in grid.models:
    if m.params["R"] > 1400:
        break

plt.plot(
    np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age))
    / np.cumsum(10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age))
)
# plt.plot(np.cumsum((10**m.lg_wind_mdot_1 /(1- m.beta_accretion))[1:] *np.diff(m.age)))
plt.show()
# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 15

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    val = (
        np.sum((10**r.lg_wind_mdot_1 / (1 - r.beta_accretion))[1:] * np.diff(r.age))
    ) / (np.sum(10 ** r.lg_mstar_dot_1[1:] * np.diff(r.age)))
    z.append(val)
    z_mesa.append(val)


# for r in sgrid.models:
#
#     if r.params["R"] > 1000:
#         continue
#
#     if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
#         print(r)
#         continue
#     if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
#         continue
#     x.append(period_days(0.9, r.params["m"], r.params["R"]))
#     x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
#     y.append(r.params["m"])
#     y_simple.append(r.params["m"])
#     z.append(1)
#     z_simple.append(1)

print(max(z))

# norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
    vmin=0,
    vmax=1,
)

# plt.scatter(
#     x_simple,
#     y_simple,
#     c=z_simple,
#     cmap=cmap,
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     norm=norm,
#     s=15,
#     marker="s",
# )

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    # norm=norm,
    zorder=-20,
    vmin=0,
    vmax=1,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="minor")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(1.25, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-8.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 15

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    val = (
        np.sum((10**r.lg_wind_mdot_1 / (1 - r.beta_accretion))[1:] * np.diff(r.age))
    ) / (np.sum(10 ** r.lg_mstar_dot_1[1:] * np.diff(r.age)))

    mi = r.params["m"]
    mf = r.star_1_mass[-1]
    qs = np.linspace(0.9, q_f(0.9, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=val,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    val = wind.P_over_P0[-1]

    z.append(val)
    z_mesa.append(val)


# for r in sgrid.models:
#
#     if r.params["R"] > 1000:
#         continue
#
#     if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
#         print(r)
#         continue
#     if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
#         continue
#     x.append(period_days(0.9, r.params["m"], r.params["R"]))
#     x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
#     y.append(r.params["m"])
#     y_simple.append(r.params["m"])
#     z.append(1)
#     z_simple.append(1)

print(max(z))

# norm = TwoSlopeNorm(1, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
)

# plt.scatter(
#     x_simple,
#     y_simple,
#     c=z_simple,
#     cmap=cmap,
#     zorder=1000,
#     edgecolor="w",
#     linewidth=0.75 / 2,
#     norm=norm,
#     s=15,
#     marker="s",
# )

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap="viridis",
    # norm=norm,
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="minor")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(1.25, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-9.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

cmap = "coolwarm"

levels = 17

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

x = []
x_mesa = []
x_simple = []
y = []
y_mesa = []
y_simple = []
z = []
z_mesa = []
z_simple = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    z.append(r.s)
    z_mesa.append(r.s)


for r in sgrid.models:

    if r.age[-1] < get_star(m=r.params["m"]).age[-1]:
        print(r)
        continue
    if np.isclose(r.params["m"], 3) and r.params["R"] < 20000:
        continue
    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_simple.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_simple.append(r.params["m"])
    z.append(r.s)
    z_simple.append(r.s)

norm = TwoSlopeNorm(0.25, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="cubic",
)

plt.scatter(
    x_mesa,
    y_mesa,
    c=z_mesa,
    cmap=cmap,
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
    s=20,
)

plt.scatter(
    x_simple,
    y_simple,
    c=z_simple,
    cmap=cmap,
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    norm=norm,
    s=15,
    marker="s",
)

c = axs.contourf(
    10**X,
    Y,
    Z,
    levels=levels,
    cmap=cmap,
    norm=norm,
    zorder=-20,
)
axs.set_rasterization_zorder(-10)
axs.contour(10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11)

# for x in [1.4,1.8,2.2,2.6,3.0]:
#     plt.axhline(x, c="w", alpha=0.5, linewidth=0.75/2, zorder=100)

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x", which="both")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y", which="major")


plt.yticks(
    [1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="[s/Fe]")

plt.xlim(np.min(x) - 20, 10e5)
plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w37-mass-10.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
