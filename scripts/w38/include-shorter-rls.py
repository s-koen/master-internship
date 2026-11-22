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
        if name in ("bin", "params"):
            raise AttributeError(name)

        try:
            return getattr(self.bin, name)
        except AttributeError:
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


# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/grid-M-R-2026-11-02/")
gridshort = MesaGrid(
    "/home/koen/master-internship/mesa-models/grid-M-R-low-2026-11-15/"
)
grid.merge(gridshort)
# %%
grid.axes
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

levels = 7

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

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    z.append(np.max(r.lg_mstar_dot_1))

print(min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="nearest",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
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

plt.xlim(np.min(x) - 20, np.max(x) + 20)
plt.ylim(0.9, np.max(y) + 0.05)
plt.xscale("log")

axs.xaxis.set_label_position("top")
axs.xaxis.tick_top()
axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial Roche lobe radius ($R_\\odot$)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-test.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

print(r.bulk_names)
# %%

for m in grid.filter(R=450):
    if m.envelope_mass[-1] > 0.01:
        plt.plot(m.envelope_mass, m.log_LHe)
plt.show()
# %%

for m in grid.filter(R=450):
    if m.envelope_mass[-1] > 0.01:
        plt.plot(m.age - m.age[0], m.rl_1, c="r")
        plt.plot(m.age - m.age[0], m.R, c="r", alpha=0.2, linewidth=3)
    else:
        plt.plot(m.age - m.age[0], m.rl_1, c="g")
        plt.plot(m.age - m.age[0], m.R, c="g", alpha=0.2, linewidth=3)
plt.show()
# %%

for m in grid.filter(R=450):
    if m.envelope_mass[-1] > 0.01:
        plt.plot(m.envelope_mass, m.rl_1, c="r")
        plt.plot(m.envelope_mass, m.R, c="r", alpha=0.2, linewidth=3)
    else:
        plt.plot(m.envelope_mass, m.rl_1, c="g")
        plt.plot(m.envelope_mass, m.R, c="g", alpha=0.2, linewidth=3)
plt.show()
# %%

for m in grid.filter(R=450):
    if m.envelope_mass[-1] > 0.01:
        plt.plot(m.envelope_mass, m.log_dt, c="r")
    else:
        plt.plot(m.envelope_mass, m.log_dt, c="g")
plt.show()

# %%

levels = 7

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

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
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
    method="linear",
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
c = axs.contourf(10**X, Y, Z, levels=levels, cmap="coolwarm", zorder=-20, norm=norm)
axs.set_rasterization_zorder(-10)
axs.contour(
    10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11, norm=norm
)

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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-test.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

ts = []
for m_i in np.arange(1, 3.1, 0.1):
    t = []
    for m in grid.filter(m=m_i):
        t.append(np.log10((m.elapsed_time[-1] - m.elapsed_time[0]) / 3600 / 8))
    ts.append(t)

plt.hist(ts, bins=10, stacked=True)
plt.show()
# %%

for m in grid.models:
    plt.scatter(
        np.log10((m.elapsed_time[-1] - m.elapsed_time[0]) / 3600 / 8),
        np.max(m.envelope_fraction_left),
    )
plt.show()
# %%

levels = 7

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

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    z.append(np.nanmax(r.R / r.rl_1))

print(min(z), max(z))
norm = TwoSlopeNorm(1.0, min(z), max(z))

xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="linear",
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
c = axs.contourf(10**X, Y, Z, levels=levels, cmap="coolwarm", zorder=-20, norm=norm)
axs.set_rasterization_zorder(-10)
axs.contour(
    10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11, norm=norm
)

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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-test.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

levels = 7

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

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    z.append(np.nanmax(r.lg_mstar_dot_1))


xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 1000)
yi = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="linear",
)

plt.scatter(
    x,
    y,
    c=z,
    cmap="viridis",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
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
axs.contour(
    10**X, Y, Z, levels=levels, colors="w", linewidths=0.75 / 2, zorder=-11, norm=norm
)

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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-test.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

for m in grid.filter(m=1.8):
    plt.plot(m.star_2_mass / m.star_1_mass, m.rl_1 / m.rl_1[0])
plt.show()

# %%

for m in grid.filter(m=1.8):
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age))
        / np.cumsum(10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age)),
    )
plt.show()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=1.8):
    print(m.params["R"])

norm = plt.Normalize(200, 1000)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")

for m in grid.filter(m=1.8):
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age))
        / np.cumsum(10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age)),
        c=cmap(norm(m.params["R"])),
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$(\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot})|_{q=0}^{q=q}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-cum.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=1.8):
    print(m.params["R"])

norm = plt.Normalize(200, 1000)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")

for m in grid.filter(m=1.8):
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        (10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:]
        * np.diff(m.age)
        / (10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age)),
        c=cmap(norm(m.params["R"])),
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$\\dot{M}_\\textrm{wind} / \\dot{M}_\\textrm{tot}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs.pgf", format="pgf")
plt.show()
plt.close()

# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

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
    method="linear",
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
    [1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

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

    arg = np.argmax(r.R / r.rl_1)
    arg2 = np.argwhere((r.R[:arg] / r.rl_1[:arg]) < 0.9)[-1][-1]
    if arg2 == 0:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])
    val = (
        np.sum(
            (10 ** r.lg_wind_mdot_1[arg2:] / (1 - r.beta_accretion[arg2:]))[1:]
            * np.diff(r.age[arg2:])
        )
    ) / (np.sum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])))
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
    method="nearest",
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
    [1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
mass = 2.4

norm = plt.Normalize(200, 1000)
cmap = Colormap("colorcet:CET_R1").to_mpl()
# color = cmap(norm(x))


for m in grid.filter(m=mass):
    plt.plot(m.age, m.rl_1, c=cmap(norm(m.params["R"])), zorder=1e3 - m.params["R"])
    plt.plot(m.age, m.R, c=cmap(norm(m.params["R"])), zorder=1e3 - m.params["R"])

    arg = np.argmax(m.R / m.rl_1)
    arg2 = np.argwhere((m.R[:arg] / m.rl_1[:arg]) < 0.9)[-1][-1]
    if arg2 == 0:
        continue
    plt.scatter(
        m.age[arg2],
        m.rl_1[arg2],
        marker="o",
        c=cmap(norm(m.params["R"])),
        zorder=1e3 - m.params["R"],
    )

plt.xlim(plt.gca().get_xlim())

print(m.bulk_names)
star = get_star(m=mass)
plt.plot(star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", zorder=-10)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w37-interpulse-pulse.pgf", format="pgf")
plt.show()
plt.close()
