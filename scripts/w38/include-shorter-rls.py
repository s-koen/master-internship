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

for m in grid.filter(m=2.0):
    print(m.params["R"])

norm = plt.Normalize(200, 1000)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")

for m in grid.filter(m=2.0):
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age))
        / np.cumsum(10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age)),
        c=cmap(norm(m.params["R"])),
    )

    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    plt.scatter(
        (m.star_2_mass[1:] / m.star_1_mass[1:])[arg2],
        (
            np.cumsum(
                (10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age)
            )
            / np.cumsum(10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age))
        )[arg2],
        c=cmap(norm(m.params["R"])),
        zorder=100,
        s=15,
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

for m in grid.filter(m=2.0):
    print(m.params["R"])

norm = plt.Normalize(200, 1000)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")

for m in grid.filter(m=2.0):
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        (10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:]
        * np.diff(m.age)
        / (10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age)),
        c=cmap(norm(m.params["R"])),
    )

    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    plt.scatter(
        (m.star_2_mass[1:] / m.star_1_mass[1:])[arg2],
        (
            ((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age))
            / (10 ** m.lg_mstar_dot_1[1:] * np.diff(m.age))
        )[arg2],
        c=cmap(norm(m.params["R"])),
        zorder=100,
        s=15,
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

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-3.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
mass = 1.8

norm = plt.Normalize(200, 1000)
cmap = Colormap("colorcet:CET_R1").to_mpl()
# color = cmap(norm(x))


for m in grid.filter(m=mass):
    # if m.params["R"] > 770:
    #     continue
    # if m.params["R"] < 750:
    #     continue
    axs[0].plot(m.age, m.rl_1, c=cmap(norm(m.params["R"])), zorder=1e3 - m.params["R"])
    axs[0].plot(
        m.age,
        m.R,
        c=cmap(norm(m.params["R"])),
        zorder=1e3 - m.params["R"],
        alpha=0.5,
        linewidth=3,
    )
    axs[1].plot(
        m.age,
        m.R / m.rl_1,
        c=cmap(norm(m.params["R"])),
        zorder=1e3 - m.params["R"],
        alpha=0.5,
        linewidth=3,
    )

    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue
    axs[1].scatter(
        m.age[arg2],
        m.R[arg2] / m.rl_1[arg2],
        marker="o",
        c=cmap(norm(m.params["R"])),
        zorder=1e3 - m.params["R"],
    )
    axs[0].scatter(
        m.age[arg2],
        m.rl_1[arg2],
        marker="o",
        c=cmap(norm(m.params["R"])),
        zorder=1e3 - m.params["R"],
    )

plt.xlim(plt.gca().get_xlim())

print(m.bulk_names)
star = get_star(m=mass)
axs[0].plot(
    star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :], c="C9", zorder=-10
)


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w37-interpulse-pulse.pgf", format="pgf")
plt.show()
plt.close()
# %%

for i, m in enumerate(grid.filter(m=2)):

    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue
    plt.plot(m.star_2_mass / m.star_1_mass, m.period_days, c=f"C{i%9}")

    val = (
        np.sum(
            (10 ** m.lg_wind_mdot_1[arg2:] / (1 - m.beta_accretion[arg2:]))[1:]
            * np.diff(m.age[arg2:])
        )
    ) / (np.sum(10 ** m.lg_mstar_dot_1[arg2:][1:] * np.diff(m.age[arg2:])))

    mi = m.star_1_mass[arg2]
    mf = m.star_1_mass[-1]
    qi = m.star_2_mass[arg2] / m.star_1_mass[arg2]
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=val,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )

    # print(val )
    val = (
        wind.P_over_P0[-1] * m.period_days[arg2] - m.period_days[-1]
    ) / m.period_days[-1]

    plt.plot(
        qs, wind.P_over_P0 * m.period_days[arg2], c=f"C{i%9}", alpha=0.5, linewidth=2
    )

plt.show()
# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 10

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

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
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

    mi = r.star_1_mass[arg2]
    mf = r.star_1_mass[-1]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=val,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )

    # print(val )
    val = (
        wind.P_over_P0[-1] * r.period_days[arg2] - r.period_days[-1]
    ) / r.period_days[-1]

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

norm = TwoSlopeNorm(0, min(z), max(z))

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
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
    norm=norm,
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
    cmap="coolwarm",
    # norm=norm,
    zorder=-20,
    norm=norm,
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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-4.pgf", format="pgf", dpi=600)
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
    # plt.plot(
    #     m.star_2_mass[1:] / m.star_1_mass[1:],
    #     np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age)),
    #     c=cmap(norm(m.params["R"])),
    # )

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    plt.scatter(
        m.star_2_mass[arg2] / m.star_1_mass[arg2],
        (m.beta_accretion)[arg2],
        c=cmap(norm(m.params["R"])),
        linewidth=5,
        alpha=0.5,
    )
    plt.plot(
        m.star_2_mass[1:] / m.star_1_mass[1:],
        (m.beta_accretion)[1:],
        c=cmap(norm(m.params["R"])),
        linewidth=5,
        alpha=0.5,
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$(\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot})|_{q=0}^{q=q}$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-cum.pgf", format="pgf")
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
    # plt.plot(
    #     m.star_2_mass[1:] / m.star_1_mass[1:],
    #     np.cumsum((10**m.lg_wind_mdot_1 / (1 - m.beta_accretion))[1:] * np.diff(m.age)),
    #     c=cmap(norm(m.params["R"])),
    # )

    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    plt.scatter(
        m.star_1_mass[arg2],
        (m.beta_accretion)[arg2],
        c=cmap(norm(m.params["R"])),
        linewidth=5,
        alpha=0.5,
    )
    plt.plot(
        m.star_1_mass[1:],
        (m.beta_accretion)[1:],
        c=cmap(norm(m.params["R"])),
        linewidth=5,
        alpha=0.5,
    )


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$(\\Delta M_\\textrm{wind} / \\Delta M_\\textrm{tot})|_{q=0}^{q=q}$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-cum.pgf", format="pgf")
plt.show()
plt.close()

# %%

for i, m in enumerate(grid.filter(m=1.8)):
    try:
        arg = np.argwhere(m.R / m.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((m.R[:arg] / m.rl_1[:arg]) < 0.9) & (m.log_LHe[:arg] - m.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    plt.plot(((10**m.lg_wind_mdot_1) / (1 - m.beta_accretion))[1:], c=f"C{i%10}")
    plt.plot(((10**m.lg_wind_mdot_1))[1:], c=f"C{i%10}", linestyle=":")
    plt.plot((10**m.lg_mstar_dot_1)[1:], c=f"C{i%10}", linewidth=4, alpha=0.5)
plt.show()
# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 10

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

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
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
    print(1 - 0.7 * val - (1 - val) * 0.8 - (1 - val) * 0.1)

    mi = r.star_1_mass[arg2]
    mf = r.star_1_mass[-1]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=val,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )

    # print(val )
    val = (
        wind.P_over_P0[-1] * r.period_days[arg2] - r.period_days[-1]
    ) / r.period_days[-1]

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


norm = TwoSlopeNorm(0, min(z), max(z))

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
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
    norm=norm,
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
    cmap="coolwarm",
    # norm=norm,
    zorder=-20,
    norm=norm,
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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 10

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

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    # val = (
    #     np.sum(
    #         (10 ** r.lg_wind_mdot_1[arg2:] / (1 - r.beta_accretion[arg2:]))[1:]
    #         * np.diff(r.age[arg2:])
    #     )
    # ) / (np.sum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])))
    # print(1 - 0.7*val - (1-val)*0.8 - (1-val)*0.1)
    val = 0

    mi = r.star_1_mass[arg2]
    mf = r.star_1_mass[-1]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=0,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )

    # print(val )
    val = (
        wind.P_over_P0[-1] * r.period_days[arg2] - r.period_days[-1]
    ) / r.period_days[-1]

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


norm = TwoSlopeNorm(0, min(z), max(z))

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
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
    norm=norm,
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
    cmap="coolwarm",
    # norm=norm,
    zorder=-20,
    norm=norm,
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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    2, 2, figsize=set_size(full), sharey="row", sharex="col", constrained_layout=True
)

axs = axs.flatten()

for m in grid.filter(m=1.8):
    star = get_star(m=m.params["m"])
    rl1 = rl(m.sb.a, m.sb.m2 / star.mass)
    axs[0].plot(star.age, rl1, c="C0", linewidth=1)
    axs[2].plot(
        star.age,
        get_period(m.sb.a, star.mass, m.sb.m2 / star.mass),
        c="C0",
        linewidth=1,
    )
    arg = np.argwhere(np.isnan(rl1) != True)[-1][-1]
    print(arg)
    axs[0].scatter(star.age[arg], rl1[arg], c="C0", zorder=1000, s=20)

    axs[1].scatter((m.sb.m2 / star.mass)[arg], rl1[arg], c="C0", s=20)
    axs[2].scatter(
        star.age[arg],
        get_period(m.sb.a, star.mass, m.sb.m2 / star.mass)[arg],
        c="C0",
        s=20,
    )
    axs[3].scatter(
        (m.sb.m2 / star.mass)[arg],
        get_period(m.sb.a, star.mass, m.sb.m2 / star.mass)[arg],
        c="C0",
        s=20,
    )

    mi = star.mass[arg]
    mf = star.mass[-1]
    qi = (m.sb.m2 / star.mass)[arg]
    val = 0
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=0,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    axs[1].plot(
        qs,
        rl1[arg] * rl(wind.a_over_a0, qs) / rl(wind.a_over_a0, qs)[0],
        c="C0",
        linewidth=1,
    )
    axs[3].plot(
        qs,
        get_period(m.sb.a[arg], star.mass[arg], (m.sb.m2 / star.mass)[arg])
        * wind.P_over_P0,
        linewidth=1,
        c="C0",
    )

    # plt.plot(m.age, m.rl_1, c="C0", alpha=0.5, linewidth=2)
axs[0].plot(star.age, 10**star.log_R, c="C9")
axs[0].set_xlim(1326224390.6157308, 1327957157.3249505)
axs[2].set_xlim(1326224390.6157308, 1327957157.3249505)
axs[0].set_ylim(0, 1050)
axs[1].set_ylim(0, 1050)

axs[2].set_xlabel("Age")
axs[2].set_ylabel("Period (days)")
axs[0].set_ylabel("Radius ($R_\\odot$)")
axs[3].set_xlabel("$q$")


fig.suptitle("$M=1.8\\;M_\\odot$")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w38-direct.pgf", format="pgf")
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

    star = get_star(m=r.params["m"])
    rl1 = rl(r.sb.a, r.sb.m2 / star.mass)
    arg = np.argwhere(np.isnan(rl1) != True)[-1][-1]

    mi = star.mass[arg]
    mf = star.mass[-1]
    qi = (r.sb.m2 / star.mass)[arg]
    val = 0
    qs = np.linspace(qi, q_f(qi, mi, mf, val, (1 - val) * 0.8, (1 - val) * 0.1), 100)
    wind = CombineEvolve(
        alpha=0,
        beta=(1 - val) * 0.8,
        delta=(1 - val) * 0.1,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    # val = (
    #     np.sum(
    #         (10 ** r.lg_wind_mdot_1[arg2:] / (1 - r.beta_accretion[arg2:]))[1:]
    #         * np.diff(r.age[arg2:])
    #     )
    # ) / (np.sum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])))
    # print(1 - 0.7*val - (1-val)*0.8 - (1-val)*0.1)

    # print(val )
    val = (
        wind.P_over_P0[-1]
        * get_period(r.sb.a[arg], star.mass[arg], (r.sb.m2 / star.mass)[arg])
        - r.period_days[-1]
    ) / r.period_days[-1]

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


norm = TwoSlopeNorm(0, min(z), max(z))

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
    cmap="coolwarm",
    zorder=1000,
    edgecolor="w",
    linewidth=0.75 / 2,
    s=20,
    norm=norm,
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
    cmap="coolwarm",
    # norm=norm,
    zorder=-20,
    norm=norm,
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

plt.colorbar(
    c,
    ax=axs,
    label="$(P_\\textrm{f;simple} - P_\\textrm{f;MESA}) / P_\\textrm{f;MESA}$",
)

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-fracs-5.pgf", format="pgf", dpi=600)
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

    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-mass-8.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
