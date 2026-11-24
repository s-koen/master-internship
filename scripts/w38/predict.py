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

grid = MesaGrid("/home/koen/master-internship/mesa-models/grid-M-R-2026-11-02/", fresh=True)
gridshort = MesaGrid(
    "/home/koen/master-internship/mesa-models/grid-M-R-low-2026-11-15/", fresh=True
)
grid.merge(gridshort)
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
        (
            np.sum(
                (10 ** r.lg_wind_mdot_1[arg2:] / (1 - r.beta_accretion[arg2:]))[1:]
                * np.diff(r.age[arg2:])
            )
        )
        - np.sum((10 ** r.lg_wind_mdot_1[arg2:])[1:] * np.diff(r.age[arg2:]))
    ) / (
        np.sum(
            (10 ** r.lg_wind_mdot_1[arg2:] / (1 - r.beta_accretion[arg2:]))[1:]
            * np.diff(r.age[arg2:])
        )
    )
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
    zorder=-20,
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

plt.colorbar(c, ax=axs, label="Average wind mass transfer efficiency after RLOF")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-beta.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%


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

    plt.plot(
        np.cumsum(10 ** r.lg_mstar_dot_2[arg2:][1:] * np.diff(r.age[arg2:]))
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    z.append(val)
    z_mesa.append(val)
    break

plt.show()

# %%


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

    plt.plot(np.cumsum(10 ** r.lg_mstar_dot_2[arg2:][1:] * np.diff(r.age[arg2:])))
    plt.plot(np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])))

    z.append(val)
    z_mesa.append(val)
    break

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

    val = np.sum(10 ** r.lg_mstar_dot_2[arg2:][1:] * np.diff(r.age[arg2:])) / np.sum(
        10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])
    )
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
    zorder=-20,
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

plt.colorbar(c, ax=axs, label="Average mass transfer efficiency after RLOF")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-eps.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%


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

    # delta m star 1
    plt.plot(np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:])))
    plt.plot(
        np.cumsum(
            10 ** r.lg_wind_mdot_1[arg2:][1:]
            / (1 - r.beta_accretion[arg2:][1:])
            * np.diff(r.age[arg2:])
        )
        + np.cumsum(10 ** r.lg_mtransfer_rate[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    # delta m star 2
    plt.plot(np.cumsum(10 ** r.lg_mstar_dot_2[arg2:][1:] * np.diff(r.age[arg2:])))
    plt.plot(
        np.cumsum(
            10 ** r.lg_wind_mdot_1[arg2:][1:]
            / (1 - r.beta_accretion[arg2:][1:])
            * r.beta_accretion[arg2:][1:]
            * np.diff(r.age[arg2:])
        )
        + np.cumsum(10 ** r.lg_mtransfer_rate[arg2:][1:] * 0.1 * np.diff(r.age[arg2:]))
    )

    z.append(val)
    z_mesa.append(val)
    break

plt.show()


# %%
# %%


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

    # delta m star 2
    plt.plot(
        (
            np.cumsum(
                10 ** r.lg_wind_mdot_1[arg2:][1:]
                / (1 - r.beta_accretion[arg2:][1:])
                * r.beta_accretion[arg2:][1:]
                * np.diff(r.age[arg2:])
            )
            + np.cumsum(
                10 ** r.lg_mtransfer_rate[arg2:][1:] * 0.1 * np.diff(r.age[arg2:])
            )
        )
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    plt.plot(
        (np.cumsum(10 ** r.lg_mtransfer_rate[arg2:][1:] * 0.1 * np.diff(r.age[arg2:])))
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    plt.plot(
        (
            np.cumsum(
                10 ** r.lg_wind_mdot_1[arg2:][1:]
                / (1 - r.beta_accretion[arg2:][1:])
                * r.beta_accretion[arg2:][1:]
                * np.diff(r.age[arg2:])
            )
        )
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    plt.plot(
        (np.cumsum(10 ** r.lg_mtransfer_rate[arg2:][1:] * 0.1 * np.diff(r.age[arg2:])))
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
        + (
            np.cumsum(
                10 ** r.lg_wind_mdot_1[arg2:][1:]
                / (1 - r.beta_accretion[arg2:][1:])
                * r.beta_accretion[arg2:][1:]
                * np.diff(r.age[arg2:])
            )
        )
        / np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * np.diff(r.age[arg2:]))
    )

    z.append(val)
    z_mesa.append(val)
    break

plt.show()

# %%


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

    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    # soberman fractions
    alpha = dm_wind_lost[-1] / dm1[-1]

    beta_delta = dm_rlof_lost[-1] / dm1[-1]

    # split using the assumed relative proportions
    beta = beta_delta * 8 / 9
    delta = beta_delta * 1 / 9

    eps = (dm_wind_acc[-1] + dm_rlof_acc[-1]) / dm1[-1]

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

    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    # soberman fractions
    alpha = dm_wind_lost[-1] / dm1[-1]

    beta_delta = dm_rlof_lost[-1] / dm1[-1]

    # split using the assumed relative proportions
    beta = beta_delta * 8 / 9
    delta = beta_delta * 1 / 9

    eps = (dm_wind_acc[-1] + dm_rlof_acc[-1]) / dm1[-1]

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    mi = r.star_1_mass[arg2]
    mf = r.star_1_mass[-1]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qs = np.linspace(qi, q_f(qi, mi, mf, alpha, beta, delta), 100)
    wind = CombineEvolve(
        alpha=alpha,
        beta=beta,
        delta=delta,
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

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue
    if r.params["R"] < 1500:
        continue
    print(r)
    break

plt.plot(r.star_2_mass / r.star_1_mass, r.binary_separation)

import numpy as np
from scipy.integrate import quad


def a_ratio(q, q0, beta, eta):
    A = 2 * beta**2 * eta - 4 * beta * eta + 3 * beta + 2 * eta
    D = 1 + eta * (beta - 1)

    if beta == 0:
        return np.exp(2 * eta * (q - q0)) * (1 + q) / (1 + q0) * (q / q0) ** (-2 * D)

    return (
        ((1 + beta * q) / (1 + beta * q0)) ** (A / beta)
        * (1 + q)
        / (1 + q0)
        * (q / q0) ** (-2 * D)
    )


plt.plot(
    r.star_2_mass / r.star_1_mass,
    [
        r.binary_separation[0]
        * a_ratio(q, r.star_1_mass[0] / r.star_2_mass[0], 0.18, 0.6)
        for q in r.star_1_mass / r.star_2_mass
    ],
)

plt.annotate(
    "MESA model",
    xy=(3, 3750),
    xycoords="data",
    xytext=(50, -20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C0"),
    fontsize=8,
    va="center",
    ha="center",
)

plt.annotate(
    "constant $\\beta$ and $\\delta$ approximation",
    xy=(3, 3900),
    xycoords="data",
    xytext=(-50, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C1"),
    fontsize=8,
    va="center",
    ha="center",
)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$a$ ($R_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-compare-simple-sal-mesa.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

import sympy as sp

q, b, e = sp.symbols("q b e")
B = (
    -2
    * (1 - b * q - e * (1 - b) * (1 + q) - (1 - b) * q / (2 * (1 + q)))
    / (q * (1 + b * q))
)
sp.apart(B, q)

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

    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    # soberman fractions
    alpha = 0

    beta = 0.8
    delta = 0.1

    eps = 0.1

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    mi = r.star_1_mass[arg2]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qf = q_f(qi, mi, mi - m_rlof_tot, alpha, beta, delta)
    qs = np.linspace(qi, qf, 100)
    wind = CombineEvolve(
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    a_over_a0 = wind.a_over_a0[-1]

    star = get_star(m=r.params["m"])
    alpha = r.beta_accretion[arg2]
    print(alpha)
    beta = 0
    delta = 0
    qs = 1 / np.linspace(
        qf, q_f(qf, mi - m_rlof_tot, star.mass[-1], alpha, beta, delta), 100
    )
    final_a = [a_ratio(q, qs[0], alpha, 0.6) for q in qs][-1]
    val = r.binary_separation[arg2] * final_a * a_over_a0 / r.binary_separation[-1]
    z.append(val)
    z_mesa.append(val)

print(z)

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

print(min(z))
print(max(z))

norm = TwoSlopeNorm(1, min(z), max(z))

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

plt.colorbar(c, ax=axs, label="$a_\\textrm{simple;f} / a_\\textrm{MESA;f}$")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-a-orbit-two-way-vs-mesa.pgf",
    format="pgf",
    dpi=600,
)
plt.show()
plt.close()

# %%

cmap = Colormap("cmasher:wildfire").to_mpl()

levels = 10

from scipy.interpolate import griddata

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
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

for r in grid.filter(m=1.8):
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

    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    # soberman fractions
    alpha = 0

    beta = 0.8
    delta = 0.1

    eps = 0.1

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    mi = r.star_1_mass[arg2]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qf = q_f(qi, mi, mi - m_rlof_tot, alpha, beta, delta)
    qs = np.linspace(qi, qf, 100)
    wind = CombineEvolve(
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    a_over_a0 = wind.a_over_a0
    plt.plot(r.star_2_mass / r.star_1_mass, r.binary_separation)
    plt.scatter(
        r.star_2_mass[arg2] / r.star_1_mass[arg2], r.binary_separation[arg2], zorder=100
    )
    plt.plot(qs, a_over_a0 * r.binary_separation[arg2])
    plt.scatter(qs[-1], a_over_a0[-1] * r.binary_separation[arg2], zorder=100)

    star = get_star(m=r.params["m"])
    alpha = r.beta_accretion[arg2]
    print(alpha)
    beta = 0
    delta = 0
    qs = 1 / np.linspace(
        qf, q_f(qf, mi - m_rlof_tot, star.mass[-1], alpha, beta, delta), 100
    )
    final_a = np.array([a_ratio(q, qs[0], alpha, 0.6) for q in qs])
    print(final_a)

    plt.plot(1 / qs, final_a * a_over_a0[-1] * r.binary_separation[arg2])

    val = r.binary_separation[arg2] * final_a * a_over_a0 / r.binary_separation[-1]
    z.append(val)
    z_mesa.append(val)
    break

plt.annotate(
    "Saladino winds",
    xy=(2.6, 1900),
    xycoords="data",
    xytext=(50, -20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C2"),
    fontsize=8,
    va="center",
    ha="center",
)

plt.annotate(
    "Soberman RLOF",
    xy=(1.8, 1700),
    xycoords="data",
    xytext=(-60, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C1"),
    fontsize=8,
    va="center",
    ha="center",
)

plt.annotate(
    "MESA",
    xy=(2.1, 1700),
    xycoords="data",
    xytext=(50, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=0.75, color="C0"),
    fontsize=8,
    va="center",
    ha="center",
)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$a$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-mesa-vs-rlof-then-wind.pgf", format="pgf")
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

    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    # soberman fractions
    alpha = 0

    beta = 0.8
    delta = 0.1

    eps = 0.1

    x.append(period_days(0.9, r.params["m"], r.params["R"]))
    x_mesa.append(period_days(0.9, r.params["m"], r.params["R"]))
    y.append(r.params["m"])
    y_mesa.append(r.params["m"])

    mi = r.star_1_mass[arg2]
    qi = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    qf = q_f(qi, mi, mi - m_rlof_tot, alpha, beta, delta)
    qs = np.linspace(qi, qf, 100)
    wind = CombineEvolve(
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    a_over_a0 = wind.a_over_a0[-1]

    star = get_star(m=r.params["m"])
    alpha = r.beta_accretion[arg2]
    print(alpha)
    beta = 0
    delta = 0
    qs = 1 / np.linspace(
        qf, q_f(qf, mi - m_rlof_tot, star.mass[-1], alpha, beta, delta), 100
    )
    final_a = [a_ratio(q, qs[0], alpha, 0.6) for q in qs][-1]
    val = r.binary_separation[arg2] * final_a * a_over_a0 / r.binary_separation[-1]
    z.append(val)
    z_mesa.append(val)

print(z)

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

print(min(z))
print(max(z))

norm = TwoSlopeNorm(1, min(z), max(z))

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
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-rlof-then-wind.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

cmap = Colormap("colorcet:CET_D4").to_mpl()

for r in grid.filter(m=1.8):
    if r.envelope_mass[-1] > 0.01:
        continue
    if r.params["R"] % 100 != 0:
        continue

    x = cplot(
        r.star_2_mass / r.star_1_mass,
        r.binary_separation,
        10**r.lg_wind_mdot_1 / (1 - r.beta_accretion) / 10**r.lg_mstar_dot_1,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

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
        r.star_2_mass[arg2] / r.star_1_mass[arg2],
        r.binary_separation[arg2],
        c=10 ** r.lg_wind_mdot_1[arg2]
        / (1 - r.beta_accretion[arg2])
        / 10 ** r.lg_mstar_dot_1[arg2],
        cmap=cmap,
        vmin=0,
        vmax=1,
        zorder=1000,
    )


plt.colorbar(label="$\\dot{M}_\\textrm{wind} / \\dot{M}_\\textrm{tot}$")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$a$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w38-frac-wind.pgf", format="pgf")
plt.show()
plt.close()

# %%

for r in grid.filter(m=1.8):
    if r.envelope_mass[-1] > 0.01:
        continue
    if r.params["R"] % 100 != 0:
        continue

    plt.plot(
        r.star_2_mass / r.star_1_mass,
        10**r.lg_wind_mdot_1 / (1 - r.beta_accretion) / 10**r.lg_mstar_dot_1,
    )

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue
    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9) & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]
    if arg2 == 0:
        continue

    # plt.scatter(r.star_2_mass[arg2] / r.star_1_mass[arg2], r.binary_separation[arg2])

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


    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    val = m_rlof_tot
    z.append(val)
    z_mesa.append(val)

print(z)

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

print(min(z))
print(max(z))

norm = TwoSlopeNorm(1, min(z), max(z))

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

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{RLOF}$ ($M_\\odot$)")

# plt.xlim(np.min(x) - 20, 2e5)
plt.ylim(0.95, np.max(y) + 0.05)
plt.xscale("log")

axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("Initial period (days)")
plt.ylabel("Initial mass ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-m_rlof.pgf", format="pgf", dpi=600
)
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


    x.append(r.star_2_mass[arg2] / r.star_1_mass[arg2])
    x_mesa.append(r.star_2_mass[arg2] / r.star_1_mass[arg2])
    y.append(r.envelope_mass[arg2])
    y_mesa.append(r.envelope_mass[arg2])


    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    val = m_rlof_tot
    z.append(val)
    z_mesa.append(val)

print(z)

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

print(min(z))
print(max(z))

norm = TwoSlopeNorm(1, min(z), max(z))

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


plt.yticks(
    [0,0.5,1.0, 1.5, 2, 2.5],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{RLOF}$ ($M_\\odot$)")


axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y")


plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-m_rlof-2.pgf", format="pgf", dpi=600
)
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


    x.append(r.star_2_mass[arg2] / r.star_1_mass[arg2])
    x_mesa.append(r.star_2_mass[arg2] / r.star_1_mass[arg2])
    y.append(r.envelope_mass[arg2])
    y_mesa.append(r.envelope_mass[arg2])


    # delta m star 2
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(10 ** r.lg_mstar_dot_1[arg2:][1:] * dt)

    dm_wind_acc = np.cumsum(mdot_wind / (1 - beta_wind) * beta_wind * dt)

    dm_wind_lost = np.cumsum(mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt)

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)

    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    val = m_rlof_tot
    z.append(val)
    z_mesa.append(val)

print(z)

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

print(min(z))
print(max(z))

norm = TwoSlopeNorm(1, min(z), max(z))

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


plt.yticks(
    [0,0.5,1.0, 1.5, 2, 2.5],
)
plt.tick_params(color="w", axis="both", which="both")

plt.colorbar(c, ax=axs, label="$\\Delta M_\\textrm{RLOF}$ ($M_\\odot$)")


axs.spines[["right", "top", "left", "bottom"]].set_visible(False)
plt.xlabel("$q$")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
# plt.title("$q=0.85,\\;\\varepsilon=0.1,\\;f_\\delta=0.1$")

plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="x")
plt.grid(c="w", alpha=0.5, linewidth=0.75 / 2, axis="y")


plt.savefig(
    "/home/koen/LaTeX-setup/plots/w38-m_rlof-2.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()

# %%
cmap = plt.get_cmap("coolwarm")

fig = plt.figure(
    figsize=set_size(full, height=0.8),
    constrained_layout=True,
)
axs = fig.add_subplot(111, projection="3d")

# interpolate
xi = np.linspace(np.log10(min(x)), np.log10(max(x)), 200)
yi = np.linspace(min(y), max(y), 200)
X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (np.log10(x), y),
    z,
    (X, Y),
    method="linear",
)

X = 10**X

# normalization
norm = TwoSlopeNorm(1, min(z), max(z))

# surface
surf = axs.plot_surface(
    X,
    Y,
    Z,
    facecolors=cmap(norm(Z)),
    linewidth=0,
    antialiased=True,
    alpha=0.85,
    shade=False,
)

# original MESA points
sc = axs.scatter(
    x_mesa,
    y_mesa,
    z_mesa,
    c=z_mesa,
    cmap=cmap,
    norm=norm,
    edgecolor="k",
    linewidth=0.4,
    s=20,
    depthshade=False,
)

# colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig.colorbar(
    sm,
    ax=axs,
    label="$\\Delta M_\\textrm{RLOF}$ ($M_\\odot$)",
    shrink=0.7,
    pad=0.12,
)

# labels
axs.set_xlabel("$q$", labelpad=10)
axs.set_ylabel("$m_\\textrm{env}$ ($M_\\odot$)", labelpad=10)
axs.set_zlabel("$\\Delta M_\\textrm{RLOF}$ ($M_\\odot$)", labelpad=10)

# viewing angle
axs.view_init(
    elev=25,
    azim=-55,
)

# aspect ratio
axs.set_box_aspect((1.3, 1, 0.8))

axs.tick_params(axis="both", colors="black")
axs.grid(True, alpha=0.3)

plt.show()
plt.close()
# %%
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

cmap = plt.get_cmap("coolwarm")

# collect data
x = []
y = []
z = []

for r in grid.models:
    if r.envelope_mass[-1] > 0.01:
        continue

    try:
        arg = np.argwhere(r.R / r.rl_1 > 1)[0][0]
    except:
        continue

    arg2 = np.argwhere(
        ((r.R[:arg] / r.rl_1[:arg]) < 0.9)
        & (r.log_LHe[:arg] - r.log_LH[:arg] < 4)
    )[-1][-1]

    if arg2 == 0:
        continue

    # coordinates
    q = r.star_2_mass[arg2] / r.star_1_mass[arg2]
    m_env = r.envelope_mass[arg2]

    # mass transfer calculations
    dt = np.diff(r.age[arg2:])

    mdot_wind = 10 ** r.lg_wind_mdot_1[arg2:][1:]
    beta_wind = r.beta_accretion[arg2:][1:]
    mdot_rlof = 10 ** r.lg_mtransfer_rate[arg2:][1:]

    dm1 = np.cumsum(
        10 ** r.lg_mstar_dot_1[arg2:][1:] * dt
    )

    dm_wind_acc = np.cumsum(
        mdot_wind / (1 - beta_wind) * beta_wind * dt
    )

    dm_wind_lost = np.cumsum(
        mdot_wind / (1 - beta_wind) * (1 - beta_wind) * dt
    )

    dm_rlof_acc = np.cumsum(0.1 * mdot_rlof * dt)
    dm_rlof_lost = np.cumsum(0.9 * mdot_rlof * dt)

    m_rlof_tot = dm_rlof_acc[-1] + dm_rlof_lost[-1]

    x.append(q)
    y.append(m_env)
    z.append(m_rlof_tot)


x = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)

print(f"Number of models: {len(z)}")
print(f"Minimum: {z.min():.4f}")
print(f"Maximum: {z.max():.4f}")

# --------------------------------------------------
# fit plane: z = a*x + b*y + c
# --------------------------------------------------

from scipy.optimize import least_squares

def model(params, x, y):
    a, b, c = params
    return np.clip(a*x + b*y + c, 0, None)


def residuals(params, x, y, z):
    return model(params, x, y) - z


# initial guess from ordinary linear least squares
A = np.column_stack([x, y, np.ones(len(x))])
p0 = np.linalg.lstsq(A, z, rcond=None)[0]

# nonlinear least-squares fit
result = least_squares(
    residuals,
    p0,
    args=(x, y, z),
)

a, b, c = result.x

z_pred = model(result.x, x, y)

rmse = np.sqrt(np.mean((z - z_pred)**2))

ss_res = np.sum((z - z_pred)**2)
ss_tot = np.sum((z - z.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R² = {r2:.6f}")

# --------------------------------------------------
# generate fitted plane
# --------------------------------------------------

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)

X, Y = np.meshgrid(xi, yi)
Z = np.clip(a * X + b * Y + c,0,1e99)

# --------------------------------------------------
# plot
# --------------------------------------------------

norm = TwoSlopeNorm(
    vcenter=1,
    vmin=z.min(),
    vmax=z.max(),
)

fig = plt.figure(
    figsize=set_size(full, height=0.8),
    constrained_layout=True,
)

axs = fig.add_subplot(111, projection="3d")

# fitted plane
surf = axs.plot_surface(
    X,
    Y,
    Z,
    cmap=cmap,
    norm=norm,
    alpha=0.7,
    edgecolor="none",
    antialiased=True,
)

# original MESA points
sc = axs.scatter(
    x,
    y,
    z,
    c=z,
    cmap=cmap,
    norm=norm,
    edgecolor="w",
    linewidth=0.5,
    s=25,
    depthshade=False,
)

# colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig.colorbar(
    sm,
    ax=axs,
    label="$\\Delta M_{\\rm RLOF}$ ($M_\\odot$)",
    shrink=0.5,
    pad=0.02,
)

# labels
axs.set_xlabel("$q$", labelpad=10)
axs.set_ylabel("$m_{\\rm env}$ ($M_\\odot$)", labelpad=10)
axs.set_zlabel("$\\Delta M_{\\rm RLOF}$ ($M_\\odot$)", labelpad=10)

# viewing angle

axs.set_box_aspect((1.3, 1, 0.8))

axs.grid(True, alpha=0.3)

line_x = np.linspace(0.9,3,300)
line_y = np.linspace(2.4,0,300)
line_z = np.clip(a * line_x + b * line_y + c,0,1e99)
plt.plot(line_x,line_y,line_z,  c="k", zorder=1000, linewidth=1)

axs.view_init(elev=-1.94, azim=-154.47)
def print_view(event):
    if event.inaxes == axs:
        print(
            f"\relev = {axs.elev:.2f}, azim = {axs.azim:.2f}",
            end="",
            flush=True,
        )

fig.canvas.mpl_connect("motion_notify_event", print_view)



plt.savefig("/home/koen/LaTeX-setup/plots/w38-3d-fit.pgf", format="pgf")

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

    # soberman fractions
    alpha = 0

    beta = 0.8
    delta = 0.1

    eps = 0.1

    mi = star.mass[arg]
    qi = (r.sb.m2 / star.mass)[arg]
    print(np.clip(-0.550794 * qi + 0.820426 * star.m_core[arg] + 0.557864,0,1e99))
    mf = star.mass[arg] - np.clip(-0.550794 * qi + 0.820426 * star.m_env[arg] + 0.557864,0,1e99)
    qf = q_f(qi, mi, mf, alpha, beta, delta)
    qs = np.linspace(qi, qf, 100)
    wind = CombineEvolve(
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=1.44,
        qs=qs,
        M_1_i=mi,
    )
    a_over_a0 = wind.a_over_a0[-1]

    betas = np.interp(star.age, r.age, r.beta_accretion, 0.3, 0.3)
    alpha = betas[arg] 
    print(alpha)
    beta = 0
    delta = 0
    qs = 1 / np.linspace(
        qf, q_f(qf, mf, star.mass[-1], alpha, beta, delta), 100
    )
    final_a = np.array([a_ratio(q, qs[0], alpha, 0.6) for q in qs])[-1]


    val = r.binary_separation[arg2] * final_a * a_over_a0 / r.binary_separation[-1]
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


norm = TwoSlopeNorm(1, min(z), max(z))

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
plt.savefig("/home/koen/LaTeX-setup/plots/w38-uninformed.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
