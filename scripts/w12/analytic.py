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


# %%
Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


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

qs = np.linspace(0.5, 3, 100)
orbit = CombineEvolve(
    alpha=1e-10,
    beta=0,
    delta=0.0,
    gamma=2,
    qs=qs,
)

plt.plot(qs, orbit.a_over_a0)

M_initial = 2
M_final = M_initial * (1 + qs[0]) / (1 + qs)

plt.plot(qs, a_final(1, M_initial, M_final, qs[0]))

plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

plt.xlabel("Star mass $M_\\odot$")
plt.ylabel("$P / P_i$")

qs = np.linspace(0.5, 1.5, 100)
wind = CombineEvolve(
    alpha=1 - 1e-10,
    beta=0.0,
    delta=0,
    gamma=0,
    qs=qs,
)

qs = np.linspace(0.5, 3.5, 100)
transfer = CombineEvolve(
    alpha=1e-10,
    beta=0.0,
    delta=0,
    gamma=0,
    qs=qs,
)


plt.plot(wind.M_d, wind.P_over_P0, c="C0", label="Isotropic wind, $q_i=0.5$")
plt.plot(wind.M_a, wind.P_over_P0, c="C0", alpha=0.3, linewidth=4)


plt.plot(
    transfer.M_d,
    transfer.P_over_P0,
    c="C1",
    label="Conservative mass transfer, $q_i=0.5$",
)
plt.plot(transfer.M_a, transfer.P_over_P0, c="C1", alpha=0.3, linewidth=4)


qs = np.linspace(1, 3, 100)
wind = CombineEvolve(
    alpha=1 - 1e-10,
    beta=0.0,
    delta=0,
    gamma=0,
    qs=qs,
)

qs = np.linspace(1, 5, 100)
transfer = CombineEvolve(
    alpha=1e-10,
    beta=0.0,
    delta=0,
    gamma=0,
    qs=qs,
)


plt.plot(wind.M_d, wind.P_over_P0, c="C2", label="Isotropic wind, $q_i=1$")
plt.plot(wind.M_a, wind.P_over_P0, c="C2", alpha=0.3, linewidth=4)


plt.plot(
    transfer.M_d,
    transfer.P_over_P0,
    c="C3",
    label="Conservative mass transfer, $q_i=1$",
)
plt.plot(transfer.M_a, transfer.P_over_P0, c="C3", alpha=0.3, linewidth=4)

plt.text(
    3.25,
    5.7,
    r"\textbf{Accretor}",
    c="C3",
    alpha=0.5,
    ha="left",
    rotation=-66,
    va="top",
)
plt.text(0.75, 5.7, r"Donor", c="C3", ha="right", rotation=66, va="top")

fig.legend(loc="outside upper center", ncols=2)
axs.set_yscale("log")
axs.invert_xaxis()

plt.yticks(
    [1, 2, 3, 4, 6],
    ["1", "2", "3", "4", "6"],
)
axs.tick_params(axis="y", which="minor", length=0)
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-1.pgf", format="pgf")
plt.show()
plt.close()
# %%
