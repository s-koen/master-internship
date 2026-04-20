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
        self.q0 = qs[0]

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
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("$q = M_\\textrm{a} / M_\\textrm{d} $")
plt.ylabel(r"$a_\textrm{f} / a_\textrm{i}$")

qs = np.linspace(0.5, 3, 100)
orbit = CombineEvolve(
    alpha=1e-10,
    beta=0,
    delta=0.0,
    gamma=2,
    qs=qs,
)


M_initial = 2
M_final = M_initial * (1 + qs[0]) / (1 + qs)

plt.plot(
    qs,
    a_final(1, M_initial, M_final, qs[0]),
    alpha=0.5,
    linewidth=5,
    c="C9",
    label="Derivation week 10",
)

plt.plot(qs, orbit.a_over_a0, label="Soberman")
plt.legend(ncols=2)

plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-test.pgf", format="pgf")
plt.show()
plt.close()

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

# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Roche lobe radius ($R_\\odot$)")
plt.ylabel("$q$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

RLs = np.logspace(np.log10(150), np.log10(650), 150)
qs = np.linspace(0.4, 1, 150)

logR = np.log10(RLs)
dlogR = np.diff(logR)

logR_edges = np.concatenate(
    [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
)
R_edges = 10**logR_edges

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(RLs)), 10.0)

for i, q in enumerate(qs):
    for j, RL in enumerate(RLs):

        # alpha_test = RL / RLs[-1] * 0.45
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha_test, beta_test, delta_test), 100)
        orbit = CombineEvolve(alpha_test, beta_test, delta_test, 0, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = get_period(
            get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1]
        )

mesh = plt.pcolormesh(
    R_edges, q_edges, Z, cmap="viridis", shading="auto", rasterized=True
)

plt.colorbar(mesh, label="Period (days)")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid1.pgf", format="pgf")
plt.show()
plt.close()

# %%

alpha_test = 0.2
beta_test = 0.2
delta_test = 0
M_i = 2
M_f = 0.6
q_i = 0.5

qs = np.linspace(q_i, q_f(q_i, M_i, M_f, alpha_test, beta_test, delta_test), 100)

orbit = CombineEvolve(alpha_test, beta_test, delta_test, 0, qs)
plt.plot(orbit.M_d, orbit.a_over_a0)
plt.show()
# %%

# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Roche lobe radius ($R_\\odot$)")
plt.ylabel("$q$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 1 - 1e-10
beta_test = 0.0
delta_test = 0

RLs = np.logspace(np.log10(150), np.log10(650), 150)
qs = np.linspace(0.4, 1, 150)

logR = np.log10(RLs)
dlogR = np.diff(logR)

logR_edges = np.concatenate(
    [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
)
R_edges = 10**logR_edges

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(RLs)), 10.0)

for i, q in enumerate(qs):
    for j, RL in enumerate(RLs):

        # alpha_test = RL / RLs[-1] * 0.45
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha_test, beta_test, delta_test), 100)
        orbit = CombineEvolve(alpha_test, beta_test, delta_test, 0, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = get_period(
            get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1]
        )

mesh = plt.pcolormesh(
    R_edges, q_edges, Z, cmap="viridis", shading="auto", rasterized=True
)

plt.colorbar(mesh, label="Period (days)")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid2.pgf", format="pgf")
plt.show()
plt.close()


# %%

# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 1 - 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 2000)
betas = np.linspace(1e-10, 1 - 1e-10, 2000)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)


dbeta = np.diff(betas)
beta_edges = np.concatenate(
    [[betas[0] - dbeta[0] / 2], betas[:-1] + dbeta / 2, [betas[-1] + dbeta[-1] / 2]]
)

RL = 500
q = 0.75

Z = np.full((len(alphas), len(betas)), np.nan)

for i, beta in enumerate(betas):
    for j, alpha in enumerate(alphas):

        if alpha + beta >= 1 - 1e-9:
            continue

        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = orbit.P_over_P0[-1]

mesh = plt.pcolormesh(
    alpha_edges, beta_edges, Z, cmap="viridis", shading="auto", rasterized=True
)

axs.spines[["right", "top"]].set_visible(False)

plt.colorbar(mesh, label=r"$P_\textrm{f}/P_\textrm{i}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid3.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%


# reproduce the grid like figures using q and RL.

qss = np.linspace(0.125, 1.125, 9)

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)
axs = axs.flatten()


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 1000)
betas = np.linspace(1e-10, 1 - 1e-10, 1000)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)


dbeta = np.diff(betas)
beta_edges = np.concatenate(
    [[betas[0] - dbeta[0] / 2], betas[:-1] + dbeta / 2, [betas[-1] + dbeta[-1] / 2]]
)

RL = 500


for a, q in enumerate(qss):
    Z = np.full((len(alphas), len(betas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):

            if alpha + beta >= 1 - 1e-9:
                continue

            qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100)
            orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = orbit.P_over_P0[-1]

    mesh = axs[a].pcolormesh(
        alpha_edges, beta_edges, Z, cmap="viridis", shading="auto", rasterized=True
    )

    plt.colorbar(
        mesh, label=r"$P_\textrm{f}/P_\textrm{i}$", ax=axs[a], orientation="horizontal"
    )

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.9, 0.9, f"$q_\\textrm{{i}} = {qss[i]}$", va="top", ha="right")


axs[-1].set_xlabel(r"$\alpha$")
axs[-2].set_xlabel(r"$\alpha$")
axs[-3].set_xlabel(r"$\alpha$")
axs[0].set_ylabel(r"$\beta$")
axs[3].set_ylabel(r"$\beta$")
axs[6].set_ylabel(r"$\beta$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid4.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

# reproduce the grid like figures using q and RL.

qss = np.linspace(0.125, 1.125, 9)

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)
axs = axs.flatten()


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 1000)
betas = np.linspace(1e-10, 1 - 1e-10, 1000)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)


dbeta = np.diff(betas)
beta_edges = np.concatenate(
    [[betas[0] - dbeta[0] / 2], betas[:-1] + dbeta / 2, [betas[-1] + dbeta[-1] / 2]]
)

RL = 500


for a, q in enumerate(qss):
    Z = np.full((len(alphas), len(betas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):

            if alpha + beta >= 1 - 1e-9:
                continue

            qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100)
            orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = orbit.a_over_a0[-1]

    mesh = axs[a].pcolormesh(
        alpha_edges, beta_edges, Z, cmap="viridis", shading="auto", rasterized=True
    )

    plt.colorbar(
        mesh, label=r"$a_\textrm{f}/a_\textrm{i}$", ax=axs[a], orientation="horizontal"
    )

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.9, 0.9, f"$q_\\textrm{{i}} = {qss[i]}$", va="top", ha="right")


axs[-1].set_xlabel(r"$\alpha$")
axs[-2].set_xlabel(r"$\alpha$")
axs[-3].set_xlabel(r"$\alpha$")
axs[0].set_ylabel(r"$\beta$")
axs[3].set_ylabel(r"$\beta$")
axs[6].set_ylabel(r"$\beta$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid5.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%


# reproduce the grid like figures using q and RL.

qss = np.linspace(0.125, 1.125, 9)

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)
axs = axs.flatten()


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 1000)
betas = np.linspace(1e-10, 1 - 1e-10, 1000)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)


dbeta = np.diff(betas)
beta_edges = np.concatenate(
    [[betas[0] - dbeta[0] / 2], betas[:-1] + dbeta / 2, [betas[-1] + dbeta[-1] / 2]]
)

RL = 500


for a, q in enumerate(qss):
    Z = np.full((len(alphas), len(betas)), 0.3)
    axs[a].pcolormesh(
        alpha_edges,
        beta_edges,
        Z,
        cmap="Greys",
        shading="auto",
        rasterized=True,
        vmin=0,
        vmax=1,
    )

    Z = np.full((len(alphas), len(betas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):

            if alpha + beta >= 1 - 1e-9:
                continue

            qs = np.log10(np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100))
            # orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = qs[-1]

    mesh = axs[a].pcolormesh(
        alpha_edges,
        beta_edges,
        Z,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        vmin=-0.7,
        vmax=0.7,
    )

plt.colorbar(mesh, label=r"$\log(q_\textrm{f})$", ax=axs[5])

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(
        0.9,
        0.9,
        f"$q_\\textrm{{i}} = {qss[i]}$",
        va="top",
        ha="right",
    )


axs[-1].set_xlabel(r"$\alpha$")
axs[-2].set_xlabel(r"$\alpha$")
axs[-3].set_xlabel(r"$\alpha$")
axs[0].set_ylabel(r"$\beta$")
axs[3].set_ylabel(r"$\beta$")
axs[6].set_ylabel(r"$\beta$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid6.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

# reproduce the grid like figures using q and RL.

qss = np.linspace(0.125, 1.125, 9)

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)
axs = axs.flatten()


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 1000)
betas = np.linspace(1e-10, 1 - 1e-10, 1000)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)


dbeta = np.diff(betas)
beta_edges = np.concatenate(
    [[betas[0] - dbeta[0] / 2], betas[:-1] + dbeta / 2, [betas[-1] + dbeta[-1] / 2]]
)

RL = 500


for a, q in enumerate(qss):
    Z = np.full((len(alphas), len(betas)), 0.3)
    axs[a].pcolormesh(
        alpha_edges,
        beta_edges,
        Z,
        cmap="Greys",
        shading="auto",
        rasterized=True,
        vmin=0,
        vmax=1,
    )

    Z = np.full((len(alphas), len(betas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):

            if alpha + beta >= 1 - 1e-9:
                continue

            qs = np.log10(np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100))
            # orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = np.mean(qs)

    mesh = axs[a].pcolormesh(
        alpha_edges,
        beta_edges,
        Z,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        vmin=-0.7,
        vmax=0.7,
    )

plt.colorbar(mesh, label=r"$\log\langle q\rangle$", ax=axs[5])

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(
        0.9,
        0.9,
        f"$q_\\textrm{{i}} = {qss[i]}$",
        va="top",
        ha="right",
    )


axs[-1].set_xlabel(r"$\alpha$")
axs[-2].set_xlabel(r"$\alpha$")
axs[-3].set_xlabel(r"$\alpha$")
axs[0].set_ylabel(r"$\beta$")
axs[3].set_ylabel(r"$\beta$")
axs[6].set_ylabel(r"$\beta$")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid6.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
