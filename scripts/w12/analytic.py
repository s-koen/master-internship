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

plt.xlabel("Roche lobe radius ($R_\\odot$)")
plt.ylabel("$q$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 0
beta_test = 1 - 1e-10
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
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid-beta-only.pgf", format="pgf"
)
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


alpha_test = 0
beta_test = 1 - 1e-10
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
        ref_qs = np.linspace(q, q_f(q, 2, 0.6, 1e-10, 1e-10, delta_test), 100)

        orbit = CombineEvolve(alpha_test, beta_test, delta_test, 0, qs)
        ref_orbit = CombineEvolve(1e-10, 1e-10, delta_test, 0, ref_qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = np.log10(
            get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
            / get_period(
                get_separation(RL, q) * ref_orbit.a_over_a0[-1], ref_orbit.M_tot[-1]
            )
        )


mesh = plt.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="RdBu_r",
    shading="auto",
    rasterized=True,
    vmin=-np.nanmax([-np.nanmin(Z), np.nanmax(Z)]),
    vmax=np.nanmax([-np.nanmin(Z), np.nanmax(Z)]),
)

cbar = plt.colorbar(
    mesh,
    label="$P_{\\textrm{f}, \\beta = 1} / P_{\\textrm{f}, \\beta = 0}$",
)
cbar.set_ticks(
    ticks=[-0.3, -0.15, 0, 0.15, 0.3],
    labels=[
        f"{10**-0.3:.2f}",
        f"{10**-0.15:.2f}",
        "1",
        f"{10**0.15:.2f}",
        f"{10**0.3:.2f}",
    ],
)
plt.xscale("log")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid-beta-only-2.pgf", format="pgf"
)
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
    figsize=set_size(full, height=0.9),
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
    figsize=set_size(full, height=0.9),
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
        vmin=-0.6,
        vmax=0.6,
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
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid7.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

plt.xlabel("Star mass $M_\\odot$")
plt.ylabel("$P / P_i$")


for q_i in np.linspace(0.125, 1.125, 3):
    alpha = 1e-10
    beta = 1e-10
    qs = np.linspace(q_i, q_f(q_i, 2, 0.6, alpha, beta, 0), 100)
    wind = CombineEvolve(alpha=alpha, beta=beta, delta=0, gamma=0, qs=qs)
    plt.plot(
        wind.M_d,
        wind.P_over_P0,
        label=f"$\\alpha={alpha:.2f}$, $\\beta = {beta:.2f}$,  $q_i={q_i}$",
    )

    alpha = 0.25
    beta = 0.25
    qs = np.linspace(q_i, q_f(q_i, 2, 0.6, alpha, beta, 0), 100)
    wind = CombineEvolve(alpha=alpha, beta=beta, delta=0, gamma=0, qs=qs)
    plt.plot(
        wind.M_d,
        wind.P_over_P0,
        label=f"$\\alpha={alpha:.2f}$, $\\beta = {beta:.2f}$,  $q_i={q_i}$",
    )

    alpha = 0.5
    beta = 0.5 - 1e-10
    qs = np.linspace(q_i, q_f(q_i, 2, 0.6, alpha, beta, 0), 100)
    wind = CombineEvolve(alpha=alpha, beta=beta, delta=0, gamma=0, qs=qs)
    plt.plot(
        wind.M_d,
        wind.P_over_P0,
        label=f"$\\alpha={alpha:.2f}$, $\\beta = {beta:.2f}$,  $q_i={q_i}$",
    )


fig.legend(loc="outside upper center", ncols=2)
axs.set_yscale("log")
axs.invert_xaxis()

axs.tick_params(axis="y", which="minor", length=0)
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-2.pgf", format="pgf")
plt.show()
plt.close()

# %%
import matplotlib as mpl

# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


plt.xlabel(r"$\log(q)$")
plt.ylabel("$\\alpha$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, 150)
qs = np.linspace(-1.5, 1.5, 150)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(alphas)), 10.0)

for i, q in enumerate(qs):
    for j, alpha in enumerate(alphas):

        # alpha_test = RL / RLs[-1] * 0.45
        qs = np.linspace(10**q, q_f(10**q, 2, 0.6, alpha, beta_test, delta_test), 100)
        orbit = CombineEvolve(alpha, beta_test, delta_test, 0, qs)
        qs_ref = np.linspace(10**q, q_f(10**q, 2, 0.6, 1e-10, 0, 0), 100)
        orbit_ref = CombineEvolve(1e-10, beta_test, delta_test, 0, qs_ref)
        val = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]
        Z[i, j] = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]


mesh = plt.pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap=cmap,
    shading="auto",
    rasterized=True,
    norm=mpl.colors.LogNorm(vmin=0.05, vmax=20),
)

plt.colorbar(mesh, label="$a_{\\textrm{f},\\alpha} / a_{\\textrm{f},0}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid11.pgf", format="pgf")
plt.show()
plt.close()


# %%

import matplotlib as mpl

# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"$\log(q)$")
plt.ylabel("$\\alpha$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 0.5 - 1e-10, 200)
qs = np.linspace(-1.5, 1.5, 200)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(alphas)), 10.0)

for i, q in enumerate(qs):
    for j, alpha in enumerate(alphas):

        # alpha_test = RL / RLs[-1] * 0.45
        beta = alpha
        qs = np.linspace(10**q, q_f(10**q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)
        qs_ref = np.linspace(10**q, q_f(10**q, 2, 0.6, 1e-10, 0, 0), 100)
        orbit_ref = CombineEvolve(1e-10, 0, delta_test, 0, qs_ref)
        val = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]
        if val < 0.05:
            Z[i, j] = np.nan
            continue
        Z[i, j] = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]


mesh = plt.pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap=cmap,
    shading="auto",
    rasterized=True,
    norm=mpl.colors.LogNorm(vmin=0.05, vmax=20),
)

plt.colorbar(mesh, label="$a_{\\textrm{f},\\alpha} / a_{\\textrm{f},0}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid12.pgf", format="pgf")
plt.show()
plt.close()


# %%


import matplotlib as mpl

qmin = -1
qmax = 0.5
num = 500

vmin = 0.333333
vmax = 3

cmap = "RdBu_r"
# reproduce the grid like figures using q and RL.

fig, axs = plt.subplots(
    1, 3, sharex=True, figsize=set_size(full, height=0.38), constrained_layout=True
)

axs[0].set_xlabel(r"$\log(q_\textrm{i})$")
axs[1].set_xlabel(r"$\log(q_\textrm{i})$")
axs[2].set_xlabel(r"$\log(q_\textrm{i})$")
axs[0].set_ylabel("$\\alpha$")
axs[1].set_ylabel("$\\alpha = \\beta$")
axs[2].set_ylabel("$\\beta$")


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, num)
qs = np.linspace(qmin, qmax, num)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])

# ---

Z = np.full((len(qs), len(alphas)), 10.0)

for i, q in enumerate(qs):
    for j, alpha in enumerate(alphas):

        # alpha_test = RL / RLs[-1] * 0.45
        beta = 0
        qs = np.linspace(10**q, q_f(10**q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)
        qs_ref = np.linspace(10**q, q_f(10**q, 2, 0.6, 1e-10, 0, 0), 100)
        orbit_ref = CombineEvolve(1e-10, 0, delta_test, 0, qs_ref)
        val = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]
        if val < 0.05:
            Z[i, j] = np.nan
            continue
        Z[i, j] = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]


mesh = axs[0].pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap=cmap,
    shading="auto",
    rasterized=True,
    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
)

# ---

alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, num)
qs = np.linspace(qmin, qmax, num)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(alphas)), 10.0)

mesh = axs[1].pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap="Greys",
    shading="auto",
    rasterized=True,
    vmin=0,
    vmax=40,
)

for i, q in enumerate(qs):
    for j, alpha in enumerate(alphas):
        if alpha > 0.5 - 1e-10:
            Z[i, j] = np.nan
            continue

        # alpha_test = RL / RLs[-1] * 0.45
        beta = alpha
        qs = np.linspace(10**q, q_f(10**q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)
        qs_ref = np.linspace(10**q, q_f(10**q, 2, 0.6, 1e-10, 0, 0), 100)
        orbit_ref = CombineEvolve(1e-10, 0, delta_test, 0, qs_ref)
        val = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]
        # if val < vmin:
        #     Z[i, j] = np.nan
        #     continue
        Z[i, j] = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]


mesh = axs[1].pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap=cmap,
    shading="auto",
    rasterized=True,
    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
)

# ---

alpha_test = 0 + 1e-10
beta_test = 0.0
delta_test = 0

alphas = np.linspace(1e-10, 1 - 1e-10, num)
qs = np.linspace(qmin, qmax, num)

dalpha = np.diff(alphas)
alpha_edges = np.concatenate(
    [
        [alphas[0] - dalpha[0] / 2],
        alphas[:-1] + dalpha / 2,
        [alphas[-1] + dalpha[-1] / 2],
    ]
)

dq = np.diff(qs)
q_edges = np.concatenate([[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]])


Z = np.full((len(qs), len(alphas)), 10.0)

for i, q in enumerate(qs):
    for j, alpha in enumerate(alphas):

        # alpha_test = RL / RLs[-1] * 0.45
        beta = alpha
        alpha = 0
        qs = np.linspace(10**q, q_f(10**q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)
        qs_ref = np.linspace(10**q, q_f(10**q, 2, 0.6, 1e-10, 0, 0), 100)
        orbit_ref = CombineEvolve(1e-10, 0, delta_test, 0, qs_ref)
        val = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]
        # if val < vmin:
        #     Z[i, j] = np.nan
        #     continue
        Z[i, j] = orbit.a_over_a0[-1] / orbit_ref.a_over_a0[-1]


mesh = axs[2].pcolormesh(
    q_edges,
    alpha_edges,
    Z.T,
    cmap=cmap,
    shading="auto",
    rasterized=True,
    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
)

norm = mpl.colors.Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))

plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    label="$\log(a_{\\textrm{f},\\alpha} / a_{\\textrm{f},0})$",
    ax=axs[1],
    orientation="horizontal",
    location="top",
    extend="both",
)

cmap = mpl.cm.bwr
bounds = [-1, 15]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    label="$\\beta = 0$",
    ax=axs[0],
    orientation="horizontal",
    location="top",
    ticks=[],
    aspect=100,
    pad=0.05,
    shrink=0,
)
cb.outline.set_visible(False)
cb.ax.tick_params(size=0)

cb = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    label="$\\alpha = 0$",
    ax=axs[2],
    orientation="horizontal",
    location="top",
    ticks=[],
    aspect=100,
    pad=0.05,
    shrink=0,
)
cb.outline.set_visible(False)
cb.ax.tick_params(size=0)

axs[1].text((qmin + qmax) / 2, 0.75, r"$\alpha = \beta$", ha="center", va="center")


plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid12.pgf", format="pgf")
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

            ref_qs = np.linspace(q, q_f(q, 2, 0.6, 1e-10, 1e-10, delta_test), 100)

            ref_orbit = CombineEvolve(1e-10, 1e-10, delta_test, 0, ref_qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = np.log10(orbit.a_over_a0[-1] / ref_orbit.a_over_a0[-1])

    mesh = axs[a].pcolormesh(
        alpha_edges,
        beta_edges,
        Z,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        vmin=-0.95 * np.nanmax([-np.nanmin(Z), np.nanmax(Z)]),
        vmax=0.95 * np.nanmax([-np.nanmin(Z), np.nanmax(Z)]),
    )

    plt.colorbar(
        mesh,
        label=r"$a_{\textrm{f,}\alpha,\beta}/a_\textrm{f, cons}$",
        ax=axs[a],
        orientation="horizontal",
        extend="both",
        format=lambda x, _: f"{10**x:.2f}",
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
    "/home/koen/LaTeX-setup/plots/w12-analytic-grid15.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

axs[1].set_xlabel("Star mass $M_\\odot$")
axs[0].set_ylabel("Roche lobe radius ($R_\\odot$)")
axs[1].set_ylabel("$P/ P_\\textrm{i}$")

alpha = 0
beta = 0
delta = 1 - 1e-10
q = 0.5

for delta in [1e-10, 0.2, 0.4, 0.6, 0.8, 1 - 1e-10]:
    qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
    ring = CombineEvolve(
        alpha=0,
        beta=0.0,
        delta=delta,
        gamma=1.5,
        qs=qs,
    )
    axs[0].plot(ring.M_d, rl(get_separation(450, 0.5) * ring.a_over_a0, qs))
    axs[0].text(
        ring.M_d[-1] - 0.05,
        rl(get_separation(450, 0.5) * ring.a_over_a0, qs)[-1],
        rf"$\delta = {delta:.1f}$",
        ha="left",
        va="center",
    )
    axs[1].plot(ring.M_d, ring.P_over_P0)
    axs[1].text(
        ring.M_d[-1] - 0.05,
        ring.P_over_P0[-1],
        rf"$\delta = {delta:.1f}$",
        ha="left",
        va="center",
    )


axs[-1].invert_xaxis()
axs[0].invert_xaxis()
axs[0].set_xlim(2.05, 0.3)
axs[0].set_ylim(0.4, 1000)
axs[1].set_ylim(7e-5, 6)
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-ring.pgf", format="pgf")
plt.show()
plt.close()
# %%


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


positions = [
    [
        (np.float64(428.06901989756756), np.float64(0.7085373579340621)),
        (np.float64(269.1736876696436), np.float64(0.5432892764895114)),
        (np.float64(170.49890575413892), np.float64(0.42594204439488637)),
    ],
    [
        (np.float64(485.0383783711634), np.float64(0.7745059724950553)),
        (np.float64(317.47443473844396), np.float64(0.5997727186811518)),
        (np.float64(216.71515712060773), np.float64(0.4658965605264783)),
    ],
    [
        (np.float64(542.5558387269865), np.float64(0.8503847536269648)),
        (np.float64(383.62762833015165), np.float64(0.6508319539606479)),
        (np.float64(281.69665536959235), np.float64(0.5064695039927067)),
        (np.float64(173.09440481685044), np.float64(0.4409670308748386)),
    ],
    [
        (np.float64(607.2153810699735), np.float64(0.9275658613078355)),
        (np.float64(422.91634322922437), np.float64(0.7406391229950012)),
        (np.float64(311.09207151109104), np.float64(0.5947977554374118)),
        (np.float64(232.34058402309154), np.float64(0.4878628821254416)),
        (np.float64(167.85899909449566), np.float64(0.4135071755609904)),
    ],
    [
        (np.float64(464.6829960167138), np.float64(0.8335984695972181)),
        (np.float64(361.4229744382027), np.float64(0.6732436729425755)),
        (np.float64(291.15845969402733), np.float64(0.554110570631264)),
        (np.float64(235.1783093390419), np.float64(0.4671688214825733)),
        (np.float64(163.54760493738547), np.float64(0.41748706831971727)),
    ],
    [
        (np.float64(530.0680458489131), np.float64(0.9145815910758792)),
        (np.float64(414.46357270834403), np.float64(0.7582880834085093)),
        (np.float64(351.6202120483639), np.float64(0.6319406662275726)),
        (np.float64(290.8237282502163), np.float64(0.5441887970816868)),
        (np.float64(248.04565970509668), np.float64(0.4742432519292394)),
        (np.float64(197.0578152488714), np.float64(0.42558876506338283)),
    ],
]

# reproduce the grid like figures using q and RL.

plt.rcParams["contour.negative_linestyle"] = "solid"
fig, axss = plt.subplots(
    2, 3, sharex=True, sharey=True, figsize=set_size(full), constrained_layout=True
)

axs = axss.flatten()
fig.supxlabel("Roche lobe radius ($R_\\odot$)", size=11)
fig.supylabel("$q$", size=11)


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


Zmin = 1e99
Zmax = -1e-99
for c, delta_test in enumerate([1e-10, 0.2, 0.4, 0.6, 0.8, 1 - 1e-10]):
    alpha_test = 0
    beta_test = 0.0
    gamma = 1.5

    RLs = np.logspace(np.log10(150), np.log10(650), 150)
    qs = np.linspace(0.4, 1, 150)

    logR = np.log10(RLs)
    dlogR = np.diff(logR)

    logR_edges = np.concatenate(
        [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
    )
    R_edges = 10**logR_edges

    dq = np.diff(qs)
    q_edges = np.concatenate(
        [[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]]
    )

    Z = np.full((len(qs), len(RLs)), 10.0)

    for i, q in enumerate(qs):
        for j, RL in enumerate(RLs):

            # alpha_test = RL / RLs[-1] * 0.45
            qs = np.linspace(q, q_f(q, 2, 0.6, alpha_test, beta_test, delta_test), 100)
            orbit = CombineEvolve(alpha_test, beta_test, delta_test, gamma, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = np.log10(
                get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
            )
    if np.nanmin(Z) < Zmin:
        Zmin = np.nanmin(Z)

    if np.nanmax(Z) > Zmax:
        Zmax = np.nanmax(Z)

for c, delta_test in enumerate([1e-10, 0.2, 0.4, 0.6, 0.8, 1 - 1e-10]):
    alpha_test = 0
    beta_test = 0.0
    gamma = 1.5

    RLs = np.logspace(np.log10(150), np.log10(650), 150)
    qs = np.linspace(0.4, 1, 150)
    Q = np.linspace(0.4, 1, 150)

    logR = np.log10(RLs)
    dlogR = np.diff(logR)

    logR_edges = np.concatenate(
        [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
    )
    R_edges = 10**logR_edges

    dq = np.diff(qs)
    q_edges = np.concatenate(
        [[qs[0] - dq[0] / 2], qs[:-1] + dq / 2, [qs[-1] + dq[-1] / 2]]
    )

    Z = np.full((len(qs), len(RLs)), 10.0)

    for i, q in enumerate(qs):
        for j, RL in enumerate(RLs):

            # alpha_test = RL / RLs[-1] * 0.45
            qs = np.linspace(q, q_f(q, 2, 0.6, alpha_test, beta_test, delta_test), 100)
            orbit = CombineEvolve(alpha_test, beta_test, delta_test, gamma, qs)

            # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
            Z[i, j] = np.log10(
                get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
            )
    mesh = axs[c].pcolormesh(
        R_edges,
        q_edges,
        Z,
        cmap="viridis",
        shading="auto",
        rasterized=True,
        vmin=Zmin,
        vmax=Zmax,
    )

    X, Y = np.meshgrid(RLs, Q)

    if c < 3:
        axs[c].text(
            0.1,
            0.9,
            f"$\\delta= {delta_test:.1f}$",
            transform=axs[c].transAxes,
            ha="left",
            va="top",
            color="k",
        )
        CS = axs[c].contour(
            X,
            Y,
            Z,
            levels=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
            colors="k",
            alpha=0.2,
        )

        # Define a nice function of distance from individual pts
        def f(x, y, pts):
            z = np.zeros_like(x)
            for p in pts:
                z = z + 1 / (np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2))
            return 1 / z

        CL = plt.clabel(CS, manual=positions[c], fontsize=9)

    else:
        axs[c].text(
            0.1,
            0.9,
            f"$\\delta= {delta_test:.1f}$",
            transform=axs[c].transAxes,
            ha="left",
            va="top",
            color="w",
        )
        CS = axs[c].contour(
            X,
            Y,
            Z,
            levels=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
            colors="w",
            alpha=0.2,
        )
        CL = plt.clabel(CS, manual=positions[c], fontsize=9)


plt.colorbar(mesh, label="$\log(\\textrm{Period / days})$ ", ax=axss[:, 2])
plt.savefig("/home/koen/LaTeX-setup/plots/w12-analytic-grid-delta.pgf", format="pgf")
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


RLs = np.logspace(np.log10(150), np.log10(650), 150)
qs = np.linspace(0.4, 1, 150)
QS = np.linspace(0.4, 1, 150)

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
        term1 = np.min([(q / QS[-1]) ** 1, 1])
        term2 = np.min([(RL / RLs[-3]), 1]) ** 1.3
        alpha = term1 * term2**2 * 0.3 + 1e-10 + 0.1
        beta = alpha * 0.3

        # alpha_test = RL / RLs[-1] * 0.45
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

        # Z[i, j] = orbit.a_over_a0[-1]
        Z[i, j] = get_period(
            get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1]
        )

mesh = plt.pcolormesh(
    R_edges, q_edges, Z, cmap="viridis", shading="auto", rasterized=True
)

plt.colorbar(mesh, label="Period (days)")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-match-mesa-1.pgf", format="pgf")
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


RLs = np.logspace(np.log10(150), np.log10(650), 150)
qs = np.linspace(0.4, 1, 150)
QS = np.linspace(0.4, 1, 150)

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
        term1 = np.min([(q / QS[-1]) ** 1, 1])
        term2 = np.min([(RL / RLs[-3]), 1]) ** 1.3
        alpha = term1 * term2**2 * 0.3 + 1e-10 + 0.1
        beta = alpha * 0.3

        # alpha_test = RL / RLs[-1] * 0.45
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta_test), 100)
        orbit = CombineEvolve(alpha, beta, delta_test, 0, qs)

        # Z[i, j] = orbit.a_over_a0[-1]
        Z[i, j] = (orbit.M_a[-1] - orbit.M_a[0]) / (orbit.M_d[0] - orbit.M_d[-1])

mesh = plt.pcolormesh(
    R_edges, q_edges, Z, cmap="viridis", shading="auto", rasterized=True
)

plt.colorbar(mesh, label=r"$- \Delta M_\textrm{a} / \delta M_\textrm{d}$")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w12-match-mesa-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
