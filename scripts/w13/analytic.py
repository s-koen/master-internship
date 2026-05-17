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


plt.rcParams["contour.negative_linestyle"] = "solid"
fig, axss = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

fig.supxlabel("Roche lobe radius ($R_\\odot$)", size=11)
fig.supylabel("$q$", size=11)


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


Zmin = 1e99
Zmax = -1e-99
delta_test = 0.2
alpha_test = 0
beta_test = 0.0
gamma = 1.25

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
        orbit = CombineEvolve(alpha_test, beta_test, delta_test, gamma, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = np.log10(
            get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
        )
if np.nanmin(Z) < Zmin:
    Zmin = np.nanmin(Z)

if np.nanmax(Z) > Zmax:
    Zmax = np.nanmax(Z)

mesh = plt.pcolormesh(
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
CS = plt.contour(
    X,
    Y,
    Z,
    levels=[
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
    colors="w",
    alpha=0.2,
)

plt.colorbar(
    mesh,
    label="Period  (days)",
    format=lambda x, _: f"{10**x:.0f}",
    ticks=[
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
)
plt.savefig("/home/koen/LaTeX-setup/plots/w13-analytic-grid-delta.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.rcParams["contour.negative_linestyle"] = "solid"
fig, axss = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

fig.supxlabel("Roche lobe radius ($R_\\odot$)", size=11)
fig.supylabel("$q$", size=11)


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


Zmin = 1e99
Zmax = -1e-99
delta_test = 0.1
alpha_test = 0
beta_test = 0.0
gamma = 1.25

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
        orbit = CombineEvolve(alpha_test, beta_test, delta_test, gamma, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = np.log10(
            get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
        )
if np.nanmin(Z) < Zmin:
    Zmin = np.nanmin(Z)

if np.nanmax(Z) > Zmax:
    Zmax = np.nanmax(Z)

mesh = plt.pcolormesh(
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
CS = plt.contour(
    X,
    Y,
    Z,
    levels=[
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
    colors="w",
    alpha=0.2,
)

plt.colorbar(
    mesh,
    label="$\log(\\textrm{Period / days})$ ",
    format=lambda x, _: f"{10**x:.0f}",
    ticks=[
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
)
# plt.savefig("/home/koen/LaTeX-setup/plots/w13-analytic-grid-delta.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.rcParams["contour.negative_linestyle"] = "solid"
fig, axss = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

fig.supxlabel("Roche lobe radius ($R_\\odot$)", size=11)
fig.supylabel("$q$", size=11)


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


Zmin = 1e99
Zmax = -1e-99
delta_test = 0.5
alpha_test = 0
beta_test = 0.0
gamma = 1.0

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
        orbit = CombineEvolve(alpha_test, beta_test, delta_test, gamma, qs)

        # Z[i, j] = get_separation(RL, q) * orbit.a_over_a0[-1]
        Z[i, j] = np.log10(
            get_period(get_separation(RL, q) * orbit.a_over_a0[-1], orbit.M_tot[-1])
        )
if np.nanmin(Z) < Zmin:
    Zmin = np.nanmin(Z)

if np.nanmax(Z) > Zmax:
    Zmax = np.nanmax(Z)

mesh = plt.pcolormesh(
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
CS = plt.contour(
    X,
    Y,
    Z,
    levels=[
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
    colors="w",
    alpha=0.2,
)

plt.colorbar(
    mesh,
    label="$\log(\\textrm{Period / days})$ ",
    format=lambda x, _: f"{10**x:.0f}",
    ticks=[
        np.log10(100),
        np.log10(250),
        np.log10(500),
        np.log10(1000),
        np.log10(2500),
        np.log10(5000),
        np.log10(9500),
    ],
)
# plt.savefig("/home/koen/LaTeX-setup/plots/w13-analytic-grid-delta.pgf", format="pgf")
plt.show()
plt.close()


# %%
