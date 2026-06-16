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

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

# %%

cons = MesaGrid(f"{MASTER}tides-grid-6")

# %%

cct = MesaGrid(f"{MASTER}tides-grid-8")

# %%

non_cons = MesaGrid(f"{MASTER}delta0.5_beta0.5")

# %%
from matplotlib import ticker

fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=1 / 3), constrained_layout=True
)

Z_min = 1e99
Z_max = -1e99
for i, grid in enumerate([cons, cct, non_cons]):
    for R, q, model in grid.iter_models():
        if model.env_mass[-1] > 0.1:
            continue
        else:
            if Z_min > model.star.period_days[-1]:
                Z_min = model.star.period_days[-1]
            if Z_max < model.star.period_days[-1]:
                Z_max = model.star.period_days[-1]


for i, grid in enumerate([cons, cct, non_cons]):
    ax = axs[i]
    ax.set_title(titles[i])
    R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
    q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

    Z = np.full((len(q_vals), len(R_vals)), np.nan)
    mask_bad = np.zeros_like(Z)

    for R, q, model in grid.iter_models():
        i = np.where(q_vals == q)[0][0]
        j = np.where(R_vals == R)[0][0]

        if model.env_mass[-1] > 0.1:
            if model.star.period_days[-1] < 50:
                mask_bad[i, j] = 0.9
            elif model.star.model_number[-1] < 500:
                mask_bad[i, j] = 0.5
            else:
                mask_bad[i, j] = 0.1
        else:
            Z[i, j] = np.log10(model.star.period_days[-1])

    logR = np.log10(R_vals)
    dlogR = np.diff(logR)

    logR_edges = np.concatenate(
        [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
    )
    R_edges = 10**logR_edges

    dq = np.diff(q_vals)
    q_edges = np.concatenate(
        [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
    )

    print(mask_bad)
    ax.pcolormesh(
        R_edges,
        q_edges,
        mask_bad,
        shading="auto",
        cmap="PiYG",
        vmin=0,
        vmax=1,
        alpha=0.4,
    )
    mesh = ax.pcolormesh(
        R_edges,
        q_edges,
        Z,
        cmap="viridis",
        shading="auto",
        vmin=np.log10(Z_min),
        vmax=np.log10(Z_max),
    )

    ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")

    ax.set_xscale("log")
    ax.set_xticks([150, 200, 300, 400, 600])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


axs[0].set_ylabel("$q$")
plt.colorbar(mesh, label=r"$\log (P/\textrm{days})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-grid-non-cons.pgf", format="pgf")
plt.show()
plt.close()


# %%


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
)

fig.update_layout(
    autosize=True,
    title="",
)


names = [
    "Conservative",
    "Circumbinary ring",
    "Non-conservative",
]

colorscales = [
    "Blues",
    "Reds",
    "Greens",
]

for grid, name, colorscale in zip(
    [cons, cct, non_cons],
    names,
    colorscales,
):

    R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
    q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

    Z = np.full((len(q_vals), len(R_vals)), np.nan)

    for R, q, model in grid.iter_models():
        i = np.where(q_vals == q)[0][0]
        j = np.where(R_vals == R)[0][0]

        if model.env_mass[-1] <= 0.1:
            Z[i, j] = model.star.period_days[-1]

    X, Y = np.meshgrid(np.log10(R_vals), q_vals)

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            name=name,
            colorscale=colorscale,
            opacity=0.65,
            showscale=False,
            showlegend=True,
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=np.log10(Z),
            name=name,
            colorscale=colorscale,
            opacity=0.65,
            showscale=False,
            legendgroup=name,
        ),
        row=1,
        col=2,
    )

fig.update_layout(
    scene=dict(
        aspectmode="cube",
        yaxis_title="q",
        xaxis_title="Roche lobe (Rsun)",
        zaxis_title="period (days)",
    )
)
fig.update_layout(
    scene2=dict(
        aspectmode="cube",
        yaxis_title="q",
        xaxis_title="Roche lobe (Rsun)",
        zaxis_title="log period (days)",
    )
)

fig.write_html(
    "/home/koen/figures/plots/master-internship/w20/periods.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)

fig.show()

# %%
# %%


fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=1 / 3), constrained_layout=True
)

titles = [
    "Fully conservative",
    "$\\delta = 0.5$, $\\beta = 0$",
    "$\\delta = 0.5$, $\\beta = 0.5$",
]

Z_min = 1e99
Z_max = -1e99
for i, grid in enumerate([cons, cct, non_cons]):
    for R, q, model in grid.iter_models():
        if model.env_mass[-1] > 0.1:
            continue
        else:
            m_initial = q * 2
            m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
            m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

            dt = np.diff(model.star.star_age)
            dm = np.diff(model.star.star_2_mass)

            Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
            Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
            M_c_transfer = np.cumsum(Xc * dm)
            M_o_transfer = np.cumsum(Xo * dm)
            M_c = m_C_initial + M_c_transfer
            M_o = m_O_initial + M_o_transfer
            Xc_final = M_c / model.star.star_2_mass[1:]
            Xo_final = M_o / model.star.star_2_mass[1:]

            Z = Xc_final[-1] / Xo_final[-1] * 16 / 12

            if Z_min > Z:
                Z_min = Z
            if Z_max < Z:
                Z_max = Z


divnorm = mpl.colors.TwoSlopeNorm(vmin=Z_min, vcenter=1, vmax=Z_max)

for i, grid in enumerate([cons, cct, non_cons]):
    ax = axs[i]
    ax.set_title(titles[i])
    R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
    q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

    Z = np.full((len(q_vals), len(R_vals)), np.nan)
    mask_bad = np.zeros_like(Z)

    for R, q, model in grid.iter_models():
        i = np.where(q_vals == q)[0][0]
        j = np.where(R_vals == R)[0][0]

        if model.env_mass[-1] > 0.1:
            if model.star.period_days[-1] < 50:
                mask_bad[i, j] = 0.9
            elif model.star.model_number[-1] < 500:
                mask_bad[i, j] = 0.5
            else:
                mask_bad[i, j] = 0.1
        else:
            m_initial = q * 2
            m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
            m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

            dt = np.diff(model.star.star_age)
            dm = np.diff(model.star.star_2_mass)

            Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
            Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
            M_c_transfer = np.cumsum(Xc * dm)
            M_o_transfer = np.cumsum(Xo * dm)
            M_c = m_C_initial + M_c_transfer
            M_o = m_O_initial + M_o_transfer
            Xc_final = M_c / model.star.star_2_mass[1:]
            Xo_final = M_o / model.star.star_2_mass[1:]

            Z[i, j] = Xc_final[-1] / Xo_final[-1] * 16 / 12

    logR = np.log10(R_vals)
    dlogR = np.diff(logR)

    logR_edges = np.concatenate(
        [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
    )
    R_edges = 10**logR_edges

    dq = np.diff(q_vals)
    q_edges = np.concatenate(
        [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
    )

    print(mask_bad)
    ax.pcolormesh(
        R_edges,
        q_edges,
        mask_bad,
        shading="auto",
        cmap="PiYG",
        vmin=0,
        vmax=1,
        alpha=0.4,
    )
    mesh = ax.pcolormesh(
        R_edges,
        q_edges,
        Z,
        cmap="coolwarm",
        shading="auto",
        norm=divnorm,
    )

    ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")

    ax.set_xscale("log")
    ax.set_xticks([150, 200, 300, 400, 600])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


axs[0].set_ylabel("$q$")
cmap = plt.colorbar(mesh, label=r"C/O-number ratio")
cmap.ax.set_yscale("linear")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-grid-non-cons-CO.pgf", format="pgf")
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


def quad2d(X, a, b, c, d, e, f):
    R, q = X
    return a * R**2 + b * q**2 + c * R * q + d * R + e * q + f


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

    def a_over_a0(self, q=None):
        # table 5 soberman1997
        if q == None:
            q = self.qs
        term_1 = (q / self.q0) ** (2 * self.curA - 2)
        term_2 = ((1 + q) / (1 + self.q0)) ** (1 - 2 * self.curB)
        term_3 = ((1 + self.epsilon * q) / (1 + self.epsilon * self.q0)) ** (
            2 * self.curC + 3
        )
        return term_1 * term_2 * term_3

    def da_over_a0_dq(self, q=None):
        if q == None:
            q = self.qs

        f = self.a_over_a0()

        n1 = 2 * self.curA - 2
        n2 = 1 - 2 * self.curB
        n3 = 2 * self.curC + 3

        dlogf = n1 / q + n2 / (1 + q) + n3 * self.epsilon / (1 + self.epsilon * q)

        return f * dlogf

    def gprime(self, q):
        eps = self.epsilon
        n1 = 2 * self.curA - 2
        n2 = 1 - 2 * self.curB
        n3 = 2 * self.curC + 3

        gprime = -n1 / q**2 - n2 / (1 + q) ** 2 - n3 * eps**2 / (1 + eps * q) ** 2
        return gprime > 0

    @property
    def stationary_points(self):

        n1 = 2 * self.curA - 2
        n2 = 1 - 2 * self.curB
        n3 = 2 * self.curC + 3

        eps = self.epsilon

        a = eps * (n1 + n2 + n3)
        b = n1 * (1 + eps) + n2 + n3 * eps
        c = n1

        disc = b**2 - 4 * a * c

        if disc < 0:
            argmin = np.argmin(self.a_over_a0())
            return self.qs[argmin]

        sqrt_disc = np.sqrt(disc)

        q1 = (-b + sqrt_disc) / (2 * a)
        q2 = (-b - sqrt_disc) / (2 * a)
        roots = [q for q in [q1, q2] if q > 0]
        if len(roots) == 0:
            argmin = np.argmin(self.a_over_a0())
            return self.qs[argmin]
        roots = np.sort(roots)
        for root in roots:
            if self.gprime(root):
                argmin = np.argmin(self.a_over_a0())
                return np.max([root, self.qs[argmin]])

        argmin = np.argmin(self.a_over_a0())
        return self.qs[argmin]

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


def get_period(a, M):
    return np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * M * (Msun))) / 3600 / 24


# %%

import matplotlib as mpl
from matplotlib.cm import ScalarMappable

alpha = 0
beta = 0
gamma = 1
RL = 600
Rsun = 600
n = 100
colormap = plt.cm.inferno(np.linspace(0, 1, 1000001))

fig, axs = plt.subplots(
    3, 3, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

fig.suptitle("$\\alpha=0$", size=10)

deltas = np.linspace(1e-10, 1 - 1e-10, 100)
for a, ax in enumerate(axs.flatten()):

    q = a * 0.1 + 0.2
    for b, beta in enumerate(np.linspace(1e-10, 1 - 1e-10, n)):

        a_ratios = []
        for d, delta in enumerate(deltas):

            if delta + beta > 0.999:
                break

            qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs, A=10)
            a_ratios.append(orbit.P_over_P0[-1])

        length = len(a_ratios)
        ax.plot(
            deltas[:length], a_ratios, color=colormap[int(beta * 1e6)], rasterized=True
        )
    ax.text(
        0.5,
        0.9,
        f"$q_\\textrm{{i}} = {q:.1f}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )

plt.colorbar(
    ScalarMappable(cmap=plt.cm.inferno), ax=axs[:, :], label="$\\beta$", aspect=30
)
fig.supxlabel("$\\delta$", size=10)
fig.supylabel("$a_\\textrm{f}/a_\\textrm{i}$", size=10)
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w20-beta.pgf", format="pgf")
plt.show()
plt.close()


# %%
grids = [cons, cct, non_cons]
# %%
for grid in grids:
    for i, (R, q, model) in enumerate(grid.iter_models()):
        if i != 30:
            continue

        # plt.plot(model.age, model.star.R)
        # plt.plot(model.age, model.star.rl_1)
        plt.plot(model.age, model.star.star_2_mass)
        plt.plot(model.age, model.star.star_1_mass)

plt.show()
# %%

alpha = 0
delta = 0.5
q = 0.5

for beta in np.linspace(0, 0.49, 10):
    print(beta)
    qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
    orbit = CombineEvolve(alpha, beta, delta, gamma, qs, A=10)
    plt.plot(qs, orbit.P_over_P0)

plt.show()

# %%
delta = 0.5
gamma = 1

beta = 0
q = 0.5


alpha = 0
beta = 0
qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
plt.plot(qs, orbit.P_over_P0)


alpha = 0
beta = 0.5 - 1e-10
qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
plt.plot(qs, orbit.P_over_P0)


plt.gca().invert_xaxis()
plt.show()


# %%
