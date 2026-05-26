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
            return []

        sqrt_disc = np.sqrt(disc)

        q1 = (-b + sqrt_disc) / (2 * a)
        q2 = (-b - sqrt_disc) / (2 * a)
        roots = [q for q in [q1, q2] if q > 0]
        for root in roots:
            if self.gprime(root):
                return root

        return self.qs[-1]

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

alpha = 0
beta = 0
delta = 0.3
gamma = 1.25
q = 0.5
RL = 600
Rsun = 600

qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
# %%

# %%

grid_max = MesaGrid(f"{MASTER}/tides-grid-7")

# %%

grid_min = MesaGrid(f"{MASTER}/tides-grid-4")

# %%

grid_none = MesaGrid(f"{MASTER}/tides-grid-6")

# %%


for i, (R, q, model) in enumerate(grid_max.get_R1_index(-2)):
    plt.plot(model.env_mass, model.star.R / model.star.binary_separation)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

grids = [grid_none, grid_min, grid_max]

Z_min = 1e99
Z_max = -1
for grid, ax in zip(grids, axs):

    R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
    q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

    Z = np.full((len(q_vals), len(R_vals)), np.nan)
    mask_bad = np.zeros_like(Z, dtype=bool)

    for R, q, model in grid.iter_models():
        print(R)
        i = np.where(q_vals == q)[0][0]
        j = np.where(R_vals == R)[0][0]

        if model.env_mass[-1] > 0.1:
            mask_bad[i, j] = True
        else:
            Z[i, j] = np.max(model.star.R) / RL
        if model.star.period_days[-1] < 50:
            mask_bad[i, j] = True
            Z[i, j] = np.nan
    if np.nanmin(Z) < Z_min:
        Z_min = np.nanmin(Z)
    if np.nanmax(Z) > Z_max:
        Z_max = np.nanmax(Z)


for grid, ax in zip(grids, axs):

    R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
    q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

    Z = np.full((len(q_vals), len(R_vals)), np.nan)
    mask_bad = np.zeros_like(Z, dtype=bool)

    for R, q, model in grid.iter_models():
        print(R)
        i = np.where(q_vals == q)[0][0]
        j = np.where(R_vals == R)[0][0]

        if model.env_mass[-1] > 0.1:
            mask_bad[i, j] = True
        else:
            Z[i, j] = np.max(model.star.R) / RL
        if model.star.period_days[-1] < 50:
            mask_bad[i, j] = True
            Z[i, j] = np.nan

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

    overlay = np.where(
        mask_bad,
        1,
        0,
    )

    ax.pcolormesh(
        R_edges,
        q_edges,
        overlay,
        shading="auto",
        cmap="Greys",
        alpha=0.3,
        vmin=Z_min,
        vmax=Z_max,
    )
    mesh = ax.pcolormesh(
        R_edges,
        q_edges,
        Z,
        cmap="viridis",
        shading="auto",
    )

    ax.set_xscale("log")

plt.colorbar(
    mesh,
    ax=axs,
    label=r"$\textrm{max}(R_\textrm{star}) / R_\textrm{RL,initial}$",
    orientation="horizontal",
    location="top",
    aspect=50,
)
axs[1].set_xlabel(r"Roche lobe radius ($R_\odot$)")
axs[0].set_ylabel(r"$q = M_\textrm{acc} / M_\textrm{donor}$")
axs[0].set_title(r"$\delta=0$")
axs[1].set_title(r"$\delta=0.2$, $\gamma=1.25$")
axs[2].set_title(r"$\delta=0.5$, $\gamma=1.0$")
plt.savefig("/home/koen/LaTeX-setup/plots/w17-grid-strong-ratio.pgf", format="pgf")
plt.show()
plt.close()

# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid_min.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid_min.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid_min.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


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

fig, ax = plt.subplots(1, 1, sharex=True, constrained_layout=True)

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="viridis",
    shading="auto",
)

plt.colorbar(
    mesh,
)
ax.set_xscale("log")

plt.show()
plt.close()
# %%

import pickle
import numpy as np
from scipy.optimize import curve_fit

# Build coordinate grids
RR, QQ = np.meshgrid(R_vals, q_vals)
print(QQ)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

print("coefficients:", popt)
a, b, c, d, e, f = popt
Z_fit = quad2d((RR, QQ), *popt)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Data points
ax.scatter(x, y, z, color="red")

# Fitted quadratic surface
ax.plot_surface(RR, QQ, Z_fit, alpha=0.5)

ax.set_xlabel("R")
ax.set_ylabel("q")
ax.set_zlabel("Z")

ax.contour(RR, QQ, Z_fit, zdir="z", offset=-100, cmap="coolwarm")
ax.contour(RR, QQ, Z_fit, zdir="x", offset=-40, cmap="coolwarm")
ax.contour(RR, QQ, Z_fit, zdir="y", offset=40, cmap="coolwarm")

with open("/home/koen/master-internship/scripts/w17/figure1.pickle", "wb") as f:
    pickle.dump(fig, f)

plt.close()

# %%

fig = pickle.load(open("/home/koen/master-internship/scripts/w17/figure1.pickle", "rb"))
plt.show()


# %%
from matplotlib.colors import Normalize

# Evaluate fitted surface
Z_fit = quad2d((RR, QQ), *popt)

fig, ax = plt.subplots(figsize=(7, 6))

norm = Normalize(
    vmin=min([np.nanmin(z), np.nanmin(Z_fit)]),
    vmax=max([np.nanmax(z), np.nanmax(Z_fit)]),
)
# Smooth fitted colormap
pcm = ax.pcolormesh(RR, QQ, Z_fit, shading="auto", cmap="viridis", norm=norm)

# Original data points
sc = ax.scatter(x, y, c=z, cmap="viridis", s=100, norm=norm)

# Colorbar
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("Z")

ax.set_xlabel("R")
ax.set_ylabel("q")
ax.set_xscale("log")
plt.show()
# %%

# Build coordinate grids
grid = grid_max
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


print(Z)
RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt

# %%
alpha = 0
beta = 0
delta = 0.5
gamma = 1
q = 0.5
RL = 600
Rsun = 600

qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
orbit = CombineEvolve(alpha, beta, delta, gamma, qs)

max_R = RL * quad2d((RL, q), *popt)
print(max_R)
# %%

plt.plot(qs, orbit.a_over_a0() * get_separation(RL, q))
plt.plot(qs, orbit.da_over_a0_dq())
plt.scatter(
    1 / orbit.stationary_points,
    orbit.a_over_a0(orbit.stationary_points) * get_separation(RL, q),
)
plt.plot([qs[0], qs[-1]], [max_R, max_R])
plt.show()

# %%
a, b, c, d, e, f = popt
alpha = 0
beta = 0
delta = 0.5
gamma = 1.0
qss = np.linspace(0.1, 1, 100)
RLs = np.linspace(150, 750, 100)

Z = np.zeros([100, 100])

rlgrid, qgrid = np.meshgrid(RLs, qs)

fig, ax = plt.subplots(figsize=(7, 6))
for i, RL in enumerate(RLs):
    for j, q in enumerate(qss):
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        orbit = CombineEvolve(alpha, beta, delta, gamma, qs)

        max_R = RL * quad2d((RL, q), *popt)
        Z[j, i] = max_R / np.min(orbit.a_over_a0() * get_separation(RL, q))

pcm = ax.pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
    cmap="bwr",
    vmin=np.min(Z),
    vmax=1 + (1 - np.min(Z)),
)

fig.colorbar(pcm)

plt.show()

# %%

# Build coordinate grids
grid = grid_max
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


print(Z)
RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt
alpha = 0
beta = 0
delta = 0.2
gamma = 1.25
qss = np.linspace(0.1, 1, 100)
RLs = np.linspace(150, 750, 100)

Z = np.zeros([100, 100])

rlgrid, qgrid = np.meshgrid(RLs, qs)

fig, ax = plt.subplots(figsize=(7, 6))
for i, RL in enumerate(RLs):
    for j, q in enumerate(qss):
        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        orbit = CombineEvolve(alpha, beta, delta, gamma, qs)

        max_R = RL * quad2d((RL, q), *popt)
        Z[j, i] = max_R / np.min(orbit.a_over_a0() * get_separation(RL, q))

pcm = ax.pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
    cmap="bwr",
    vmin=np.min(Z),
    vmax=1 + (1 - np.min(Z)),
)

fig.colorbar(pcm)

plt.show()

# %%

# Build coordinate grids
grid = grid_max
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt


alpha = 0
beta = 0
delta = 0.2
gamma = 1.25
q = 0.5
RL = 600
Rsun = 600
qss = np.linspace(0.1, 1, 100)
RLs = np.linspace(150, 750, 100)

Z = np.zeros([100, 100])

rlgrid, qgrid = np.meshgrid(RLs, qs)

fig, ax = plt.subplots(figsize=(7, 6))
for i, RL in enumerate(RLs):
    print(i)
    for j, q in enumerate(qss):
        max_R = RL * quad2d((RL, q), *popt)

        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        for delta in np.linspace(1e-10, 1 - 1e-10, 100):
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
            if orbit.a_over_a0(orbit.stationary_points) * get_separation(RL, q) < max_R:
                break
        Z[j, i] = delta


pcm = ax.pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
)

fig.colorbar(pcm)

plt.show()


# %%


# Build coordinate grids
grid = grid_max
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt


alpha = 0
beta = 0
delta = 0.25
gamma = 1.25
q = 0.5
RL = 600
Rsun = 600
qss = np.linspace(0.1, 1, 100)
RLs = np.linspace(150, 750, 100)

Z = np.zeros([100, 100])

rlgrid, qgrid = np.meshgrid(RLs, qs)

fig, ax = plt.subplots(figsize=(7, 6))
for i, RL in enumerate(RLs):
    print(i)
    for j, q in enumerate(qss):
        max_R = RL * quad2d((RL, q), *popt)

        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        for gamma in np.linspace(0, 5, 100):
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
            if orbit.a_over_a0(orbit.stationary_points) * get_separation(RL, q) < max_R:
                break
        Z[j, i] = gamma


pcm = ax.pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
)

fig.colorbar(pcm)

plt.show()


# %%

# Build coordinate grids
grid = grid_min
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt

qss = np.linspace(0, 1, 100)
RLs = np.linspace(150, 750, 100)

Z = np.zeros([100, 100])

fig, ax = plt.subplots(3, 3, figsize=(7, 6), constrained_layout=True)

for i, RL in enumerate(RLs):
    for j, q in enumerate(qss):
        max_R = RL * quad2d((RL, q), *popt)
        Z[j, i] = np.log10(max_R)


pcm = ax[1, 1].pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
)

fig.colorbar(pcm)

RLs = [300, 450, 600]
qss = [0.75, 0.5, 0.25]

print(RLs)
# panels

for ii in range(3):
    for jj in range(3):
        RL = RLs[ii]
        q = qss[jj]

        if ii == 1 and jj == 1:
            continue

        max_R = RL * quad2d((RL, q), *popt)
        deltagrid = np.linspace(1e-10, 1 - 1e-10, 100)
        gammagrid = np.linspace(1e-10, 5, 100)

        Z = np.zeros([100, 100])

        dgrid, ggrid = np.meshgrid(deltagrid, gammagrid)

        for i, d in enumerate(deltagrid):
            for j, g in enumerate(gammagrid):
                qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
                orbit = CombineEvolve(alpha, beta, d, g, qs)
                Z[j, i] = np.log10(
                    orbit.a_over_a0(orbit.stationary_points)
                    * get_separation(RL, q)
                    / max_R
                )

        pcm = ax[ii, jj].pcolormesh(
            deltagrid,
            gammagrid,
            Z,
            cmap="bwr",
            vmin=-0.5,
            vmax=0.5,
            shading="auto",
        )

        # ax[ii,jj].contour(
        #     deltagrid,
        #     gammagrid,
        #     Z,
        #     levels=[0,1]
        # )

        fig.colorbar(pcm)


plt.show()


# %%
import pickle
from matplotlib.gridspec import GridSpec

# Build coordinate grids
grid = grid_min
R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = np.max(model.star.R) / RL
    if model.star.period_days[-1] < 50:
        mask_bad[i, j] = False
        Z[i, j] = np.nan


RR, QQ = np.meshgrid(R_vals, q_vals)
# Keep only valid points
valid = (~np.isnan(Z)) & (~mask_bad)

x = RR[valid]
y = QQ[valid]
z = Z[valid]

# 2D quadratic model
# Fit
popt, pcov = curve_fit(quad2d, (x, y), z)

a, b, c, d, e, f = popt

qss = np.linspace(0, 1, 30)
RLs = np.linspace(150, 750, 30)

Z = np.zeros([30, 30])

fig = plt.figure(constrained_layout=True)

gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 2, 1])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0:, 1:])
ax = [ax1, ax2]

for i, RL in enumerate(RLs):
    for j, q in enumerate(qss):
        max_R = RL * quad2d((RL, q), *popt)
        Z[j, i] = np.log10(max_R)


pcm = ax[0].pcolormesh(
    RLs,
    qss,
    Z,
    shading="auto",
)

fig.colorbar(pcm)

contours = dict()
colormeshes = dict()

for ii in range(len(RLs)):
    print(ii)
    for jj in range(len(qss)):
        RL = RLs[ii]
        q = qss[jj]

        max_R = RL * quad2d((RL, q), *popt)
        deltagrid = np.linspace(1e-10, 1 - 1e-10, 30)
        gammagrid = np.linspace(1e-10, 5, 30)

        Z = np.zeros([30, 30])

        dgrid, ggrid = np.meshgrid(deltagrid, gammagrid)

        for i, d in enumerate(deltagrid):
            for j, g in enumerate(gammagrid):
                qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
                orbit = CombineEvolve(alpha, beta, d, g, qs)
                Z[j, i] = np.log10(
                    orbit.a_over_a0(orbit.stationary_points)
                    * get_separation(RL, q)
                    / max_R
                )
        contour = ax[1].contour(
            deltagrid,
            gammagrid,
            Z,
            levels=[0, 100],
        )
        contour.set_gid(f"{RL:.3f}, {q:.3f} contour")
        contour.set(visible=False)
        contours[f"{RL:.3f}, {q:.3f}"] = contour
        colormesh = ax[1].pcolormesh(
            deltagrid,
            gammagrid,
            Z,
            cmap="bwr",
            vmin=-1.5,
            vmax=1.5,
            shading="auto",
        )
        colormesh.set(visible=False)
        colormesh.set_gid(f"{RL:.3f}, {q:.3f} colormesh")
        colormeshes[f"{RL:.3f}, {q:.3f}"] = colormesh


with open("/home/koen/master-internship/scripts/w17/figure.pickle", "wb") as f:
    pickle.dump(fig, f)

# %%

fig = pickle.load(open("/home/koen/master-internship/scripts/w17/figure.pickle", "rb"))


def mouse_move(event):
    x = event.xdata
    y = event.ydata
    if x != None and y != None:
        if event.inaxes == ax[0]:
            for contour in contours.values():
                contour.set(visible=False)
            for colormesh in colormeshes.values():
                colormesh.set(visible=False)
            RL = find_nearest(RLs, x)
            q = find_nearest(qss, y)

            contours[f"{RL:.3f}, {q:.3f}"].set(visible=True)
            colormeshes[f"{RL:.3f}, {q:.3f}"].set(visible=True)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


while True:
    cid = fig.canvas.mpl_connect("motion_notify_event", mouse_move)
    plt.draw()
    if plt.waitforbuttonpress(0.01):
        break

# %%
