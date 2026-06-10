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

grid_non_cons = MesaGrid(f"{MASTER}/tides-grid-4")

# %%

grid_very_non_cons = MesaGrid(f"{MASTER}/tides-grid-8")

# %%

grid_cons = MesaGrid(f"{MASTER}/tides-grid-6")

# %%

grids = [grid_cons, grid_non_cons, grid_very_non_cons]
# %%
fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

grid = grid_very_non_cons

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

        Z[i, j] = model.star.period_days[-1]


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
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="viridis",
    shading="auto",
    # vmin=np.log10(5 / 8),
    # vmax=np.log10(1.6),
)


print(Zmax)
cbar = plt.colorbar(
    mesh,
    # label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
)


ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()


# %%

grid = grid_cons
for R, q, model in grid.get_R1_index(8):
    plt.plot(model.age, model.star.R)

plt.show()

# %%

for i, grid in enumerate(grids):
    for R, q, model in grid.get_R1_index(7):
        plt.plot(model.env_mass, model.star.R, c=f"C{i}")

plt.show()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    for R, q, model in grid.get_R1_index(8):
        (l,) = plt.plot(
            model.env_mass, model.star.R, c=f"C{i}", label=labels[i], linewidth=1
        )
    ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=3)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-1.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    if i == 0:
        continue
    for R, q, model in grid.iter_models():
        (l,) = plt.plot(
            model.env_mass, model.star.R, c=f"C{i}", label=labels[i], linewidth=1
        )
    ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=3)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    for R, q, model in grid.get_R1_index(4):
        (l,) = plt.plot(
            model.env_mass, model.star.R, c=f"C{i}", label=labels[i], linewidth=1
        )
        print(R)
    ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=3)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-3.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    for R, q, model in grid.get_R1_index(8):

        index = np.argwhere(model.star.R > model.star.rl_1)[0][0]
        age = model.age[index]

        (l,) = plt.plot(
            model.age - age, model.star.R, c=f"C{i}", label=labels[i], linewidth=1
        )
        plt.xlim(-100, 500)
    ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=3)

plt.xlabel("time (yr)")
plt.ylabel("Star radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-4.pgf", format="pgf")
plt.show()
plt.close()


# %%

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "none"

colors = px.colors.qualitative.D3

fig = make_subplots(rows=3, cols=1, shared_xaxes=False)

for i, grid in enumerate(grids):

    for j, (R, q, model) in enumerate(grid.get_R1_index(8)):
        # top subplot
        if j == 0:
            fig.add_trace(
                go.Scatter(
                    x=model.env_mass,
                    y=model.star.R,
                    name=labels[i],
                    legendgroup=labels[i],
                    showlegend=True,
                    line=dict(color=colors[i]),
                    hovertemplate="<b>M</b>: %{x}<br>"
                    + "<b>R</b>: %{y}<br>"
                    + f"<b>q</b>: {q}",
                ),
                row=1,
                col=1,
            )

        else:
            fig.add_trace(
                go.Scatter(
                    x=model.env_mass,
                    y=model.star.R,
                    legendgroup=labels[i],
                    showlegend=False,
                    line=dict(color=colors[i]),
                    hovertemplate="<b>M</b>: %{x}<br>"
                    + "<b>R</b>: %{y}<br>"
                    + f"<b>q</b>: {q}",
                ),
                row=1,
                col=1,
            )

        # bottom subplot

        index = np.argwhere(model.star.R > model.star.rl_1)[0][0]
        age = model.age[index]

        fig.add_trace(
            go.Scatter(
                x=model.age - age,
                y=model.star.R,
                legendgroup=labels[i],
                showlegend=False,
                line=dict(color=colors[i]),
                hovertemplate="<b>t</b>: %{x}<br>"
                + "<b>R</b>: %{y}<br>"
                + f"<b>q</b>: {q}",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=model.star.log_Teff,
                y=model.star.log_L,
                legendgroup=labels[i],
                showlegend=False,
                line=dict(color=colors[i]),
                hovertemplate="<b>T</b>: %{x}<br>"
                + "<b>L</b>: %{y}<br>"
                + f"<b>q</b>: {q}",
            ),
            row=3,
            col=1,
        )


# solar values
Tsun = 5772

# grid in teff
teff = np.linspace(10**3.34, 10**3.55, 500)

# radii to plot (in solar radii)
radii = np.logspace(np.log10(100), np.log10(700), 9)
for R in radii:
    L = (R**2) * (teff / Tsun) ** 4
    fig.add_trace(
        go.Scatter(
            x=np.log10(teff),
            y=np.log10(L),
            name=f"R = {R:.0f}",
            showlegend=False,
            line=dict(color="gray", width=0.5),
        ),
        row=3,
        col=1,
    )


fig.update_xaxes(range=[-100, 500], row=2, col=1)
fig.update_xaxes(type="log", row=1, col=1)
fig.update_yaxes(type="log", row=1, col=1)

fig.update_layout(
    autosize=True,
    title="Low envelope mass-radius relation",
)

fig.update_xaxes(title_text="envelope mass", row=1, col=1)
fig.update_yaxes(title_text="radius", row=1, col=1)

fig.update_xaxes(title_text="time since rlof", row=2, col=1)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.update_xaxes(title_text="log T", row=3, col=1)
fig.update_xaxes(autorange="reversed", row=3, col=1)
fig.update_yaxes(title_text="log L", row=3, col=1)
fig.update_xaxes(range=[3.55, 3.34], row=3, col=1)
fig.update_yaxes(range=[3.3, 4.2], row=3, col=1)

fig.write_html(
    "/home/koen/figures/plots/master-internship/w19/low-mass-radius.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)

fig.show()

# %%

further_path = "/home/koen/master-internship/mesa-models/run-further/R552.27_q0.500/LOGS/TPAGB/history.data"

further = mr.MesaData(further_path)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    # if i == 0:
    #     continue
    for R, q, model in grid.get_R1_index(8):

        index = np.argwhere(model.star.R > model.star.rl_1)[0][0]
        age = model.age[index]

        (l,) = plt.plot(
            model.star.log_Teff,
            model.star.log_L,
            c=f"C{i}",
            label=labels[i],
            linewidth=1,
        )
        print(model.star.he_core_mass[-1])
        print(R)
    ls.append(l)

# solar values
Tsun = 5772

# grid in teff
teff = np.linspace(10**3.3, 10**3.6, 500)

# radii to plot (in solar radii)
radii = np.logspace(np.log10(100), np.log10(700), 9)
for R in radii:
    L = (R**2) * (teff / Tsun) ** 4
    plt.plot(np.log10(teff), np.log10(L), c="C9", linewidth=0.8, zorder=-1)

plt.gca().invert_xaxis()
fig.legend(loc="outside upper center", handles=ls, ncols=3)

plt.ylim(3.3, 4.2)
plt.xlim(3.6, 3.3)
plt.xlabel(r"$\log (T_\textrm{eff} / K)$")
plt.ylabel(r"$\log (L/L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-5.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]
for i, grid in enumerate(grids):
    for R, q, model in grid.get_R1_index(8):

        index = np.argwhere(model.star.R > model.star.rl_1)[0][0]
        age = model.age[index]

        (l,) = plt.plot(
            model.age - age,
            model.star.R / model.star.R[0],
            c=f"C{i}",
            label=labels[i],
            linewidth=1,
        )

        (l,) = plt.plot(
            model.age - age,
            model.star.log_L - model.star.log_L[0],
            c=f"C{i}",
            label=labels[i],
            linewidth=1,
        )

        plt.xlim(-100, 500)
    ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=3)

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-8.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]

n = 7
colormap = plt.cm.inferno(np.linspace(0, 0.8, n))

for i, grid in enumerate(grids):
    if i in [0, 1]:
        continue
    for j, (R, q, model) in enumerate(grid.get_R1_index(8)):
        (l,) = axs[0].plot(
            model.env_mass,
            model.star.log_L,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[1].plot(
            model.env_mass,
            model.star.log_Teff,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[2].plot(
            model.env_mass,
            model.star.log_R,
            c=colormap[j],
            label=f"q={q:.2f}",
            linewidth=1,
        )
        ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=4)
plt.xscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
axs[0].set_ylabel(r"$\log (L / L_\odot)$")
axs[1].set_ylabel(r"$\log (T_\textrm{eff} / \textrm{K})$")
axs[2].set_ylabel(r"$\log (R / R_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-6.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]

n = 7
colormap = plt.cm.inferno(np.linspace(0, 0.8, n))

for i, grid in enumerate(grids):
    if i in [0, 1]:
        continue
    for j, (R, q, model) in enumerate(grid.get_R1_index(8)):
        (l,) = axs[0].plot(
            model.env_mass,
            model.star.Lnuc / model.star.L,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[1].plot(
            model.env_mass,
            model.star.log_L,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[2].plot(
            model.env_mass,
            model.star.log_Lnuc,
            c=colormap[j],
            label=f"q={q:.2f}",
            linewidth=1,
        )
        ls.append(l)


axs[0].set_ylim(0.8, 2)
axs[1].set_ylim(3.6, 4.05)
axs[2].set_ylim(3.6, 4.05)
fig.legend(loc="outside upper center", handles=ls, ncols=4)
plt.xscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
axs[0].set_ylabel(r"$L_\textrm{nuc} / L$")
axs[1].set_ylabel(r"$\log (L / L_\odot)$")
axs[2].set_ylabel(r"$\log (L_\textrm{nuc} / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-7.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]

n = 7
colormap = plt.cm.inferno(np.linspace(0, 0.8, n))

for i, grid in enumerate(grids):
    if i in [0, 1]:
        continue
    for j, (R, q, model) in enumerate(grid.get_R1_index(8)):
        (l,) = axs[0].plot(
            model.env_mass,
            model.star.log_abs_mdot,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[1].plot(
            model.env_mass,
            model.star.log_L,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[2].plot(
            model.env_mass,
            model.star.log_Lnuc,
            c=colormap[j],
            label=f"q={q:.2f}",
            linewidth=1,
        )
        ls.append(l)


axs[0].set_ylim(-6, 0)
axs[1].set_ylim(3.6, 4.05)
axs[2].set_ylim(3.6, 4.05)
fig.legend(loc="outside upper center", handles=ls, ncols=4)
plt.xscale("log")

plt.xlabel("Envelope mass ($M_\odot$)")
plt.ylabel("Star radius ($R_\odot$)")
axs[0].set_ylabel(r"$L_\textrm{nuc} / L$")
axs[1].set_ylabel(r"$\log (L / L_\odot)$")
axs[2].set_ylabel(r"$\log (L_\textrm{nuc} / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-9.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

ls = []
labels = [
    "fully conservative",
    "$\delta = 0.2,\; \gamma = 1.25$",
    "$\delta = 0.5,\; \gamma=1.0$",
]

n = 7
colormap = plt.cm.inferno(np.linspace(0, 0.8, n))

for i, grid in enumerate(grids):
    if i in [0, 1]:
        continue
    for j, (R, q, model) in enumerate(grid.get_R1_index(8)):
        (l,) = axs[0].plot(
            model.star.star_2_mass / model.star.star_1_mass,
            model.star.log_L,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[1].plot(
            model.star.star_2_mass / model.star.star_1_mass,
            model.star.log_Teff,
            c=colormap[j],
            label=labels[i],
            linewidth=1,
        )

        (l,) = axs[2].plot(
            model.star.star_2_mass / model.star.star_1_mass,
            model.star.log_R,
            c=colormap[j],
            label=f"q={q:.2f}",
            linewidth=1,
        )
        ls.append(l)


fig.legend(loc="outside upper center", handles=ls, ncols=4)

plt.xlabel("$q$")
plt.ylabel("Star radius ($R_\odot$)")
axs[0].set_ylabel(r"$\log (L / L_\odot)$")
axs[1].set_ylabel(r"$\log (T_\textrm{eff} / \textrm{K})$")
axs[2].set_ylabel(r"$\log (R / R_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-shrinking-10.pgf", format="pgf")
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

fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

grids = [grid_cons, grid_non_cons, grid_very_non_cons]

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
            Z[i, j] = np.max(model.star.R) / R
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
            Z[i, j] = np.max(model.star.R) / R
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
plt.savefig("/home/koen/LaTeX-setup/plots/w19-grid-strong-ratio.pgf", format="pgf")
plt.show()
plt.close()


# %%

from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

grid = grid_non_cons
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
        Z[i, j] = np.max(model.star.R) / R
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
Z_fit = quad2d((RR, QQ), *popt)


fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

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
cbar.set_label(r"$\textrm{max}(R_\textrm{star}) / R_\textrm{RL}$")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")
ax.set_xscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w19-compare-fit.pgf", format="pgf")
plt.show()
plt.close()

# %%
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

alpha = 0
beta = 0
gamma = 1
q = 0.5
RL = 600
Rsun = 600
n = 10
colormap = plt.cm.inferno(np.linspace(0, 1, 1000001))

fig, axs = plt.subplots(
    3, 3, sharex=False, figsize=set_size(full, height=1), constrained_layout=True
)

fig.suptitle("$q_\\textrm{i}= 0.5$, $\\alpha=0$", size=10)
for a, ax in enumerate(axs.flatten()):

    delta = a * 0.1

    for i, beta in enumerate(np.linspace(0, 1 - delta + 1e-9, n)):

        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        orbit = CombineEvolve(alpha, beta, delta, gamma, qs)

        ax.plot(qs, orbit.a_over_a0(), c=colormap[int(beta * 1e6)], linewidth=2)
    ax.text(
        0.5,
        0.9,
        f"$\delta = {delta:.1f}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )

plt.colorbar(
    ScalarMappable(cmap=plt.cm.inferno), ax=axs[:, :], label="$\\beta$", aspect=30
)
fig.supxlabel("$q$", size=10)
fig.supylabel("$a/a_\\textrm{i}$", size=10)
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-beta.pgf", format="pgf")
plt.show()
plt.close()

# %%

import matplotlib as mpl
from matplotlib.cm import ScalarMappable

alpha = 0
beta = 0
gamma = 1
delta = 0.4
RL = 600
Rsun = 600
n = 10
colormap = plt.cm.inferno(np.linspace(0, 1, 1000001))

fig, axs = plt.subplots(
    3, 3, sharex=False, figsize=set_size(full, height=1), constrained_layout=True
)

fig.suptitle("$\\delta = 0.4$, $\\alpha=0$", size=10)
for a, ax in enumerate(axs.flatten()):

    q = a * 0.1 + 0.2

    for i, beta in enumerate(np.linspace(0, 1 - delta + 1e-9, n)):

        qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
        orbit = CombineEvolve(alpha, beta, delta, gamma, qs)

        ax.plot(qs, orbit.a_over_a0(), c=colormap[int(beta / 0.6 * 1e6)], linewidth=2)
    ax.text(
        0.5,
        0.9,
        f"$q_\\textrm{{i}} = {q:.1f}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )

map = ScalarMappable(cmap=plt.cm.inferno)
map.set_clim(0, 0.6)
plt.colorbar(
    map,
    ax=axs[:, :],
    label="$\\beta$",
    aspect=30,
)
fig.supxlabel("$q$", size=10)
fig.supylabel("$a/a_\\textrm{i}$", size=10)
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-beta-2.pgf", format="pgf")
plt.show()
plt.close()


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
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
            a_ratios.append(orbit.a_over_a0()[-1])

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
plt.savefig("/home/koen/LaTeX-setup/plots/w19-beta-3.pgf", format="pgf")
plt.show()
plt.close()


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

q_grid = np.linspace(1e-10, 1 - 1e-10, 100)
for a, ax in enumerate(axs.flatten()):

    delta = a * 0.1
    for b, beta in enumerate(np.linspace(1e-10, 1 - 1e-10, n)):

        a_ratios = []
        for d, q in enumerate(q_grid):

            if delta + beta > 0.999:
                break

            qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
            a_ratios.append(orbit.a_over_a0()[-1])

        length = len(a_ratios)
        ax.plot(
            deltas[:length], a_ratios, color=colormap[int(beta * 1e6)], rasterized=True
        )
    ax.text(
        0.5,
        0.9,
        f"$\delta = {delta:.1f}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )

plt.colorbar(
    ScalarMappable(cmap=plt.cm.inferno), ax=axs[:, :], label="$\\beta$", aspect=30
)
fig.supxlabel("$q_\\textrm{i}$", size=10)
fig.supylabel("$a_\\textrm{f}/a_\\textrm{i}$", size=10)
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-beta-4.pgf", format="pgf")
plt.show()
plt.close()


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

q_grid = np.linspace(1e-10, 1 - 1e-10, 100)
for a, ax in enumerate(axs.flatten()):

    delta = a * 0.1
    for b, beta in enumerate(np.linspace(1e-10, 1 - 1e-10, n)):

        a_ratios = []
        q_finals = []
        for d, q in enumerate(q_grid):

            if delta + beta > 0.999:
                break

            qs = np.linspace(q, q_f(q, 2, 0.6, alpha, beta, delta), 100)
            orbit = CombineEvolve(alpha, beta, delta, gamma, qs)
            a_ratios.append(orbit.a_over_a0()[-1])
            q_finals.append(qs[-1])

        length = len(a_ratios)
        ax.plot(
            deltas[:length], a_ratios, color=colormap[int(beta * 1e6)], rasterized=True
        )
    ax.text(
        0.5,
        0.9,
        f"$\delta = {delta:.1f}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )

plt.colorbar(
    ScalarMappable(cmap=plt.cm.inferno), ax=axs[:, :], label="$\\beta$", aspect=30
)
fig.supxlabel("$q_\\textrm{i}$", size=10)
fig.supylabel("$a_\\textrm{f}/a_\\textrm{i}$", size=10)
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w19-beta-4.pgf", format="pgf")
plt.show()
plt.close()
