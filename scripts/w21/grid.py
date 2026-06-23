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

cct = MesaGrid(f"{MASTER}tides-grid-4")

# %%

non_cons = MesaGrid(f"{MASTER}delta0.25_beta0.75")

# %%
from matplotlib import ticker
import matplotlib as mpl

titles = [
    "Fully conservative",
    "$\\delta = 0.5$, $\\beta = 0$",
    "$\\delta = 0.5$, $\\beta = 0.5$",
]
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
            print(model.star.fixed_xfer_fraction[-1])
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
    "/home/koen/figures/plots/master-internship/w21/periods.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)

fig.show()


# %%

for R, q, model in non_cons.get_R1_index(-2):
    print(q)
    plt.plot(model.env_mass, model.star.R)

plt.xscale("log")
plt.yscale("log")
plt.show()
# %%
