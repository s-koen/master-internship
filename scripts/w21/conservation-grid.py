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

d0_1b0_9 = mr.MesaData(
    f"{MASTER}conservation-grid/delta0.1_beta0.9/LOGS/TPAGB/history.data"
)
d0_2b0_8 = mr.MesaData(
    f"{MASTER}conservation-grid/delta0.2_beta0.8/LOGS/TPAGB/history.data"
)

# %%

plt.plot(d0_1b0_9.age, d0_1b0_9.R)
plt.plot(d0_2b0_8.age, d0_2b0_8.R)
plt.plot(d0_1b0_9.age, d0_1b0_9.rl_1)
plt.plot(d0_2b0_8.age, d0_2b0_8.rl_1)
plt.show()

# %%

deltas = [0.0, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

models = []
for d in deltas:
    b = 1 - d
    try:
        models.append(
            mr.MesaData(
                f"{MASTER}conservation-grid/delta{np.round(d,3):.3f}_beta{np.round(b,3):.3f}/LOGS/TPAGB/history.data"
            )
        )
    except:
        models.append(
            mr.MesaData(
                f"{MASTER}conservation-grid/delta{np.round(d,1)}_beta{np.round(b,1)}/LOGS/TPAGB/history.data"
            )
        )

# %%


for i, m in enumerate(models):
    plt.plot(m.envelope_mass, m.R, c=f"C{i}")
    plt.plot(m.envelope_mass, m.rl_1, c=f"C{i}", alpha=0.4, linewidth=4)

plt.show()

# %%
betas = np.linspace(0.0, 0.7, 8)

models2 = []
for b in betas:
    d = 0.3
    models2.append(
        mr.MesaData(
            f"{MASTER}conservation-grid/delta{np.round(d,1)}_beta{np.round(b,1)}/LOGS/TPAGB/history.data"
        )
    )


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0, vmax=0.7)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in betas]

for i, m in enumerate(models2):
    plt.plot(np.log10(m.envelope_mass), m.log_R, c=colors[i])
    plt.plot(
        np.log10(m.envelope_mass), np.log10(m.rl_1), c=colors[i], alpha=0.4, linewidth=4
    )

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\beta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)


plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, "$\delta = 0.3$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$\log(M_\textrm{env}/M_\odot)$")
plt.ylabel(r"$\log(R/R_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-beta-r.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0, vmax=0.7)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in betas]

for i, m in enumerate(models2):
    plt.plot(np.log10(m.envelope_mass), np.log10(m.period_days), c=colors[i])

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\beta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)


plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, "$\delta = 0.3$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$\log(M_\textrm{env}/M_\odot)$")
plt.ylabel(r"$\log(P/\textrm{days})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-beta-periods.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0, vmax=0.7)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in betas]

for i, m in enumerate(models2):
    plt.plot(m.star_2_mass / m.star_1_mass, np.log10(m.binary_separation), c=colors[i])

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\beta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)


plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, "$\delta = 0.3$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$q$")
plt.ylabel(r"$a$ ($R_\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-beta-periods-q.pgf",
    format="pgf",
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0, vmax=0.7)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in betas]

for i, m in enumerate(models2):
    plt.plot(m.log_Teff, m.log_L, c=colors[i])

plt.gca().invert_xaxis()

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\beta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)

plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, "$\delta = 0.3$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$\log(T_\textrm{eff}/\textrm{K})$")
plt.ylabel(r"$\log(L/L_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-beta-HR.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0.0, vmax=0.3)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in deltas]


for i, m in enumerate(models):
    plt.plot(m.log_Teff, m.log_L, c=colors[i])

plt.gca().invert_xaxis()
sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\delta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)

plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, r"$\beta = 1-\delta$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$\log(T_\textrm{eff}/\textrm{K})$")
plt.ylabel(r"$\log(L/L_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-delta-HR.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm2 = mpl.colors.Normalize(vmin=0.0, vmax=0.3)

cmap = plt.cm.viridis

colors = [cmap(norm2(x)) for x in deltas]

for i, m in enumerate(models):
    plt.plot(np.log10(m.envelope_mass), m.log_R, c=colors[i])
    plt.plot(
        np.log10(m.envelope_mass), np.log10(m.rl_1), c=colors[i], alpha=0.4, linewidth=4
    )

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap)
fig.colorbar(
    sm,
    ax=plt.gca(),
    label=r"$\delta$",
    orientation="horizontal",
    location="top",
    aspect=50,
)


plt.ylim(2)
plt.gca().set_facecolor("C9")
plt.text(0.07, 0.9, r"$\beta = 1-\delta$", transform=axs.transAxes, ha="left", va="top")

plt.xlabel(r"$\log(M_\textrm{env}/M_\odot)$")
plt.ylabel(r"$\log(R/R_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-conservation-grid-delta-r.pgf", format="pgf"
)
plt.show()
plt.close()

# %%
list_of_models = []
deltas = [0.0, 0.1, 0.2, 0.3]

for i, d in enumerate(deltas):
    b = 1 - d
    try:
        l = mr.MesaLogDir(
            f"{MASTER}conservation-grid/delta{np.round(d,3):.3f}_beta{np.round(b,3):.3f}/LOGS/TPAGB"
        )
        print(
            f"{MASTER}conservation-grid/delta{np.round(d,3):.3f}_beta{np.round(b,3):.3f}/LOGS/TPAGB"
        )
    except:
        l = mr.MesaLogDir(
            f"{MASTER}conservation-grid/delta{np.round(d,1)}_beta{np.round(b,1)}/LOGS/TPAGB"
        )
        print(
            f"{MASTER}conservation-grid/delta{np.round(d,1)}_beta{np.round(b,1)}/LOGS/TPAGB"
        )
    for profile in l.profile_numbers:
        profile = l.profile_data(profile_number=profile)

    if np.round(d, 3) in [0.100, 0.200, 0.300]:
        profiles = list(l.profile_dict.values())[1:]
    elif np.round(d, 3) in [0.0]:
        profiles = list(l.profile_dict.values())
        profiles[0] = profiles[1]
    elif np.round(d, 3) in [0.125]:
        profiles = list(l.profile_dict.values())
        for t in range(11):
            profiles[t] = profiles[11]

    elif np.round(d, 3) in [0.150]:
        profiles = list(l.profile_dict.values())
        for t in range(13):
            profiles[t] = profiles[13]

    elif np.round(d, 3) in [0.175]:
        profiles = list(l.profile_dict.values())
        for t in range(19):
            profiles[t] = profiles[19]

    elif np.round(d, 3) in [0.225]:
        profiles = list(l.profile_dict.values())
        for t in range(30):
            profiles[t] = profiles[30]

    elif np.round(d, 3) in [0.275]:
        profiles = list(l.profile_dict.values())
        for t in range(41):
            profiles[t] = profiles[41]

    else:
        profiles = list(l.profile_dict.values())[1:]

    list_of_models.append(profiles)

true_list = list_of_models.copy()
# %%


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import savgol_filter

import matplotlib.cm as cm
from matplotlib.colors import to_hex

norm2 = mpl.colors.Normalize(vmin=0.0, vmax=0.3)

cmap = plt.cm.viridis

deltas = [0.0, 0.1, 0.2, 0.3]
colors = [cmap(norm2(x)) for x in deltas]


colors = [to_hex(c) for c in colors]
# --------------------------------------------------
# input
# --------------------------------------------------

n_models = len(list_of_models)

# --------------------------------------------------
# helper: cumulative L_grav profile
# --------------------------------------------------

true_list = list_of_models.copy()
for i in range(n_models):
    if i in [5, 10, 11, 12]:
        true_list[i] = list_of_models[i][1:]


def compute_lgrav(profile):
    """
    returns:
        m_coord: mass coordinate (inward)
        lgrav: cumulative integral of eps_grav_ad
    """

    m = profile.mass[::-1]
    r = 10 ** profile.logR[::-1]
    eps = profile.eps_grav[::-1][1:]

    dm = np.diff(m)

    lgrav = np.concatenate([[0.0], np.cumsum(eps * dm)])
    lgrav = savgol_filter(lgrav, 52, 3)

    env = np.log10(profile.star_mass - profile.he_core_mass)
    t = np.log10(profile.Teff)
    lum = np.log10(profile.photosphere_L)
    print(profile.bulk_names)
    return m, r, lgrav, env, t, lum


# --------------------------------------------------
# preprocess models
# --------------------------------------------------

model_m = []
model_r = []
model_lgrav = []
model_labels = []
model_env = []
model_t = []
model_lum = []

for i, model in enumerate(true_list):

    m_series = []
    r_series = []
    l_series = []
    env_series = []
    t_series = []
    lum_series = []

    for p in model:
        m, r, lgrav, env, t, lum = compute_lgrav(p)
        m_series.append(m)
        r_series.append(r)
        l_series.append(lgrav)
        env_series.append(env)
        t_series.append(t)
        lum_series.append(lum)

    model_m.append(m_series)
    model_r.append(r_series)
    model_lgrav.append(l_series)
    model_env.append(env_series)
    model_t.append(t_series)
    model_lum.append(lum_series)
    model_labels.append(f"1e{rates[i]:.1f} Msun / yr")


# --------------------------------------------------
# common grid for plotting
# --------------------------------------------------

n_grid = 150

m_min = 0.3
m_max = 1

m_grid = np.linspace(m_min, m_max, n_grid)


R_min = min(np.min(10**p.logR) for p in profiles)
R_max = max(np.max(10**p.logR) for p in profiles)

R_grid = np.logspace(np.log10(R_min), np.log10(R_max), n_grid)


# --------------------------------------------------
# figure layout (UNCHANGED STRUCTURE)
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.5, 0.5],
    specs=[
        [{}, {}],
        [{}, {}],
    ],
)

# --------------------------------------------------
# static traces (initial timestep only)
# --------------------------------------------------
marker_indices = []

for i in range(n_models):

    m0 = model_m[i][0]
    r0 = model_r[i][0]
    l0 = model_lgrav[i][0]
    env0 = model_env[i]
    t0 = model_t[i]
    lum0 = model_lum[i]

    l0_interp = np.interp(m_grid, m0, l0, left=np.nan, right=np.nan)

    fig.add_trace(
        go.Scatter(
            x=m_grid,
            y=l0_interp,
            mode="lines",
            name=model_labels[i],
            legendgroup=model_labels[i],
            line=dict(color=colors[i]),
        ),
        row=1,
        col=1,
    )

    l0_interp = np.interp(R_grid, r0, l0, left=np.nan, right=np.nan)

    fig.add_trace(
        go.Scatter(
            x=np.log10(R_grid),
            y=l0_interp,
            mode="lines",
            showlegend=False,
            legendgroup=model_labels[i],
            line=dict(color=colors[i]),
        ),
        row=1,
        col=2,
    )

    radius_proxy = np.array([np.max(p.logR) for p in true_list[i]])

    fig.add_trace(
        go.Scatter(
            x=[env0[0]],
            y=[radius_proxy[0]],
            mode="markers",
            marker=dict(size=10, color=colors[i]),
            showlegend=False,
            legendgroup=model_labels[i],
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[t0[0]],
            y=[lum0[0]],
            mode="markers",
            marker=dict(size=10, color=colors[i]),
            showlegend=False,
            legendgroup=model_labels[i],
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=env0,
            y=radius_proxy,
            mode="lines",
            showlegend=False,
            legendgroup=model_labels[i],
            line=dict(color=colors[i]),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t0,
            y=lum0,
            mode="lines",
            showlegend=False,
            legendgroup=model_labels[i],
            line=dict(color=colors[i]),
        ),
        row=2,
        col=2,
    )

    # moving marker

    marker_index = len(fig.data) - 1
    marker_indices.append(marker_index)


# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []
n_steps = max(len(m) for m in model_m)

for t in range(n_steps):

    frame_data = []

    for i in range(n_models):

        if t < len(model_m[i]):
            m = model_m[i][t]
            r = model_r[i][t]
            l = model_lgrav[i][t]
            env = model_env[i]
            T = model_t[i]
            lum = model_lum[i]
        else:
            m = model_m[i][-1]
            r = model_r[i][-1]
            l = model_lgrav[i][-1]
            env = model_env[i]
            T = model_t[i]
            lum = model_lum[i]

        l_interp = np.interp(m_grid, m, l, left=np.nan, right=np.nan)

        frame_data.append(go.Scatter(x=m_grid, y=l_interp, mode="lines"))

        l_interp = np.interp(R_grid, r, l, left=np.nan, right=np.nan)

        frame_data.append(go.Scatter(x=np.log10(R_grid), y=l_interp, mode="lines"))

        frame_data.append(
            go.Scatter(
                x=[env[t] if t < len(model_m[i]) else env[-1]],
                y=[
                    (
                        np.log10(np.nanmax(r))
                        if t < len(model_m[i])
                        else np.log10(np.nanmax(r))
                    )
                ],
                mode="markers",
            )
        )
        frame_data.append(
            go.Scatter(
                x=[T[t] if t < len(model_m[i]) else T[-1]],
                y=[lum[t] if t < len(model_m[i]) else lum[-1]],
                mode="markers",
            )
        )
    # frame_data.append(go.Scatter())
    traces = []

    for i in range(n_models):
        base = 6 * i
        traces.extend(
            [
                base,
                base + 1,
                base + 2,
                base + 3,
            ]
        )
    frames.append(
        go.Frame(
            name=str(t),
            data=frame_data,
            traces=traces,
        )
    )

fig.frames = frames


# --------------------------------------------------
# layout
# --------------------------------------------------
steps = [
    dict(
        method="animate",
        args=[
            [str(i)],
            {
                "mode": "immediate",
                "frame": {"duration": 20, "redraw": False},
                "transition": {"duration": 0},
            },
        ],
        label=str(i),
    )
    for i in range(n_steps)
]

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 30, "redraw": False},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"mode": "immediate"}],
                ),
            ],
        )
    ],
    sliders=[dict(active=0, steps=steps)],
)

fig.update_xaxes(title_text="mass coordinate", row=1, col=1)
fig.update_xaxes(title_text="log r", row=1, col=2)
fig.update_xaxes(title_text="log envelope mass", row=2, col=1)
fig.update_xaxes(title_text="log Teff", row=2, col=2, autorange="reversed")

fig.update_yaxes(
    title_text="cumulative L_grav",
    row=1,
    col=1,
    range=[-7000, 1000],
)
fig.update_yaxes(
    title_text="cumulative L_grav",
    row=1,
    col=2,
    range=[-7000, 1000],
)
fig.update_yaxes(title_text="log radius", row=2, col=1)
fig.update_yaxes(title_text="log L", row=2, col=2)

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w21/losing-envelope-opacity.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )


fig.show()
# %%

print(models[0].date)
# %%
