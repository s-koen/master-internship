import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

from scipy.signal import savgol_filter

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

grid = MesaGrid(f"{MASTER}delta0.25_beta0.75")
# %%
m0_01 = mr.MesaData(f"{MASTER}constant-mass-loss-rate/1e-2/LOGS/TPAGB/history.data")
# %%

plt.plot(m0_01.envelope_mass, m0_01.R)
plt.xscale("log")
plt.xlim(1e-2, 2)
plt.show()

# %%

plt.plot(m0_01.log_Teff, m0_01.log_L)
plt.gca().invert_xaxis()
plt.show()

# %%

for R, q, model in grid.get_R1_index(-1):
    plt.plot(model.env_mass, model.star.R, c="C9")

plt.plot(m0_01.envelope_mass, m0_01.R)
plt.xscale("log")
plt.show()
# %%

for R, q, model in grid.get_R1_index(-1):
    plt.plot(model.env_mass, model.star.log_abs_mdot, c="C9")

plt.plot(m0_01.envelope_mass, m0_01.log_abs_mdot)
plt.xscale("log")
plt.show()

# %%

for R, q, model in grid.get_R1_index(-1):
    plt.plot(model.star.log_Teff, model.star.log_L, c="C9")

plt.plot(m0_01.log_Teff, m0_01.log_L)
plt.scatter(m0_01.log_Teff[-1], m0_01.log_L[-1])
plt.gca().invert_xaxis()
plt.show()

# %%
l = mr.MesaLogDir(f"{MASTER}/constant-mass-loss-rate/1e-2/LOGS/TPAGB")

# %%
for profile in l.profile_numbers:
    profile = l.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)

# %%


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())
n = len(profiles)

n_grid = 300

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = min(np.min(p.logR) for p in profiles)
logR_max = max(np.max(p.logR) for p in profiles)
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = 0.0
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.linspace(m_min, m_max, n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid + 1))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.logR[::-1]
    yR = p.gradT[::-1]

    xM = p.mass[::-1]
    yM = p.gradT[::-1]

    m_grid = np.linspace(m_min, m_max, n_grid)
    max_S = p.mass[::-1][np.argmax(yM)]
    m_grid = np.append(m_grid, max_S)
    m_grid.sort()

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    entropy_M[i] = np.interp(m_grid, xM, yM, left=np.nan, right=np.nan)

# radius evolution
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
)

# use index instead of age for stability
idx = np.arange(n)

print(entropy_R[0])

# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
    subplot_titles=[
        "Entropy vs log R",
        "Entropy vs mass",
        "Radius evolution",
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
    ),
    row=1,
    col=1,
)

# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=m_grid,
        y=entropy_M[0],
        mode="lines",
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2,
    col=1,
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=m_grid, y=entropy_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
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
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
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

# axis labels
fig.update_xaxes(title_text="log R", row=1, col=1)
fig.update_xaxes(title_text="mass coordinate", row=1, col=2)
fig.update_xaxes(title_text="profile index", row=2, col=1)

fig.update_yaxes(title_text="entropy", row=1, col=1)
fig.update_yaxes(title_text="entropy", row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w20/post-AGB-instabilities.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )


# %%
from scipy.signal import savgol_filter

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())
n = len(profiles)

n_grid = 200

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = 0.1
logR_max = 2
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = 0.0
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.logspace(-9, np.log10(m_max), n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
hydrogen_R = np.empty((n, n_grid))
helium_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid))
hydrogen_M = np.empty((n, n_grid))
helium_M = np.empty((n, n_grid))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.mass[::-1][1:]
    yR = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    yR = savgol_filter(yR, 52, 3)
    HR = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    HeR = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])

    logR_grid = np.linspace(logR_min, logR_max, n_grid)
    logR_grid.sort()
    xM = p.mass[0] - p.mass[1:]
    yM = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    yM = savgol_filter(yM, 52, 3)
    HM = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    HeM = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])

    print(xR, yR)
    print(xM, yM)

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=-100, right=np.nan)
    hydrogen_R[i] = np.interp(logR_grid, xR, HR, left=-100, right=np.nan)
    hydrogen_R[i] = np.nan_to_num(hydrogen_R[i], neginf=-99, nan=np.nan)
    helium_R[i] = np.interp(logR_grid, xR, HeR, left=-100, right=np.nan)
    entropy_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), yM, left=np.nan, right=np.nan
    )
    hydrogen_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), HM, left=np.nan, right=np.nan
    )
    helium_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), HeM, left=np.nan, right=np.nan
    )

# radius evolution
radius = np.array([p.logR.max() for p in profiles])
print(p.bulk_names)
age = np.array(
    [
        np.log10(p.star_mass - p.he_core_mass) if hasattr(p, "star_mass") else i
        for i, p in enumerate(profiles)
    ]
)

# use index instead of age for stability
idx = np.arange(n)


# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
        name="Z",
        legendgroup="Z",
        line=dict(color=colors[2]),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=hydrogen_R[0],
        mode="lines",
        name="H",
        legendgroup="H",
        line=dict(color=colors[0]),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=helium_R[0],
        mode="lines",
        name="He",
        legendgroup="He",
        line=dict(color=colors[1]),
    ),
    row=1,
    col=1,
)
# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=entropy_M[0],
        mode="lines",
        legendgroup="Z",
        showlegend=False,
        line=dict(color=colors[2]),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=hydrogen_M[0],
        mode="lines",
        legendgroup="H",
        showlegend=False,
        line=dict(color=colors[0]),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=helium_M[0],
        mode="lines",
        legendgroup="He",
        showlegend=False,
        line=dict(color=colors[1]),
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2,
    col=1,
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=logR_grid, y=hydrogen_R[i]),  # trace 0
                go.Scatter(x=logR_grid, y=helium_R[i]),  # trace 0
                go.Scatter(x=np.log10(m_grid), y=entropy_M[i]),  # trace 1
                go.Scatter(x=np.log10(m_grid), y=hydrogen_M[i]),  # trace 1
                go.Scatter(x=np.log10(m_grid), y=helium_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, 2, 3, 4, 5, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
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
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
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

# axis labels
fig.update_xaxes(title_text="internal mass", row=1, col=1)
fig.update_xaxes(title_text="log(M - m)", row=1, col=2)
fig.update_xaxes(title_text="age", row=2, col=1)

# fig.update_yaxes(title_text="mass ratio", range=[-2.5, 0.2], row=1, col=1)
# fig.update_yaxes(range=[-2.5, 0.2], row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w21/dredge-up.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )
#


# %%
from scipy.signal import savgol_filter

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())
n = len(profiles)

n_grid = 200

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = 0.3
logR_max = 1.0
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = min(np.min(10**p.logR) for p in profiles)
m_max = max(np.max(10**p.logR) for p in profiles)
m_grid = np.logspace(np.log10(m_min), np.log10(m_max), n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
hydrogen_R = np.empty((n, n_grid))
helium_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid))
hydrogen_M = np.empty((n, n_grid))
helium_M = np.empty((n, n_grid))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.mass[::-1][1:]
    yR = np.cumsum(p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1]))
    HR = savgol_filter(yR, 52, 3)
    HeR = savgol_filter(HR, 52, 3)
    # HR = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    # HeR = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])

    logR_grid = np.linspace(logR_min, logR_max, n_grid)
    logR_grid.sort()
    xM = 10 ** p.logR[::-1][1:]
    yM = np.cumsum(p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1]))
    HM = savgol_filter(yM, 52, 3)
    HeM = savgol_filter(HM, 52, 3)
    # HM = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])
    # HeM = p.eps_grav_ad[::-1][1:] * np.diff(p.mass[::-1])

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=-100, right=np.nan)
    hydrogen_R[i] = np.interp(logR_grid, xR, HR, left=-100, right=np.nan)
    hydrogen_R[i] = np.nan_to_num(hydrogen_R[i], neginf=-99, nan=np.nan)
    helium_R[i] = np.interp(logR_grid, xR, HeR, left=-100, right=np.nan)
    entropy_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), yM, left=np.nan, right=np.nan
    )
    hydrogen_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), HM, left=np.nan, right=np.nan
    )
    helium_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), HeM, left=np.nan, right=np.nan
    )

# radius evolution
radius = np.array([p.logR.max() for p in profiles])
print(p.bulk_names)
age = np.array(
    [
        np.log10(p.star_mass - p.he_core_mass) if hasattr(p, "star_mass") else i
        for i, p in enumerate(profiles)
    ]
)

# use index instead of age for stability
idx = np.arange(n)


# --------------------------------------------------
# figure layout
# --------------------------------------------------

fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.7, 0.3],
    specs=[
        [{}, {}],
        [{"colspan": 2}, None],
    ],
)

# --------------------------------------------------
# static traces
# --------------------------------------------------

# entropy vs logR
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=entropy_R[0],
        mode="lines",
        name="Z",
        legendgroup="Z",
        line=dict(color=colors[2]),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=hydrogen_R[0],
        mode="lines",
        name="H",
        legendgroup="H",
        line=dict(color=colors[0]),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=helium_R[0],
        mode="lines",
        name="He",
        legendgroup="He",
        line=dict(color=colors[1]),
    ),
    row=1,
    col=1,
)
# entropy vs mass
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=entropy_M[0],
        mode="lines",
        legendgroup="Z",
        showlegend=False,
        line=dict(color=colors[2]),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=hydrogen_M[0],
        mode="lines",
        legendgroup="H",
        showlegend=False,
        line=dict(color=colors[0]),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=helium_M[0],
        mode="lines",
        legendgroup="He",
        showlegend=False,
        line=dict(color=colors[1]),
    ),
    row=1,
    col=2,
)

# radius evolution curve
fig.add_trace(
    go.Scatter(
        x=age,
        y=radius,
        mode="lines",
        name="radius",
        line=dict(color="black"),
    ),
    row=2,
    col=1,
)

# moving marker
marker_trace_index = len(fig.data)

fig.add_trace(
    go.Scatter(
        x=[age[0]],
        y=[radius[0]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="current",
    ),
    row=2,
    col=1,
)

# --------------------------------------------------
# frames
# --------------------------------------------------

frames = []

for i in range(n):

    frames.append(
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=logR_grid, y=entropy_R[i]),  # trace 0
                go.Scatter(x=logR_grid, y=hydrogen_R[i]),  # trace 0
                go.Scatter(x=logR_grid, y=helium_R[i]),  # trace 0
                go.Scatter(x=np.log10(m_grid), y=entropy_M[i]),  # trace 1
                go.Scatter(x=np.log10(m_grid), y=hydrogen_M[i]),  # trace 1
                go.Scatter(x=np.log10(m_grid), y=helium_M[i]),  # trace 1
                # marker ONLY (must stay marker mode!)
                go.Scatter(
                    x=[age[i]],
                    y=[radius[i]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                ),  # trace 3 (marker)
            ],
            traces=[0, 1, 2, 3, 4, 5, marker_trace_index],
        )
    )

fig.frames = frames

# --------------------------------------------------
# slider
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
    for i in range(n)
]

# --------------------------------------------------
# layout
# --------------------------------------------------

fig.update_layout(
    template="simple_white",
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

# axis labels
fig.update_xaxes(title_text="internal mass", row=1, col=1)
fig.update_xaxes(title_text="log(M - m)", row=1, col=2)
fig.update_xaxes(title_text="age", row=2, col=1)

# fig.update_yaxes(title_text="mass ratio", range=[-2.5, 0.2], row=1, col=1)
# fig.update_yaxes(range=[-2.5, 0.2], row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w21/dredge-up.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )
#


# %%
profiles = list(l.profile_dict.values())
# %%


list_of_models = []

for i, name in enumerate(
    [
        "1e-1",
        "1e-1.1",
        "1e-1.2",
        "1e-1.3",
        "1e-1.4",
        "1e-1.5",
        "1e-1.6",
        "1e-1.7",
        "1e-1.8",
        "1e-1.9",
        "1e-2",
        "1e-2.5",
        "1e-3",
    ]
):
    l = mr.MesaLogDir(f"{MASTER}/constant-mass-loss-rate/{name}/LOGS/TPAGB")
    for profile in l.profile_numbers:
        profile = l.profile_data(profile_number=profile)

    profiles = list(l.profile_dict.values())

    list_of_models.append(profiles)

true_list = list_of_models.copy()
for i in range(len(list_of_models)):
    if i in [5, 10, 11, 12]:
        true_list[i] = list_of_models[i][1:]

# %%
for i in range(len(list_of_models)):
    true_list[i] = true_list[i][::2]


# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.cm as cm

rates = [-1, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.5, -3]
mass_loss = np.array(rates)

norm = (mass_loss - mass_loss.min()) / (mass_loss.max() - mass_loss.min() + 1e-12)

cmap = cm.viridis
from matplotlib.colors import to_hex

colors = [cmap(v) for v in norm]
colors = [to_hex(c) for c in colors]
print(colors)
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
    eps = profile.eps_grav_ad[::-1][1:]

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

fig.update_yaxes(title_text="cumulative L_grav", row=1, col=1, range=[-8000, 2000])
fig.update_yaxes(title_text="cumulative L_grav", row=1, col=2, range=[-8000, 2000])
fig.update_yaxes(title_text="log radius", row=2, col=1)
fig.update_yaxes(title_text="log L", row=2, col=2)
#
# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w21/losing-envelope.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )
#


fig.show()
# %%


models = []

for name in [
    "1e-1",
    "1e-1.1",
    "1e-1.2",
    "1e-1.3",
    "1e-1.4",
    "1e-1.5",
    "1e-1.6",
    "1e-1.7",
    "1e-1.8",
    "1e-1.9",
    "1e-2",
    "1e-2.5",
    "1e-3",
]:
    model = mr.MesaData(
        f"{MASTER}/constant-mass-loss-rate/{name}/LOGS/TPAGB/history.data"
    )
    models.append(model)

# %%

for model in models:
    plt.plot(model.envelope_mass, model.R)

plt.yscale("log")
plt.xscale("log")
plt.show()
# %%
for model in models:
    plt.plot(model.log_Teff, model.log_L)

plt.gca().invert_xaxis()

plt.show()

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.cm as cm

rates = [-1, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.5, -3]
mass_loss = np.array(rates)

norm = (mass_loss - mass_loss.min()) / (mass_loss.max() - mass_loss.min() + 1e-12)

cmap = cm.viridis
from matplotlib.colors import to_hex

colors = [cmap(v) for v in norm]
colors = [to_hex(c) for c in colors]
print(colors)
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
    eps = profile.eps_grav_ad[::-1][1:]

    dm = np.diff(m)

    lgrav = profile.logS_per_baryon[::-1]
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
    title_text="log S",
    row=1,
    col=1,
    # range=[-8000, 2000],
)
fig.update_yaxes(
    title_text="log S",
    row=1,
    col=2,
    # range=[-8000, 2000],
)
fig.update_yaxes(title_text="log radius", row=2, col=1)
fig.update_yaxes(title_text="log L", row=2, col=2)

fig.write_html(
    "/home/koen/figures/plots/master-internship/w21/losing-envelope-entropy.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)


fig.show()

# %%

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.cm as cm

rates = [-1, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.5, -3]
mass_loss = np.array(rates)

norm = (mass_loss - mass_loss.min()) / (mass_loss.max() - mass_loss.min() + 1e-12)

cmap = cm.viridis
from matplotlib.colors import to_hex

colors = [cmap(v) for v in norm]
colors = [to_hex(c) for c in colors]
print(colors)
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
    eps = profile.eps_grav_ad[::-1][1:]

    dm = np.diff(m)

    lgrav = profile.logRho[::-1]
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
    title_text="log Rho",
    row=1,
    col=1,
    # range=[-8000, 2000],
)
fig.update_yaxes(
    title_text="log Rho",
    row=1,
    col=2,
    # range=[-8000, 2000],
)
fig.update_yaxes(title_text="log radius", row=2, col=1)
fig.update_yaxes(title_text="log L", row=2, col=2)

fig.write_html(
    "/home/koen/figures/plots/master-internship/w21/losing-envelope-density.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)


fig.show()
# %%

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.cm as cm

rates = [-1, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.5, -3]
mass_loss = np.array(rates)

norm = (mass_loss - mass_loss.min()) / (mass_loss.max() - mass_loss.min() + 1e-12)

cmap = cm.viridis
from matplotlib.colors import to_hex

colors = [cmap(v) for v in norm]
colors = [to_hex(c) for c in colors]
print(colors)
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
    eps = profile.eps_grav_ad[::-1][1:]

    dm = np.diff(m)

    lgrav = profile.logT[::-1]
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
    title_text="log opacity",
    row=1,
    col=1,
    # range=[-8000, 2000],
)
fig.update_yaxes(
    title_text="log opacity",
    row=1,
    col=2,
    # range=[-8000, 2000],
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

models = []

for name in [
    "1e-1",
    "1e-1.1",
    "1e-1.2",
    "1e-1.3",
    "1e-1.4",
    "1e-1.5",
    "1e-1.6",
    "1e-1.7",
    "1e-1.8",
    "1e-1.9",
    "1e-2",
    "1e-2.5",
    "1e-3",
]:
    model = mr.MesaData(
        f"{MASTER}/constant-mass-loss-rate/{name}/LOGS/TPAGB/history.data"
    )
    models.append(model)

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
for i, model in enumerate(models):
    plt.plot(model.envelope_mass, model.R, c=colors[i])

plt.gca().set_facecolor("C9")
plt.yscale("log")
plt.xscale("log")


plt.xlabel(r"$\log(M_\textrm{env}/M_\odot)$")
plt.ylabel(r"$\log(R/R_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-constant-mass-loss-rate-radius.pgf", format="pgf"
)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
for i, model in enumerate(models):
    plt.plot(model.log_Teff, model.log_L, c=colors[i])

plt.gca().set_facecolor("C9")
plt.gca().invert_xaxis()

plt.xlabel(r"$\log(T_\textrm{eff}/\textrm{K})$")
plt.ylabel(r"$\log(L/L_\odot)$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-constant-mass-loss-rate-HR.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
