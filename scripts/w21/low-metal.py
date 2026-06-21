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

low = mr.MesaData(f"{MASTER}single-stars/z0.00453/M2.0/LOGS/TPAGB/history.data")
# %%
med = mr.MesaData(f"{MASTER}single-stars/z0.00883/M2.0/LOGS/TPAGB/history.data")

# %%
sol = mr.MesaData(f"{MASTER}standard-2msun-v3/LOGS/TPAGB/history.data")
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")

plt.plot(low.star_age, low.R, label="$Z=0.00453$")
plt.plot(med.star_age, med.R, label="$Z=0.00883$")
plt.plot(sol.star_age, sol.R, label="$Z=0.014$")

fig.legend(loc="outside upper center", ncols=3)
plt.savefig("/home/koen/LaTeX-setup/plots/w21-radius.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(low.log_Teff, low.log_L)
plt.gca().invert_xaxis()
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Time (yr)")
plt.ylabel("C/O-number ratio")
plt.plot(low.star_age, low.surface_c12 / low.surface_o16 * 16 / 12, label="$Z=0.00453$")
plt.plot(med.star_age, med.surface_c12 / med.surface_o16 * 16 / 12, label="$Z=0.00883$")
plt.plot(sol.star_age, sol.surface_c12 / sol.surface_o16 * 16 / 12, label="$Z=0.014$")
fig.legend(loc="outside upper center", ncols=3)
plt.savefig("/home/koen/LaTeX-setup/plots/w21-co.pgf", format="pgf")
plt.show()
plt.close()
# %%

low.bulk_names
# %%
prof = mr.MesaData(f"{MASTER}single-stars/z0.00453/M2.0/LOGS/TPAGB/profile1000.data")
# %%
prof.bulk_names

# %%

plt.plot(prof.mass, prof.z_mass_fraction_metals)
plt.plot(prof.mass, prof.x_mass_fraction_H)
plt.plot(prof.mass, prof.y_mass_fraction_He)
plt.ylim(1e-3, 1.2)
plt.yscale("log")
plt.show()

# %%

plt.plot(prof.logR, prof.z_mass_fraction_metals)
plt.plot(prof.logR, prof.x_mass_fraction_H)
plt.plot(prof.logR, prof.y_mass_fraction_He)
plt.ylim(1e-3, 1.2)
plt.yscale("log")
plt.show()
# %%
l = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00453/M2.0/LOGS/TPAGB")

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

profiles = list(l.profile_dict.values())[60:]
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
    yR = p.logz_mass_fraction_metals[::-1]

    xM = p.mass[::-1]
    yM = p.logz_mass_fraction_metals[::-1]

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
profile.bulk_names
# %%

plt.plot(low.model_number, low.surface_c12 / low.surface_o16 * 16 / 12)
plt.show()
# %%

low_zoom = mr.MesaData(
    f"{MASTER}single-stars/z0.00453/M2.0-zoom/LOGS/TPAGB/history.data"
)
# %%

plt.plot(low.model_number, low.surface_c12 / low.surface_o16 * 16 / 12)
plt.plot(low_zoom.model_number, low_zoom.surface_c12 / low_zoom.surface_o16 * 16 / 12)
plt.show()
# %%


plt.plot(low.model_number, low.varcontrol)
plt.plot(low_zoom.model_number, low_zoom.varcontrol)
plt.show()
# %%
l2 = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00453/M2.0-zoom/LOGS/TPAGB")

# %%
for profile in l2.profile_numbers:
    profile = l2.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)

# %%

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l2.profile_dict.values())
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
m_grid = np.logspace(-13, np.log10(m_max), n_grid)

print(m_grid)
# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid))
entropy_M = np.empty((n, n_grid))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.logR[::-1]
    yR = p.c12[::-1] / p.o16[::-1] * 16 / 12

    xM = p.mass[0] - p.mass
    print(xM)
    yM = p.c12 / p.o16 * 16 / 12

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    entropy_M[i] = np.interp(
        np.log10(m_grid), np.log10(xM), yM, left=np.nan, right=np.nan
    )

# radius evolution
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
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
        x=np.log10(m_grid),
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
                go.Scatter(x=np.log10(m_grid), y=entropy_M[i]),  # trace 1
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

plt.plot(profiles[-100].logR, (profiles[-100].c12 / profiles[-100].o16))
plt.plot(profiles[-1].logR, (profiles[-1].c12 / profiles[-1].o16))
plt.show()

# %%

plt.plot(profiles[0].zone[1:], np.abs(np.diff(profiles[0].mass) / profiles[0].mass[0]))
plt.plot(
    profiles[-1].zone[1:], np.abs(np.diff(profiles[-1].mass) / profiles[-1].mass[0])
)
plt.plot(profiles[-1].zone, profiles[-1].c12 / profiles[-1].o16 * 16 / 12)
plt.plot(profiles[0].zone, profiles[0].c12 / profiles[0].o16 * 16 / 12)
plt.show()
# %%

plt.plot(profiles[-1].mass[0] - profiles[-1].mass, profiles[-1].z_mass_fraction_metals)
plt.plot(profiles[-1].mass[0] - profiles[0].mass, profiles[0].z_mass_fraction_metals)
plt.show()

# %%

plt.plot(
    profiles[0].mass[0] - profiles[0].mass, profiles[0].c12 / profiles[0].o16 * 16 / 12
)
plt.plot(
    profiles[-1].mass[0] - profiles[-1].mass,
    profiles[-1].c12 / profiles[-1].o16 * 16 / 12,
)
plt.show()


# %%

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
m_grid = np.logspace(-9, np.log10(m_max), n_grid)

print(m_grid)
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
    xR = p.logR[::-1]
    yR = np.log10(p.z_mass_fraction_metals[::-1])
    HR = np.log10(p.x_mass_fraction_H[::-1])
    HeR = np.log10(p.y_mass_fraction_He[::-1])
    xM = p.mass[0] - p.mass
    yM = np.log10(p.z_mass_fraction_metals)
    HM = np.log10(p.x_mass_fraction_H)
    HeM = np.log10(p.y_mass_fraction_He)

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=np.nan, right=np.nan)
    hydrogen_R[i] = np.interp(logR_grid, xR, HR, left=np.nan, right=np.nan)
    helium_R[i] = np.interp(logR_grid, xR, HeR, left=np.nan, right=np.nan)
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
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
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
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=hydrogen_R[0],
        mode="lines",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=logR_grid,
        y=helium_R[0],
        mode="lines",
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
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=hydrogen_M[0],
        mode="lines",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.log10(m_grid),
        y=helium_M[0],
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
fig.update_xaxes(title_text="log R", row=1, col=1)
fig.update_xaxes(title_text="mass coordinate", row=1, col=2)
fig.update_xaxes(title_text="profile index", row=2, col=1)

fig.update_yaxes(title_text="entropy", range=[-3, 0.2], row=1, col=1)
fig.update_yaxes(title_text="entropy", range=[-3, 0.2], row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

# fig.write_html(
#     "/home/koen/figures/plots/master-internship/w20/post-AGB-instabilities.html",
#     include_plotlyjs="cdn",
#     include_mathjax="cdn",
#     config={"responsive": True},
# )


# %%

pio.templates.default = "simple_white"

colors = px.colors.qualitative.D3


# --------------------------------------------------
# data
# --------------------------------------------------

profiles = list(l.profile_dict.values())[::2]
n = len(profiles)

n_grid = 200

# --------------------------------------------------
# grids
# --------------------------------------------------

logR_min = 0.45
logR_max = 0.70
logR_grid = np.linspace(logR_min, logR_max, n_grid)

m_min = 0.0
m_max = max(np.max(p.mass) for p in profiles)
m_grid = np.logspace(-9, np.log10(m_max), n_grid)

# --------------------------------------------------
# precompute interpolations
# --------------------------------------------------

entropy_R = np.empty((n, n_grid + 1))
hydrogen_R = np.empty((n, n_grid + 1))
helium_R = np.empty((n, n_grid + 1))
entropy_M = np.empty((n, n_grid))
hydrogen_M = np.empty((n, n_grid))
helium_M = np.empty((n, n_grid))

for i, p in enumerate(profiles):

    # ensure monotonic x for interpolation
    xR = p.mass[::-1]
    yR = np.log10(p.z_mass_fraction_metals[::-1])
    HR = np.log10(p.x_mass_fraction_H[::-1])
    HeR = np.log10(p.y_mass_fraction_He[::-1])

    logR_grid = np.linspace(logR_min, logR_max, n_grid)
    index = np.argwhere(HR > -10)[0][0] - 1
    logR_grid = np.append(logR_grid, p.mass[index])
    logR_grid.sort()
    xM = p.mass[0] - p.mass
    yM = np.log10(p.z_mass_fraction_metals)
    HM = np.log10(p.x_mass_fraction_H)
    HeM = np.log10(p.y_mass_fraction_He)

    entropy_R[i] = np.interp(logR_grid, xR, yR, left=-100, right=np.nan)
    hydrogen_R[i] = np.interp(logR_grid, xR, HR, left=-100, right=np.nan)
    hydrogen_R[i] = np.nan_to_num(hydrogen_R[i], neginf=-99, nan=np.nan)
    print(hydrogen_R[i])
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
radius = np.array([10 ** p.logR.max() for p in profiles])
age = np.array(
    [p.star_age if hasattr(p, "star_age") else i for i, p in enumerate(profiles)]
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

fig.update_yaxes(title_text="mass ratio", range=[-2.5, 0.2], row=1, col=1)
fig.update_yaxes(range=[-2.5, 0.2], row=1, col=2)
fig.update_yaxes(title_text="radius", row=2, col=1)

fig.show()

fig.write_html(
    "/home/koen/figures/plots/master-internship/w21/dredge-up.html",
    include_plotlyjs="cdn",
    include_mathjax="cdn",
    config={"responsive": True},
)


# %%

profiles = list(l.profile_dict.values())

age = []
z = []
for profile in profiles:
    age.append(profile.star_age)
    z.append(profile.z_mass_fraction_metals[0])

co_interp = np.interp(age, low.star_age, low.surface_c12 / low.surface_o16 * 16 / 12)
plt.plot(age, z)
plt.show()
# %%
profiles = list(l.profile_dict.values())

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Time (yr)")
plt.ylabel("$Z$-fraction")

age = []
z = []
for profile in profiles:
    age.append(profile.star_age)
    z.append(profile.z_mass_fraction_metals[0])

co_interp = np.interp(age, low.star_age, low.surface_c12 / low.surface_o16 * 16 / 12)
plt.plot(age, z)
plt.yscale("log")

plt.savefig("/home/koen/LaTeX-setup/plots/w21-Z-surface.pgf", format="pgf")
plt.show()
plt.close()
# %%

low.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model, m in zip([low, med, sol], [r"$Z=0.00453$", r"$Z=0.00883$", r"$Z=0.014$"]):
    index = []

    for i in range(2, model.TP_count[-1] + 1):
        for j in range(len(model.model_number)):
            if (
                model.TP_count[j] == i
                and np.argmax(model.lambda_DUP[j - 1000 : j + 1000]) == 1000
            ):
                index.append(j)
                break

    plt.plot(model.TP_count[index], model.lambda_DUP[index], ".-", label=m)


fig.legend(loc="outside upper center", ncols=3)
plt.xlim(0)
plt.xticks([2, 5, 10, 15, 20])
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\lambda_\textrm{DUP}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w21-lambda.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(low.star_age / 20, low.he_core_mass)
plt.plot(med.star_age / 18, med.he_core_mass)
plt.plot(sol.star_age / 17, sol.he_core_mass)
plt.show()
# %%

profiles = list(l.profile_dict.values())

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Time (yr)")
plt.ylabel("$Z$-fraction")

age = []
z = []
for profile in profiles:
    age.append(profile.star_age)
    vals = np.nan_to_num(profile.x_mass_fraction_H, neginf=-99, nan=-99)
    index = np.argwhere(np.log10(vals) > -0.17)
    z.append(np.mean(profile.z_mass_fraction_metals[index].flatten()))


plt.plot(age, z, label="envelope")
age = []
z = []
for profile in profiles:
    age.append(profile.star_age)
    z.append(profile.z_mass_fraction_metals[0])

plt.plot(age, z, label="surface")

plt.yscale("log")


fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w21-Z-envelope.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.show()

# %%
