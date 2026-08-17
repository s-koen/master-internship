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
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/const-ba-star-mass-grid")


# %%


def find_CO(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    try:
        ind = np.argwhere(r.rl_1 < r.R)[0][0]
        return r.period_days[-1]
    except:
        return np.nan


fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=0.8),
    constrained_layout=True,
)


minn = 1e99
maxx = -1e-99

axs = axs.flatten()
axs[5].axis("off")

for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])

    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])
    c = axs[i].pcolormesh(
        RL,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, r in enumerate(RL):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{ratio[k, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\beta / (1-\epsilon) = {grid.axes["beta"][i]}$")


axs[4].set_xlabel(r"$R_\textrm{Roche lobe, initial}$ ($R_\odot$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Final period (days)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-period.pgf", format="pgf")
plt.show()
plt.close()


plt.show()

# %%

for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    print(i, m.params["q"], m.params["R"])

# %%


def find_CO(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    try:
        ind = np.argwhere(r.rl_1 < r.R)[0][0]
        return r.star_2_mass[-1]
    except:
        return np.nan


fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=0.9),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e-99

axs = axs.flatten()

for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])

    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])
    c = axs[i].pcolormesh(
        RL,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, r in enumerate(RL):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{np.round(ratio[k, j], 2):.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\beta / (1-\epsilon) = {grid.axes["beta"][i]}$")


axs[4].set_xlabel(r"$R_\textrm{Roche lobe, initial}$ ($R_\odot$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Observed CO-ratio (complete mixing in barium star)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-mass.pgf", format="pgf")
plt.show()
plt.close()


plt.show()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.25), constrained_layout=True
)
masses = []
for m in grid.models:
    if m.envelope_mass[-1] <= 0.01:
        masses.append(m.star_2_mass[-1])

np.random.shuffle(masses)
masses.sort()
plt.scatter(
    masses[0::4],
    np.array(masses)[0::4] * 0 + 0.3,
    s=1.5,
    marker="o",
    color="k",
)
plt.scatter(
    masses[1::4],
    np.array(masses)[1::4] * 0 + 0.1,
    s=1.5,
    marker="o",
    color="k",
)
plt.scatter(
    masses[2::4],
    np.array(masses)[2::4] * 0 - 0.1,
    s=1.5,
    marker="o",
    color="k",
)
plt.scatter(
    masses[3::4],
    np.array(masses)[3::4] * 0 - 0.3,
    s=1.5,
    marker="o",
    color="k",
)

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
plt.xlim(1.175, 1.375)
plt.xticks([1.175, 1.2, 1.25, 1.3, 1.35, 1.375])
plt.annotate(
    "wind",
    (1.24, 1.2),
    (1.29, 1.2),
    arrowprops=dict(arrowstyle="-|>", color="k"),
    ha="center",
    va="center",
)
plt.annotate(
    "wind",
    (1.34, 1.2),
    (1.29, 1.2),
    arrowprops=dict(arrowstyle="-|>", color="k"),
    ha="center",
    va="center",
)

plt.plot([1.29, 1.29], [-1, 0.75], c="C9", zorder=-1000, linewidth=0.8)
plt.xlabel(r"Final Barium star mass ($M_\odot$)")
plt.yticks([0, 1], "")
plt.ylim(-1, 1.5)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-scatter.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.25), constrained_layout=True
)
masses = []
radii = []
for m in grid.models:
    if m.envelope_mass[-1] <= 0.01:
        masses.append(m.star_2_mass[-1])
        radii.append(m.params["R"])

plt.scatter(
    radii,
    np.array(masses),
    s=1.5,
    marker="o",
    color="k",
)

# plt.fill_between([1.175, 1.250], [-1.5, -1.5], [1.5, 1.5], zorder=-1, alpha=0.2)

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
# plt.xlim(1.175, 1.375)
# plt.xticks([1.175, 1.2, 1.25, 1.3, 1.35, 1.375])
# plt.annotate(
#     "wind",
#     (1.225, 1.2),
#     (1.2625, 1.2),
#     arrowprops=dict(arrowstyle="-|>", color="k"),
#     ha="center",
#     va="center",
# )

plt.xlabel(r"Final Barium star mass ($M_\odot$)")
# plt.yticks([0, 1], "")
# plt.ylim(-1, 1.5)
# plt.savefig("/home/koen/LaTeX-setup/plots/w23-scatter.pgf", format="pgf")
plt.show()
plt.close()
# %%


def find_CO(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    try:
        ind = np.argwhere(r.rl_1 < r.R)[0][0]
        return r.star_2_mass[-1] - r.star_2_mass[0]
    except:
        return np.nan


fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=0.8),
    constrained_layout=True,
)


minn = 1e99
maxx = -1e-99

axs = axs.flatten()
axs[5].axis("off")

for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])

    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])
    c = axs[i].pcolormesh(
        RL,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, r in enumerate(RL):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{ratio[k, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\beta / (1-\epsilon) = {grid.axes["beta"][i]}$")


axs[4].set_xlabel(r"$R_\textrm{Roche lobe, initial}$ ($R_\odot$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Final period (days)",
)
plt.xlabel("")
plt.ylabel("")
# plt.savefig("/home/koen/LaTeX-setup/plots/w23-period.pgf", format="pgf")
plt.show()
plt.close()


plt.show()


# %%

for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    if i > 0:
        continue
    plt.plot(m.age, ((-m.wind_mass_transfer / (1.989 * 10**33 / 3.15576e7))))
    plt.plot(m.age, 10**m.lg_wind_mdot_1 * (1 - m.beta_accretion))
plt.show()


# %%

m.bulk_names
# %%

norm = plt.Normalize(400, 600)
cmap = plt.cm.viridis
# color = cmap(norm(x))


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")

for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    try:
        ind = np.where(0.9 * m.R > m.rl_1)[0][0]
    except:
        continue
    plt.plot(
        m.star_1_mass,
        np.cumsum(10**m.lg_wind_mdot_1 * (1 - 0 * m.beta_accretion) * 10 * m.dt)
        - np.sum(
            10 ** m.lg_wind_mdot_1[:ind]
            * (1 - 0 * m.beta_accretion[:ind])
            * 10
            * m.dt[:ind]
        ),
        color=cmap(norm(m.params["R"])),
    )
plt.show()
# %%

plt.plot(m.star_1_mass, 10 * m.dt)
plt.show()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    plt.plot(m.age, m.R, c="C9")
    plt.plot(m.age, m.rl_1, c="k")

plt.xlim(2271694821.770604, 2271695636.1188197)
plt.ylim(100, 500)
plt.xlabel("Time (yr)")
plt.ylabel("Radius (yr)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-wrong-start.pgf", format="pgf")
plt.show()
plt.close()

# %%


def find_CO(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    if r.period_days[-1] < 50:
        return np.nan
    try:
        ind = np.argwhere(r.rl_1 < r.R)[0][0]
        return r.star_2_mass[-1] - 1.29
    except:
        return np.nan


fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=0.9),
    constrained_layout=True,
)

minn = 1e99
maxx = -1e-99

axs = axs.flatten()

for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])

    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):
    if i == 5:
        continue

    RL, q, ratio = grid.array(find_CO, x="R", y="q", f_beta=grid.axes["beta"][i])
    c = axs[i].pcolormesh(
        RL,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, r in enumerate(RL):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    r,
                    qq,
                    f"{np.round(ratio[k, j], 2):.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\beta / (1-\epsilon) = {grid.axes["beta"][i]}$")


axs[4].set_xlabel(r"$R_\textrm{Roche lobe, initial}$ ($R_\odot$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Observed CO-ratio (complete mixing in barium star)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-mass.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(400, 600)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i, m in enumerate(grid.models):
    try:
        ind = np.where(0.9 * m.R > m.rl_1)[0][0]
    except:
        continue

    plt.plot(
        m.envelope_mass[ind:],
        np.cumsum(
            10 ** m.lg_mstar_dot_1[ind:]
            * (1 - m.fixed_xfer_fraction[ind:])
            * 10
            * m.dt[ind:]
        )
        - np.cumsum(
            10 ** m.lg_mstar_dot_1[ind:]
            * (1 - m.eff_xfer_fraction[ind:])
            * 10
            * m.dt[ind:]
        ),
        c=cmap(norm(m.params["R"])),
    )

# plt.xlim(2271694821.770604, 2271695636.1188197)
# plt.ylim(100,500)
plt.xlabel("Time (yr)")
plt.ylabel("Radius (yr)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-wrong-start.pgf", format="pgf")
plt.show()
plt.close()


# %%
m.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(400, 600)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i, m in enumerate(grid.models):
    try:
        ind = np.where(0.9 * m.R > m.rl_1)[0][0]
    except:
        continue

    plt.plot(
        m.envelope_mass[ind:],
        np.cumsum(
            (10 ** m.lg_mstar_dot_1[ind:] * m.dt[ind:] * 10 * m.eff_xfer_fraction[ind:])
        )
        - np.cumsum((10 ** m.lg_mstar_dot_1[ind:] * m.dt[ind:] * 10 * m.params["eps"])),
        c=cmap(norm(m.params["R"])),
    )

# plt.xlim(2271694821.770604, 2271695636.1188197)
# plt.ylim(100,500)
plt.xlabel("Time (yr)")
plt.ylabel("Radius (yr)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-wrong-start.pgf", format="pgf")
plt.show()
plt.close()


# %%

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
giant = data[
    data["class"].isin(
        [
            "Ba 0",
            "Ba 0.5",
            "Ba 1",
            "Ba 2",
            "Ba mild",
            "sgCH",
            "sgCH / Ba 1",
            "Ba 3",
            "Ba 4",
            "Ba 5",
            "Ba strong",
            "Ba 5 / eS",
        ]
    )
]


d = dwarf["M"].dropna().to_numpy().astype(np.float64)


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.25), constrained_layout=True
)


plt.scatter(d, d * 0 + 1, s=1.5, marker="o", color="k")

masses = []
for m in grid.models:
    if m.envelope_mass[-1] <= 0.01:
        masses.append(m.star_2_mass[-1])

np.random.shuffle(masses)
# masses.sort()
rows = 5
for i in range(rows):

    plt.scatter(
        masses[i::rows],
        np.array(masses)[i::rows] * 0 - 0.15 * rows / 2 + 0.15 * i,
        s=1.5,
        marker="o",
        color="k",
    )

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
plt.xlim(1.175, 1.375)
plt.xticks([0.7, 1, 1.25, 1.5, 1.75])
plt.xlabel(r"Final Barium star mass ($M_\odot$)")
plt.yticks([0, 1], ["MESA", "actual"])
plt.ylim(-0.75, 1.5)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-scatter-compare.pgf", format="pgf")
plt.show()
plt.close()


# %%
data = pd.read_csv("presentation-1/Ba_star_orbits.csv")
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
giant = data[
    data["class"].isin(
        [
            "Ba 0",
            "Ba 0.5",
            "Ba 1",
            "Ba 2",
            "Ba mild",
            "sgCH",
            "sgCH / Ba 1",
            "Ba 3",
            "Ba 4",
            "Ba 5",
            "Ba strong",
            "Ba 5 / eS",
        ]
    )
]


d = dwarf["M"].dropna().to_numpy().astype(np.float64)
d.sort()
d
# %%

import pandas as pd

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
giant = data[
    data["class"].isin(
        [
            "Ba 0",
            "Ba 0.5",
            "Ba 1",
            "Ba 2",
            "Ba mild",
            "sgCH",
            "sgCH / Ba 1",
            "Ba 3",
            "Ba 4",
            "Ba 5",
            "Ba strong",
            "Ba 5 / eS",
        ]
    )
]


d = dwarf["M"].dropna().to_numpy().astype(np.float64)


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.175), constrained_layout=True
)


plt.scatter(d, d * 0, s=1.5, marker="o", color="k")

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
plt.xlim(1.175, 1.375)
plt.xticks([0.7, 1, 1.25, 1.5, 1.75])
plt.xlabel(r"Final Barium star mass ($M_\odot$)")
plt.yticks([])
plt.ylim(-0.5, 0.5)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-scatter-dwarf-masses.pgf", format="pgf")
plt.show()
plt.close()


# %%
