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
    figsize=set_size(full, height=0.5),
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
# plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist-CO.pgf", format="pgf")
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

plt.fill_between([1.175, 1.250], [-1.5, -1.5], [1.5, 1.5], zorder=-1, alpha=0.2)

axs.spines[["right", "top", "left"]].set_visible(False)
axs.tick_params(axis="y", which="both", length=0)
plt.xlim(1.175, 1.375)
plt.xticks([1.175, 1.2, 1.25, 1.3, 1.35, 1.375])
plt.annotate(
    "wind",
    (1.225, 1.2),
    (1.2625, 1.2),
    arrowprops=dict(arrowstyle="-|>", color="k"),
    ha="center",
    va="center",
)

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
plt.show()


# %%

m.bulk_names
# %%


for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    plt.plot(
        m.age,
        np.cumsum((-m.wind_mass_transfer / (1.989 * 10**33 / 3.15576e7)) * 10 * m.dt),
    )
plt.show()
# %%

plt.plot(m.age, 10 * m.dt)
plt.show()

# %%
for i, m in enumerate(grid.filter(f_beta=grid.axes["beta"][3])):
    plt.plot(m.age, m.rl_1)
    plt.plot(m.age, m.R)

plt.xlim(2271694821.770604, 2271695636.1188197)
plt.show()

# %%
