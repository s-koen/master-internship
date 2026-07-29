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

grid = MesaGrid(f"{MASTER}epsilon-grid")


# %%
def find_period(r):
    if r.period_days[-1] < 50:
        return np.nan
    else:
        return r.period_days[-1]


def find_mass(r):
    if r.period_days[-1] < 50:
        return np.nan
    else:
        return r.star_2_mass[-1]


epss = grid.axes["eps"]
delta, q, ratio = grid.array(
    find_mass,
    x="delta",
    y="q",
    eps=epss[-1],
)

print(q, delta, ratio)
plt.pcolormesh(delta, q, ratio.T, shading="auto", cmap="viridis")
plt.colorbar()
plt.show()
# %%
fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_period,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_period, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Period (days)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist.pgf", format="pgf")
plt.show()
plt.close()


# %%
def find_accreted_mass(r):
    if r.period_days[-1] < 50:
        return np.nan
    else:
        return r.star_2_mass[-1] - r.star_2_mass[0]


fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_accreted_mass,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_accreted_mass, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(delta):
        for j, qq in enumerate(q):
            try:
                if str(ratio[k, j]) == str(np.nan):
                    continue
                ax.text(
                    d * (1 - epss[i]),
                    qq,
                    np.round(ratio[k, j], 2),
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label=r"$\Delta M_\textrm{a}$ ($M_\odot$)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist-acc.pgf", format="pgf")
plt.show()
plt.close()
# %%


def find_accreted_mass(r):
    if r.period_days[-1] < 50:
        return np.nan
    else:
        return r.star_2_mass[-1]


fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_accreted_mass,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_accreted_mass, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(delta):
        for j, qq in enumerate(q):
            try:
                if str(ratio[k, j]) == str(np.nan):
                    continue
                ax.text(
                    d * (1 - epss[i]),
                    qq,
                    np.round(ratio[k, j], 2),
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label=r"$M_\textrm{a}$ ($M_\odot$)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist-mass.pgf", format="pgf")
plt.show()
plt.close()
# %%

for i, m in enumerate(grid.filter(eps=grid.axes["eps"][-1])):

    print(m.params)
    plt.plot(m.age, m.R, c=f"C{i}")
    plt.plot(m.age, m.rl_1, c=f"C{i}")

plt.show()


# %%
for i, m in enumerate(grid.filter(eps=grid.axes["eps"][-1])):
    print(len(m.profiles))
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
max = -1e99
min = 1e99

print(profile.bulk_names)

for profile in m.profiles:
    max = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) > max
        else max
    )
    min = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) < min
        else min
    )

print(min, max)
norm = plt.Normalize(min, max)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for profile in m.profiles[::-1]:
    # plt.plot(profile.mass - profile.he_core_mass, profile.entropy, c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    plt.plot(
        profile.mass,
        10**profile.logS_per_baryon,
        c=cmap(norm(np.log10(profile.star_mass - profile.he_core_mass))),
    )

# plt.xscale("log")
# plt.xlim(0.01)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M_\textrm{env}$ ($M_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$m$ ($M_\odot$)")
plt.ylabel(r"$S_\textrm{baryon}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-check-profiles.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
max = -1e99
min = 1e99

print(profile.bulk_names)

for profile in m.profiles:
    max = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) > max
        else max
    )
    min = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) < min
        else min
    )

norm = plt.Normalize(min, max)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for profile in m.profiles[::-3]:
    # plt.plot(profile.mass - profile.he_core_mass, profile.entropy, c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    # plt.plot(np.log(1 - profile.mass / profile.mass[0]), profile.gradT - profile.grada,c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    inds = np.where(profile.gradT - profile.grada > 0)[0]
    groups = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)
    print(groups)
    for group in groups:
        plt.plot(
            np.log(1 - profile.mass[group] / profile.mass[0]),
            profile.log_thermal_time_to_surface[group],
            c="k",
            alpha=0.4,
            linewidth=4,
        )
    plt.plot(
        np.log(1 - profile.mass / profile.mass[0]),
        profile.log_thermal_time_to_surface,
        c=cmap(norm(np.log10(profile.star_mass - profile.he_core_mass))),
    )

plt.gca().invert_xaxis()
plt.xlim(0, -7)
plt.ylim(4, 12)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$\log(M_\textrm{env} / M_\odot)$")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$\log(1 - m/M)$")
plt.ylabel(r"$\log(\tau_\textrm{th} / \textrm{yr})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-check-thermal.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
max = -1e99
min = 1e99

print(profile.bulk_names)

for profile in m.profiles:
    max = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) > max
        else max
    )
    min = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) < min
        else min
    )

norm = plt.Normalize(min, max)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for profile in m.profiles[::-3]:
    # plt.plot(profile.mass - profile.he_core_mass, profile.entropy, c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    # plt.plot(np.log(1 - profile.mass / profile.mass[0]), profile.gradT - profile.grada,c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    inds = np.where(profile.gradT - profile.grada > 0)[0]
    groups = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)
    print(groups)
    for group in groups:
        plt.plot(
            profile.mass[group],
            profile.log_thermal_time_to_surface[group],
            c="k",
            alpha=0.4,
            linewidth=4,
        )
    plt.plot(
        profile.mass,
        profile.log_thermal_time_to_surface,
        c=cmap(norm(np.log10(profile.star_mass - profile.he_core_mass))),
    )

# plt.gca().invert_xaxis()
# plt.xlim(0, -7)
plt.ylim(4, 12)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$\log(M_\textrm{env} / M_\odot)$")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$m$ ($M_\odot$)")
plt.ylabel(r"$\log(\tau_\textrm{th} / \textrm{yr})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-check-thermal-mass.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
max = -1e99
min = 1e99

print(profile.bulk_names)

for profile in m.profiles:
    max = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) > max
        else max
    )
    min = (
        np.log10(profile.star_mass - profile.he_core_mass)
        if np.log10(profile.star_mass - profile.he_core_mass) < min
        else min
    )

norm = plt.Normalize(min, max)
cmap = plt.cm.viridis
# color = cmap(norm(x))


ms = []
rs = []
for profile in m.profiles:
    # plt.plot(profile.mass - profile.he_core_mass, profile.entropy, c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    # plt.plot(np.log(1 - profile.mass / profile.mass[0]), profile.gradT - profile.grada,c= cmap(norm(np.log(profile.star_mass- profile.he_core_mass))))
    inds = np.where(profile.gradT - profile.grada > 0)[0]
    groups = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)
    print(groups)
    # for group in groups:
    #     plt.plot(
    #         profile.mass[group] - profile.he_core_mass,
    #         10**profile.logR[group],
    #         c="k",
    #         alpha=0.4,
    #         linewidth=4,
    #     )
    ms.append(profile.mass[0] - profile.he_core_mass)
    rs.append(10 ** profile.logR[0])


plt.scatter(ms, rs, color="w", s=200, zorder=10, marker=".")
plt.scatter(ms, rs, color="k", zorder=11, marker=".")
plt.plot(ms, rs, c="k", linewidth=0.8)

plt.xscale("log")
# plt.xlim(10**(-2.5))
# plt.gca().invert_xaxis()
# plt.xlim(0, -7)
# plt.ylim(4, 12)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label(r"$M_\textrm{env}$ ($M_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel(r"$R$ ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-check-radius.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_period,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_period, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(delta):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    d * (1 - epss[i]),
                    qq,
                    int(ratio[k, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Period (days)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


for m in grid.filter(eps=grid.axes["eps"][0], q=grid.axes["q"][3]):
    print(m.params)
    print(m.bulk_names)
    plt.plot(m.q, m.period_days, label=f"$\delta = {m.params["delta"]:.2f}$")


fig.legend(loc="outside upper center", ncols=4)
axs.spines[["right", "top"]].set_visible(False)
plt.yscale("log")
plt.xlabel("$q$")
plt.ylabel("Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-darwin-instability.pgf", format="pgf")
plt.show()
plt.close()


# %%
sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

import pickle

proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M1.5"
try:
    with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
        Star = pickle.load(f)
    print("read back combined_star.pkl")
except:

    Star = read_stellar_models(single_star_dir)[0]
    with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
        pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%


def find_CO(r):
    if r.period_days[-1] < 50:
        return np.nan

    m_initial = r.params["q"] * 2
    m_C_initial = m_initial * Star.surf_c12[Star.ntams]
    m_O_initial = m_initial * Star.surf_o16[Star.ntams]

    dt = np.diff(r.star_age)
    dm = np.diff(r.star_2_mass)

    Xc = 0.5 * (r.surface_c12[:-1] + r.surface_c12[1:])
    print(r.params["q"], r.surface_c12[-1] / r.surface_o16[-1] * 16 / 12)
    Xo = 0.5 * (r.surface_o16[:-1] + r.surface_o16[1:])
    M_c_transfer = np.cumsum(Xc * dm)
    M_o_transfer = np.cumsum(Xo * dm)
    M_c = m_C_initial + M_c_transfer
    M_o = m_O_initial + M_o_transfer
    Xc_final = M_c / r.star_2_mass[1:]
    Xo_final = M_o / r.star_2_mass[1:]

    return Xc_final[-1] / Xo_final[-1] * 16 / 12


fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_CO,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_CO, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(delta):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    d * (1 - epss[i]),
                    qq,
                    np.round(ratio[k, j], 2),
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
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
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist-CO.pgf", format="pgf")
plt.show()
plt.close()
# %%


def find_CO(r):
    if r.period_days[-1] < 50:
        return np.nan

    ind = np.argwhere(r.rl_1 < r.R)[0][0]
    return r.surface_c12[ind] / r.surface_o16[ind] * 16 / 12


fig, axs = plt.subplots(
    1, 3, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

minn = 1e99
maxx = -1e-99

for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(
        find_CO,
        x="delta",
        y="q",
        eps=epss[i],
    )
    minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
    maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx


for i, ax in enumerate(axs):

    delta, q, ratio = grid.array(find_CO, x="delta", y="q", eps=epss[i])

    c = axs[i].pcolormesh(
        delta * (1 - epss[i]),
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(delta):
        for j, qq in enumerate(q):
            try:
                ax.text(
                    d * (1 - epss[i]),
                    qq,
                    np.round(ratio[k, j], 2),
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                )
            except:
                pass
    ax.set_title(rf"$\epsilon = {epss[i]}$")


axs[1].set_xlabel(r"$\delta$ ($1 - \beta$)")
axs[0].set_ylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)")
plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="Observed CO-ratio (no mixing in barium star)",
)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-eps-hist-CO-no-mixing.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


for m in grid.filter(eps=grid.axes["eps"][0], q=grid.axes["q"][3]):
    print(m.params)
    print(m.bulk_names)
    plt.plot(m.q, np.abs(m.jdot_ls), label=f"$\delta = {m.params["delta"]:.2f}$")


fig.legend(loc="outside upper center", ncols=4)
axs.spines[["right", "top"]].set_visible(False)
plt.yscale("log")
plt.xlabel("$q$")
plt.ylabel(r"$\dot{J}_\textrm{ls}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w22-darwin-instability-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
