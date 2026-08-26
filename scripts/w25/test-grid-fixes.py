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
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import compute_m_DUP, track_DUP

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid


from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

# %%
grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16-clean")
# %%

grid_old = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16", loc="LOGS/TPAGB/")

# %%

rng = np.random.default_rng(seed=9)
models = rng.permutation(grid.models)
for model in models:

    if model.period_days[-1] < 50:
        print(model.params)
        break

plt.plot(model.age, model.R)
plt.plot(model.age, model.rl_1)
plt.show()


# %%

fix_model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/grid-masses-2026-08-14-clean/R800.00_q0.400_eps0.250_delta0.200_M3.0/LOGS/history.data"
)
# %%

plt.plot(fix_model.star_age, fix_model.R)
plt.plot(fix_model.star_age, fix_model.rl_1)
plt.plot(model.star_age, model.R)
plt.show()

# %%

grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16-clean")

# %%

grid_old = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16", loc="LOGS/TPAGB/")
grid_old.models
# %%
fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(full), constrained_layout=True
)

axs = axs.flatten()

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0.01, 3)
    ax.set_yscale("log")

for ax, m in zip(axs, [1.8, 2.2, 2.6, 3.0]):
    m = np.round(m, 1)
    ax.text(
        0.95,
        0.05,
        f"$M = {m:.1f}\\;M_\\odot$",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
    )
    for model in grid.filter(m=m):
        if model.env_mass[-1] > 0.01:
            ((l),) = ax.plot(
                model.rl_1,
                model.env_mass,
                c="C3",
                linewidth=2,
                zorder=1000,
                label="Unsuccessful",
            )
            ax.plot(model.rl_1, model.env_mass, c="white", linewidth=5, zorder=100)
        else:
            (n,) = ax.plot(model.rl_1, model.env_mass, c="C2", label="Successful model")

fig.legend(handles=[n, l], loc="outside upper center", ncols=2)

fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel(r"Envelope mass ($M_\odot$)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-1-success.pgf", format="pgf")
plt.show()
plt.close()

# %%
for m in grid.filter(m=[2.2]):
    if m.log_LH[-1] < -12:
        print(m.params)
        print(m.log_LH[-1])
        print("")
        plt.plot(m.log_Teff, m.log_L, c="C3")
    elif m.log_LH[-1] > -10:
        plt.plot(m.log_Teff, m.log_L, c="C2")

plt.gca().invert_xaxis()
plt.show()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=[1.8, 2.2, 2.6, 3.0]):
    try:
        ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
        ind_0_05Msun = np.argwhere(m.envelope_mass < 0.05)[0][0]
    except:
        ind_0_1Msun = -1
        ind_0_05Msun = -1
        print(m.params)
    print(ind_0_1Msun)
    if np.max([m.log_LHe[ind_0_05Msun:]]) > 6:
        print(m.params)
        print(m.log_LH[-1])
        print("")
        (k,) = plt.plot(
            m.envelope_mass,
            m.log_LHe,
            c="C2",
            zorder=100,
            label=r"Thermal pulse after $M_\textrm{env} < 0.05\;M_\odot$",
            rasterized=True,
        )
        plt.plot(
            m.envelope_mass,
            m.log_LHe,
            c="white",
            zorder=99,
            linewidth=4,
            rasterized=True,
        )

    elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
        (l,) = plt.plot(
            m.envelope_mass,
            m.log_LHe,
            c="C8",
            zorder=90,
            label=r"Thermal pulse after $M_\textrm{env} < 0.1\;M_\odot$",
            rasterized=True,
        )
        plt.plot(
            m.envelope_mass,
            m.log_LHe,
            c="white",
            zorder=89,
            linewidth=4,
            rasterized=True,
        )
    else:
        print(m.log_LHe[-100] / m.log_LHe[-1])
        plt.plot(m.envelope_mass, m.log_LHe, c="C3", alpha=0.2, rasterized=True)


plt.xscale("log")
fig.legend(loc="outside upper center", handles=[k, l], ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel(r"$\log(L_\textrm{He} / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-AFTP.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
for m in grid.filter(m=[1.8, 2.2, 2.6, 3.0]):
    try:
        ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
    except:
        ind_0_1Msun = -1
        print(m.params)
    print(ind_0_1Msun)
    if np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
        print(m.params)
        print(m.log_LH[-1])
        print("")
        plt.plot(m.log_Teff, m.log_L, c="C2", zorder=100, label="")
        plt.plot(m.log_Teff, m.log_L, c="white", zorder=99, linewidth=4)
    # elif m.log_LHe[-100]  / m.log_LHe[-1] < 0.978 :
    #     plt.plot(m.log_Teff, m.log_L, c="C8", zorder=100, label= "")
    #     plt.plot(m.log_Teff, m.log_L, c="white", zorder=99, linewidth=4)
    else:
        print(m.log_LHe[-100] / m.log_LHe[-1])
        plt.plot(m.log_Teff, m.log_L, c="C3", alpha=0.2)

plt.gca().invert_xaxis()
plt.show()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=[1.8, 2.2, 2.6, 3.0]):
    try:
        ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
        ind_0_05Msun = np.argwhere(m.envelope_mass < 0.05)[0][0]
    except:
        ind_0_1Msun = -1
        ind_0_05Msun = -1
        print(m.params)
    print(ind_0_1Msun)
    if np.max([m.log_LHe[ind_0_05Msun:]]) > 6:
        print(m.params)
        print(m.log_LH[-1])
        print("")
        plt.plot(
            m.envelope_mass,
            m.period_days,
            c="C2",
            zorder=100,
            label="",
            rasterized=True,
        )
        plt.plot(
            m.envelope_mass,
            m.period_days,
            c="white",
            zorder=99,
            linewidth=4,
            rasterized=True,
        )
    elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
        plt.plot(
            m.envelope_mass,
            m.period_days,
            c="C8",
            zorder=100,
            label="",
            rasterized=True,
        )
        plt.plot(
            m.envelope_mass,
            m.period_days,
            c="white",
            zorder=99,
            linewidth=4,
            rasterized=True,
        )
    else:
        print(m.log_LHe[-100] / m.log_LHe[-1])
        plt.plot(m.envelope_mass, m.period_days, c="C3", alpha=0.2, rasterized=True)

plt.xscale("log")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel("Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-LTP-period.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    4, 1, sharex=False, figsize=set_size(column, height=2), constrained_layout=True
)

for i, m_i in enumerate([1.8, 2.2, 2.6, 3.0]):
    min_age = 1e99
    for m in grid.filter(m=m_i):
        if m.envelope_mass[-1] > 0.01:
            continue
        min_age = np.min([min_age, m.age[0]])
        try:
            ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
            ind_0_05Msun = np.argwhere(m.envelope_mass < 0.05)[0][0]
        except:
            ind_0_1Msun = -1
            ind_0_05Msun = -1
            print(m.params)
        print(ind_0_1Msun)
        if np.max([m.log_LHe[ind_0_05Msun:]]) > 6:
            print(m.params)
            print(m.log_LH[-1])
            print("")
            axs[i].plot(m.age, m.rl_1, c="C2", zorder=100, label="", rasterized=True)
            axs[i].plot(
                m.age, m.rl_1, c="white", zorder=99, linewidth=4, rasterized=True
            )
        elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
            axs[i].plot(m.age, m.rl_1, c="C8", zorder=90, label="", rasterized=True)
            axs[i].plot(
                m.age, m.rl_1, c="white", zorder=89, linewidth=4, rasterized=True
            )
        else:
            print(m.log_LHe[-100] / m.log_LHe[-1])
            axs[i].plot(m.age, m.rl_1, c="C3", alpha=0.2, rasterized=True)
    axs[i].set_xlim(min_age)

    star = get_star(m=m_i)
    axs[i].plot(star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :])
    axs[i].text(
        0.05,
        0.95,
        rf"$M_\textrm{{TPAGB,i}} = {m_i}\;M_\odot$",
        transform=axs[i].transAxes,
    )


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(200)
plt.xlabel("Age (yr)")
fig.supylabel("Radius ($R_\odot$)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w25-LTP-predictor.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

grid = MesaGrid(f"{MASTER}grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16-clean")
grid.merge(grid2)
# %%


def get_mass_limit(envelope_mass):
    return np.clip(envelope_mass * 1e-4, a_min=7.5e-6, a_max=1e-3)


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=2.2, q=0.651, R=800):

    plt.plot(m.envelope_mass, get_mass_limit(m.envelope_mass), label="Target")
    plt.plot(m.envelope_mass[1:], -np.diff(m.star_mass) / 10, label="Actual")


fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel(r"$\Delta M_\textrm{star}$ per timestep ($M_\odot$)")

plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-6)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-dm-target.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.filter(m=2.2, q=0.651, R=800):

    # plt.plot(m.envelope_mass, get_mass_limit(m.envelope_mass),label="Target")
    plt.plot(m.envelope_mass, m.num_zones)


fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel(r"$\Delta M_\textrm{star}$ per timestep ($M_\odot$)")

plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-6)

# plt.savefig("/home/koen/LaTeX-setup/plots/w25-dm-target.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    4, 1, sharex=True, figsize=set_size(column, height=2), constrained_layout=True
)

for i, m_i in enumerate([1.8, 2.2, 2.6, 3.0]):
    min_age = 1e99
    for m in grid.filter(m=m_i):
        if m.envelope_mass[-1] > 0.01:
            continue
        min_age = np.min([min_age, m.age[0]])
        try:
            ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
            ind_0_05Msun = np.argwhere(m.envelope_mass < 0.05)[0][0]
        except:
            ind_0_1Msun = -1
            ind_0_05Msun = -1
            print(m.params)
        print(ind_0_1Msun)
        if np.max([m.log_LHe[ind_0_05Msun:]]) > 6:
            print(m.params)
            print(m.log_LH[-1])
            print("")
            axs[i].plot(
                m.log_Teff, m.log_L, c="C2", zorder=100, label="", rasterized=True
            )
            axs[i].plot(
                m.log_Teff, m.log_L, c="white", zorder=99, linewidth=4, rasterized=True
            )
        elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
            axs[i].plot(
                m.log_Teff, m.log_L, c="C8", zorder=90, label="", rasterized=True
            )
            axs[i].plot(
                m.log_Teff, m.log_L, c="white", zorder=89, linewidth=4, rasterized=True
            )
        else:
            print(m.log_LHe[-100] / m.log_LHe[-1])
            axs[i].plot(m.log_Teff, m.log_L, c="C3", alpha=0.2, rasterized=True)
    # axs[i].set_xlim(min_age)

    star = get_star(m=m_i)
    axs[i].plot(
        star.log_Teff, star.log_L, c="C9", linewidth=0.75, zorder=-1, rasterized=True
    )
    axs[i].text(
        0.05,
        0.95,
        rf"$M_\textrm{{TPAGB,i}} = {m_i}\;M_\odot$",
        transform=axs[i].transAxes,
    )


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
ax.invert_xaxis()
plt.xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
fig.supylabel("$\log(L / L_\odot)$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w25-LTP-HR.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    4, 1, sharex=True, figsize=set_size(column, height=2), constrained_layout=True
)

for i, m_i in enumerate([1.8, 2.2, 2.6, 3.0]):
    min_age = 1e99
    for m in grid.filter(m=m_i):
        if m.envelope_mass[-1] > 0.01:
            continue
        min_age = np.min([min_age, m.age[0]])
        try:
            ind_0_1Msun = np.argwhere(m.envelope_mass < 0.1)[0][0]
            ind_0_05Msun = np.argwhere(m.envelope_mass < 0.05)[0][0]
        except:
            ind_0_1Msun = -1
            ind_0_05Msun = -1
            print(m.params)
        print(ind_0_1Msun)
        if np.max([m.log_LHe[ind_0_05Msun:]]) > 6:
            print(m.params)
            print(m.log_LH[-1])
            print("")
            axs[i].plot(
                m.log_Teff, m.log_L, c="C2", zorder=100, label="", rasterized=True
            )
            axs[i].plot(
                m.log_Teff, m.log_L, c="white", zorder=99, linewidth=4, rasterized=True
            )
        elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
            axs[i].plot(
                m.log_Teff, m.log_L, c="C8", zorder=90, label="", rasterized=True
            )
            axs[i].plot(
                m.log_Teff, m.log_L, c="white", zorder=89, linewidth=4, rasterized=True
            )
        else:
            print(m.log_LHe[-100] / m.log_LHe[-1])
            axs[i].plot(m.log_Teff, m.log_L, c="C3", alpha=0.2, rasterized=True)
    # axs[i].set_xlim(min_age)

    star = get_star(m=m_i)
    axs[i].plot(
        star.log_Teff, star.log_L, c="C9", linewidth=0.75, zorder=-1, rasterized=True
    )
    axs[i].text(
        0.05,
        0.95,
        rf"$M_\textrm{{TPAGB,i}} = {m_i}\;M_\odot$",
        transform=axs[i].transAxes,
    )


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(3.45)
ax.invert_xaxis()
ax.set_xlim(3.62)
plt.xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
fig.supylabel("$\log(L / L_\odot)$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w25-LTP-HR-zoom.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%
import matplotlib


def z(r):
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        if r.period_days[-1] < 50:
            return 0.5
        else:
            return 0

    return 1


from matplotlib.colors import ListedColormap

# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
col_dict = {
    0: "C3",
    0.3: "C1",
    0.6: "C2",
}

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

# Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
labels = np.array(["Convergence problems", "Darwin Instability", "Completed"])
len_lab = len(labels)
# prepare normalizer
# Prepare bins for the normalizer
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
print(norm_bins)
# Make normalizer and formatter
norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


R, q, ratio = grid.array(z, x="R", y="q")

minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

ms = [1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    R, q, ratio = grid.array(z, x="R", y="q", m=ms[i])

    c = axs[i].pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap=cm,
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)

fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    ticks=[0.333 / 2, 0.5, 1 - 0.333 / 2],
    format=fmt,
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-fate.pgf", format="pgf")
plt.show()
plt.close()

# %%
import matplotlib


def z(r):
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan
    return r.period_days[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


R, q, ratio = grid.array(z, x="R", y="q")

minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

ms = [1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    R, q, ratio = grid.array(z, x="R", y="q", m=ms[i])

    c = axs[i].pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(R):
        for j, qq in enumerate(q):
            try:
                axs[i].text(
                    d,
                    qq,
                    f"{ratio[k,j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                    zorder=1000,
                )
                print(ratio[k, j])
            except:
                pass


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)


plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label=r"Final period (days)",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-period.pgf", format="pgf")
plt.show()
plt.close()

# %%
import matplotlib


def z(r):
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan
    return r.star_2_mass[-1]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


R, q, ratio = grid.array(z, x="R", y="q")

minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

ms = [1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    R, q, ratio = grid.array(z, x="R", y="q", m=ms[i])

    c = axs[i].pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(R):
        for j, qq in enumerate(q):
            try:
                axs[i].text(
                    d,
                    qq,
                    f"{ratio[k,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                    zorder=1000,
                )
                print(ratio[k, j])
            except:
                pass


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)


plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label=r"Final barium star mass ($M_\odot$)",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-star_2_mass.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(full), constrained_layout=True
)

seeds = [0, 2, 100, 1000]

for i, ax in enumerate(axs.flatten()):
    rng = np.random.default_rng(seed=seeds[i])
    models = rng.permutation(grid.models)
    for model in models:

        if model.envelope_mass[-1] < 0.01:
            break

    dup = track_DUP(model)

    size = np.shape(dup["tracer_history_combined"])[1]
    print(size)

    winds_s = np.zeros(size)
    cores_s = np.zeros(size)
    traces_s = np.zeros(size)
    accreted_s = np.zeros(size)

    for dup_count in range(len(dup["tracer_history_combined"])):
        traces_s += dup["tracer_history_combined"][dup_count]
        winds_s += dup["wind_history_combined"][dup_count]
        cores_s += dup["core_history_combined"][dup_count]

    dup = track_DUP(model)

    winds = np.zeros(len(model.model_number))
    cores = np.zeros(len(model.model_number))
    traces = np.zeros(len(model.model_number))
    accreted = np.zeros(len(model.model_number))

    for dup_count in range(len(dup["tracer_history_binary"])):
        traces += dup["tracer_history_binary"][dup_count]
        winds += dup["wind_history_binary"][dup_count]
        cores += dup["core_history_binary"][dup_count]
        accreted += dup["accreted_history_binary"][dup_count]

    star = get_star(m=model.params["m"])

    ax.plot(
        star.m_env[:size][star.ntpagb :], traces_s[star.ntpagb :], c="C0", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :], winds_s[star.ntpagb :], c="C1", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :], cores_s[star.ntpagb :], c="C2", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :],
        0 * winds_s[star.ntpagb :],
        c="C3",
        linestyle=":",
    )
    (l1,) = ax.plot(model.envelope_mass[1:], traces[1:], c="C0", label="envelope")
    (l2,) = ax.plot(model.envelope_mass[1:], winds[1:], c="C1", label="wind")
    (l3,) = ax.plot(model.envelope_mass[1:], cores[1:], c="C2", label="core")
    (l4,) = ax.plot(model.envelope_mass[1:], accreted[1:], c="C3", label="RLOF")
    ax.set_xscale("log")


for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=[l1, l2, l3, l4])
fig.supxlabel(r"$M_\textrm{env}$ ($M_\odot$)", fontsize=10)
fig.supylabel(r"$M_\textrm{DUP}$ ($M_\odot$)", fontsize=10)
# plt.savefig("/home/koen/LaTeX-setup/plots/w24-dup-combined-m_env.pgf", format="pgf")
plt.show()
plt.close()

plt.show()
# %%
dup = track_DUP(model)

winds_real = np.zeros(len(model.model_number))
winds = np.zeros(len(model.model_number))
cores = np.zeros(len(model.model_number))
traces = np.zeros(len(model.model_number))
accreted = np.zeros(len(model.model_number))

for dup_count in range(len(dup["tracer_history_binary"])):
    traces += dup["tracer_history_binary"][dup_count]
    winds_real += dup["wind_history_binary"][dup_count] * model.beta_accretion
    winds += dup["wind_history_binary"][dup_count]
    cores += dup["core_history_binary"][dup_count]
    accreted += dup["accreted_history_binary"][dup_count] * 0.25

print(winds)
# %%

fix = mr.MesaData(
    "/home/koen/master-internship/mesa-models/grid-masses-3-2026-08-26/R800.00_q0.700_eps0.250_delta0.200_M2.2/LOGS/history.data"
)

# %%
print(model.params)
R = model.params["R"]
q = model.params["q"]
a_init = inv_roche_lobe(R, q)
star = get_star(m=2.2)
[star, Options, q_init, a_init, e_init, Bins] = call_evolution(
    star, q, a_init, simple_only=True
)
bin = Bins[0]
# plt.plot(bin.age[star.ntpagb+1:], -np.diff(bin.m2[star.ntpagb:]) / np.diff(bin.m1[star.ntpagb:]))

plt.plot(bin.age[star.ntpagb + 2 :] - bin.age[star.ntpagb + 2], bin.eta[star.ntpagb :])
plt.plot(model.star_age, model.eta_accretion)
plt.plot(fix.star_age, fix.eta_accretion)
# plt.plot(model.age, winds)
# plt.plot(model.age, winds_real)
plt.show()

# %%

model.bulk_names
# %%
plt.plot(bin.age, star.vwind[:90756] / v_orbit(bin.m1 + bin.m2, bin.a))
plt.plot(model.age, model.vwind_over_vorbit)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


plt.plot(bin.age, bin.a)
plt.plot(model.age, model.binary_separation)
plt.xlim(model.age[0] - 300000, model.age[-1] + 10000)
axs.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.ylabel("Separation ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-wrong-separation.pgf", format="pgf")

plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


plt.plot(bin.age, bin.a)
plt.plot(model.age, model.binary_separation)
plt.plot(fix.age + model.age[0], fix.binary_separation, c="C3")
plt.scatter(
    fix.age[0] + model.age[0], fix.binary_separation[0], color="C2", zorder=3, s=50
)

plt.xlim(model.age[0] - 300000, model.age[-1] + 10000)
axs.spines[["right", "top"]].set_visible(False)

plt.xlabel("Star age (yr)")
plt.ylabel("Separation ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-wrong-separation-fix.pgf", format="pgf")

plt.show()
plt.close()
# %%


import os
import re


def get_model_dict(params, evolution):

    single_star_dir = params["single_star_dir"]
    models_dir = f"{single_star_dir}/models/"
    models = [os.path.basename(f) for f in os.scandir(models_dir)]
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

    models_R = []
    models_TP = []

    for model in models:
        match = pattern.search(model)
        R = float(match.group(1))
        TP = int(match.group(2))
        models_R.append(R)
        models_TP.append(TP)

    models_dict = {}
    index = 0

    while len(models_R) > 0:
        arg = np.argmin(models_R)
        models_dict[f"model {index}"] = {
            "name": models.pop(arg),
            "R": models_R.pop(arg),
            "TP": models_TP.pop(arg),
        }

        index += 1
    star = get_star(m=2.2)

    for key in models_dict.keys():
        R = models_dict[key]["R"]
        arg = np.where(10**star.log_R >= R - 1e-3)[0][0]
        M = star.mass[arg]
        phase = star.phase[arg]
        R = 10 ** star.log_R[arg]
        age = star.age[arg]

        models_dict[key]["M"] = M
        models_dict[key]["phase"] = phase
        models_dict[key]["age"] = age
        models_dict[key]["R"] = R

    return models_dict


x = {
    "single_star_dir": "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M2.2/"
}
models_dict = get_model_dict(x, None)
# %%
print(models_dict)


# %%
def get_model_dict2(params, evolution):

    single_star_dir = params["single_star_dir"]
    models_dir = f"{single_star_dir}/models/"
    models = [os.path.basename(f) for f in os.scandir(models_dir)]
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

    models_R = []
    models_TP = []
    models_age = []

    for model in models:
        p = mr.MesaData(f"{models_dir}/{model}")
        match = pattern.search(model)
        R = float(match.group(1))
        TP = int(match.group(2))
        models_R.append(R)
        models_TP.append(TP)
        models_age.append(p.star_age)

    models_dict = {}
    index = 0

    while len(models_R) > 0:
        arg = np.argmin(models_R)
        models_dict[f"model {index}"] = {
            "name": models.pop(arg),
            "R": models_R.pop(arg),
            "TP": models_TP.pop(arg),
            "age": models_age.pop(arg),
        }

        index += 1
    star = get_star(m=2.2)

    for key in models_dict.keys():

        R = models_dict[key]["R"]
        naive_arg = np.where(np.abs(10**star.log_R - R) <= 1e-2)[0][0]
        phase = int(star.phase[naive_arg])

        match phase:
            case 0:
                offset = 0
            case 1:
                offset = star.age[star.ntams]
            case 2:
                offset = star.age[star.nzacheb]
            case 3:
                offset = star.age[star.ntacheb]
            case 4:
                offset = star.age[star.ntpagb]
            case _:
                raise Exception("invalid phase")

        real_age = models_dict[key]["age"] + offset
        print(real_age)
        arg = np.argmin(np.abs(star.age - real_age))
        print(arg)
        R = 10 ** star.log_R[arg]
        age = star.age[arg]
        M = star.mass[arg]

        models_dict[key]["M"] = M
        models_dict[key]["phase"] = phase
        models_dict[key]["age"] = age
        models_dict[key]["real_R"] = R

    return models_dict


models_dict2 = get_model_dict2(x, None)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.75), constrained_layout=True
)


for model1, model2 in zip(models_dict.values(), models_dict2.values()):
    c = plt.scatter(
        model1["age"],
        model1["R"],
        color="C3",
        zorder=100,
        marker="x",
        label="incorrect starting point",
    )
    w = plt.scatter(
        model2["age"],
        model2["real_R"],
        color="C2",
        zorder=90,
        marker="s",
        label="correct starting point",
    )

fig.legend(loc="outside upper center", ncols=2, handles=[c, w])
plt.yscale("log")
plt.xlim(0.997 * star.age[star.ntpagb], star.age[-1] * 1.0005)
plt.ylim(10, 1000)
plt.plot(star.age, 10**star.log_R)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("Radius ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-incorrect-setup.pgf", format="pgf")
plt.show()
plt.close()

# %%
grid3 = MesaGrid(f"{MASTER}grid-masses-3-2026-08-24")
# grid.merge(grid3, overwrite=True)
# %%
grid4 = MesaGrid(f"{MASTER}grid-masses-4-2026-08-25")
# grid.merge(grid4, overwrite=True)
# %%
print(grid4.models)
for model in grid4.models:
    print(model.envelope_mass[-1])

# %%
import matplotlib


def z(r):
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        if r.period_days[-1] < 50:
            return 0.5
        else:
            return 0

    return 1


from matplotlib.colors import ListedColormap

# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
col_dict = {
    0: "C3",
    0.2: "C1",
    1: "C2",
}

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

# Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
labels = np.array(["Still simulating / forgotten", "Darwin Instability", "Completed"])
len_lab = len(labels)
# prepare normalizer
# Prepare bins for the normalizer
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
print(norm_bins)
# Make normalizer and formatter
norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 0
maxx = -1e99


R, q, ratio = grid.array(z, x="R", y="q")

minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

print(minn, maxx)
ms = [1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    R, q, ratio = grid.array(z, x="R", y="q", m=ms[i])
    print(ratio.T)

    c = axs[i].pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap=cm,
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)

fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    ticks=[0.333 / 2, 0.5, 1 - 0.333 / 2],
    format=fmt,
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-fate-2.pgf", format="pgf")
plt.show()
plt.close()


# %%


def z(r):
    try:
        r.period_days[-1]
    except IndexError:
        return np.nan

    if r.envelope_mass[-1] > 1e-2:
        return np.nan

    return r.star_2_mass[-1] - r.params["q"] * r.params["m"]


fig, axs = plt.subplots(
    2,
    2,
    sharey=True,
    sharex=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

minn = 1e99
maxx = -1e99


R, q, ratio = grid.array(z, x="R", y="q")

minn = np.nanmin(ratio) if np.nanmin(ratio) < minn else minn
maxx = np.nanmax(ratio) if np.nanmax(ratio) > maxx else maxx

print(minn, maxx)
ms = [1.8, 2.2, 2.6, 3.0]

for i, ax in enumerate(axs):

    ax.set_title(f"$M_\\textrm{{TPAGB}} = {ms[i]:.1f}\\;M_\\odot$")
    R, q, ratio = grid.array(z, x="R", y="q", m=ms[i])
    print(ratio.T)

    c = axs[i].pcolormesh(
        R,
        q,
        ratio.T,
        shading="auto",
        cmap="viridis",
        vmin=minn,
        vmax=maxx,
        rasterized=True,
    )
    for k, d in enumerate(R):
        for j, qq in enumerate(q):
            try:
                axs[i].text(
                    d,
                    qq,
                    f"{ratio[k,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    c="k" if (ratio[k, j] - minn) / (maxx - minn) > 0.5 else "w",
                    zorder=1000,
                )
                print(ratio[k, j])
            except:
                pass


fig.supxlabel(r"Roche lobe radius ($R_\odot$)", fontsize=10)
fig.supylabel("$q$ ($M_\\textrm{a} / M_\\textrm{d}$)", fontsize=10)

fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

plt.colorbar(
    c,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label=r"$\Delta M_\textrm{Ba-star}$ ($M_\odot$)",
)
plt.yticks(
    [0.7, 0.66, 0.63, 0.6, 0.5, 0.44, 0.4][::-1], [f"{q:.2f}" for q in grid.axes["q"]]
)

plt.savefig("/home/koen/LaTeX-setup/plots/w25-grid-accr.pgf", format="pgf")
plt.show()
plt.close()


# %%
