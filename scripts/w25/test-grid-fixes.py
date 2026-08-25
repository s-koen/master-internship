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

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

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
        )
        plt.plot(m.envelope_mass, m.log_LHe, c="white", zorder=99, linewidth=4)

    elif np.max([m.log_LHe[ind_0_1Msun:]]) > 6:
        (l,) = plt.plot(
            m.envelope_mass,
            m.log_LHe,
            c="C8",
            zorder=90,
            label=r"Thermal pulse after $M_\textrm{env} < 0.1\;M_\odot$",
        )
        plt.plot(m.envelope_mass, m.log_LHe, c="white", zorder=89, linewidth=4)
    else:
        print(m.log_LHe[-100] / m.log_LHe[-1])
        plt.plot(m.envelope_mass, m.log_LHe, c="C3", alpha=0.2)


plt.xscale("log")
fig.legend(loc="outside upper center", handles=[k, l], ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel(r"$M_\textrm{env}$ ($M_\odot$)")
plt.ylabel(r"$\log(L_\textrm{He} / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-AFTP.pgf", format="pgf")
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
        plt.plot(m.envelope_mass, m.period_days, c="C2", zorder=100, label="")
        plt.plot(m.envelope_mass, m.period_days, c="white", zorder=99, linewidth=4)
    elif m.log_LHe[-100] / m.log_LHe[-1] < 0.963:
        plt.plot(m.envelope_mass, m.period_days, c="C8", zorder=100, label="")
        plt.plot(m.envelope_mass, m.period_days, c="white", zorder=99, linewidth=4)
    else:
        print(m.log_LHe[-100] / m.log_LHe[-1])
        plt.plot(m.envelope_mass, m.period_days, c="C3", alpha=0.2)

plt.xscale("log")
plt.show()


# %%
fig, axs = plt.subplots(
    4, 1, sharex=False, figsize=set_size(column), constrained_layout=True
)

for i, m_i in enumerate([1.8, 2.2, 2.6, 3.0]):
    min_age = 1e99
    for m in grid.filter(m=m_i):
        if m.envelope_mass[-1] > 0.01:
            continue
        min_age = np.min([min_age, m.age[0]])
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
            axs[i].plot(m.age, m.rl_1, c="C2", zorder=100, label="")
            axs[i].plot(m.age, m.rl_1, c="white", zorder=99, linewidth=4)
        # elif m.log_LHe[-100]  / m.log_LHe[-1] < 0.975 :
        #     axs[i].plot(m.age, m.rl_1, c="C8", zorder=100, label= "")
        #     axs[i].plot(m.age, m.rl_1, c="white", zorder=99, linewidth=4)
        else:
            print(m.log_LHe[-100] / m.log_LHe[-1])
            axs[i].plot(m.age, m.rl_1, c="C3", alpha=0.2)
    axs[i].set_xlim(min_age)

    star = get_star(m=m_i)
    axs[i].plot(star.age[star.ntpagb :], 10 ** star.log_R[star.ntpagb :])


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(200)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-LTP-predictor.pgf", format="pgf")
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
