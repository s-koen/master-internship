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

grid = MesaGrid("/home/koen/master-internship/mesa-models/binary-tpagb-grid-5/")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for R, q, model in grid.iter_models():
    if model.env_mass[-1] > 0.01:
        plt.scatter(R, q, c="C3", s=80, marker="x")
    else:
        plt.scatter(R, q, c="C2", s=80)

plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.xscale("log")
plt.ylabel("$q$")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-grid-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    periods = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            periods.append(model.star.period_days[-1])
            Rs.append(R)

    plt.plot(Rs, periods, ".-", label=f"$q={qs}$")

fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-grid-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    fig, axs = plt.subplots(
        2, 1, sharex=True, figsize=set_size(full, height=0.7), constrained_layout=True
    )

    plt.xlabel("Star age (yr)")
    axs[0].set_ylabel(r"$\log(R / R_\odot)$")
    axs[1].set_ylabel(r"$\log|(M_\textrm{env,b} - M_\textrm{env,s}) / M_\odot|$")

    for i, (R, model) in enumerate(grid.get_q_slice(q)):
        axs[0].plot(
            model.age, model.star.R, c=f"C{i}", label=f"$R_\\textrm{{RL}}={R:.0f}$"
        )
        axs[0].plot(model.age, model.star.rl_1, c=f"C{i}", alpha=0.5, linewidth=3)
        axs[0].plot(
            grid.ref_tpagb.star_age,
            grid.ref_tpagb.R,
            zorder=-10,
            linewidth=5,
            color="C9",
        )
    axs[0].set_ylim(50, 1000)

    for R, model in grid.get_q_slice(q):
        mass_ref_interp = np.interp(
            model.age,
            grid.ref_tpagb.star_age,
            grid.ref_tpagb.star_mass - grid.ref_tpagb.he_core_mass,
            left=np.nan,
        )
        plt.plot(model.age, np.abs(model.env_mass - mass_ref_interp))
    # plt.plot(
    #     grid.ref_tpagb.star_age,
    #     grid.ref_tpagb.star_mass - grid.ref_tpagb.he_core_mass,
    #     zorder=-10,
    #     linewidth=5,
    #     c=r"C9",
    # )

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    fig.legend(loc="outside upper center", ncols=5)
    plt.savefig(f"/home/koen/LaTeX-setup/plots/w8-mass-{qs:.1f}-2.pgf", format="pgf")
    plt.show()
    plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    max_ratio = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            max_ratio.append(np.max(model.star.R / model.star.rl_1))
            Rs.append(R)

    plt.plot(Rs, max_ratio, ".-", label=f"$q={qs}$")

fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-grid-3.pgf", format="pgf")
plt.show()
plt.close()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    max_ratio = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            max_ratio.append(np.max(model.star.R / rol(model.star)))
            Rs.append(R)

    plt.plot(Rs, max_ratio, ".-", label=f"$q={qs}$")


plt.xlim(plt.gca().get_xlim())
plt.ylim(plt.gca().get_ylim())
plt.fill_between([0, 1000], [1, 1], [3, 3], color="C9", zorder=-10, alpha=0.3)

fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{ROL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-grid-4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

models = []
qs = []
Rs = []
for R, q, model in grid.iter_models():
    if model.env_mass[-1] > 0.1:
        plt.scatter(R, q, c="C9", s=80, marker="x")
    # elif np.max(model.star.R / rol(model.star)) > 1:
    #     plt.scatter(R, q, c="C3", s=80, marker="^")
    else:
        qs.append(q)
        Rs.append(R)
        models.append(model)


plt.scatter(
    Rs,
    qs,
    c=[np.max(model.star.R / model.star.rl_1) for model in models],
    s=80,
    cmap="viridis",
)
print(model.star.bulk_names)
plt.colorbar(label=r"$\textrm{max}(R_\textrm{star} / R_\textrm{RL})$")
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.xscale("log")
plt.ylabel("$q$")
plt.savefig("/home/koen/LaTeX-setup/plots/w9-grid-1.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

models = []
qs = []
Rs = []
for R, q, model in grid.iter_models():
    if model.env_mass[-1] > 0.1:
        (l1,) = plt.scatter(R, q, c="C9", s=80, marker="x", label="Crashed")
    elif np.max(model.star.R / rol(model.star)) > 1:
        (l2,) = plt.scatter(R, q, c="C3", s=80, marker="^", label="Outer lobe overflow")
    else:
        (l3,) = plt.scatter(R, q, c="C0", s=80, label="No outer lobe overflow")


fig.legend(loc="outside upper center", ncols=3, handles=[l1, l2, l3])
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.xscale("log")
plt.ylabel("$q$")
plt.savefig("/home/koen/LaTeX-setup/plots/w9-grid-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
