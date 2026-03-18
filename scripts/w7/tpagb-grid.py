import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.general_utils.mesa_grid import MesaGrid
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

grid = MesaGrid(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/", R1=300
)

# %%

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model.tpagb
    plt.plot(hist.star_mass - hist.he_core_mass, hist.R, c=f"C{count}")
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        hist.rl_1,
        c=f"C{count}",
        linewidth=5,
        alpha=0.5,
    )
    count += 1

plt.gca().invert_xaxis()
plt.show()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%

fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(full), constrained_layout=True
)

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model.tpagb
    axs[0].plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / hist.rl_1,
        c=f"C{count}",
        label=f"$q = {q}$",
    )
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / hist.rl_1,
        c=f"C{count}",
    )
    count += 1

axs[0].invert_xaxis()
axs[0].set_ylim(0.7, 3)
axs[1].set_yscale("log")
axs[1].set_ylim(10 ** (-0.5), 12)

fig.legend(loc="outside upper center", ncols=4)
axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[1].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R_\textrm{star} / R_\textrm{RL}$")

axs[0].grid(which="both", axis="y")
axs[1].grid(which="major", axis="y")
plt.savefig("/home/koen/LaTeX-setup/plots/relativerocheloberadius.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(full), constrained_layout=True
)

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model.tpagb
    axs[0].plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / rol(hist),
        c=f"C{count}",
        label=f"$q = {q}$",
    )
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / rol(hist),
        c=f"C{count}",
    )
    count += 1

axs[0].invert_xaxis()
axs[0].set_ylim(0.7, 3)
axs[1].set_yscale("log")
axs[1].set_ylim(10 ** (-0.5), 12)
axs[0].set_xlim(axs[0].get_xlim())
axs[1].set_xlim(axs[0].get_xlim())
axs[0].fill_between([100, -1], [1, 1], [3, 3], color="C9", zorder=-10, alpha=0.3)
axs[1].fill_between([100, -1], [1, 1], [30, 30], color="C9", zorder=-10, alpha=0.3)

fig.legend(loc="outside upper center", ncols=4)
axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[1].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R_\textrm{star} / R_\textrm{ROL}$")

axs[0].grid(which="both", axis="y")
axs[1].grid(which="major", axis="y")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/relativeouterrocheloberadius.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

count = 0
for m1, q, model in grid.iter_models():
    hist = model.tpagb
    plt.plot(hist.star_age, hist.R, c=f"C{count}")
    plt.plot(
        hist.star_age,
        hist.rl_1,
        c=f"C{count}",
        linewidth=5,
        alpha=0.5,
    )
    count += 1

plt.show()


# %%


count = 0
for m1, q, model in grid.iter_models():
    print(q)
    hist = model.tpagb
    plt.plot(hist.log_Teff, hist.log_L, c=f"C{count}")
    count += 1

plt.gca().invert_xaxis()
plt.show()
# %%


fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(full), constrained_layout=True
)

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model.tpagb
    axs[0].plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / hist.binary_separation,
        c=f"C{count}",
        label=f"$q = {q}$",
    )
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        hist.R / hist.binary_separation,
        c=f"C{count}",
    )
    count += 1

axs[0].invert_xaxis()
axs[0].set_ylim(0, 2.2)
axs[1].set_yscale("log")
axs[1].set_ylim(10 ** (-1), 10)
axs[0].set_xlim(axs[0].get_xlim())
axs[1].set_xlim(axs[0].get_xlim())
axs[0].fill_between([100, -1], [2, 2], [3, 3], color="C9", zorder=-10, alpha=0.3)
axs[1].fill_between([100, -1], [2, 2], [30, 30], color="C9", zorder=-10, alpha=0.3)


fig.legend(loc="outside upper center", ncols=4)
axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[1].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R_\textrm{star} / R_\textrm{orbit}$")

axs[0].grid(which="both", axis="y")
axs[1].grid(which="major", axis="y")
plt.savefig("/home/koen/LaTeX-setup/plots/R300period.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(full), constrained_layout=True
)

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model.tpagb
    print(hist.bulk_names)
    axs[0].plot(
        hist.star_mass - hist.he_core_mass,
        10 ** (hist.lg_mstar_dot_1) / 10 ** (hist.quasi_adiabatic_Mdot),
        c=f"C{count}",
        label=f"$q = {q}$",
    )
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        10 ** (hist.lg_mstar_dot_1) / 10 ** (hist.quasi_adiabatic_Mdot),
        c=f"C{count}",
    )
    count += 1

axs[0].invert_xaxis()
# axs[0].set_ylim(0.7, 3)
axs[1].set_yscale("log")
axs[1].set_ylim(10 ** (-4), 1.2)

fig.legend(loc="outside upper center", ncols=4)
axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[1].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$\dot{M} / \dot{M}_\textrm{qad}$")

plt.savefig("/home/koen/LaTeX-setup/plots/w7-mdot.pgf", format="pgf")
plt.show()
plt.close()
# %%


grid2 = MesaGrid(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/", q=0.500
)
# %%

for i, (R, model) in enumerate(grid2.get_q_slice(q=0.625)):
    plt.plot(model.star.star_age + model.initial_age, model.star.log_R, c=f"C{i}")
    plt.plot(
        model.star.star_age + model.initial_age,
        np.log10(model.star.rl_1),
        c=f"C{i}",
        alpha=0.5,
        linewidth=5,
    )
plt.plot(
    grid2.ref_tpagb.star_age, grid2.ref_tpagb.log_R, color="C9", linewidth=5, zorder=-10
)


plt.show()
# %%

for i, (R, model) in enumerate(grid2.get_q_slice(q=0.625)):
    plt.plot(
        model.star.star_age + model.initial_age,
        model.star.surface_c12 / model.star.surface_o16 * 16 / 12,
        c=f"C{i}",
    )

plt.plot(
    grid2.ref_tpagb.star_age,
    grid2.ref_tpagb.surface_c12 / grid2.ref_tpagb.surface_o16 * 16 / 12,
    color="C9",
    linewidth=5,
    zorder=-10,
)


plt.show()
# %%


for i, (R, model) in enumerate(grid2.get_q_slice(q=0.625)):
    plt.plot(
        model.star.star_mass - model.star.he_core_mass,
        model.star.surface_c12 / model.star.surface_o16 * 16 / 12,
        c=f"C{i}",
    )

plt.plot(
    grid2.ref_tpagb.star_mass - grid2.ref_tpagb.he_core_mass,
    grid2.ref_tpagb.surface_c12 / grid2.ref_tpagb.surface_o16 * 16 / 12,
    color="C9",
    linewidth=5,
    zorder=-10,
)

plt.gca().invert_xaxis()

plt.show()
# %%

print(grid2.models)
# %%
for R1, q, model in grid2.iter_models():
    print(f"{R1 = }, {q = }, {model}")
# %%

grid = MesaGrid(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/",
    q=[0.125, 0.250, 0.375],
    R1=[150, 225, 300],
)
for R1, q, model in grid.iter_models():
    print(f"{R1 = }, {q = }, {model}")
# %%
for R1, q, model in grid.iter_models():
    print(f"{R1 = }, {q = :.3f}, {model}")
# %%

print("slicing q")
for R1, model in grid.get_q_slice(q=0.125):
    print(model)

print("\nslicing R1")
for R1, model in grid.get_R1_slice(R1=225):
    print(model)

# %%
grid = MesaGrid(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/",
    q=[0.875],
    R1=[300],
)
# %%
model = grid.get(R1=300, q=0.875)
plt.plot(grid.ref_tpagb.star_age, grid.ref_tpagb.R)
plt.plot(model.age, model.star.R)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

model = grid.get(R1=300, q=0.875)
plt.plot(grid.ref_tpagb.star_age, grid.ref_tpagb.log_R, c="C9", linewidth=5)
plt.plot(model.age, model.star.log_R)

plt.ylim(1.75)
plt.xlabel("Star age (yr)")
plt.ylabel(r"$\log(R / R_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w7-compare-with-ref.pgf", format="pgf")
plt.show()
plt.close()
# %%

model = grid.get(R1=300, q=0.875)
plt.plot(model.env_mass, model.star.R)
plt.plot(model.env_mass, model.star.rl_1)
plt.gca().invert_xaxis()
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
model = grid.get(R1=300, q=0.875)
plt.plot(
    model.env_mass,
    model.star.log_R,
    label=r"$R_\textrm{star}$",
)
plt.plot(
    model.env_mass,
    np.log10(model.star.rl_1),
    label=r"$R_\textrm{RL}$",
    c="C0",
    alpha=0.5,
    linewidth=5,
)
plt.gca().invert_xaxis()

plt.ylim(2)

plt.legend()
plt.xlabel(r"Envelope mass ($M_\textrm{env}$")
plt.ylabel(r"$\log(R / R_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w7-show-mass.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for id in range(24, 25):
    model = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/compare-opacity-mass-transfer/ferguson/runs/R150.00_q0.250/LOGS/TPAGB/profile{id}.data"
    )

    plt.plot(model.R, model.gradT, zorder=1000)

plt.xlabel(r"$r$ ($R_\odot$)")
plt.ylabel(r"$\nabla$")
plt.xlim(2.5, 15)
plt.gca().invert_xaxis()
plt.ylim(-1.5, 4)
# plt.savefig("/home/koen/LaTeX-setup/plots/problem-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/compare-opacity-mass-transfer/ferguson/runs/R150.00_q0.250/LOGS/TPAGB/profile{24}.data"
)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [
    [[15, 2.5], [4.575, 4.225]],
    [[0.75, 1], [0.85, 0.89]],
    [[-1, -6], [-2.25, -2.6]],
]
ylims = [[-1.9, 4], [-1.9, 2]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        if i == 0 and j == 0:
            ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
            ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
            ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")
        else:
            ax.plot(xaxis[i], model.gradT, zorder=1000)
            ax.plot(xaxis[i], model.gradr)
            ax.plot(xaxis[i], model.grada)


plt.ylabel(r"$\nabla$")
fig.legend(loc="outside upper center", ncols=3)
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/w7-problem-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/compare-opacity-mass-transfer/ferguson/runs/R150.00_q0.250/LOGS/TPAGB/profile{23}.data"
)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [[[15, 2.5], [5, 4.65]], [[0.75, 1], [0.81, 0.85]], [[-1, -6], [-2.08, -2.4]]]
ylims = [[-1.5, 4], [-1.5, 1]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
        ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
        ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")


plt.ylabel(r"$\nabla$")
axss[0, 1].legend()
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/w7-problem-gradTgradR.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axss = plt.subplots(
    3, 2, figsize=set_size(column, height=1.1), constrained_layout=True
)

model = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/compare-opacity-mass-transfer/aesopus/runs/R150.00_q0.250/LOGS/TPAGB/profile{30}.data"
)

xaxis = [model.logP, model.R / model.R[0], np.log10(1 - model.mass / model.mass[0])]
xaxis_label = [r"$\log P$ (dyn/cm$^2$)", r"$r / R$", r"$\log(1 - m / M)$"]

xlims = [
    [[15, 2.5], [4.625, 4.275]],
    [[0.75, 1], [0.835, 0.875]],
    [[-1, -6], [-2.2, -2.5]],
]
ylims = [[0, 1.5], [0, 0.6]]
plt.gca().invert_xaxis()

for i, axs in enumerate(axss):
    for j, ax in enumerate(axs):
        ax.set_xlim(*xlims[i][j])
        ax.set_ylim(*ylims[j])
        ax.set_xlabel(xaxis_label[i])
        ax.set_ylabel(r"$\nabla$")
        if i == 0 and j == 0:
            ax.plot(xaxis[i], model.gradT, zorder=1000, label=r"$\nabla$")
            ax.plot(xaxis[i], model.gradr, label=r"$\nabla_\textrm{rad}$")
            ax.plot(xaxis[i], model.grada, label=r"$\nabla_\textrm{ad}$")
        else:
            ax.plot(xaxis[i], model.gradT, zorder=1000)
            ax.plot(xaxis[i], model.gradr)
            ax.plot(xaxis[i], model.grada)


plt.ylabel(r"$\nabla$")
fig.legend(loc="outside upper center", ncols=3)
# plt.xlim(2.5, 15)
# plt.gca().invert_xaxis()
# plt.ylim(-1.5, 4)
plt.savefig("/home/koen/LaTeX-setup/plots/w7-problem-gradTgradR-aeso.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, (R, q, model) in enumerate(ferguson.iter_models()):
    if model.env_mass[-1] > 0.2:
        plt.plot(
            model.star.elapsed_time / 3600,
            model.env_mass,
            c=f"C{i}",
            label=f"$R_\\textrm{{RL}} = {R:.0f}, q = {q}$",
            zorder=1000 - i,
        )
    else:
        plt.plot(
            model.star.elapsed_time / 3600,
            model.env_mass,
            c=f"C9",
            alpha=0.5,
            zorder=-10,
        )

for i, (R, q, model) in enumerate(aesopus.iter_models()):
    print(R)
    if model.env_mass[-1] > 0.2:
        plt.plot(
            model.star.elapsed_time / 3600,
            model.env_mass,
            c=f"C{i}",
            linewidth=5,
            alpha=0.5,
            zorder=1000 - i,
        )
    else:
        plt.plot(
            model.star.elapsed_time / 3600,
            model.env_mass,
            c=f"C9",
            alpha=0.5,
            zorder=-10,
        )

plt.ylim(1.175, 1.525)
plt.xlim(-0.25, 3.5)

fig.legend(loc="outside upper center", ncols=3)


plt.xlabel("Elapsed time (hr)")
plt.ylabel("Envelope mass ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w7-compare-opacity.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

count = -1
ref_R = 0
for i, (R, q, model) in enumerate(ferguson.iter_models()):
    if R != ref_R:
        count += 1
        ref_R = R

        plt.plot(
            model.star.elapsed_time / 3600,
            np.log10(model.env_mass),
            c=f"C{count}",
            label=f"$R_\\textrm{{RL}} = {R}, q = {q}$",
        )

    else:
        plt.plot(
            model.star.elapsed_time / 3600,
            np.log10(model.env_mass),
            c=f"C{count}",
        )

count = -1
ref_R = 0

for i, (R, q, model) in enumerate(aesopus.iter_models()):
    if R != ref_R:
        count += 1
        ref_R = R

        plt.plot(
            model.star.elapsed_time / 3600,
            np.log10(model.env_mass),
            c=f"C{count}",
            label=f"$R_\\textrm{{RL}} = {R:.0f}, q = {q}$",
        )

    else:
        plt.plot(
            model.star.elapsed_time / 3600,
            np.log10(model.env_mass),
            c=f"C{count}",
        )


fig.legend(loc="outside upper center", ncols=3)


plt.xlabel("Elapsed time (hr)")
plt.ylabel("Envelope mass ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w7-all-compare-opacity.pgf", format="pgf")
plt.show()
plt.close()
# %%


grid = MesaGrid("/home/koen/master-internship/mesa-models/binary-tpagb-grid-4/", R1=300)

# %%

for R, q, model in grid.iter_models():
    plt.plot(model.age, model.R)
plt.show()
# %%
