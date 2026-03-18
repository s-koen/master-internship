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

ferguson = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/ferguson/")
aesopus = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/aesopus/")
# %%

for i, (q, model) in enumerate(ferguson.get_r1_slice(r1=420)):
    plt.plot(model.age, model.star.R, c=f"c{i}")
    plt.plot(model.age, model.star.rl_1, c=f"c{i}")

for i, (q, model) in enumerate(aesopus.get_r1_slice(r1=420)):
    plt.plot(model.age, model.star.R, c=f"c{i}", alpha=0.5, linewidth=5)
    plt.plot(model.age, model.star.rl_1, c=f"c{i}", alpha=0.5, linewidth=5)

plt.show()
# %%

for i, (q, model) in enumerate(ferguson.get_q_slice(q=0.25)):
    plt.plot(model.age, model.star.R, c=f"C{i}")
plt.plot(ferguson.ref_tpagb.star_age, ferguson.ref_tpagb.R, c=f"C9")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=0.25)):
    plt.plot(model.age, model.star.R, c=f"C{i}", alpha=0.5, linewidth=5)
plt.plot(aesopus.ref_tpagb.star_age, aesopus.ref_tpagb.R, c=f"C9")

plt.show()
# %%

for i, (q, model) in enumerate(ferguson.get_q_slice(q=0.25)):
    plt.plot(model.env_mass, model.star.R, c=f"C{i}")
    plt.plot(model.env_mass, model.star.rl_1, c=f"C{i}")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=0.25)):
    plt.plot(model.env_mass, model.star.R, c=f"C{i}", alpha=0.5, linewidth=5)
    plt.plot(model.env_mass, model.star.rl_1, c=f"C{i}", alpha=0.5, linewidth=5)

plt.show()
# %%
q_choice = 1

for i, (q, model) in enumerate(ferguson.get_q_slice(q=q_choice)):
    plt.plot(model.env_mass, model.star.R / model.star.rl_1, c=f"C{i}")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=q_choice)):
    plt.plot(
        model.env_mass,
        model.star.R / model.star.rl_1,
        c=f"C{i}",
        alpha=0.5,
        linewidth=5,
    )

plt.show()
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
            label=f"$R_\\textrm{{RL}} = {R:.0f}$",
        )

    else:
        plt.plot(
            model.star.elapsed_time / 3600,
            np.log10(model.env_mass),
            c=f"C{count}",
        )

count = -1
ref_R = 0

fig.legend(loc="outside upper center", ncols=3)


plt.xlabel("Elapsed time (hr)")
plt.ylabel("Envelope mass ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w7-all-compare-opacity.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"Envelope mass ($M_\odot$)")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$ ")

for R, model in ferguson.get_q_slice(0.25):
    plt.plot(
        model.env_mass,
        model.star.R / model.star.rl_1,
        label=f"$R_\\textrm{{RL}} = {R:.0f}$",
    )
fig.legend(loc="outside upper center", ncols=3)
plt.gca().invert_xaxis()
plt.savefig("/home/koen/LaTeX-setup/plots/w7-compare-q.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel(r"Envelope mass ($M_\odot$)")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$ ")

for R, model in ferguson.get_q_slice(0.25):
    plt.plot(
        model.env_mass,
        model.star.period_days,
        label=f"$R_\\textrm{{RL}} = {R:.0f}$",
    )
fig.legend(loc="outside upper center", ncols=3)
plt.gca().invert_xaxis()
plt.savefig("/home/koen/LaTeX-setup/plots/w7-compare-q-orbit.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Age (yr)")
plt.ylabel("$\log(R / R_\\odot)$")
for i, (q, model) in enumerate(ferguson.get_R1_slice(600)):
    plt.plot(model.age, np.log10(model.star.rl_1), c=f"C{i}", label=f"$q={q}$")

plt.plot(model.age, model.star.log_R, c="C9")
fig.legend(loc="outside upper center", ncols=2)

plt.savefig("/home/koen/LaTeX-setup/plots/w7-large-R.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

plt.xlabel("Age (yr)")
axs[0].set_ylabel("$R / R_\\odot$")
axs[1].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
ls = []
for i, (q, model) in enumerate(ferguson.get_R1_slice(600)):
    (l,) = axs[0].plot(model.age, model.star.rl_1, c=f"C{i}", label=f"$q={q}$")
    ls.append(l)

axs[0].plot(model.age, model.star.R, c="C9")

(l1,) = plt.plot(model.age, model.star.lg_wind_mdot_1, c=f"C0", label="wind")
(l2,) = plt.plot(model.age, model.star.lg_mtransfer_rate, c=f"C1", label="transfer")
plt.ylim(-12)
fig.legend(loc="outside upper center", ncols=4, handles=ls)
plt.legend(loc="upper center", ncols=4, handles=[l1, l2])

plt.savefig("/home/koen/LaTeX-setup/plots/w7-large-R.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Age (yr)")
plt.ylabel("$R / R_\\odot$")
for i, (q, model) in enumerate(ferguson.get_R1_slice(600)):
    plt.plot(model.age, np.log10(model.star.rl_1), c=f"C{i}", label=f"$q={q}$")


plt.plot(model.age, model.star.lg_wind_mdot_1, c=f"C{i}", label=f"$q={q}$")

fig.legend(loc="outside upper center", ncols=4)

plt.savefig("/home/koen/LaTeX-setup/plots/w7-large-R-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/binary-tpagb-grid-4/")

# %%
import matplotlib as mpl

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=5, vmax=10)

colors = plt.cm.viridis(np.linspace(0, 1, 12))

for i, (R, q, model) in enumerate(grid.iter_models()):
    plt.plot(model.env_mass, model.star.R / model.star.rl_1, color=colors[i])


plt.xlabel(r"Envelope mass ($M_\odot$)")
plt.ylabel(r"$R_\textrm{star} / R_\textrm{RL}$ ")


plt.xlim(plt.gca().get_xlim())
plt.plot([-100, 100], [1, 1], c="C9", linewidth=0.75)
plt.gca().invert_xaxis()
fig.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(506, 550), cmap=cmap),
    ax=plt.gca(),
    label=r"$R_\textrm{RL}$",
)

plt.savefig("/home/koen/LaTeX-setup/plots/w7-end.pgf", format="pgf")
plt.show()
plt.close()

# %%
