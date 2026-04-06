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

ferg = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/ferguson", q=0.5)
# %%

aeso = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/aesopus")

# %%

# interpolate reference onto data grid


for R, model in aeso.get_q_slice(0.75):
    mass_ref_interp = np.interp(
        model.age,
        aeso.ref_tpagb.star_age,
        aeso.ref_tpagb.star_mass - aeso.ref_tpagb.he_core_mass,
        left=np.nan,
    )
    plt.plot(model.age, np.abs(model.env_mass - mass_ref_interp))
# plt.plot(
#     aeso.ref_tpagb.star_age,
#     aeso.ref_tpagb.star_mass - aeso.ref_tpagb.he_core_mass,
#     zorder=-10,
#     linewidth=5,
#     c=r"C9",
# )

plt.show()

# %%

for R, model in ferg.get_q_slice(0.5):
    plt.plot(model.age, model.env_mass)
plt.plot(
    ferg.ref_tpagb.star_age,
    ferg.ref_tpagb.star_mass - ferg.ref_tpagb.he_core_mass,
    zorder=-10,
    linewidth=5,
    c=r"C9",
)

plt.show()


# %%

for R, model in aeso.get_q_slice(0.5):
    plt.plot(model.age, model.env_mass)

plt.show()

# %%

for q in [0.25, 0.5, 0.75, 1]:

    fig, axs = plt.subplots(
        2, 1, sharex=True, figsize=set_size(full, height=0.7), constrained_layout=True
    )

    plt.xlabel("Star age (yr)")
    axs[0].set_ylabel(r"$\log(R / R_\odot)$")
    axs[1].set_ylabel(r"$\log|(M_\textrm{env,b} - M_\textrm{env,s}) / M_\odot|$")

    for i, (R, model) in enumerate(aeso.get_q_slice(q)):
        axs[0].plot(
            model.age, model.star.R, c=f"C{i}", label=f"$R_\\textrm{{RL}}={R:.0f}$"
        )
        axs[0].plot(model.age, model.star.rl_1, c=f"C{i}", alpha=0.5, linewidth=3)
        axs[0].plot(
            aeso.ref_tpagb.star_age,
            aeso.ref_tpagb.R,
            zorder=-10,
            linewidth=5,
            color="C9",
        )
    axs[0].set_ylim(50, 1000)

    for R, model in aeso.get_q_slice(q):
        mass_ref_interp = np.interp(
            model.age,
            aeso.ref_tpagb.star_age,
            aeso.ref_tpagb.star_mass - aeso.ref_tpagb.he_core_mass,
            left=np.nan,
        )
        plt.plot(model.age, np.abs(model.env_mass - mass_ref_interp))
    # plt.plot(
    #     aeso.ref_tpagb.star_age,
    #     aeso.ref_tpagb.star_mass - aeso.ref_tpagb.he_core_mass,
    #     zorder=-10,
    #     linewidth=5,
    #     c=r"C9",
    # )

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    fig.legend(loc="outside upper center", ncols=3)
    plt.savefig(f"/home/koen/LaTeX-setup/plots/w8-mass-{q}.pgf", format="pgf")
    plt.show()
    plt.close()


# %%
