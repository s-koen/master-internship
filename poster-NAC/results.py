import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import os
import re
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

sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

# %%
Star3 = read_stellar_models(
    f"/home/koen/master-internship/mesa-models/standard-2msun-v3/"
)[0]


# %%
#
# def find_model(R):
#     req_TP = 0
#     lowering_R = True
#     for key in list(models_dict.keys())[::-1]:
#         model = models_dict[key]
#         if model["R"] < (0.8 * R) and model["TP"] <= req_TP:
#             print(model["R"])
#             return model
#         elif lowering_R:
#             if model["R"] < (0.8 * R):
#                 req_TP = model["TP"] - 1
#                 lowering_R = False
#
def find_model(R):
    searching_TP = True
    for key in list(models_dict.keys())[::-1]:
        model = models_dict[key]

        # searching for the thermal pulse count
        # for the first presumable RLOF
        if searching_TP:
            if model["R"] < 0.95 * R:
                TP_collision = model["TP"]
                searching_TP = False

        # after finding the thermal pulse count,
        # finding a model 2 thermal pulses earlier
        if not searching_TP:
            if model["TP"] <= (np.max([TP_collision - 2, 0])):
                if model["TP"] == 0 and model["R"] < 0.95 * R:
                    return model
                if model["R"] < 0.9 * R:
                    return model


def get_model_dict():

    models = [os.path.basename(f) for f in os.scandir(f"{grid_dir}/models")]
    print(f"\nloading reference histories and models: {len(models)}\n")
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

    ms_history = mr.MesaData(f"{grid_dir}/reference-histories/ms.data")
    gb_history = mr.MesaData(f"{grid_dir}/reference-histories/gb.data")
    cheb_history = mr.MesaData(f"{grid_dir}/reference-histories/cheb.data")
    eagb_history = mr.MesaData(f"{grid_dir}/reference-histories/eagb.data")
    tpagb_history = mr.MesaData(f"{grid_dir}/reference-histories/tpagb.data")
    histories = [ms_history, gb_history, cheb_history, eagb_history, tpagb_history]

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

    for key in models_dict.keys():
        R = models_dict[key]["R"]
        for history in histories:
            try:
                arg = np.where(history.R >= R - 1e-3)[0][0]
                M = history.star_mass[arg]
                R = history.R[arg]
                break
            except:
                continue

        models_dict[key]["M"] = M
        models_dict[key]["R"] = R
    return models_dict


def get_separation(R, q):
    return (
        (0.6 * (q) ** (-2 / 3) + np.log(1 + (q) ** (-1 / 3)))
        / (0.49 * (q) ** (-2 / 3))
        * R
    )


def change_inlist(R, q, mass, inlist_path):

    with open(inlist_path, "r") as f:
        lines = f.readlines()

        new = []
        for line in lines:
            if "m1" in line:
                new.append(f"\tm1 = {mass}d0  ! donor mass in Msun\n")
            elif "m2" in line:
                new.append(f"\tm2 = {2*q}d0  ! donor mass in Msun\n")
            elif "initial_separation_in_Rsuns" in line:
                new.append(
                    f"\tinitial_separation_in_Rsuns = {get_separation(R, q)}d0 ! in Rsun units\n"
                )
            else:
                new.append(line)

    with open(inlist_path, "w") as f:
        f.writelines(new)


proj_dir = "/home/koen/master-internship"
grid_dir = f"{proj_dir}/mesa-models/tides-grid"

ref_dir = f"{grid_dir}/reference-histories"


def _get_tpagb_age():

    tpagb_age = 0
    for history in ["ms.data", "gb.data", "cheb.data", "eagb.data"]:
        history = mr.MesaData(f"{ref_dir}/{history}")
        tpagb_age += history.star_age[-1]
    return tpagb_age


def _find_initial_age(model):
    if model.header("net_name") == "c13.net":
        return model.header("star_age")
    else:
        return model.header("star_age") - ref_eagb_age


models_dict = get_model_dict()
tpagb_age = _get_tpagb_age()
ref_eagb_age = mr.MesaData(f"{ref_dir}/eagb.data").star_age[-1]

Rs = np.linspace(150, 675, 1)
qs = np.linspace(0.5, 1, 1)


for R in Rs:

    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in qs:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
        bin = Bins[0]
        index = np.argwhere(bin.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m2 = bin.m2[index]


# %%

grid = MesaGrid(f"{MASTER}/tides-grid-2")
# %%

deltat = 1.2474427e9 + 83 - 94400

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
#
# for i, (R, q, model) in enumerate( grid.get_R1_index(-1)):
#     if q != 0.8:
#         continue
#     plt.plot(model.age, model.star.R, c="C9")
#     plt.plot(model.age, model.star.rl_1)
#
for i, (R, q, model) in enumerate(grid.get_R1_index(-3)):
    if q != 0.4:
        continue

    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    (l,) = axs.plot(
        model.age + grid.tpagb_age - deltat,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    (l,) = axs.plot(
        model.age + grid.tpagb_age - deltat,
        model.star.R,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    axs.plot(bin.age - deltat, rl1, c=f"C{i}", alpha=0.4, linewidth=3)
    axs.plot(Star.age - deltat, 10**Star.log_R, c=f"C{i}", alpha=0.4, linewidth=3)


plt.ylim(150, 800)
plt.xlim(0, 400000)
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("results1.svg", format="svg")
plt.show()
plt.close()

# %%

model = grid.get(469.24, 0.9)

model.star.bulk_names
# %%

new_model = mr.MesaData(f"{MASTER}/tides/R469.24_q0.900/LOGS/TPAGB/history.data")
# %%

plt.plot(model.star.age, model.star.rl_1)
plt.plot(model.star.age, model.star.R)
plt.plot(new_model.age, new_model.rl_1)
plt.plot(new_model.age, new_model.R)
plt.show()

# %%

plt.plot(model.star.model_number, model.star.rl_1)
plt.plot(model.star.model_number, model.star.R)
plt.plot(new_model.model_number, new_model.rl_1)
plt.plot(new_model.model_number, new_model.R)
plt.show()

# %%

plt.plot(model.star.age, model.star.Omega_star)
plt.plot(new_model.age, new_model.Omega_star)
plt.plot(model.star.age, model.star.Omega_orb)
plt.plot(new_model.age, new_model.Omega_orb)
plt.show()

# %%


plt.plot(model.star.model_number, model.star.quasi_adiabatic_Mdot)
plt.plot(model.star.model_number, model.star.log_abs_mdot)
plt.plot(new_model.model_number, new_model.quasi_adiabatic_Mdot)
plt.plot(new_model.model_number, new_model.log_abs_mdot)
plt.show()

# %%

plt.plot(model.star.age, model.star.quasi_adiabatic_Mdot)
plt.plot(model.star.age, model.star.log_abs_mdot)
plt.plot(new_model.age, new_model.quasi_adiabatic_Mdot)
plt.plot(new_model.age, new_model.log_abs_mdot)
plt.show()

# %%

plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.rl_1)
plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.R)
plt.plot(new_model.star_2_mass / new_model.star_1_mass, new_model.rl_1)
plt.plot(new_model.star_2_mass / new_model.star_1_mass, new_model.R)
plt.show()

# %%

plt.plot(model.star.star_2_mass / model.star.star_1_mass, -model.star.jdot_ls)
plt.plot(model.star.star_2_mass / model.star.star_1_mass, -model.star.jdot_ml)
plt.plot(new_model.star_2_mass / new_model.star_1_mass, -new_model.jdot_ls)
plt.plot(new_model.star_2_mass / new_model.star_1_mass, -new_model.jdot_ml)
plt.show()

# %%

plt.plot(model.star.model_number, -model.star.jdot_ls)
plt.plot(model.star.model_number, -model.star.jdot_ml)
plt.plot(new_model.model_number, -new_model.jdot_ls)
plt.plot(new_model.model_number, -new_model.jdot_ml)
plt.show()

# %%

plt.plot(model.star.age, -model.star.jdot_ls)
plt.plot(model.star.age, -model.star.jdot_ml)
plt.plot(new_model.age, -new_model.jdot_ls)
plt.plot(new_model.age, -new_model.jdot_ml)
plt.show()

# %%

model = grid.get(650, 0.7)

model.star.bulk_names

# %%

plt.plot(
    model.star.star_2_mass / model.star.star_1_mass, model.star.quasi_adiabatic_Mdot
)
plt.plot(model.star.star_2_mass / model.star.star_1_mass, model.star.log_abs_mdot)
plt.show()

# %%

plt.plot(model.star.star_age, model.star.quasi_adiabatic_Mdot)
plt.plot(model.star.star_age, model.star.log_abs_mdot)
plt.show()


# %%

R_vals = np.array(sorted(set(R for R, q, _ in grid.iter_models())))
q_vals = np.array(sorted(set(q for R, q, _ in grid.iter_models())))

Z = np.full((len(q_vals), len(R_vals)), np.nan)
mask_bad = np.zeros_like(Z, dtype=bool)

for R, q, model in grid.iter_models():
    print(R)
    i = np.where(q_vals == q)[0][0]
    j = np.where(R_vals == R)[0][0]

    if model.env_mass[-1] > 0.1:
        mask_bad[i, j] = True
    else:
        Z[i, j] = model.star.period_days[-1]

logR = np.log10(R_vals)
dlogR = np.diff(logR)

logR_edges = np.concatenate(
    [[logR[0] - dlogR[0] / 2], logR[:-1] + dlogR / 2, [logR[-1] + dlogR[-1] / 2]]
)
R_edges = 10**logR_edges

dq = np.diff(q_vals)
q_edges = np.concatenate(
    [[q_vals[0] - dq[0] / 2], q_vals[:-1] + dq / 2, [q_vals[-1] + dq[-1] / 2]]
)

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges, q_edges, overlay, shading="auto", cmap="Greys", alpha=0.3, rasterized=True
)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="Blues_r", shading="auto", rasterized=True
)


plt.colorbar(mesh, label=r"Orbital period (days)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("results2.svg", format="svg", dpi=600)
# plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-4.pgf", format="pgf")
plt.show()
plt.close()

# %%
