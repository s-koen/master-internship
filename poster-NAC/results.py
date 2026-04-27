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
#
# %%

plt.plot(model.star.age, model.star.period_days)
plt.plot(model.star.age, model.star.R)
plt.plot(new_model.age, new_model.period_days)
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
plt.plot(model.env_mass, model.star.rl_1)
plt.plot(model.env_mass, model.star.R)
plt.plot(new_model.star_mass - new_model.he_core_mass, new_model.rl_1)
plt.plot(new_model.star_mass - new_model.he_core_mass, new_model.R)
plt.show()



# %%

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   version="1.1"
   id="svg1"
   xml:space="preserve"
   width="1449.515"
   height="968.61285"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg"><defs
     id="defs1" /><sodipodi:namedview
     id="namedview1"
     pagecolor="#ffffff"
     bordercolor="#000000"
     borderopacity="0.25"
     inkscape:showpageshadow="2"
     inkscape:pageopacity="0.0"
     inkscape:pagecheckerboard="0"
     inkscape:deskcolor="#d1d1d1" /><inkscape:clipboard
     style="stroke-width:1.33333338;stroke-linecap:butt;stroke-linejoin:round"
     min="1589.1764,2842.3543"
     max="3038.6914,3810.9671"
     geom-min="1589.1764,2842.3543"
     geom-max="3038.6914,3810.9671" /><g
     id="g199"
     transform="matrix(3.7795276,0,0,3.7795276,-1589.1764,-2958.6066)"><g
       id="figure_1-4-4"
       transform="matrix(0.84356642,0,0,0.84356642,486.77092,868.70999)"
       style="stroke-width:0.418198;stroke-linecap:butt;stroke-linejoin:round"
       inkscape:label="figure 4"><rect
         style="fill:#ffffff;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         id="rect362-5-4"
         width="454.63818"
         height="303.80396"
         x="-78.596458"
         y="-101.84378"
         ry="4.273286e-16" /><g
         id="axes_1-4-8"
         style="opacity:0.8;stroke-width:0.178431;stroke-linecap:butt;stroke-linejoin:round"
         transform="matrix(1.061675,0,0,0.9048738,-21.983233,-2.6792508)"><g
           id="patch_2-2"
           style="stroke-width:0.178431;stroke-linecap:butt;stroke-linejoin:round"><path
             d="M 33.676606,162.85168 H 247.45859 V 3.00024 H 33.676606 Z"
             style="fill:#ffffff;stroke-width:0.178431;stroke-linecap:butt;stroke-linejoin:round"
             id="path2-3" /></g><image
           xlink:href="data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAABvUAAAU0CAYAAADoiJuQAAAx8klEQVR4nOzbz2pcZQCH4STMNKczySSpIdX+sZWCG0EEwZ3gXtx7D16E4C14J7kuFy6UtjZNXbh38w1875TnuYLf4vBxOO/5jo+Ojn46AgDYg+WbX25nb4BhNy9nL4Bh66ub2RNgL7a77ewJMOzxk8vZE2DYycnx7AkwbLdbZk+AYSezBwAAAAAAAAD/T9QDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBuNXsAAP9Zvv/1dvYGGHb12ewFMGy1LLMnwLDHz29mT4C9uL7ezJ4Aw756+Wj2BBh2d38/ewIM+/JmO3sCDHNTDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4lazB8A+LD/8djt7Awzb3cxeAMO2VxezJ8Cw60+vZk+AYc+fOY/5ODxY+xeZw/fdi/PZE2DYm7v72RNg2KvL7ewJMMzbMQAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAELeaPQD2YnczewEM21zsZk+AYctmmT0Bhp2fn86eAMN2m/XsCbAX3764nD0Bhj05847M4Ts+Op49AYY9v9jMngDD3NQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIC41fLj77ezR8CoB9vN7AkwbHuxnT0Bhn1yfTZ7Agz78OHD7Akw7O6955iPw7I+nj0Bhv397m72BBj26vJ89gQYdrVdz54Aw9zUAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAOFEPAAAAAAAA4kQ9AAAAAAAAiBP1AAAAAAAAIE7UAwAAAAAAgDhRDwAAAAAAAOJEPQAAAAAAAIgT9QAAAAAAACBO1AMAAAAAAIA4UQ8AAAAAAADiRD0AAAAAAACIE/UAAAAAAAAgTtQDAAAAAACAuNXJw+3sDTBsc7aZPQGGnS4PZk+AYeu1/4U4fF88vZg9AYbtHq5nT4C9+OOvu9kTYNjPXz+bPQGGvb27nz0Bhp0tq9kTYJgvbwAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGiHgAAAAAAAMSJegAAAAAAABAn6gEAAAAAAECcqAcAAAAAAABxoh4AAAAAAADEiXoAAAAAAAAQJ+oBAAAAAABAnKgHAAAAAAAAcaIeAAAAAAAAxIl6AAAAAAAAECfqAQAAAAAAQJyoBwAAAAAAAHGi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"
           id="imaged17c4b5057-0"
           transform="matrix(1,0,0,-1,0,159.84)"
           x="33.720001"
           y="-2.8800001"
           width="213.72"
           height="159.84"
           style="stroke-width:0.178431;stroke-linecap:butt;stroke-linejoin:round" /></g><g
         id="axes_2-3"
         style="opacity:0.8;stroke-width:0.172129;stroke-linecap:butt;stroke-linejoin:round"
         transform="matrix(1.1419599,0,0,0.90398789,-43.988566,-2.5361368)"><image
           xlink:href="data:image/png;base64, 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"
           id="image96e2c0d74d-7"
           transform="matrix(1,0,0,-1,0,159.84)"
           x="258.12"
           y="-2.8800001"
           width="8.04"
           height="159.84"
           style="stroke-width:0.172129;stroke-linecap:butt;stroke-linejoin:round" /></g><g
         id="g455-5-6"
         transform="matrix(-1.1854431,0,0,0.98715023,405.74287,-562.85318)"
         style="stroke-width:0.458279;stroke-linecap:butt;stroke-linejoin:round"><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.547923;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 118.37865,586.09735 h -4.01082 v 126.16438 h 4.01082"
           id="path318-0-4-4"
           sodipodi:nodetypes="cccc" /><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.547923;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36783,670.16234 4.01082,2e-5"
           id="path319-9-6-6"
           sodipodi:nodetypes="cc" /><path
           style="fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.547923;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36783,628.13206 h 4.06697"
           id="path323-5-5-7-6"
           sodipodi:nodetypes="cc" /></g><g
         id="g330-7-7"
         inkscape:label="normal axis"
         transform="matrix(1.1854431,0,0,1.1854431,-88.092097,-697.70413)"
         style="stroke-linecap:butt;stroke-linejoin:round"><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="88.29705"
           y="730.91553"
           id="text324-7-5-2-1"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="88.29705"
             y="730.91553"
             id="tspan325-8-2-0-1">150 <tspan
   style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;baseline-shift:baseline"
   id="tspan640">R</tspan><tspan
   style="font-size:65%;baseline-shift:sub"
   id="tspan639">ⵙ</tspan></tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="260.19995"
           y="730.91553"
           id="text324-7-5-3-1-9-9"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="260.19995"
             y="730.91553"
             id="tspan325-8-2-9-5-4-5">650 <tspan
   style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;baseline-shift:baseline"
   id="tspan641">R</tspan><tspan
   style="font-size:65%;baseline-shift:sub"
   id="tspan638">ⵙ</tspan></tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="58.414356"
           y="705.07703"
           id="text324-7-3-09-4"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="58.414356"
             y="705.07703"
             id="tspan325-8-1-5-0">0.4</tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="58.515987"
           y="670.15906"
           id="text324-7-3-09-9-6"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="58.515987"
             y="670.15906"
             id="tspan325-8-1-5-5-6">0.6</tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="58.394756"
           y="635.2984"
           id="text324-7-3-09-9-5-0"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="58.394756"
             y="635.2984"
             id="tspan325-8-1-5-5-8-6">0.8</tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="60.370182"
           y="599.9787"
           id="text324-7-3-0-2-7"><tspan
             sodipodi:role="line"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1"
             x="60.370182"
             y="599.9787"
             id="tspan325-8-1-0-4-9">1.0</tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:8.46667px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;opacity:1;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.600001;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="141.3428"
           y="730.9494"
           id="text452-5"><tspan
             sodipodi:role="line"
             id="tspan452-6"
             style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.6"
             x="141.3428"
             y="730.9494">Initial Roche lobe radius </tspan></text></g><g
         id="g457-5"
         transform="matrix(1.1016314,0,0,1.1854431,-124.2121,-712.13203)"
         style="stroke-width:0.433815;stroke-linecap:butt;stroke-linejoin:round"><path
           style="fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.518671;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 135.54631,728.44552 v 4.01082 h 185.44135 v -4.01082"
           id="path318-2-9-9"
           sodipodi:nodetypes="cccc" /></g><g
         id="g455-1"
         transform="matrix(1.1854431,0,0,0.98322716,-134.76443,-566.02347)"
         style="stroke-width:0.459193;stroke-linecap:butt;stroke-linejoin:round"><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 118.37865,586.09735 h -4.01082 v 126.16438 h 4.01082"
           id="path318-0-5"
           sodipodi:nodetypes="cccc" /><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36783,670.16234 4.01082,2e-5"
           id="path319-9-2"
           sodipodi:nodetypes="cc" /><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36248,691.23743 h 1.75672"
           id="path321-0-6"
           sodipodi:nodetypes="cc" /><path
           style="opacity:1;fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36248,649.13917 h 1.75077"
           id="path323-5-57"
           sodipodi:nodetypes="cc" /><path
           style="fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36783,628.13206 h 4.06697"
           id="path323-5-5-5"
           sodipodi:nodetypes="cc" /><path
           style="fill:none;fill-opacity:1;stroke:#606060;stroke-width:0.549014;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           d="m 114.36248,607.14027 h 1.75077"
           id="path323-5-5-2-3"
           sodipodi:nodetypes="cc" /></g><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;text-align:start;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;text-anchor:start;opacity:1;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1.18544;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="-45.492928"
         y="-12.311503"
         id="text324-79-1"><tspan
           sodipodi:role="line"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:1.18544"
           x="-45.492928"
           y="-12.311503"
           id="tspan352-5-2">Initial mass ratio [<tspan
   style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal"
   id="tspan642">M</tspan><tspan
   style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:6.69116px;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;stroke-width:1.18544"
   id="tspan442-0">a</tspan>/<tspan
   style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal"
   id="tspan643">M</tspan><tspan
   style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:6.69116px;font-family:Sans;-inkscape-font-specification:'Sans, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;stroke-width:1.18544"
   id="tspan443-1">d</tspan>] </tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;white-space:pre;inline-size:352.513;display:inline;opacity:1;fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="-44.569561"
         y="-49.741219"
         id="text354-2-3"
         transform="translate(-0.97781398,-20.338121)"><tspan
           x="-44.569561"
           y="-49.741219"
           id="tspan6"><tspan
             style="font-size:12.5459px;stroke-width:0.711266"
             id="tspan4">Fully conservative RLOF mass transfer results </tspan><tspan
             style="font-size:12.5459px;stroke-width:0.711266"
             id="tspan5">in periods </tspan></tspan><tspan
           x="-44.569561"
           y="-34.686139"
           id="tspan8"><tspan
             style="font-size:12.5459px;stroke-width:0.711266"
             id="tspan7">larger than those of the observed barium star population.</tspan></tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;opacity:1;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="274.88895"
         y="144.08965"
         id="text461-3"><tspan
           sodipodi:role="line"
           id="tspan461-4"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711266"
           x="274.88895"
           y="144.08965">2500</tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="274.88895"
         y="102.53125"
         id="text461-9-8"><tspan
           sodipodi:role="line"
           id="tspan461-8-2"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711266"
           x="274.88895"
           y="102.53125">5000</tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="275.29041"
         y="60.500145"
         id="text461-9-6-92"><tspan
           sodipodi:role="line"
           id="tspan461-8-4-4"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711266"
           x="275.29041"
           y="60.500145">7500</tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="275.29041"
         y="18.196766"
         id="text461-9-6-9-5"><tspan
           sodipodi:role="line"
           id="tspan461-8-4-8-0"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711266"
           x="275.29041"
           y="18.196766">10 000</tspan></text><text
         xml:space="preserve"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.0368px;line-height:1.2;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711267;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
         x="249.85104"
         y="-12.015393"
         id="text452-4-4"><tspan
           sodipodi:role="line"
           id="tspan452-7-3"
           style="fill:#606060;fill-opacity:1;stroke:none;stroke-width:0.711266"
           x="249.85104"
           y="-12.015393">Final period [days] </tspan></text><g
         id="g650"
         style="display:none;stroke-linecap:butt;stroke-linejoin:round"><text
           xml:space="preserve"
           style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:1.4;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;baseline-shift:baseline;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1.18544;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="277.18652"
           y="-194.82767"
           id="text644-7"><tspan
             sodipodi:role="line"
             id="tspan644-1"
             style="baseline-shift:baseline;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1.18544"
             x="277.18652"
             y="-194.82767"><tspan
               style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.8731px;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal"
               id="tspan648">M</tspan><tspan
               style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:65%;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;baseline-shift:sub;fill:#606060;fill-opacity:1"
               id="tspan646">star</tspan></tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:12.5459px;line-height:1.4;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;opacity:1;fill:#606060;fill-opacity:1;stroke:none;stroke-width:1.18544;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="282.06763"
           y="-203.90823"
           id="text647"><tspan
             sodipodi:role="line"
             id="tspan647"
             style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:12.5459px;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;stroke-width:1.18544"
             x="282.06763"
             y="-203.90823">.</tspan></text></g><g
         id="g649"
         transform="translate(0,26.409965)"
         style="display:none;stroke-linecap:butt;stroke-linejoin:round"><text
           xml:space="preserve"
           style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:1.4;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;baseline-shift:baseline;opacity:1;fill:#458fbf;fill-opacity:1;stroke:none;stroke-width:1.18544;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="277.18652"
           y="-269.81143"
           id="text644"><tspan
             sodipodi:role="line"
             id="tspan644"
             style="baseline-shift:baseline;fill:#458fbf;fill-opacity:1;stroke:none;stroke-width:1.18544"
             x="277.18652"
             y="-269.81143"><tspan
               style="font-style:italic;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:10.8731px;font-family:Geist;-inkscape-font-specification:'Geist, Italic';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal"
               id="tspan649">M</tspan><tspan
               style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:7.8px;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;baseline-shift:sub;fill:#458fbf;fill-opacity:1"
               id="tspan645">th, crit</tspan></tspan></text><text
           xml:space="preserve"
           style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:12.5459px;line-height:1.4;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;letter-spacing:0px;writing-mode:lr-tb;direction:ltr;fill:#458fbf;fill-opacity:1;stroke:none;stroke-width:1.18544;stroke-linecap:butt;stroke-linejoin:miter;stroke-dasharray:none;stroke-opacity:1;paint-order:markers stroke fill"
           x="281.95868"
           y="-278.60556"
           id="text647-2"><tspan
             sodipodi:role="line"
             id="tspan647-4"
             style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:12.5459px;font-family:Geist;-inkscape-font-specification:'Geist, Normal';font-variant-ligatures:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-east-asian:normal;fill:#458fbf;fill-opacity:1;stroke-width:1.18544"
             x="281.95868"
             y="-278.60556">.</tspan></text></g></g></g></svg>
# %%

grid2 = MesaGrid(f"{MASTER}/tides-grid-3")

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
for i, (R, q, model) in enumerate(grid.get_R1_index(3)):
    if q not in [0.4, 0.5, .9]:
        continue

    (l,) = axs.plot(
        model.star.star_2_mass / model.star.star_1_mass,
        model.star.period_days / model.star.period_days[0],
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )



plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("results1.svg", format="svg")
plt.show()
plt.close()


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
for i, (R, q, model) in enumerate(grid.get_R1_index(8)):
    if q not in [0.6]:
        continue

    (l,) = axs.plot(
        model.star.star_mass,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    (l,) = axs.plot(
        model.star.star_mass,
        model.star.R,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )


plt.gca().invert_xaxis()
# plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\\odot$)")
# plt.savefig("results1.svg", format="svg")
plt.show()
plt.close()


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
for i, (R, q, model) in enumerate(grid.get_R1_index(4)):
    if q not in [0.6]:
        continue

    (l,) = axs.plot(
        model.star.star_age,
        model.star.quasi_adiabatic_Mdot,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    (l,) = axs.plot(
        model.star.star_age,
        model.star.log_abs_mdot,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )


plt.xlim(0,500000)
plt.ylim(-12,2)
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel("Radius ($R_\\odot$)")
plt.savefig("results3.svg", format="svg")
plt.show()
plt.close()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


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
        Z[i, j] = np.nanmax(model.star.R / rol(model.star))

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
    R_edges, q_edges, Z, cmap="Blues", shading="auto", vmin=0.8, vmax=1.6, rasterized=True
)


plt.colorbar(mesh, label=r"Orbital period (days)")

ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
ax.set_ylabel("$q$")

ax.set_xscale("log")

plt.savefig("results4.svg", format="svg", dpi=600)
# plt.savefig("/home/koen/LaTeX-setup/plots/w12-grid-4.pgf", format="pgf")
plt.show()
plt.close()


# %%

