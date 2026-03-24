import shutil
import subprocess
import os
import re
import numpy as np

import mesa_reader as mr


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
                print(TP_collision)
                searching_TP = False

        # after finding the thermal pulse count,
        # finding a model 2 thermal pulses earlier
        else:
            print(model["TP"], model["R"])
            if model["TP"] <= (np.max([TP_collision - 2, 0])) and model["R"] < 0.9 * R:
                print(model)
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
grid_dir = f"{proj_dir}/mesa-models/binary-tpagb-grid-test"

models_dict = get_model_dict()

Rs = np.linspace(150, 675, 8)
qs = np.linspace(1, 1, 1)
for R in Rs:

    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"

    for q in qs:
        print(f"\nR = {R:.0f}, q = {q:.2f}")
        run_subdir = f"runs/R{R:05.2f}_q{q:.3f}"
        run_dir = f"{grid_dir}/{run_subdir}"
        print(f"copying reference binary to {run_subdir}...")
        shutil.copytree(f"{grid_dir}/reference-binary", run_dir)
        inlist_path = f"{run_dir}/inlist_project"
        print(f"copying {model["name"]} to {run_subdir}...")
        shutil.copyfile(model_path, f"{run_dir}/start.mod")
        print(f"changing inlist...")
        change_inlist(R, q, mass, inlist_path)
