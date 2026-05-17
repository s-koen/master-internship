import shutil
import subprocess
import os
import re
import numpy as np

import mesa_reader as mr
import sys

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

proj_dir = "/home/koen/master-internship"
grid_dir = f"{proj_dir}/mesa-models/tides-grid-2"

ref_dir = f"{grid_dir}/reference-histories"
standard_dir = f"{proj_dir}/mesa-models/standard-2msun-v3/"

Rs = np.logspace(np.log10(150), np.log10(650), 10)
qs = np.linspace(0.4, 1, 7)


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


def change_inlist_bin(a, m1, m2, inlist_path):

    with open(inlist_path, "r") as f:
        lines = f.readlines()

        new = []
        for line in lines:
            if "m1" in line:
                new.append(f"\tm1 = {m1}d0  ! donor mass in Msun\n")
            elif "m2" in line:
                new.append(f"\tm2 = {m2}d0  ! donor mass in Msun\n")
            elif "initial_separation_in_Rsuns" in line:
                new.append(f"\tinitial_separation_in_Rsuns = {a}d0 ! in Rsun units\n")
            else:
                new.append(line)

    with open(inlist_path, "w") as f:
        f.writelines(new)


def change_inlist_star(omega, varcontrol, inlist_path):

    with open(inlist_path, "r") as f:
        lines = f.readlines()

        new = []
        for line in lines:
            if "x_ctrl(3)" in line:
                val = f"{omega:.16e}".replace("e", "d")
                new.append(f"\tx_ctrl(3) = {val}  ! initial rotation rate\n")
            elif "x_ctrl(4)" in line:
                val = f"{varcontrol:.3e}".replace("e", "d")
                new.append(f"\tx_ctrl(4) = {val}  ! initial varcontrol\n")
            else:
                new.append(line)

    with open(inlist_path, "w") as f:
        f.writelines(new)


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

Star = read_stellar_models(standard_dir)[0]

for R in Rs:

    for q in qs:
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star, q, a_init, simple_only=True
        )

        R_star = Star.R
        ages_star = Star.age
        bin = Bins[0]

        q_evolve = bin.m2 / bin.m1
        RL = roche_lobe(1 / q_evolve) * bin.a

        # interpolate stellar radius onto binary ages
        R_interp = np.interp(bin.age, ages_star, R_star)

        # find first near-contact point
        inds = np.where(R_interp > 0.9 * RL)[0]

        if len(inds) == 0:
            print("system never approaches RLOF -> skipping")
            continue

        contact_index = inds[0]
        contact_age = bin.age[contact_index]

        candidate = None

        for key in list(models_dict.keys())[::-1]:

            model = models_dict[key]

            model_path = f"{grid_dir}/models/{model['name']}"
            mod_data = mr.MesaData(model_path)

            model_age = _find_initial_age(mod_data) + tpagb_age

            if model_age < contact_age:
                candidate = model
                break

        if candidate is None:
            print("no suitable stellar model before contact -> skipping")
            continue

        model = candidate
        mass = model["M"]
        model_path = f"{grid_dir}/models/{model["name"]}"
        mod_data = mr.MesaData(model_path)
        model_age = _find_initial_age(mod_data) + tpagb_age

        index = np.argwhere(bin.age > model_age)[0][0] - 1
        index_star = np.argwhere(Star.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m1 = bin.m1[index]
        m2 = bin.m2[index]
        a = bin.a[index]
        omega = bin.spin1[index]
        varcontrol = Star.varcontrol[index_star]

        print(f"\nR = {R:.0f}, q = {q:.2f}")
        run_subdir = f"runs/R{R:05.2f}_q{q:.3f}"
        run_dir = f"{grid_dir}/{run_subdir}"
        print(f"copying reference binary to {run_subdir}...")
        shutil.copytree(f"{grid_dir}/reference-binary", run_dir)
        inlist_bin = f"{run_dir}/inlist_project"
        inlist_star = f"{run_dir}/inlist1"
        print(f"copying {model["name"]} to {run_subdir}...")
        shutil.copyfile(model_path, f"{run_dir}/start.mod")
        print(f"changing inlist...")
        change_inlist_bin(a, m1, m2, inlist_bin)
        change_inlist_star(omega, varcontrol, inlist_star)
