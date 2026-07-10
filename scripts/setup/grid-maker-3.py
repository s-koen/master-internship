import shutil
import os
import re
import numpy as np
import mesa_reader as mr
import sys
import json
import pickle

# WARNING: CHECK / MODIFY THESE PATHS
grid_name = "test-grid"
proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/new-abundances/M2.0/"
reference_binary_dir = f"{proj_dir}/mesa-models/reference-binary/2026-07-01/"

# WARNING: SETTINGS FOR THE GRID

Rs = np.logspace(np.log10(50), np.log10(2000), 10)
qs = np.linspace(0.4, 1, 2)

# WARNING: SETTINGS FOR BINARY SIMULATION
mass_transfer_alpha = 0.0
mass_transfer_beta = 0.875
mass_transfer_delta = 0.125
mass_transfer_gamma = 1.44

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

grid_dir = f"{proj_dir}/mesa-models/{grid_name}/"
models_dir = f"{single_star_dir}/models/"
os.makedirs(os.path.dirname(grid_dir), exist_ok=True)

try:
    with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
        Star = pickle.load(f)
    print("read back combined_star.pkl")
except:

    Star = read_stellar_models(single_star_dir)[0]
    with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
        pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

tpagb_age = Star.age[Star.ntpagb]


def get_initial_Z():
    model = [os.path.basename(f) for f in os.scandir(models_dir)][0]
    Z = mr.MesaData(f"{models_dir}/{model}").initial_z
    return Z


star_Z = get_initial_Z()


def generate_settings_file():

    settings = {
        "grid_name": grid_name,
        "project_directory": proj_dir,
        "single_star_directory": single_star_dir,
        "reference_binary_directory": reference_binary_dir,
        "Rs": Rs.tolist(),
        "qs": qs.tolist(),
        "mass_transfer": {
            "alpha": mass_transfer_alpha,
            "beta": mass_transfer_beta,
            "delta": mass_transfer_delta,
            "gamma": mass_transfer_gamma,
        },
        "star_Z": float(star_Z),
        "num_runs": float(len(Rs) * len(qs)),
    }

    with open(f"{grid_dir}/grid_settings.json", "w") as f:
        json.dump(settings, f, indent=4)


def generate_run_file(R, q, model_name, age, m1, m2, a, omega, varcontrol):

    settings = {
        "R": R,
        "q": q,
        "model_name": model_name,
        "starting_age": age,
        "m1": m1,
        "m2": m2,
        "a": a,
        "omega": omega,
        "varcontrol": varcontrol,
    }

    with open(f"{grid_dir}/{run_subdir}/settings.json", "w") as f:
        json.dump(settings, f, indent=4)


generate_settings_file()


def get_model_dict():
    models = [os.path.basename(f) for f in os.scandir(models_dir)]
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

    models_R = []
    models_TP = []
    models_age = []

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
        arg = np.where(10**Star.log_R >= R - 1e-3)[0][0]
        M = Star.mass[arg]
        phase = Star.phase[arg]
        R = 10 ** Star.log_R[arg]
        age = Star.age[arg]

        models_dict[key]["M"] = M
        models_dict[key]["phase"] = phase
        models_dict[key]["age"] = age
        models_dict[key]["R"] = R

    return models_dict


def change_inlist_bin(
    a,
    m1,
    m2,
    mass_transfer_alpha,
    mass_transfer_beta,
    mass_transfer_delta,
    mass_transfer_gamma,
    inlist_path,
):

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
            elif "mass_transfer_alpha" in line:
                new.append(f"\tmass_transfer_alpha = {mass_transfer_alpha}d0\n")
            elif "mass_transfer_beta" in line:
                new.append(f"\tmass_transfer_beta = {mass_transfer_beta}d0\n")
            elif "mass_transfer_delta" in line:
                new.append(f"\tmass_transfer_delta = {mass_transfer_delta}d0\n")
            elif "mass_transfer_gamma" in line:
                new.append(f"\tmass_transfer_gamma = {mass_transfer_gamma}d0\n")
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


def change_inlist_common(star_Z, inlist_path):

    with open(inlist_path, "r") as f:
        lines = f.readlines()

        new = []
        for line in lines:
            if "Zbase" in line:
                val = f"{star_Z:.16e}".replace("e", "d")
                new.append(f"\tZbase = {val}  ! base metallicity\n")
            elif "initial_z " in line:
                val = f"{star_Z:.16e}".replace("e", "d")
                new.append(f"\tinitial_z = {val}  ! initial metallicity\n")
            else:
                new.append(line)

    with open(inlist_path, "w") as f:
        f.writelines(new)


print("\nloading models...")
models_dict = get_model_dict()

print("\nfinding models...")
for R in Rs:

    print(f"\n{R = :.1f}\n |-> q = ", end="")
    for q_i, q in enumerate(qs):
        print(f"{q:.1f}", end=", ", flush=True)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star, q, a_init, simple_only=True
        )

        R_star = 10**Star.log_R
        ages_star = Star.age
        bin = Bins[0]

        q_evolve = bin.m2 / bin.m1
        RL = roche_lobe(1 / q_evolve) * bin.a

        # interpolate stellar radius onto binary ages
        R_interp = np.interp(bin.age, ages_star, R_star)

        inds = np.where(R_interp > 0.9 * RL)[0]

        if len(inds) == 0:
            print("system never approaches RLOF -> skipping")
            break

        contact_index = inds[0]
        contact_age = bin.age[contact_index]

        if contact_age < tpagb_age:
            (
                print("system approaches RLOF before TPAGB -> skipping")
                if q_i == len(qs) - 1
                else print(
                    "system approaches RLOF before TPAGB -> skipping\n |-> q = ", end=""
                )
            )
            continue

        candidate = None

        for key in list(models_dict.keys())[::-1]:

            model = models_dict[key]
            if model["age"] < contact_age:
                candidate = model
                break

        if candidate is None:
            print("no suitable stellar model before contact -> skipping")
            continue

        model = candidate
        mass = model["M"]
        model_path = f"{models_dir}/{model["name"]}"
        mod_data = mr.MesaData(model_path)
        model_age = model["age"]

        index = np.argwhere(bin.age > model_age)[0][0] - 1
        index_star = np.argwhere(Star.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m1 = bin.m1[index]
        m2 = bin.m2[index]
        a = bin.a[index]
        omega = bin.spin1[index]
        varcontrol = Star.varcontrol[index_star]

        run_subdir = f"R{R:05.2f}_q{q:.3f}"
        run_dir = f"{grid_dir}/{run_subdir}"
        shutil.copytree(reference_binary_dir, run_dir)
        inlist_bin = f"{run_dir}/inlist_project"
        inlist_star = f"{run_dir}/inlist1"
        inlist_common = f"{run_dir}/inlist_common"
        shutil.copyfile(model_path, f"{run_dir}/start.mod")

        generate_run_file(R, q, model_path, model_age, m1, m2, a, omega, varcontrol)

        change_inlist_bin(
            a,
            m1,
            m2,
            mass_transfer_alpha,
            mass_transfer_beta,
            mass_transfer_delta,
            mass_transfer_gamma,
            inlist_bin,
        )
        change_inlist_star(
            omega,
            varcontrol,
            inlist_star,
        )
        change_inlist_common(
            star_Z,
            inlist_common,
        )

    else:
        continue
    break
