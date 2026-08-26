import shutil
import os
import re
import numpy as np
import mesa_reader as mr
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

# ---

# WARNING: CHECK / MODIFY THESE PATHS
name = "grid-masses-3"
grid_name = f"{name}-{datetime.today().strftime('%Y-%m-%d')}"
proj_dir = "/home/koen/master-internship"
reference_binary_dir = f"{proj_dir}/mesa-models/reference-binary/2026-08-15/"
binary_exe_dir = f"{proj_dir}/mesa-models/reference-binary/"

# WARNING: SETTINGS FOR THE GRID
single_star_masses = np.array([2.2])
Rs = np.array([800.0])
qs = np.array([0.7])
epss = np.array([0.25])

mass_transfer_beta = np.array([1 - 0.266666666667])
mass_transfer_delta = 1 - mass_transfer_beta

# WARNING: SETTINGS FOR BINARY SIMULATION
mass_transfer_alpha = 0.0
mass_transfer_gamma = 1.44

# ---

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

single_star_dirs = np.array(
    [
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i}"
        for m_i in single_star_masses
    ]
)

grid_dir = f"{proj_dir}/mesa-models/{grid_name}/"
models_dir = f"{single_star_dirs[0]}/models/"
os.makedirs(os.path.dirname(grid_dir), exist_ok=True)
shutil.copyfile(f"{binary_exe_dir}/binary", f"{grid_dir}/binary")


def get_star(full_path):

    directory = Path(full_path)

    path = directory / "combined_star.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    star = read_stellar_models(directory)[0]

    with path.open("wb") as f:
        pickle.dump(star, f, protocol=pickle.HIGHEST_PROTOCOL)

    return star


def get_initial_Z():
    model = [os.path.basename(f) for f in os.scandir(models_dir)][0]
    Z = mr.MesaData(f"{models_dir}/{model}").initial_z
    return Z


star_Z = get_initial_Z()


class Evolution:
    def __init__(self, bin, Star, inds):
        self.bin = bin
        self.star = Star
        self.inds = inds


class GridPoint:
    def __init__(self, evolution, model):
        self.evolution = evolution
        self.model = model


def generate_settings_file():

    settings = {
        "grid_name": grid_name,
        "project_directory": proj_dir,
        "single_star_directory": single_star_dirs.tolist(),
        "reference_binary_directory": reference_binary_dir,
        "Rs": Rs.tolist(),
        "qs": qs.tolist(),
        "eps": epss.tolist(),
        "mass_transfer": {
            "alpha": mass_transfer_alpha,
            "beta": mass_transfer_beta.tolist(),
            "delta": mass_transfer_delta.tolist(),
            "gamma": mass_transfer_gamma,
        },
        "star_Z": float(star_Z),
        "num_runs": float(len(Rs) * len(qs) * len(epss) * len(mass_transfer_beta)),
    }

    with open(f"{grid_dir}/grid_settings.json", "w") as f:
        json.dump(settings, f, indent=4)


def generate_run_file(
    m_i, R, q, model_name, age, m1, m2, a, omega, varcontrol, beta, delta, eps, run_dir
):

    settings = {
        "m": m_i,
        "R": R,
        "q": q,
        "model_name": model_name,
        "starting_age": age,
        "m1": m1,
        "m2": m2,
        "a": a,
        "omega": omega,
        "varcontrol": varcontrol,
        "beta": beta,
        "delta": delta,
        "eps": eps,
    }

    with open(f"{run_dir}/settings.json", "w") as f:
        json.dump(settings, f, indent=4)


generate_settings_file()


def get_model_dict(single_star_dir):

    models_dir = f"{single_star_dir}/models/"
    models = [os.path.basename(f) for f in os.scandir(models_dir)]
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

    models_R = []
    models_TP = []
    models_age = []

    for model in models:
        p = mr.MesaData(f"{models_dir}/{model}")
        match = pattern.search(model)
        R = float(match.group(1))
        TP = int(match.group(2))
        models_R.append(R)
        models_TP.append(TP)
        models_age.append(p.star_age)

    models_dict = {}
    index = 0

    while len(models_R) > 0:
        arg = np.argmin(models_R)
        models_dict[f"model {index}"] = {
            "name": models.pop(arg),
            "R": models_R.pop(arg),
            "TP": models_TP.pop(arg),
            "age": models_age.pop(arg),
        }

        index += 1
    star = get_star(single_star_dir)

    for key in models_dict.keys():

        R = models_dict[key]["R"]
        naive_arg = np.where(np.abs(10**star.log_R - R) <= 1e-2)[0][0]
        phase = int(star.phase[naive_arg])

        match phase:
            case 0:
                offset = 0
            case 1:
                offset = star.age[star.ntams]
            case 2:
                offset = star.age[star.nzacheb]
            case 3:
                offset = star.age[star.ntacheb]
            case 4:
                offset = star.age[star.ntpagb]
            case _:
                raise Exception("invalid phase")

        real_age = models_dict[key]["age"] + offset
        arg = np.argmin(np.abs(star.age - real_age))
        R = 10 ** star.log_R[arg]
        age = star.age[arg]
        M = star.mass[arg]

        models_dict[key]["M"] = M
        models_dict[key]["phase"] = phase
        models_dict[key]["age"] = age
        models_dict[key]["real_R"] = R

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


def load_star(single_star_dir):

    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        # print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)
    return Star


def evolve_binary(params):
    q = params["q"]
    R = params["R"]
    single_star_dir = params["single_star_dir"]
    try:
        Star = load_star(single_star_dir)
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
        evolution = Evolution(bin=bin, Star=Star, inds=inds)
        return evolution
    except:
        print(
            f"something went wrong during simple integration with {params=}",
        )
        return None


def find_contact(evolution):
    if len(evolution.inds) == 0:
        return False

    contact_index = evolution.inds[0]
    contact_age = evolution.bin.age[contact_index]

    if contact_age < evolution.star.age[evolution.star.ntpagb]:
        return None
    return contact_age


def select_model(contact_age, models_dict):

    candidate = None
    models = list(models_dict.values())

    for i in range(len(models) - 1, -1, -1):

        if models[i]["age"] < contact_age:

            # Return the previous model if it exists
            if i > 0:
                return models[i - 1]

    print("no suitable stellar model before contact -> skipping")
    return candidate


def prepare_run_dir(params, model):
    R = params["R"]
    q = params["q"]
    m = params["m"]
    eps = params["eps"]
    delta = params["delta"]
    single_star_dir = params["single_star_dir"]

    models_dir = f"{single_star_dir}/models/"
    model_path = f"{models_dir}/{model["name"]}"

    run_subdir = f"R{R:05.2f}_q{q:.3f}_eps{eps:.3f}_delta{delta:.3f}_M{m:.1f}"
    run_dir = f"{grid_dir}/{run_subdir}"
    shutil.copytree(reference_binary_dir, run_dir)
    shutil.copyfile(model_path, f"{run_dir}/start.mod")
    return run_dir


def write_run_files(params, model, evolution, run_dir):
    single_star_dir = params["single_star_dir"]
    m_i = params["m"]
    beta = params["beta"]
    delta = params["delta"]
    eps = params["eps"]

    model_age = model["age"]
    index = np.argwhere(evolution.bin.age > model_age)[0][0] - 1
    index_star = np.argwhere(evolution.star.age > model_age)[0][0] - 1
    m1 = evolution.bin.m1[index]
    m2 = evolution.bin.m2[index]
    a = evolution.bin.a[index]
    omega = evolution.bin.spin1[index]
    varcontrol = evolution.star.varcontrol[index_star]

    models_dir = f"{single_star_dir}/models/"
    model_path = f"{models_dir}/{model["name"]}"

    inlist_bin = f"{run_dir}/inlist_project"
    inlist_star = f"{run_dir}/inlist1"
    inlist_common = f"{run_dir}/inlist_common"
    generate_run_file(
        m_i,
        R,
        q,
        model_path,
        model_age,
        m1,
        m2,
        a,
        omega,
        varcontrol,
        beta,
        delta,
        eps,
        run_dir,
    )

    change_inlist_bin(
        a,
        m1,
        m2,
        mass_transfer_alpha,
        beta,
        delta,
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


def prepare_grid_point(params, models_dict):

    evolution = evolve_binary(params)
    if evolution is None:
        return None

    contact_age = find_contact(evolution)
    if contact_age is None:
        return None

    if contact_age is False:
        return False

    model = select_model(contact_age, models_dict)
    if model is None:
        return None

    return GridPoint(evolution, model)


def generate_run(params, grid_point):

    run_dir = prepare_run_dir(params, grid_point.model)

    write_run_files(
        params,
        grid_point.model,
        grid_point.evolution,
        run_dir,
    )


for single_star_mass in single_star_masses:

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{single_star_mass}"
    )
    print(f"getting model dict for {single_star_mass}")
    models_dict = get_model_dict(single_star_dir)
    for R in Rs:
        for q in qs:
            print(single_star_mass, R, q)
            base = {
                "R": R,
                "q": q,
                "m": single_star_mass,
                "single_star_dir": single_star_dir,
            }

            grid_point = prepare_grid_point(params=base, models_dict=models_dict)

            if grid_point is None:
                print("no suitable model")
                continue
            if grid_point is False:
                print("skipping this R q combination")
                continue

            for eps in epss:
                if eps == 1:
                    params = {
                        **base,
                        "beta": 0,
                        "delta": 0,
                        "eps": eps,
                    }

                    generate_run(params=params, grid_point=grid_point)
                    continue

                for beta, delta in zip(mass_transfer_beta, mass_transfer_delta):
                    params = {
                        **base,
                        "beta": beta * (1 - eps),
                        "delta": delta * (1 - eps),
                        "eps": eps,
                    }

                    generate_run(params=params, grid_point=grid_point)
# %%
