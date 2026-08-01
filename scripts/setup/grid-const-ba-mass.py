import shutil
import os
import re
import numpy as np
import mesa_reader as mr
import sys
import json
import pickle

# WARNING: CHECK / MODIFY THESE PATHS
grid_name = "const-ba-star-mass-grid"
proj_dir = "/home/koen/master-internship"
reference_binary_dir = f"{proj_dir}/mesa-models/reference-binary/2026-07-01/"
binary_exe_dir = f"{proj_dir}/mesa-models/reference-binary/"

# WARNING: SETTINGS FOR THE GRID
single_star_dirs = [f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M1.5"]
Rs = np.linspace(400, 600, 5)
qs = np.linspace(0.4, 0.8, 5)
barium_mass = 1.29

mass_transfer_beta = np.linspace(0, 1, 5)
mass_transfer_delta = 1 - mass_transfer_beta

# WARNING: SETTINGS FOR BINARY SIMULATION
mass_transfer_alpha = 0.0
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
models_dir = f"{single_star_dirs[0]}/models/"
os.makedirs(os.path.dirname(grid_dir), exist_ok=True)
shutil.copyfile(f"{binary_exe_dir}/binary", f"{grid_dir}/binary")

try:
    with open(f"{single_star_dirs[0]}/combined_star.pkl", "rb") as f:
        Star = pickle.load(f)
    print("read back combined_star.pkl")
except:

    Star = read_stellar_models(single_star_dirs[0])[0]
    with open(f"{single_star_dirs[0]}/combined_star.pkl", "wb") as f:
        pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

tpagb_age = Star.age[Star.ntpagb]


def get_initial_Z():
    model = [os.path.basename(f) for f in os.scandir(models_dir)][0]
    Z = mr.MesaData(f"{models_dir}/{model}").initial_z
    return Z


star_Z = get_initial_Z()


class Evolution:
    def __init__(self, bin, Star, inds, eps, CO, m_DUP):
        self.bin = bin
        self.star = Star
        self.inds = inds
        self.eps = eps
        self.CO = CO
        self.m_DUP = m_DUP


class GridPoint:
    def __init__(self, evolution, model):
        self.evolution = evolution
        self.model = model


def generate_settings_file():

    settings = {
        "grid_name": grid_name,
        "project_directory": proj_dir,
        "single_star_directory": single_star_dirs[0],
        "reference_binary_directory": reference_binary_dir,
        "Rs": Rs.tolist(),
        "qs": qs.tolist(),
        "barium_mass": barium_mass,
        "mass_transfer": {
            "alpha": mass_transfer_alpha,
            "beta": mass_transfer_beta.tolist(),
            "delta": mass_transfer_delta.tolist(),
            "gamma": mass_transfer_gamma,
        },
        "star_Z": float(star_Z),
        "num_runs": float(len(Rs) * len(qs) * len(mass_transfer_beta)),
    }

    with open(f"{grid_dir}/grid_settings.json", "w") as f:
        json.dump(settings, f, indent=4)


def compute_epsilon(barium_mass, tpagb_mass, core_mass, q):
    """
    computes the mass transfer efficiency assuming no wind mass transfer.
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = barium_mass - tpagb_mass * q

    return delta_accreted / delta_donated


def generate_run_file(
    R, q, model_name, age, m1, m2, a, omega, varcontrol, beta, delta, eps, run_dir
):

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
        "beta": beta,
        "delta": delta,
        "eps": eps,
    }

    with open(f"{run_dir}/settings.json", "w") as f:
        json.dump(settings, f, indent=4)


generate_settings_file()


def get_model_dict(params):

    single_star_dir = params["single_star_dir"]
    models_dir = f"{single_star_dir}/models/"
    models = [os.path.basename(f) for f in os.scandir(models_dir)]
    pattern = re.compile(r"([0-9.]+)Rsun_TP([0-9]+)")

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
        core_mass = Star.m_core
        bin = Bins[0]

        q_evolve = bin.m2 / bin.m1
        RL = roche_lobe(1 / q_evolve) * bin.a

        q_begin = q_evolve[-1]
        tpagb_mass = bin.m1[-1]

        # interpolate stellar radius onto binary ages
        R_interp = np.interp(bin.age, ages_star, R_star)
        core_mass_interp = np.interp(bin.age, ages_star, core_mass)
        TP_count = int(np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1])
        CO_ratio = np.interp(
            bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
        )[-1]

        eps = compute_epsilon(barium_mass, tpagb_mass, core_mass_interp[-1], q_begin)
        m_DUP = np.sum(Star.m_DUP[:TP_count])

        inds = np.where(R_interp > 0.9 * RL)[0]
        evolution = Evolution(
            bin=bin, Star=Star, inds=inds, eps=eps, CO=CO_ratio, m_DUP=m_DUP
        )
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
    eps = params["eps"]
    delta = params["delta"]
    single_star_dir = params["single_star_dir"]

    models_dir = f"{single_star_dir}/models/"
    model_path = f"{models_dir}/{model["name"]}"

    run_subdir = f"R{R:05.2f}_q{q:.3f}_eps{eps:.3f}_delta{delta:.3f}"
    run_dir = f"{grid_dir}/{run_subdir}"
    shutil.copytree(reference_binary_dir, run_dir)
    shutil.copyfile(model_path, f"{run_dir}/start.mod")
    return run_dir


def write_run_files(params, model, evolution, run_dir):
    single_star_dir = params["single_star_dir"]
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


def prepare_grid_point(params):

    evolution = evolve_binary(params)
    if evolution is None:
        return None

    if evolution.eps > 1:
        return None

    if evolution.eps < 0:
        return None

    contact_age = find_contact(evolution)
    if contact_age is None:
        return None

    if contact_age is False:
        return False

    models_dict = get_model_dict(params)
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


for single_star_dir in single_star_dirs:
    for R in Rs:
        for q in qs:
            print(R, q)
            base = {
                "R": R,
                "q": q,
                "single_star_dir": single_star_dir,
            }

            grid_point = prepare_grid_point(params=base)

            if grid_point is None:
                print("no suitable model")
                continue
            if grid_point is False:
                print("skipping this R")
                break

            # prevent double computation
            if np.abs(grid_point.evolution.eps - 1) < 0.01:
                params = {
                    **base,
                    "beta": 0,
                    "delta": 0,
                    "eps": grid_point.evolution.eps,
                }

                generate_run(params=params, grid_point=grid_point)
                continue

            for beta, delta in zip(mass_transfer_beta, mass_transfer_delta):
                params = {
                    **base,
                    "beta": beta * (1 - grid_point.evolution.eps),
                    "delta": delta * (1 - grid_point.evolution.eps),
                    "eps": grid_point.evolution.eps,
                }

                generate_run(params=params, grid_point=grid_point)
