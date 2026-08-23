import shutil
import subprocess
import os
import re
import numpy as np
import mesa_reader as mr
import sys
from pathlib import Path

restarts = 1

# folder_path = "/home/koen/master-internship/mesa-models/grid-masses-2026-08-14-clean/"
#
#
# models = []
# for dirpath, dirnames, filenames in os.walk(folder_path):
#     models.extend(dirnames)
#     break
#
# np.random.shuffle(models)
# model = models[0]
# folder_path = folder_path + model
# print(folder_path)
#
# os.chdir(folder_path)
# print(os.getcwd())
#


def get_mesh_delta_coeff(history, index):
    """
    exactly inverted logic compared to subroutine resolve_mesh_during_mass_loss(s, ierr)
    in run_dir/src/star/additional_routines.inc
    """

    tp_array = get_in_tp(history)
    in_TP = tp_array[-index]
    if in_TP:
        target_mesh_delta_coeff = 2.0

    elif (history.log_abs_mdot[-index]) > -5.5:
        target_mesh_delta_coeff = 1.0

    else:
        target_mesh_delta_coeff = 2.0

    return target_mesh_delta_coeff


def get_in_tp(history):
    """Return a boolean array indicating whether each history point is in a TP."""

    log_LHe = np.asarray(history.log_LHe)
    log_LH = np.asarray(history.log_LH)

    in_tp = np.zeros(len(log_LHe), dtype=bool)
    pulse_active = False

    for i in range(len(log_LHe)):

        if not pulse_active:
            # start thermal pulse
            if log_LHe[i] > 4:
                pulse_active = True

        else:
            # end thermal pulse
            if (log_LHe[i] < 3) and (log_LH[i] - log_LHe[i] > 1):
                pulse_active = False

        in_tp[i] = pulse_active

    return in_tp


def get_restart(prev_start_model):

    photos_path = f"photos/"
    history = mr.MesaData(f"LOGS/history.data")

    index = find_index(history, prev_start_model, verbose=1)

    final_restart_value = history.model_number[-index]
    if final_restart_value % 1000 == 0:
        photo_str = int(final_restart_value)
    else:
        photo_str = f"x{int(str(final_restart_value)[-3:])}"

    return photo_str, get_mesh_delta_coeff(history, index), final_restart_value


def check_finished():
    final_model = Path("post_AGB.mod")
    return final_model.exists()


def find_index(history, prev_start_model, verbose=0):
    """
    conditions for restart:
        1. the model restarts at least 30 model steps before its initial termination
        2. the restart model has a log_dt that is larger than -3.5
        3. the restart model has the same number of retries as the last model saved before
        4. the restart model has a DIFFERENT model number than the previous restart model.
    """

    for i in range(1, len(history.model_number) + 1):

        if history.model_number[-1] - history.model_number[-i] >= 1000:
            mod = 1000
        else:
            mod = 50

        if history.model_number[-i] % mod == 0:
            if verbose:
                print(history.model_number[-i], end=", ")

            if history.model_number[-1] - history.model_number[-i] < 30:
                if verbose:
                    print("not enough models between")
                prev_possible = False
                prev_retries = None
                prev_index = None
                continue

            if history.log_dt[-i] < -3.5:
                if verbose:
                    print("log_dt too small")
                prev_possible = False
                prev_retries = None
                prev_index = None
                continue

            if history.model_number[-i] == prev_start_model:
                if verbose:
                    print(
                        "want to select the same starting model as last time. selecting earlier model."
                    )
                prev_possible = False
                prev_retries = None
                prev_index = None
                continue

            if verbose:
                print("possible, testing retries")
            if history.num_retries[-i] == history.num_retries[-i - 5]:
                print("retries succesful")
                return i
            if verbose:
                print("retries failed")
    raise Exception("No restart model found")


def update_inlist(mesh_delta_coeff):
    """
    this function updates the inlists for the retries. it does 2 things.

    1: it ADDS the line if it is not present:
        min_timestep_limit = 30d0
    because some of my earlier binary models did not have it, and it is
    important not to let the binary models struggle too long once they
    encounter convergence problems.

    2: it changes the delta_mesh_coeff value either to 1, or to 2
    depending on the conditions of the retry model. it changes the mesh_delta_coeff
    value to the opposite value as the one we want to simulate the binary evolution with,
    such that we temporarily use a different mesh refinement and thus hopefully evolve past
    the instability.
    """
    with open("inlist_common", "r") as f:
        lines = f.readlines()
        new = []

        encountered_time_limit = False
        encountered_mesh_delta_coeff = False
        for line in lines:
            if "min_timestep_limit" in line:
                new.append(f"\tmin_timestep_limit = 30d0\n")
                encountered_time_limit = True
            elif "mesh_delta_coeff" in line:
                new.append(f"\tmesh_delta_coeff = {mesh_delta_coeff}d0\n")
                encountered_mesh_delta_coeff = True
            else:
                new.append(line)

        if not encountered_time_limit:
            new.insert(-2, "\tmin_timestep_limit = 30d0\n")

        if not encountered_mesh_delta_coeff:
            new.insert(-2, f"\tmesh_delta_coeff = {mesh_delta_coeff}d0\n")

    with open("inlist_common", "w") as f:
        f.writelines(new)


print("cleaning dir")
subprocess.run(["./clean"])
print("making binaries")
subprocess.run(["./mk"])

restarts = 1


try:
    history = mr.MesaData(f"LOGS/history.data")
except FileNotFoundError:
    print("running from scratch because data is missing")
    subprocess.run([f"./rn"])


prev_start_model = None
while restarts <= 10 and not check_finished():

    photo_int, mesh_delta_coeff, start_model = get_restart(prev_start_model)
    update_inlist(mesh_delta_coeff)

    print(photo_int)
    subprocess.run([f"./re", photo_int])

    if check_finished():
        break
    prev_start_model = start_model

    restarts += 1
# %%
