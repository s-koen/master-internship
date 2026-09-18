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


def change_inlist_common(m, inlist_path):

    with open(inlist_path, "r") as f:
        lines = f.readlines()

        new = []
        for line in lines:
            if "initial_mass " in line:
                val = f"{m:.16e}".replace("e", "d")
                new.append(f"\tinitial_mass = {val}  ! initial mass\n")
            else:
                new.append(line)

    with open(inlist_path, "w") as f:
        f.writelines(new)


reference_star = "/home/koen/master-internship/mesa-models/single-ms-stars/reference/"

ms = np.linspace(0.4, 4.4, 41)
for m in ms:
    run_dir = f"/home/koen/master-internship/mesa-models/single-ms-stars/M{m:.1f}/"
    shutil.copytree(reference_star, run_dir)

    inlist_path = run_dir + "inlist_common"
    change_inlist_common(m, inlist_path)


# %%
