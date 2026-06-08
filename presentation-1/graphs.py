import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

import pandas as pd

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("presentation")

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid


def axis_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


# %%

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")

# %%

data["Porb"] = data["Porb"][1:].astype("float")
data["ecc"] = data["ecc"][1:].astype("float")
# %%
fig = plt.figure()
plt.scatter(data["Porb"], data["ecc"], s=50)
plt.xscale("log")
plt.xlim(1e1, 1e5)
plt.ylim(0, 1)
ax = plt.gca()
ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))
plt.title("Forming barium stars", size=50)
plt.xlabel("Period (days)")
axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/period-eccentricity.svg")
plt.show()
# %%
rees = mr.MesaData(f"{MASTER}tides-grid-4/reference-histories/tpagb.data")
# %%
import re
import os

grid_dir = f"{MASTER}tides-grid-6"


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


model_dict = get_model_dict()
# %%


def _find_initial_age(model):
    if model.header("net_name") == "c13.net":
        return model.header("star_age")
    else:
        return 0


# %%

ages = []
for key in list(model_dict.keys())[::-1]:

    model = model_dict[key]

    model_path = f"{grid_dir}/models/{model['name']}"
    mod_data = mr.MesaData(model_path)

    model_age = _find_initial_age(mod_data)
    if model_age != 0:
        ages.append(model_age)

# %%

r_interp = np.interp(ages, rees.star_age, rees.R)

# %%
fig, ax = plt.subplots()
plt.plot(rees.star_age / 1e6 + 0.02, rees.R)
plt.scatter(
    np.array(ages) / 1e6 + 0.02,
    r_interp,
    marker="s",
    s=200,
    edgecolor="w",
    linewidth=2,
    zorder=1000,
)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))
print(np.max(rees.R))
plt.xlim(0, 2)
plt.yticks([0, 719])
plt.xticks([0, 2])
plt.ylim(0, 719)
axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/rees-model.svg")

plt.show()
# %%

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid
import os
import re
import numpy as np

sys.path.insert(1, "/home/koen/master-internship/")
sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa")

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

grid = MesaGrid(f"{MASTER}/tides-grid-6")


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

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    (l,) = plt.plot(
        model.age / 1e6 + 0.02,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    plt.plot(
        (bin.age - grid.tpagb_age) / 1e6 + 0.02, rl1, c=f"C{i}", alpha=0.4, linewidth=3
    )
    plt.plot(
        (Star.age - grid.tpagb_age) / 1e6 + 0.02,
        10**Star.log_R,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )
    r_interp = np.interp(
        ages,
        (Star.age - grid.tpagb_age),
        10**Star.log_R,
    )

    plt.scatter(
        np.array(ages) / 1e6 + 0.02,
        r_interp,
        marker="s",
        s=100,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )
    plt.scatter(
        np.array(ages)[4] / 1e6 + 0.02,
        r_interp[4],
        marker="s",
        s=400,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )
plt.xlim(0, 2)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.xlim(0, 2)
plt.yticks([0, 550, 719])
plt.xticks([0, 2])
plt.ylim(0, 719)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/binary-methods.svg")
plt.show()

# %%

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    delta = model.age[0]

    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.R,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    plt.plot(
        (bin.age - grid.tpagb_age - delta) / 1e6, rl1, c=f"C{i}", alpha=0.4, linewidth=3
    )
    plt.plot(
        (Star.age - grid.tpagb_age - delta) / 1e6,
        10**Star.log_R,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )
    r_interp = np.interp(
        ages,
        (Star.age - grid.tpagb_age),
        10**Star.log_R,
    )

    plt.scatter(
        (np.array(ages) - delta)[4] / 1e6,
        r_interp[4],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )

    plt.scatter(
        0,
        model.star.rl_1[0],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )
plt.xlim(0, 2)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.xlim(-0.1, 0.32296 / 28 + 0.2296)

plt.yticks([150, 550])
plt.xticks([-0.1, 0, 0.2296, 0.32296 / 28 + 0.2296], [])
plt.ylim(150, 550)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-1.svg")
plt.show()


# %%

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    delta = model.age[0]

    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.log_abs_mdot,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    (l,) = plt.plot(
        model.star.age / 1e6,
        np.log10(model.star.quasi_adiabatic_Mdot),
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    print(np.max(model.star.log_abs_mdot))

    plt.plot(
        (Star.age - grid.tpagb_age - delta) / 1e6,
        np.log10(Star.Mdot),
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )

    plt.plot(
        (Star.age - grid.tpagb_age - delta) / 1e6,
        np.log10(np.log10(Star.log_Mdot_crit)),
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )

    r_interp = np.interp(
        ages,
        (Star.age - grid.tpagb_age),
        np.log10(Star.Mdot),
    )

    plt.scatter(
        0,
        model.star.log_abs_mdot[0],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )

plt.xlim(0, 2)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.xlim(-0.1, 0.32296 / 28 + 0.2296)

plt.yticks([-10, -1.28, 0])
plt.xticks([-0.1, 0, 0.2296, 0.32296 / 28 + 0.2296], [])
plt.ylim(-10, 0)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-2.svg")
plt.show()


# %%


fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    delta = model.age[0]

    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.R,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    plt.plot(
        (bin.age - grid.tpagb_age - delta) / 1e6, rl1, c=f"C{i}", alpha=0.4, linewidth=3
    )
    plt.plot(
        (Star.age - grid.tpagb_age - delta) / 1e6,
        10**Star.log_R,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )
    r_interp = np.interp(
        ages,
        (Star.age - grid.tpagb_age),
        10**Star.log_R,
    )

    plt.scatter(
        (np.array(ages) - delta)[4] / 1e6,
        r_interp[4],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )

    plt.scatter(
        0,
        model.star.rl_1[0],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )
plt.xlim(0, 2)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.xlim(-0.1, 0.32296 / 28 + 0.2296)

plt.yticks([150, 550])
plt.xticks([-0.1, 0, 0.2296, 0.32296 / 28 + 0.2296], [])
plt.ylim(150, 550)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-1.svg")
plt.show()


# %%


fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2
    rl1 = roche_lobe(q_evolve) * bin.a

    delta = model.age[0]

    (l,) = plt.plot(
        model.star.age / 1e6,
        model.env_mass,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    plt.plot(
        (Star.age - grid.tpagb_age - delta) / 1e6,
        Star.mass - Star.m_core,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )
    r_interp = np.interp(
        ages,
        (Star.age - grid.tpagb_age),
        Star.mass - Star.m_core,
    )

    plt.scatter(
        0,
        model.env_mass[0],
        marker="s",
        s=200,
        edgecolor="w",
        linewidth=2,
        zorder=1000,
    )
plt.xlim(0, 2)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = True
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = False


ax.spines["right"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.xlim(-0.1, 0.32296 / 28 + 0.2296)

# plt.yticks([150, 550])
plt.xticks([-0.1, 0, 0.1, 0.2296, 0.32296 / 28 + 0.2296], [])
# plt.ylim(150, 550)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-3.svg")
plt.show()


# %%

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


for i, (R, q, model) in enumerate(grid.get_R1_index(-2)):

    if q not in [0.4]:
        continue
    print(R)

    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.rl_1,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )
    (l,) = plt.plot(
        model.star.age / 1e6,
        model.star.R,
        c=f"C{i}",
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    for i in range(len(model.star.age)):
        if i < 1060:
            continue
        if model.star.rl_1[i] > model.star.R[i]:
            print(i)
            break

    print(model.star.age[1463])
plt.xlim(0, 2)


ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True


plt.xlim(0.229502, 0.229852)

plt.yticks([250, 500], [])
plt.xticks([0.229502, 0.229602, 0.22966688, 0.2297158174896005, 0.229852], [])
plt.ylim(250, 500)


axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-4.svg")
plt.show()


# %%

print((0.22966688 - 0.229602) * 1e6)
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
        m_initial = q * 2
        m_C_initial = m_initial * grid.ref_ms.surface_c12[-1]
        m_O_initial = m_initial * grid.ref_ms.surface_o16[-1]

        dt = np.diff(model.star.star_age)
        dm = np.diff(model.star.star_2_mass)

        Xc = 0.5 * (model.star.surface_c12[:-1] + model.star.surface_c12[1:])
        Xo = 0.5 * (model.star.surface_o16[:-1] + model.star.surface_o16[1:])
        M_c_transfer = np.cumsum(Xc * dm)
        M_o_transfer = np.cumsum(Xo * dm)
        M_c = m_C_initial + M_c_transfer
        M_o = m_O_initial + M_o_transfer
        Xc_final = M_c / model.star.star_2_mass[1:]
        Xo_final = M_o / model.star.star_2_mass[1:]

        Z[i, j] = Xc_final[-1] / Xo_final[-1] * 16 / 12


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

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges, q_edges, Z, cmap="coolwarm", shading="auto", vmin=1 - (Zmax - 1), vmax=Zmax
)


cbar = plt.colorbar(
    mesh,
    # label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
    extend="min",
)
# ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
# ax.set_ylabel("$q$")
#
ax.set_xscale("log")

cbar.ax.spines["right"].set_position(("outward", 22.5))

ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True

plt.xticks([150, 250, 400, 650], [])
plt.yticks([0.4, 0.7, 1], [])
plt.xticks([], [], minor=True)

axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-5.svg")
plt.show()


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

        Z[i, j] = np.log10(np.nanmax(model.star.R / rol(model.star)))


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

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="PiYG_r",
    shading="auto",
    vmin=np.log10(5 / 8),
    vmax=np.log10(1.6),
)


print(Zmax)
cbar = plt.colorbar(
    mesh,
    # label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
)


cbar.set_ticks([np.log10(2 / 3), 0, np.log10(3 / 2)])
cbar.set_ticklabels(["2/3", "1", "3/2"])
# ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
# ax.set_ylabel("$q$")
#
ax.set_xscale("log")

cbar.ax.spines["right"].set_position(("outward", 22.5))

ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True

plt.xticks([150, 250, 400, 650], [])
plt.yticks([0.4, 0.7, 1], [])
plt.xticks([], [], minor=True)

axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-5.svg")
plt.show()


# %%

grid_non_cons = MesaGrid(f"{MASTER}/tides-grid-4")

# %%

grid_very_non_cons = MesaGrid(f"{MASTER}/tides-grid-7")

# %%

grid_very_non_cons_2 = MesaGrid(f"{MASTER}/tides-grid-8")

# %%


grid_3_5 = MesaGrid(f"{MASTER}/tides-M3.5")

# %%


grid_cons = MesaGrid(f"{MASTER}/tides-grid-6")

# %%


grid = grid_very_non_cons_2

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

        Z[i, j] = np.nanmax(model.star.period_days)


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

fig = plt.figure()
ax = plt.gca()
axis_size(10 / 8 * 7, 5 / 8 * 7)


overlay = np.where(
    mask_bad,
    1,
    0,
)
ax.pcolormesh(
    R_edges,
    q_edges,
    overlay,
    shading="auto",
    cmap="Greys",
    alpha=0.3,
)
Zmax = np.nanmax(Z)
mesh = ax.pcolormesh(
    R_edges,
    q_edges,
    Z,
    cmap="Spectral",
    shading="auto",
    # vmin=np.log10(5 / 8),
    # vmax=np.log10(1.6),
)


print(Zmax)
cbar = plt.colorbar(
    mesh,
    # label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
)


# cbar.set_ticks([np.log10(2 / 3), 0, np.log10(3 / 2)])
# cbar.set_ticklabels(["2/3", "1", "3/2"])
# ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
# ax.set_ylabel("$q$")
#
ax.set_xscale("log")

cbar.ax.spines["right"].set_position(("outward", 22.5))

ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True

plt.xticks([150, 250, 400, 650], [])
plt.yticks([0.4, 0.7, 1], [])
plt.xticks([], [], minor=True)

axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-6.svg")
plt.show()


# %%

fig, axs = plt.subplots(1, 2, figsize=(20, 20), width_ratios=[1, 1], sharey=True)
bins = np.arange(1.5, 5.0, 1 / 3)
axs[1].hist(
    np.log10(data["Porb"]),
    orientation="horizontal",
    bins=bins,
    edgecolor="k",
    histtype="step",
)
axs[1].set_yticks([2, 3, 4], [])

# plt.xscale("log")
# plt.xlim(1e1, 1e5)
plt.ylim(2 - 1 / 6, 4 + 3 / 6)
axs[0].spines["left"].set_position(("outward", 22.5))
axs[0].spines["bottom"].set_position(("outward", 22.5))
axs[1].tick_params(direction="inout", axis="y")
axs[1].spines["left"].set_position(("outward", 22.5))
axs[1].spines["bottom"].set_position(("outward", 22.5))
axs[1].spines["bottom"].set_visible(False)
plt.subplots_adjust(wspace=2 / 17.5, hspace=0)
axis_size(10 * 36 / 28 / 8 * 7, 5 / 8 * 7, axs[1])
axis_size(10 * 36 / 28 / 8 * 7, 5 / 8 * 7, axs[0])

axs[0].set_xticks([-1e6, -0.5e6, 0], [], minor=False)
axs[1].set_xticks([], [], minor=False)
delta = 1.245852e9 + 810 + 40000

grid = grid_cons


def orbital_period_days(obj):
    """
    Compute orbital period (in days) from semi-major axis and masses.

    Parameters
    ----------
    obj : object
        Must have attributes:
        - a  : semi-major axis [R_sun]
        - m1 : mass of primary [M_sun]
        - m2 : mass of secondary [M_sun]

    Returns
    -------
    float
        Orbital period in days.
    """
    # constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    R_sun = 6.957e8  # m
    M_sun = 1.98847e30  # kg

    # convert to SI
    a = obj.a * R_sun
    M = (obj.m1 + obj.m2) * M_sun

    # kepler's third law
    P = 2 * np.pi * np.sqrt(a**3 / (G * M))

    # convert to days
    return P / (60 * 60 * 24)


#
# for i, (R, q, model) in enumerate(grid.iter_models()):
#     if R < 240 or R > 600:
#         continue
#
#     a_init = inv_roche_lobe(R, q)
#     [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
#     bin = Bins[0]
#     q_evolve = bin.m1 / bin.m2
#
#     index = np.argwhere(model.star.R > model.star.rl_1)[-1][-1]
#     time = model.age[index]
#
#     (l,) = plt.plot(
#         model.age + grid.tpagb_age - time,
#         model.star.period_days,
#         linewidth=1,
#         label=f"$q_i = {q:.1f}$",
#         c="C9"
#     )
#     stop = np.argwhere(bin.age > model.age[0] + grid.tpagb_age)[0][0]
#     plt.plot(bin.age[:stop] - time, orbital_period_days(bin)[:stop], c="C9")
#

for i, (R, q, model) in enumerate(grid.get_R1_index(4)):
    if q != 0.4:
        continue

    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2

    index = np.argwhere(model.star.R > model.star.rl_1)[-1][-1]
    time = model.age[index]
    (l,) = axs[0].plot(
        model.age + grid.tpagb_age - time - delta,
        np.log10(model.star.period_days),
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    stop = np.argwhere(bin.age > model.age[0] + grid.tpagb_age)[0][0]
    axs[0].plot(
        bin.age[:stop] - time - delta, np.log10(orbital_period_days(bin)[:stop])
    )

for i, (R, q, model) in enumerate(grid.get_R1_index(-1)):
    if q != 1:
        continue

    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star3, q, a_init)
    bin = Bins[0]
    q_evolve = bin.m1 / bin.m2

    index = np.argwhere(model.star.R > model.star.rl_1)[-1][-1]
    time = model.age[index]

    (l,) = axs[0].plot(
        model.age + grid.tpagb_age - time - delta,
        np.log10(model.star.period_days),
        linewidth=1,
        label=f"$q_i = {q:.1f}$",
    )

    stop = np.argwhere(bin.age > model.age[0] + grid.tpagb_age)[0][0]
    axs[0].plot(
        bin.age[:stop] - time - delta, np.log10(orbital_period_days(bin)[:stop])
    )

axs[0].set_xlim(-1e6, 0)


plt.savefig("presentation-1/period-eccentricity-2.svg")
plt.show()


# %%
further = mr.MesaData(f"{MASTER}/run-further/R552.27_q0.500/LOGS/TPAGB/history.data")

delta = np.array(
    [
        1832709.270831625,
        1832474.3565732383,
        141536.33550088422,
    ]
)

q_reversal = np.array(
    [
        1832803.3871383066,
        1832542.369949226,
        141589.52183649162,
    ]
)

for g, grid in enumerate((grid_cons, grid_non_cons)):
    for i, (R, q, model) in enumerate(grid.get_R1_index(8)):
        if q != 0.5:
            continue
        plt.plot(model.age - delta[g], model.star.R)
        plt.plot(model.age - delta[g], model.star.rl_1)

        for i in range(len(model.age)):
            if model.star.R[i] > model.star.rl_1[i]:
                print("start RLOF", model.age[i])
                break
            model_cool = model

        for i in range(len(model.age)):
            if model.star.star_2_mass[i] > model.star.star_1_mass[i]:
                print("q-reveral", model.age[i])
                break

        plt.scatter(
            model.age[i] - delta[g],
            model.star.rl_1[i],
            marker="s",
            s=200,
            edgecolor="w",
            linewidth=2,
            zorder=1000,
        )


for i in range(len(further.age)):
    if further.R[i] > further.rl_1[i]:
        print("start RLOF", further.age[i])
        break

for i in range(len(further.age)):
    if further.star_2_mass[i] > further.star_1_mass[i]:
        print("start RLOF", further.age[i])
        print(i)
        break


print(i)

plt.plot(further.age - delta[-1], further.R)
plt.plot(further.age - delta[-1], further.rl_1)

plt.scatter(
    further.age[i] - delta[-1],
    further.rl_1[i],
    marker="s",
    s=200,
    edgecolor="w",
    linewidth=2,
    zorder=1000,
)

ticks = [-50, 0, 72.125, 98.5, 200]
plt.xlim(-50, 200)
plt.xticks(ticks, [])
plt.ylim(135, 535)
plt.yticks([100, 405, 550], [])


ax = plt.gca()
ax.spines["left"].set_position(("outward", 22.5))
ax.spines["bottom"].set_position(("outward", 22.5))

plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True

axis_size(10 / 8 * 7, 5 / 8 * 7)
plt.savefig("presentation-1/results-7.svg")
plt.show()

# %%

for grid in (grid_cons, grid_non_cons, grid_very_non_cons_2):
    for i, (R, q, model) in enumerate(grid.get_R1_index(8)):
        if q != 0.5:
            continue
        print(R)
        plt.plot(model.env_mass, model.star.R)
        plt.plot(model.env_mass, model.star.rl_1)

plt.plot(further.envelope_mass, further.R)
plt.show()

# %%

fig = plt.figure()

for grid in (grid_cons, grid_non_cons, grid_very_non_cons_2):
    if grid == grid_cons:
        delta = -30000
    else:
        delta = 0
    for i, (R, q, model) in enumerate(grid.get_R1_index(8)):
        if q != 0.5:
            continue
        print(R)
        plt.plot(model.star.log_Teff, model.star.log_L)


plt.plot(further.log_Teff, further.log_L)


plt.gca().invert_xaxis()
plt.show()


# %%

further = mr.MesaData(f"{MASTER}/run-further/R552.27_q0.500/LOGS/TPAGB/history.data")
# %%

plt.plot(further.age, further.R)
plt.plot(further.age, further.rl_1)
plt.show()

# %%

plt.plot(further.age, further.R)
plt.plot(further.age, further.rl_1)

plt.plot(model_cool.star.age, model_cool.star.R)
plt.plot(model_cool.star.age, model_cool.star.rl_1)


plt.show()
# %%

print(model_cool.star.Omega_star[-1])
# %%

plt.plot(further.envelope_mass, further.R)
plt.plot(further.envelope_mass, further.rl_1)

plt.plot(model_cool.env_mass, model_cool.star.R)
plt.plot(model_cool.env_mass, model_cool.star.rl_1)

plt.show()
# %%

grids = [grid_cons, grid_non_cons, grid_very_non_cons_2]

fig, axs = plt.subplots(1, 3, sharey=True)

for g, ax in enumerate(axs):
    grid = grids[g]

    axis_size(10 / 8 * 7, 5 / 2 / 8 * 7, ax)
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

            Z[i, j] = np.log10(np.nanmax(model.star.R / rol(model.star)))

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

    overlay = np.where(
        mask_bad,
        1,
        0,
    )
    ax.pcolormesh(
        R_edges,
        q_edges,
        overlay,
        shading="auto",
        cmap="Greys",
        alpha=0.3,
    )
    Zmax = np.nanmax(Z)
    mesh = ax.pcolormesh(
        R_edges,
        q_edges,
        Z,
        cmap="PiYG_r",
        shading="auto",
        vmin=np.log10(0.25),
        vmax=np.log10(4),
    )
    ax.set_xscale("log")

    cbar.ax.spines["right"].set_position(("outward", 22.5))

    if g > 0:
        ax.tick_params(tickdir="inout", axis="y")
    else:
        ax.tick_params(tickdir="in", axis="y")
    ax.spines["left"].set_position(("outward", 22.5))
    ax.spines["bottom"].set_position(("outward", 22.5))

    ax.set_xticks([150, 650], [])
    ax.set_xticks([], [], minor=True)


print(Zmax)
cbar = plt.colorbar(
    mesh,
    # label=r"$($C/O-ratio$)_\textrm{cons} / ($C/O-ratio$)_\textrm{rad}$",
)


# cbar.set_ticks([np.log10(2 / 3), 0, np.log10(3 / 2)])
# cbar.set_ticklabels(["2/3", "1", "3/2"])
# ax.set_xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
# ax.set_ylabel("$q$")
#
plt.rcParams["ytick.right"] = plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True

plt.yticks([0.4, 0.7, 1], [])

plt.savefig("presentation-1/results-8.svg")
plt.show()


# %%
