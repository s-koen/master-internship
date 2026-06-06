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

grid_3_5 = MesaGrid(f"{MASTER}/tides-M3.5")

# %%


grid_cons = MesaGrid(f"{MASTER}/tides-grid-6")

# %%


grid = grid_cons

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

fig, axs = plt.subplots(1, 2, figsize=(20, 20), width_ratios=[0.7, 0.3], sharey=True)
bins = np.arange(1.5, 5.0, 1 / 3)
axs[1].hist(np.log10(data["Porb"]), orientation="horizontal", bins=bins, edgecolor="k")
axs[1].set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10 000", "100 000"])

# plt.xscale("log")
# plt.xlim(1e1, 1e5)
# plt.ylim(0, 1)
axs[0].spines["left"].set_position(("outward", 22.5))
axs[0].spines["bottom"].set_position(("outward", 22.5))
axs[1].spines["left"].set_position(("outward", 22.5))
axs[1].spines["bottom"].set_position(("outward", 22.5))
plt.subplots_adjust(wspace=1 / 13.5, hspace=0)
axis_size(10 / 8 * 7, 5 / 8 * 7, axs[1])
axis_size(10 / 8 * 7, 5 / 8 * 7, axs[0])
plt.savefig("presentation-1/period-eccentricity-2.svg")
plt.show()

# %%
