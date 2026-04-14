import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
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
import os
import re
import numpy as np

import mesa_reader as mr

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
Star2 = read_stellar_models(
    f"/home/koen/master-internship/mesa-models/standard-2msun-v2/"
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
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star2, q, a_init)
        bin = Bins[0]
        index = np.argwhere(bin.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m2 = bin.m2[index]


# %%
Rs = np.linspace(150, 675, 1)
Rs = [300]
qs = np.linspace(0.5, 1, 1)
qs = [0.5]
for R in Rs:

    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in qs:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(Star2, q, a_init)
        bin = Bins[0]
        index = np.argwhere(bin.age > model_age)[0][0] - 1
        index_star = np.argwhere(Star.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m2 = bin.m2[index]
        rl1 = roche_lobe(q_evolve) * bin.a
        varcontrol = Star.varcontrol
        print(varcontrol[index_star])
        plt.plot(Star.age, Star.varcontrol)
        plt.scatter(Star.age[index_star], Star.varcontrol[index_star])
plt.show()


# %%


q_evolve = bin.m1 / bin.m2
m2 = bin.m2[index]
rl1 = roche_lobe(q_evolve) * bin.a

plt.plot(bin.age, rl1)
plt.show()
# %%

grid = MesaGrid(f"{MASTER}/tides-grid")
# %%

for q, model1 in grid.get_R1_slice(300):
    print(q)

for q, model2 in grid.get_R1_slice(301):
    print(q)

for q, model3 in grid.get_R1_slice(302):
    print(q)

for q, model4 in grid.get_R1_slice(303):
    print(q)

for q, model5 in grid.get_R1_slice(304):
    print(q)


# %%

# plt.plot(model1.age + grid.tpagb_age, model1.star.binary_separation)
# plt.plot(model2.age + grid.tpagb_age, model2.star.binary_separation)
# plt.plot(model3.age + grid.tpagb_age, model3.star.binary_separation)
# plt.plot(model4.age + grid.tpagb_age, model4.star.binary_separation)
# plt.plot(model5.age + grid.tpagb_age, model5.star.binary_separation)

for R in [300]:
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for boool in [True, False]:
        for q in qs:
            print(R, q)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star,
                q,
                a_init,
                simple_only=boool,
            )
            bin = Bins[0]
            index = np.argwhere(bin.age > model_age)[0][0] - 1
            q_evolve = bin.m1 / bin.m2
            m2 = bin.m2[index]
            rl1 = roche_lobe(q_evolve) * bin.a
            plt.plot(bin.age, rl1)

plt.plot(Star.age, Star.radius)
plt.show()
# %%

plt.plot(grid.ref_tpagb.star_age + grid.tpagb_age, grid.ref_tpagb.t_conv)
plt.plot(model.age + grid.tpagb_age, model.star.t_conv)
plt.plot(Star.age, Star.tconv * 3600 * 24 * 365.25)
plt.xlim(grid.tpagb_age, bin.age[-1])
plt.show()

# %%

plt.plot(model.age + grid.tpagb_age, model.star.Omega_star)
plt.plot(bin.age, bin.spin1)
plt.xlim(grid.tpagb_age, bin.age[-1])
plt.show()


# %%
from scipy.interpolate import interp1d

o_orb = omega_Kep(bin.m1 + bin.m2, bin.a)

o_tid = const.yr * abs(o_orb - bin.spin1)
p_tid = 2 * np.pi / o_tid
print(o_tid)


f = interp1d(Star.age, Star.tconv, bounds_error=False, fill_value="extrapolate")

bin.tconv = f(bin.age)

f_conv = (0.5 * p_tid / bin.tconv) ** 2
f_conv[np.where(f_conv > 1)] = 1


plt.plot(bin.age, f_conv)
plt.plot(model.age + grid.tpagb_age, model.star.f_conv)
plt.xlim(grid.tpagb_age, bin.age[-1])
plt.show()


# %%
log_z = np.log10(0.014)
a = 2.72 + 0.63 * log_z
b1 = 0.68 - 0.219 * log_z
c = 0.12 - 0.023 * log_z

M_conv = model.star.M_conv
M = model.star.star_mass

R_conv = model.star.R_conv
R = model.star.R

t_conv = model.star.t_conv
f_conv = 1
k_over_T = (R_conv / R) ** a * (M_conv / M) ** b1 * c / t_conv * f_conv


plt.plot(Star.age, Star.k_over_T_conv)
plt.plot(model.age + grid.tpagb_age, k_over_T * const.yr)
plt.xlim(grid.tpagb_age, bin.age[-1])
plt.show()

# %%

plt.plot(Star.age, Star.varcontrol)
plt.plot(model.age + grid.tpagb_age, model.star.varcontrol)
plt.xlim(grid.tpagb_age, bin.age[-1])
plt.show()
# %%

fig, axs = plt.subplots(2, 2, figsize=set_size(full, height=1), constrained_layout=True)
axs = axs.flatten()

labels = ["Simple integration only", r"Simple + \texttt{solve\_ivp}"]

for i, R in enumerate([150, 300, 450, 600]):
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    ls = []
    for j, boool in enumerate([True, False]):
        for q in [0.8]:
            print(R, q)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star3,
                q,
                a_init,
                simple_only=boool,
            )
            bin = Bins[0]
            index = np.argwhere(bin.age > model_age)[0][0] - 1
            q_evolve = bin.m1 / bin.m2
            m2 = bin.m2[index]
            rl1 = roche_lobe(q_evolve) * bin.a
            (l,) = axs[i].plot(bin.age, rl1, c=f"C{j}", label=labels[j])
            ls.append(l)
    print(ls)

fig.legend(loc="outside upper center", ncols=2, handles=ls)
axs[2].set_xlabel("Star age (yr)")
axs[3].set_xlabel("Star age (yr)")
axs[0].set_ylabel("Roche lobe radius ($R_\odot$)")
axs[2].set_ylabel("Roche lobe radius ($R_\odot$)")

axs[0].set_xlim(740000 + 1.245e9, 860000 + 1.245e9)
axs[0].set_ylim(143.25, 145.75)

axs[1].set_xlim(0 + 1.246e9, 650000 + 1.246e9)
axs[1].set_ylim(280, 300)

axs[2].set_xlim(1.2465e9, 1.2476e9)
axs[2].set_ylim(420, 447)

axs[3].set_xlim(300000 + 1.247e9, 720000 + 1.247e9)
axs[3].set_ylim(425, 600)

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-1.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    2, 2, figsize=set_size(full, height=0.96), constrained_layout=True
)
axs = axs.flatten()

labels = ["Simple integration only", r"Simple + \texttt{solve\_ivp}"]

for i, R in enumerate([150, 300, 450, 600]):
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in [0.8]:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star3,
            q,
            a_init,
            simple_only=False,
        )
        bin1 = Bins[0]
        q_evolve = bin1.m1 / bin1.m2
        m2 = bin1.m2[index]
        rl0 = roche_lobe(q_evolve) * bin1.a

        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star3,
            q,
            a_init,
            simple_only=True,
        )

        bin2 = Bins[0]
        q_evolve = bin2.m1 / bin2.m2
        rl1 = roche_lobe(q_evolve) * bin2.a

        x1 = bin1.age
        y1 = rl0

        x2 = bin2.age
        y2 = rl1

        # make sure mesa is sorted (interp1d requires this)
        idx = np.argsort(x2)
        x2_sorted = x2[idx]
        y2_sorted = y2[idx]

        # interpolation function (no extrapolation outside range)
        f_interp = interp1d(x2_sorted, y2_sorted, bounds_error=False, fill_value=np.nan)

        # evaluate mesa on evolve grid
        y2_interp = f_interp(x1)

        # residual
        residual = y1 - y2_interp

        # plot
        axs[i].plot(
            x1,
            residual,
        )
        if i == 3:
            axs[i].text(
                0.05,
                0.05,
                f"$R = {R}\;R_\\odot$",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=axs[i].transAxes,
            )
            continue

        axs[i].text(
            0.05,
            0.95,
            f"$R = {R}\;R_\\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axs[i].transAxes,
        )


axs[2].set_xlabel("Star age (yr)")
axs[3].set_xlabel("Star age (yr)")
axs[0].set_ylabel("Residual ($R_\odot$)")
axs[2].set_ylabel("Residual ($R_\odot$)")

axs[0].set_xlim(740000 + 1.245e9, 860000 + 1.245e9)

axs[1].set_xlim(0 + 1.246e9, 650000 + 1.246e9)

axs[2].set_xlim(1.2465e9, 1.2476e9)

axs[3].set_xlim(300000 + 1.247e9, 720000 + 1.247e9)

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(2, 2, figsize=set_size(full, height=1), constrained_layout=True)
axs = axs.flatten()

labels = ["Simple integration only", r"Simple + \texttt{solve\_ivp}"]

for i, R in enumerate([150, 300, 450, 600]):
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    ls = []
    for j, boool in enumerate([True, False]):
        for q in [0.8]:
            print(R, q)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star2,
                q,
                a_init,
                simple_only=boool,
            )
            bin = Bins[0]
            q_evolve = bin.m1 / bin.m2
            rl1 = roche_lobe(q_evolve) * bin.a
            (l,) = axs[i].plot(bin.age, rl1, c=f"C{j}", label=labels[j])
            ls.append(l)
    print(ls)

fig.legend(loc="outside upper center", ncols=2, handles=ls)
axs[2].set_xlabel("Star age (yr)")
axs[3].set_xlabel("Star age (yr)")
axs[0].set_ylabel("Roche lobe radius ($R_\odot$)")
axs[2].set_ylabel("Roche lobe radius ($R_\odot$)")

axs[0].set_xlim(1.09e9, 1.25e9)
axs[0].set_ylim(140, 152.5)

axs[1].set_xlim(1.245e9, 1.2468e9)
axs[1].set_ylim(282.5, 307.5)

axs[2].set_xlim(1.2458e9, 1.2477e9)
axs[2].set_ylim(420, 460)

axs[3].set_xlim(1.2466e9, 1.24775e9)
axs[3].set_ylim(425, 610)

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-3.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(2, 2, figsize=set_size(full, height=1), constrained_layout=True)
axs = axs.flatten()

labels = [
    r"Reduced - Simple integration only",
    "Complete - Simple integration only",
    r"Complete - Simple + \texttt{solve\_ivp}",
]

for i, R in enumerate([150, 300, 450, 600]):
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    ls = []
    for j, boool in enumerate([True, False]):
        for q in [0.8]:
            print(R, q)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star3,
                q,
                a_init,
                simple_only=boool,
            )
            bin = Bins[0]
            q_evolve = bin.m1 / bin.m2
            rl1 = roche_lobe(q_evolve) * bin.a
            (l,) = axs[i].plot(bin.age, rl1, c=f"C{j+1}", label=labels[j + 1])
            ls.append(l)

    for q in [0.8]:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star2,
            q,
            a_init,
            simple_only=True,
        )
        bin = Bins[0]
        q_evolve = bin.m1 / bin.m2
        rl1 = roche_lobe(q_evolve) * bin.a
        (l,) = axs[i].plot(bin.age, rl1, c=f"C0", label=labels[0])
        ls.append(l)

    print(ls)

fig.legend(loc="outside upper center", ncols=2, handles=ls)
axs[2].set_xlabel("Star age (yr)")
axs[3].set_xlabel("Star age (yr)")
axs[0].set_ylabel("Roche lobe radius ($R_\odot$)")
axs[2].set_ylabel("Roche lobe radius ($R_\odot$)")

axs[0].set_xlim(740000 + 1.245e9, 860000 + 1.245e9)
axs[0].set_ylim(143.25, 145.75)

axs[1].set_xlim(0 + 1.246e9, 650000 + 1.246e9)
axs[1].set_ylim(280, 300)

axs[2].set_xlim(1.2465e9, 1.2476e9)
axs[2].set_ylim(420, 447)

axs[3].set_xlim(300000 + 1.247e9, 720000 + 1.247e9)
axs[3].set_ylim(425, 600)

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, figsize=set_size(full, height=0.96), constrained_layout=True
)

labels = ["Simple integration only", r"Simple + \texttt{solve\_ivp}"]

R_list = []
age_diff = []

for i, R in enumerate(np.linspace(100, 200, 100)):
    print(f"\n\n{i}\n\n")
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in [0.8]:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star2,
            q,
            a_init,
            simple_only=False,
        )
        bin1 = Bins[0]
        q_evolve = bin1.m1 / bin1.m2
        rl0 = roche_lobe(q_evolve) * bin1.a

        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star2,
            q,
            a_init,
            simple_only=True,
        )

        bin2 = Bins[0]
        q_evolve = bin2.m1 / bin2.m2
        rl1 = roche_lobe(q_evolve) * bin2.a

        R_list.append(R)
        age_diff.append(np.abs(bin2.age[-1] - bin1.age[-1]))
plt.plot(R_list, age_diff)

plt.yscale("log")
plt.ylabel(r"$\Delta t_\textrm{finish}$")
plt.xlabel(r"Initial Roche lobe radius ($R_\odot$) ")

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-5.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, figsize=set_size(full, height=0.96), constrained_layout=True
)

labels = ["Simple integration only", r"Simple + \texttt{solve\_ivp}"]

R_list = []
age_diff = []

for i, R in enumerate(np.linspace(100, 1000, 300)):
    print(f"\n\n{i}\n\n")
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in [0.8]:
        print(R, q)
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star3,
            q,
            a_init,
            simple_only=False,
        )
        bin1 = Bins[0]
        q_evolve = bin1.m1 / bin1.m2
        rl0 = roche_lobe(q_evolve) * bin1.a

        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star3,
            q,
            a_init,
            simple_only=True,
        )

        bin2 = Bins[0]
        q_evolve = bin2.m1 / bin2.m2
        rl1 = roche_lobe(q_evolve) * bin2.a

        R_list.append(R)
        age_diff.append(np.abs(bin2.age[-1] - bin1.age[-1]))
plt.plot(R_list, age_diff)

plt.yscale("log")
plt.ylabel(r"$\Delta t_\textrm{finish}$")
plt.xlabel(r"Initial Roche lobe radius ($R_\odot$) ")

plt.savefig("/home/koen/LaTeX-setup/plots/w11-compare-simple-scipy-6.pgf", format="pgf")
plt.show()
plt.close()
# %%

grid = MesaGrid(f"{MASTER}/tides-grid")
# %%

for q, model1 in grid.get_R1_slice(300):
    print(q)


# %%

# plt.plot(model1.age + grid.tpagb_age, model1.star.binary_separation)
# plt.plot(model2.age + grid.tpagb_age, model2.star.binary_separation)
# plt.plot(model3.age + grid.tpagb_age, model3.star.binary_separation)
# plt.plot(model4.age + grid.tpagb_age, model4.star.binary_separation)
# plt.plot(model5.age + grid.tpagb_age, model5.star.binary_separation)

for R in [300]:
    model = find_model(R)
    mass = model["M"]
    model_path = f"{grid_dir}/models/{model["name"]}"
    mod_data = mr.MesaData(model_path)
    model_age = _find_initial_age(mod_data) + tpagb_age

    for q in [0.5]:
        a_init = inv_roche_lobe(R, q)
        [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
            Star3,
            q,
            a_init,
            simple_only=True,
        )
        bin = Bins[0]
        index = np.argwhere(bin.age > model_age)[0][0] - 1
        q_evolve = bin.m1 / bin.m2
        m2 = bin.m2[index]
        rl1 = roche_lobe(q_evolve) * bin.a
        plt.plot(bin.age, rl1)

plt.plot(Star.age, Star.radius)
plt.show()
