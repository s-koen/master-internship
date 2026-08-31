import pandas as pd
import numpy as np
import glob, os
import re

# %%

filepath = "/home/koen/master-internship/data/"  # Filepath where all information is stored/saved
species_folder = "/home/koen/master-internship/data/rob_files/"
which_folders = (
    filepath + "rob_files/z0.007_models/",
    filepath + "rob_files/z0.0028_models/",
    filepath + "rob_files/z0.014_models/",
)  # Where to find Amanda's data files
table_location = (
    filepath + "nucsyn_intershell_abundances_karakas_LMC2016_tmp.h"
)  # Where to save binary_c table

# %%
Z = []
pmz = []
TPnum = []
M = []
newGroup = [0]
newline = []
M_first = 0
num_particles = []
isotope = []

print("Finding and uploading Amanda's data files! \n")

# Uploading information about species tracked by Amanda's models
with open(species_folder + "species.dat") as f:
    for line in f.readlines()[1:]:
        line = re.split("\s+", line)
        num_particles.append(float(line[1]))
        isotope.append(line[3])

print("Number of Isotopes: ", len(isotope))

for which_folder in which_folders:
    print(which_folder)
    for filename in sorted(glob.glob(which_folder + "/intershell_*")):
        print(filename)
        first = True
        with open(filename, "r") as file:  # Opening rob.dat file
            for line in file:  # Sorting through the data
                newPulse = False
                line = re.split("\s+", line)
                if line[0] == "":
                    line = line[1:-1]
                elif line[-1] == "":
                    line = line[:-1]
                else:
                    line = line[:-1]
                line = [float(i) for i in line]

                if line[0] >= 1.0:  # All mass fractions are less than 1
                    newPulse = True
                    TPnum.append(
                        int(line[0])
                    )  # The intershell file takes Z, pmz, M, and TPnum.
                    if first == True:
                        M_first = line[3]
                    M.append(float(M_first))

                    # automatically get pmz and Z from filename
                    match_number = re.compile(
                        "-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?"
                    )
                    string_num = re.findall(match_number, filename)
                    Z.append(float(string_num[-3]))
                    pmz.append(string_num[-2])

                else:
                    if len(line) == 11:
                        if first == False:
                            if len(newGroup) != 328:
                                print("Something is off!", len(newGroup))
                            newline.append(
                                np.concatenate(
                                    ([Z[-1]], [pmz[-1]], [M[-1]], [TPprev], newGroup),
                                    axis=0,
                                )
                            )
                        # else: newline.append(np.concatenate(([Z[-1]], [pmz[-1]], [M[-1]], [TPnum[-1]], newGroup), axis = 0))
                        first = False
                        newGroup = line
                        TPprev = TPnum[-1]
                    else:
                        newGroup = np.concatenate((newGroup, line), axis=0)

            if len(newGroup) != 328:
                print("Something is off!", len(newGroup))
            newline.append(
                np.concatenate(
                    ([Z[-1]], [pmz[-1]], [M[-1]], [TPprev], newGroup), axis=0
                )
            )

print("next step")
# Column names
columns = ["Z", "pmz", "M1tp", "ntp"]
elements = isotope

columns = np.concatenate((columns, elements), axis=0)

df = pd.DataFrame(columns=columns)

print(len(newline))
for i in range(len(newline)):
    if i % 100 == 0:
        print(i)
    df.loc[len(df)] = newline[i]
print(df)

print("Finished :)")
# %%

import pickle

with open(f"data/intershell_pd_df.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%

with open(f"data/intershell_pd_df.pkl", "rb") as f:
    df = pickle.load(f)


# %%
interesting = df[df["pmz"] == "2e-3"]
interesting = interesting[interesting["M1tp"] == "1.49998"]
# %%
for col in interesting.columns:
    print(col)
# %%

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
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star

plt.cplot = cplot


# %%
c = []
o = []
p = []
tp = []
c13 = []

for x in interesting["c12"]:
    c.append(np.float64(x))

for x in interesting["o16"]:
    o.append(np.float64(x))

for x in interesting["p"]:
    p.append(np.float64(x))

for x in interesting["c13"]:
    c13.append(np.float64(x))

for x in interesting["ntp"]:
    tp.append(np.float64(x))

plt.scatter(tp, np.array(c) / np.array(o), c=np.log10(p), cmap="viridis")
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

prev_len = 0

for i in range(1, 18):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    p = []
    tp = []

    for x in pulse["p"]:
        p.append(np.float64(x))

    for x in pulse["ntp"]:
        tp.append(np.float64(x))

    cbar = plt.cplot(
        list(range(prev_len, len(p) + prev_len)),
        p,
        i * np.ones(len(p)),
        cmap="viridis",
        vmin=1,
        vmax=17,
    )

    axs.text(np.argmax(p) + prev_len, 2 * np.max(p), f"{i}", ha="center", va="bottom")
    prev_len += len(p)

plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$Y(\textrm{p}^{+})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-first-tests-intershell.pgf", format="pgf")
plt.show()
plt.close()

# %%
pulse1 = interesting[interesting["ntp"] == "17"]
# %%
pulse1

# %%
tot = 0

for i, species in enumerate(pulse):
    if i > 6:

        match = re.search(r"(\d+)$", species)
        a = int(match.group(1))
        tot += a * np.float64(pulse[species])

    if i < 20:
        print(pulse[species])
print(tot)

# %%
print(np.mean(tot))
print(0.020001)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

prev_len = 0

for i in range(1, 18):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    sum = 0
    for j, species in enumerate(pulse):

        if j < 4:
            continue

        if j in [4, 5]:
            a = 1
        elif j == 6:
            a = 2
        else:
            match = re.search(r"(\d+)$", species)
            a = int(match.group(1))

        p = []
        tp = []
        for x in pulse[species]:
            p.append(np.float64(x))

        p = a * np.array(p)
        sum += np.array(p)
        for x in pulse["ntp"]:
            tp.append(np.float64(x))

        max = -1e99
        if np.max(p) > 1e-4:
            if np.max(p) > max:
                max = np.max(p)

            plt.plot(list(range(prev_len, len(p) + prev_len)), p, c=f"C{j%9}")
            if i == 17:
                if species in ["na23", "ne20", "mg24"]:
                    pass
                elif species == "fe56":
                    axs.text(
                        len(p) + prev_len + 2,
                        p[-1],
                        f"ne20,fe56,na23, mg24",
                        ha="left",
                        va="center",
                    )
                else:
                    axs.text(
                        len(p) + prev_len + 2,
                        p[-1],
                        f"{species}",
                        ha="left",
                        va="center",
                    )

    plt.plot(list(range(prev_len, len(p) + prev_len)), sum, c="k")

    axs.text(len(p) / 2 + prev_len, 1.2, f"{i}", ha="center", va="bottom")
    prev_len += len(p)

axs.text(
    prev_len + 2,
    1,
    f"sum",
    ha="left",
    va="center",
)
# plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")
plt.ylim(1e-9)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$X_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-mass-abundance.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

prev_len = 0

for i in range(1, 18):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    sum = 0
    for j, species in enumerate(pulse):

        if j < 4:
            continue

        p = []
        tp = []
        for x in pulse[species]:
            p.append(np.float64(x))
        sum += np.array(p)
        for x in pulse["ntp"]:
            tp.append(np.float64(x))

        max = -1e99
        if np.max(p) > 1e-5:
            if np.max(p) > max:
                max = np.max(p)

            plt.plot(list(range(prev_len, len(p) + prev_len)), p, c=f"C{j%9}")
            if i == 17:
                if species == "na23":
                    axs.text(
                        len(p) + prev_len + 2,
                        p[-1],
                        f"na23, mg24",
                        ha="left",
                        va="center",
                    )
                elif species == "mg24":
                    pass
                else:
                    axs.text(
                        len(p) + prev_len + 2,
                        p[-1],
                        f"{species}",
                        ha="left",
                        va="center",
                    )

    plt.plot(list(range(prev_len, len(p) + prev_len)), sum, c="k")

    axs.text(len(p) / 2 + prev_len, 1, f"{i}", ha="center", va="bottom")
    prev_len += len(p)

axs.text(
    prev_len + 2,
    0.3,
    f"sum",
    ha="left",
    va="center",
)
# plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")
plt.ylim(1e-9, 0)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$Y_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-number-abundance.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

prev_len = 0

for i in range(1, 18):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    sum = 0
    for j, species in enumerate(pulse):

        if species not in [
            "ba134",
            "ba135",
            "ba136",
            "ba137",
            "ba138",
        ]:
            continue

        if j in [4, 5]:
            a = 1
        elif j == 6:
            a = 2
        else:
            match = re.search(r"(\d+)$", species)
            a = int(match.group(1))

        p = []
        tp = []
        for x in pulse[species]:
            p.append(np.float64(x))

        p = a * np.array(p)
        sum += np.array(p)
        for x in pulse["ntp"]:
            tp.append(np.float64(x))

        max = -1e99
        if np.max(p) > max:
            max = np.max(p)

        plt.plot(list(range(prev_len, len(p) + prev_len)), p, c=f"C{j%9}")
        if i == 17:
            if species in ["na23", "ne20", "mg24"]:
                pass
            elif species == "fe56":
                axs.text(
                    len(p) + prev_len + 2,
                    p[-1],
                    f"ne20,fe56,na23, mg24",
                    ha="left",
                    va="center",
                )
            else:
                axs.text(
                    len(p) + prev_len + 2,
                    p[-1],
                    f"{species}",
                    ha="left",
                    va="center",
                )

    plt.plot(list(range(prev_len, len(p) + prev_len)), sum, c="k")

    axs.text(len(p) / 2 + prev_len, 1.5e-10, f"{i}", ha="center", va="bottom")
    prev_len += len(p)

axs.text(
    prev_len + 2,
    sum[-1] * 1.2,
    f"sum",
    ha="left",
    va="center",
)
# plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")
plt.ylim(1e-10)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$X_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-mass-abundance-ba.pgf", format="pgf")
plt.show()
plt.close()

# %%

interesting = df[df["pmz"] == "2e-3"]
masses = np.unique(interesting["M1tp"])
print(masses)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)


ls = []

for m, mass in enumerate(masses):

    prev_len = 0
    mass_abundance = interesting[interesting["M1tp"] == mass]

    tps = np.unique(interesting["ntp"])
    for i in range(1, len(tps) + 1):
        pulse = mass_abundance[mass_abundance["ntp"] == f"{i}"]
        sum = 0
        for j, species in enumerate(pulse):

            if species not in [
                "ba134",
                "ba135",
                "ba136",
                "ba137",
                "ba138",
            ]:
                continue

            if j in [4, 5]:
                a = 1
            elif j == 6:
                a = 2
            else:
                match = re.search(r"(\d+)$", species)
                a = int(match.group(1))

            p = []
            tp = []
            for x in pulse[species]:
                p.append(np.float64(x))

            p = a * np.array(p)
            sum += np.array(p)
            for x in pulse["ntp"]:
                tp.append(np.float64(x))

        (j,) = plt.plot(
            list(range(prev_len, len(p) + prev_len)),
            sum,
            c=f"C{m}",
            label=f"$M_\\textrm{{TPAGB}} = {mass}\\;M_\\odot$",
        )

        prev_len += len(p)
    ls.append(j)

# plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")
fig.legend(handles=ls, ncols=3, loc="outside upper center")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$X(\textrm{Ba}_{138})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w25-mass-abundance-ba-per-mass.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
interesting = df[df["pmz"] == "2e-3"]
interesting = interesting[interesting["M1tp"] == "2.09998"]

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

prev_len = 0

tps = np.unique(interesting["ntp"])
for i in range(1, len(tps) + 1):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    sum = 0
    for j, species in enumerate(pulse):

        if species not in ["c12", "c13", "c14"]:
            continue

        if j in [4, 5]:
            a = 1
        elif j == 6:
            a = 2
        else:
            match = re.search(r"(\d+)$", species)
            a = int(match.group(1))

        p = []
        tp = []
        for x in pulse[species]:
            p.append(np.float64(x))

        p = a * np.array(p)
        sum += np.array(p)
        for x in pulse["ntp"]:
            tp.append(np.float64(x))

        max = -1e99
        if np.max(p) > max:
            max = np.max(p)

        plt.plot(list(range(prev_len, len(p) + prev_len)), p, c=f"C{j%9}")
        if i == len(tps):
            if species in ["na23", "ne20", "mg24"]:
                pass
            elif species == "fe56":
                axs.text(
                    len(p) + prev_len + 2,
                    p[-1],
                    f"ne20,fe56,na23, mg24",
                    ha="left",
                    va="center",
                )
            else:
                axs.text(
                    len(p) + prev_len + 2,
                    p[-1],
                    f"{species}",
                    ha="left",
                    va="center",
                    color=f"C{j%9}",
                )

    # plt.plot(list(range(prev_len, len(p) + prev_len)), sum, c="k")

    axs.text(len(p) / 2 + prev_len, 0.4, f"{i}", ha="center", va="bottom")
    prev_len += len(p)

# axs.text(
#     prev_len + 2,
#     sum[-1] * 1.2,
#     f"sum",
#     ha="left",
#     va="center",
# )
# plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")
# plt.ylim(1e-)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$X_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-carbon.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

prev_len = 0

for i in range(1, 18):
    pulse = interesting[interesting["ntp"] == f"{i}"]
    p = []
    c13 = []
    tp = []

    for x in pulse["p"]:
        p.append(np.float64(x))

    for x in pulse["c13"]:
        c13.append(np.float64(x))

    for x in pulse["ntp"]:
        tp.append(np.float64(x))

    cbar = plt.cplot(
        list(range(prev_len, len(p) + prev_len)),
        np.array(p),
        i * np.ones(len(p)),
        cmap="viridis",
        vmin=1,
        vmax=17,
    )
    cbar = plt.cplot(
        list(range(prev_len, len(p) + prev_len)),
        np.array(c13),
        i * np.ones(len(p)),
        cmap="viridis",
        vmin=1,
        vmax=17,
    )

    axs.text(
        np.argmax(c13) + prev_len, 2 * np.max(c13), f"{i}", ha="center", va="bottom"
    )
    prev_len += len(p)

axs.text(prev_len + 2, c13[-1], r"C$_{13}$", ha="left", va="center")
axs.text(prev_len + 2, p[-1], r"p$^{+}$", ha="left", va="center")
plt.colorbar(cbar, label="Thermal pulse count")

plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Zone number")
plt.ylabel(r"$Y_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-first-tests-intershell.pgf", format="pgf")
plt.show()
plt.close()


# %%

interesting = df[df["pmz"] == "2e-3"]
masses = np.unique(interesting["M1tp"])
print(masses)

fig, axs = plt.subplots(
    len(masses) + 1,
    1,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
    sharex=True,
)


ls = []

for m, mass in enumerate(masses):

    prev_len = 0
    mass_abundance = interesting[interesting["M1tp"] == mass]

    tps = np.unique(interesting["ntp"])
    for i in range(1, len(tps) + 1):
        pulse = mass_abundance[mass_abundance["ntp"] == f"{i}"]
        sum = 0
        for j, species in enumerate(pulse):
            if j < 4:
                continue

            if species in [
                "p",
                "n",
                "d",
                "he3",
                "he4",
            ]:
                continue

            if j in [4, 5]:
                a = 1
            elif j == 6:
                a = 2
            else:
                match = re.search(r"(\d+)$", species)
                a = int(match.group(1))

            p = []
            tp = []
            for x in pulse[species]:
                p.append(np.float64(x))

            p = a * np.array(p)
            sum += np.array(p)
            for x in pulse["ntp"]:
                tp.append(np.float64(x))

        (j,) = axs[m].plot(
            list(range(prev_len, len(p) + prev_len)),
            sum,
            c=f"C{m}",
            label=f"$M_\\textrm{{TPAGB}} = {mass}\\;M_\\odot$",
        )

        prev_len += len(p)
    ls.append(j)

# plt.colorbar(cbar, label="Thermal pulse count")

z_intershell = []
for p in profiles:
    z = p.z_mass_fraction_metals
    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    if np.log10(p.power_he_burn) > 4:
        print(np.log10(p.power_h_burn))
        z_intershell.append(
            np.average(
                z[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 110], prepend=m[ind + 99]),
            )
        )

axs[-1].plot(np.linspace(0, 250, len(z_intershell)), z_intershell)
for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")
    if i == len(masses):
        ax.text(
            0.95,
            0.95,
            f"$M = 2.00\\;M_\\odot$\nMESA",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        continue
    ax.text(
        0.95,
        0.95,
        f"$M = {float(masses[i]):.2f}\\;M_\\odot$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
plt.xlabel("Zone number")
fig.supylabel(r"$Z$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w25-mass-abundance-Z.pgf", format="pgf")
plt.show()
plt.close()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import mesa_reader as mr
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

l = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00453/M2.0/LOGS/TPAGB")

# %%
for profile in l.profile_numbers:
    profile = l.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)


# %%

profiles = list(l.profile_dict.values())
profiles[0].header_names
for profile in profiles:
    print(np.log10(profile.power_he_burn))

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)
z_intershell = []
z_envelope = []
age = []
m_env = []
for p in profiles:
    z = p.z_mass_fraction_metals
    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    z_intershell.append(
        np.average(
            z[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 110], prepend=m[ind + 99]),
        )
    )
    m_env.append(m[0] - m[ind])
    z_envelope.append(
        np.average(z[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
    )
    age.append(p.star_age)

# plt.plot(age, m_env)
plt.plot(z_intershell)

plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$Z$ (metal mass fraction)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w25-z-inter+env.pgf", format="pgf")
plt.show()
plt.close()

# %%


import re
import pandas as pd


def parse_surf_file(paths):

    header_regex = re.compile(
        r"Initial mass =\s*([\d.]+).*M_mix =\s*([\d.E+-]+)(?:.*N_ov =\s*([\d.E+-]+))?"
    )

    tp_regex = re.compile(
        r"#\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)"
    )

    abundance_regex = re.compile(
        r"^\s*([a-z]{1,2})\s+\d+\s+[\d.E+-]+\s+[\d.E+-]+\s+([\d.E+-]+)\s+[\d.E+-]+\s+([\d.E+-]+)",
        re.IGNORECASE,
    )

    rows = []

    # state
    current_mass = None
    current_mmix = None
    current_tp_meta = None
    in_abundance_block = False
    last_tp = None
    final = False

    for path in paths:

        Z = float("0." + re.search(r"surf_z(\d+)\.dat$", path).group(1))
        with open(path) as f:
            for line in f:

                # -------------------------
                # NEW MODEL (hard reset)
                # -------------------------
                h = header_regex.search(line)
                if h:
                    current_mass = float(h.group(1))
                    current_mmix = float(h.group(2))
                    N_ov = float(h.group(3)) if h.group(3) is not None else 1.0

                    current_tp_meta = None
                    in_abundance_block = False

                    last_tp = None
                    continue

                # -------------------------
                # TP HEADER
                # -------------------------
                t = tp_regex.match(line)
                if t:
                    tp = int(t.group(1))

                    if tp == last_tp:
                        continue

                    last_tp = tp

                    current_tp_meta = {
                        "M_init": current_mass,
                        "pmz": current_mmix,
                        "N_ov": N_ov,
                        "ntp": tp,
                        "Z": Z,
                        "Mass": float(t.group(2)),
                        "Mcore": float(t.group(3)),
                        "Menv": float(t.group(4)),
                        "logL": float(t.group(5)),
                    }

                    in_abundance_block = False
                    continue

                # -------------------------
                # ABUNDANCE BLOCK START
                # -------------------------
                if line.strip().startswith("# El") and "X(i)" in line:
                    in_abundance_block = True
                    continue

                # -------------------------
                # ABUNDANCE BLOCK END MARKERS
                # -------------------------
                if line.startswith("# Elemental abundance ratios"):
                    in_abundance_block = False
                    continue

                if line.startswith("# Initial abundances"):
                    in_abundance_block = False
                    final = False
                    continue

                if line.startswith("# Final abundances"):
                    in_abundance_block = False
                    final = True
                    continue

                # -------------------------
                # ABUNDANCES
                # -------------------------

                if final and current_tp_meta:
                    current_tp_meta["ntp"] += 1
                    final = False

                if in_abundance_block and current_tp_meta is not None:
                    a = abundance_regex.match(line)
                    if a:
                        rows.append(
                            {
                                **current_tp_meta,
                                "element": a.group(1),
                                "XFe": float(a.group(2)),
                                "massfrac": float(a.group(3)),
                            }
                        )

    return pd.DataFrame(rows)


# %%
paths = [
    "scripts/w21/surf_z007.dat",
    "scripts/w21/surf_z014.dat",
    "scripts/w21/surf_z03.dat",
]
df = parse_surf_file(paths)

with open(f"data/env_pd_df.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
