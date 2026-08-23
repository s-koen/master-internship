import pandas as pd
import numpy as np
import glob, os
import re

# %%

filepath = "/home/koen/master-internship/data/"  # Filepath where all information is stored/saved
species_folder = "/home/koen/master-internship/data/rob_files/"
which_folder = (
    filepath + "rob_files/z0.007_models/"
)  # Where to find Amanda's data files
table_location = (
    filepath + "nucsyn_intershell_abundances_karakas_LMC2016_tmp.h"
)  # Where to save binary_c table

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
                match_number = re.compile("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?")
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
            np.concatenate(([Z[-1]], [pmz[-1]], [M[-1]], [TPprev], newGroup), axis=0)
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

# with open(f"data/intershell_pd_df.pkl", "wb") as f:
#     pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
#
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
# %%
masses = np.unique(interesting["M1tp"])

# %%
