import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import compute_m_DUP

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

with open(f"data/intershell_pd_df.pkl", "rb") as f:
    inter = pickle.load(f)

with open(f"data/env_pd_df.pkl", "rb") as f:
    env = pickle.load(f)


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis

element_init = []
for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        ba138 = np.log10(138 * np.array(pulse["ba138"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), ba138, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                ba138[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )
            element_init.append(ba138[0])

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Ba}_{138}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-ba138-metal.pgf", format="pgf")
plt.show()
plt.close()

# %%

print(np.array(zs.astype(np.float64)) / np.float64(zs[0]))
print(10 ** np.array(element_init) / 10 ** element_init[0])


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        pb208 = np.log10(208 * np.array(pulse["pb208"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), pb208, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                pb208[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Pb}_{208}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-pb208-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        fe60 = np.log10(60 * np.array(pulse["fe60"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), fe60, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                fe60[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(
    0.25,
    axs.get_ylim()[-1] * 1.05,
    "$M=2.5\\;M_\\odot$",
    size=8,
    ha="right",
    va="center",
)

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Fe}_{60}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-fe60-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        ca48 = np.log10(48 * np.array(pulse["ca48"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), ca48, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                ca48[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Pb}_{208}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-ca48-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        ca46 = np.log10(46 * np.array(pulse["ca46"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), ca46, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                ca46[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Pb}_{208}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-ca46-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        bi209 = np.log10(209 * np.array(pulse["bi209"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), bi209, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                bi209[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Pb}_{208}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-bi209-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        co59 = np.log10(59 * np.array(pulse["co59"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), co59, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                co59[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(
    0.25,
    axs.get_ylim()[-1] * 1.05,
    "$M=2.5\\;M_\\odot$",
    size=8,
    ha="right",
    va="center",
)

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Co}_{59}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-co59-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        c13 = np.log10(13 * np.array(pulse["c13"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), c13, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                c13[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Ba}_{138}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-c13-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

zs = np.unique(interesting["Z"])
for z_i, z in enumerate(zs[:-1]):
    dat = interesting[interesting["Z"] == z]
    print(dat)
    mass = np.unique(dat["M_init"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    for element in np.unique(data["element"]):
        if element not in ["tc"]:
            continue
        el = data[data["element"] == element]
        plt.step(
            el["ntp"].astype(int),
            el["massfrac"].astype(np.float64),
            c=f"C{z_i+1}",
            where="mid",
        )


for p in range(0, 20):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
# plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] == 1]

zs = np.unique(interesting["Z"])

# collect the data for all elements
element_data = {}

for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M_init"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    for element in np.unique(data["element"]):

        el = data[data["element"] == element]

        element_data[element] = (
            el["ntp"].astype(int).to_numpy(),
            el["massfrac"].astype(np.float64).to_numpy(),
        )

elements = list(element_data)
i = 0


def plot_element():
    ax.clear()

    element = elements[i]
    ntp, massfrac = element_data[element]

    ax.step(ntp, massfrac, where="mid")
    ax.set_title(element)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yscale("log")

    fig.canvas.draw_idle()


def on_key(event):
    global i

    if event.key in ["right", "down", "n"]:
        i = (i + 1) % len(elements)
        plot_element()

    elif event.key in ["left", "up", "p"]:
        i = (i - 1) % len(elements)
        plot_element()


fig.canvas.mpl_connect("key_press_event", on_key)

plot_element()

plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] == 1]

zs = np.unique(interesting["Z"])
for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M_init"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    for element in np.unique(data["element"]):
        if element not in ["tc"]:
            continue
        print(element)
        el = data[data["element"] == element]
        plt.step(el["ntp"].astype(int), el["massfrac"].astype(np.float64), c=f"C{z_i}")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
# plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

zs = np.unique(interesting["Z"])

table = []
for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    print(dat)
    mass = np.unique(dat["M_init"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    el_init = []
    elements = np.unique(data["element"])
    for element in elements:
        if element in ["p", "pm"]:
            continue
        el = data[data["element"] == element]
        el_init.append(list(el["massfrac"].astype(np.float64))[0])
        print(list(el["massfrac"].astype(np.float64))[0], element)

    table.append(el_init)

table = np.array(table)

all_elements = []
for element in elements:
    if element in ["p", "pm"]:
        continue
    all_elements.append(element)

for el, name in zip(table, elements):
    plt.plot(all_elements, el / table[0])

ax_t = axs.secondary_xaxis("top")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

zs = np.unique(interesting["Z"])

table = []
elements = None

for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]

    mass = np.unique(dat["M_init"].astype(np.float64))
    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    # preserve the order from the original file
    current_elements = [
        el for el in data["element"].drop_duplicates() if el not in ["p", "pm"]
    ]

    # use the ordering from the first metallicity
    if elements is None:
        elements = current_elements

    el_init = []
    for element in elements:
        el = data[data["element"] == element]
        el_init.append(float(el["massfrac"].iloc[0]))

    table.append(el_init)

table = np.array(table)

# plot each metallicity
for el, z in zip(table, zs):
    plt.plot(elements, el / table[1], label=f"$z={z}$")


# alternate element names between bottom and top axes
x = np.arange(len(elements))

bottom_idx = x[::2]
top_idx = x[1::2]

axs.set_xticks(bottom_idx)
axs.set_xticklabels([elements[i] for i in bottom_idx])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(top_idx)
ax_t.set_xticklabels([elements[i] for i in top_idx])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in x[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

plt.xlim(x[0] - 2, x[-1] + 1)


axs.axhline(zs[0] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)
axs.axhline(zs[1] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)
axs.axhline(zs[2] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)

plt.yscale("log")

fig.legend(loc="outside upper center", ncols=3)


axs.set_xlabel("")
axs.set_ylabel(r"$X(Z) / X(Z_0)$")
axs.set_xlabel("Elements")

plt.savefig("/home/koen/LaTeX-setup/plots/w26-initial-envelope.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

zs = np.unique(interesting["Z"])

table = []
elements = None

for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]

    mass = np.unique(dat["M_init"].astype(np.float64))
    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

    # preserve the order from the original file
    current_elements = [
        el for el in data["element"].drop_duplicates() if el not in ["p", "pm"]
    ]

    # use the ordering from the first metallicity
    if elements is None:
        elements = current_elements

    el_init = []
    for element in elements:
        el = data[data["element"] == element]
        el_init.append(float(el["massfrac"].iloc[0]))

    table.append(el_init)

table = np.array(table)

# plot each metallicity
for el, z in zip(table, zs):
    plt.plot(elements, el / table[1], label=f"$z={z}$")


# alternate element names between bottom and top axes
x = np.arange(len(elements))

bottom_idx = x[::2]
top_idx = x[1::2]

axs.set_xticks(bottom_idx)
axs.set_xticklabels([elements[i] for i in bottom_idx])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(top_idx)
ax_t.set_xticklabels([elements[i] for i in top_idx])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in x[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

plt.xlim(x[0] - 2, x[-1] + 1)


axs.axhline(zs[0] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)
axs.axhline(zs[1] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)
axs.axhline(zs[2] / zs[1], c="C9", linewidth=0.75 / 2, zorder=-1)

plt.yscale("log")

fig.legend(loc="outside upper center", ncols=3)

plt.ylim(0.4, 3)


axs.set_ylabel(r"$X(Z) / X(Z_0)$")
axs.set_xlabel("Elements")

plt.savefig("/home/koen/LaTeX-setup/plots/w26-initial-envelope-zoom.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

interesting = interesting[interesting["Z"].astype(np.float64) == 0.014]

ms = np.unique(interesting["M_init"])

table = []
elements = None

for m_i, m in enumerate(ms):
    dat = interesting[interesting["M_init"] == m]

    # preserve the order from the original file
    current_elements = [
        el for el in dat["element"].drop_duplicates() if el not in ["p", "pm"]
    ]

    # use the ordering from the first metallicity
    if elements is None:
        elements = current_elements

    el_init = []
    for element in elements:
        el = dat[dat["element"] == element]
        el_init.append(float(el["massfrac"].iloc[0]))

    table.append(el_init)

table = np.array(table)

# plot each metallicity
for el, m in zip(table, ms):
    plt.plot(elements, el / table[4], label=f"$M={m}$")


# alternate element names between bottom and top axes
x = np.arange(len(elements))

bottom_idx = x[::2]
top_idx = x[1::2]

axs.set_xticks(bottom_idx)
axs.set_xticklabels([elements[i] for i in bottom_idx])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(top_idx)
ax_t.set_xticklabels([elements[i] for i in top_idx])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in x[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

plt.xlim(x[0] - 2, x[-1] + 1)


axs.axhline(ms[0] / ms[4], c="C0", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[1] / ms[4], c="C1", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[2] / ms[4], c="C2", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[3] / ms[4], c="C3", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[4] / ms[4], c="C4", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[5] / ms[4], c="C5", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[6] / ms[4], c="C6", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[7] / ms[4], c="C7", linewidth=0.75 / 2, zorder=-1)

plt.yscale("log")

fig.legend(loc="outside upper center", ncols=4)

plt.ylim(0.5, 1.5)


axs.set_ylabel(r"$X(Z) / X(Z_0)$")
axs.set_xlabel("Elements")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w26-initial-envelope-zoom-mass.pgf", format="pgf"
)
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

interesting = interesting[interesting["Z"].astype(np.float64) == 0.014]

ms = np.unique(interesting["M_init"])

table = []
elements = None

for m_i, m in enumerate(ms):
    dat = interesting[interesting["M_init"] == m]

    # preserve the order from the original file
    current_elements = [
        el for el in dat["element"].drop_duplicates() if el not in ["p", "pm"]
    ]

    # use the ordering from the first metallicity
    if elements is None:
        elements = current_elements

    el_init = []
    for element in elements:
        el = dat[dat["element"] == element]
        el_init.append(float(el["massfrac"].iloc[-1]))

    table.append(el_init)

table = np.array(table)

# plot each metallicity
for el, m in zip(table, ms):
    plt.plot(elements, el / table[4], label=f"$M={m}$")


# alternate element names between bottom and top axes
x = np.arange(len(elements))

bottom_idx = x[::2]
top_idx = x[1::2]

axs.set_xticks(bottom_idx)
axs.set_xticklabels([elements[i] for i in bottom_idx])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(top_idx)
ax_t.set_xticklabels([elements[i] for i in top_idx])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in x[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

plt.xlim(x[0] - 2, x[-1] + 1)


axs.axhline(ms[0] / ms[4], c="C0", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[1] / ms[4], c="C1", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[2] / ms[4], c="C2", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[3] / ms[4], c="C3", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[4] / ms[4], c="C4", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[5] / ms[4], c="C5", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[6] / ms[4], c="C6", linewidth=0.75 / 2, zorder=-1)
axs.axhline(ms[7] / ms[4], c="C7", linewidth=0.75 / 2, zorder=-1)

plt.yscale("log")

fig.legend(loc="outside upper center", ncols=4)

plt.ylim(0.1, 3)


axs.set_ylabel(r"$X(Z) / X(Z_0)$")
axs.set_xlabel("Elements")

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w26-final-envelope-zoom-mass.pgf", format="pgf"
)
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


interesting = inter[inter["pmz"] == "2e-3"]

zs = np.unique(interesting["Z"])
ntpsmax = 0
for z in zs:
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntpsmax = np.max([np.max(data["ntp"].astype(int)), ntpsmax])
norm = plt.Normalize(1, ntpsmax)
cmap = plt.cm.viridis


for z_i, z in enumerate(zs):
    dat = interesting[interesting["Z"] == z]
    mass = np.unique(dat["M1tp"].astype(np.float64))

    ind = np.argmin(np.abs(mass - 2.5))
    data = dat[dat["M1tp"].astype(np.float64) == mass[ind]]

    ntps = np.max(data["ntp"].astype(int))
    for p in range(1, ntps + 1):
        pulse = data[data["ntp"] == f"{p}"]
        tl203 = np.log10(203 * np.array(pulse["tl203"].astype(np.float64)))

        plt.plot(p + np.linspace(-0.5, 0.5, len(pulse)), tl203, c=f"C{z_i}")
        if p == 1:
            plt.text(
                p - 0.75,
                tl203[0],
                f"$z={z}\\;\\;$",
                va="center",
                ha="right",
                fontsize=8,
            )

for p in range(1, ntpsmax + 1):
    plt.axvline(p - 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.axvline(p + 0.5, color="C9", linewidth=0.75, zorder=-1)
plt.xticks(np.arange(1, ntpsmax + 1, 2))
ax_t = axs.secondary_xaxis("top")

ax_t.set_xticks(np.arange(2, ntpsmax + 1, 2))

plt.xlim(-4, ntpsmax + 0.5)
plt.text(0.25, -4.5, "$M=2.5\\;M_\\odot$", size=8, ha="right")

# axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel(r"$\log(X(\textrm{Tl}_{206}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-tl203-metal.pgf", format="pgf")
plt.show()
plt.close()


# %%
