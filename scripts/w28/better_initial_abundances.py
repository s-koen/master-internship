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
from scripts.general_utils.m_dup import compute_m_DUP, AbundanceTables, Abundances

plt.cplot = cplot
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid3 = MesaGrid(f"{MASTER}/grid-masses-3-2026-08-24")
grid4 = MesaGrid(f"{MASTER}/grid-masses-4-2026-08-25")
grid.merge(grid2)
grid.merge(grid3, overwrite=True)
grid.merge(grid4, overwrite=True)

# %%

df = AbundanceTables()

model = grid.models[50]

ab = Abundances(model=model, df=df)

ab.o.envelope
# %%
for model in grid.filter(m=3, R=1000, q=0.6):
    print(model)
    star = get_star(m=3)

ab = Abundances(model, df, method="tp")

element_names = ["he", "c", "n", "o", "ne", "mg"]

init_old = []
for element in element_names:
    init_old.append(ab.__getattr__(element).envelope[star.ntpagb])

# %%
ab = Abundances(model=model, df=df)

init_new = []
for element in element_names:
    init_new.append(ab.__getattr__(element).envelope[star.ntpagb])

# %%


fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

isos = [
    "envelope_he3",
    "envelope_he4",
    "envelope_c12",
    "envelope_c13",
    "envelope_n14",
    "envelope_o16",
    "envelope_ne20",
    "envelope_mg24",
]

init_MESA = []
for element in isos:
    if element in ["envelope_he4", "envelope_c13"]:
        init_MESA[-1] += star.__getattribute__(element)[star.ntpagb]
    else:
        init_MESA.append(star.__getattribute__(element)[star.ntpagb])


elements = ["He", "C", "N", "O", "Ne", "Mg"]
axs[0].plot(elements, init_MESA, label="MESA", linewidth=3, alpha=0.5)
axs[0].plot(elements, init_old, label="Monash post-processing old", linewidth=1)
axs[0].plot(elements, init_new, label="Monash post-processing new", linewidth=1)
axs[0].set_yscale("log")

axs[1].plot(
    elements,
    np.abs(np.array(init_MESA) - np.array(init_MESA)) / np.array(init_MESA),
    linewidth=1,
)
axs[1].plot(
    elements,
    np.abs(np.array(init_old) - np.array(init_MESA)) / np.array(init_MESA),
    linewidth=1,
)
axs[1].plot(
    elements,
    np.abs(np.array(init_new) - np.array(init_MESA)) / np.array(init_MESA),
    linewidth=1,
)

fig.legend(loc="outside upper center", ncols=3)
plt.yscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Element")
axs[0].set_ylabel("Mass fraction $X$")
axs[1].set_ylabel(" $|X_\\textrm{monash} - X_\\textrm{MESA}| / X_\\textrm{MESA}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-initial-envelope-abundance.pgf", format="pgf"
)
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.he.envelope[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j])
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_he4,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_he4[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )
    ax.set_title(f"$M={m}\\;M_\\odot,\\;q=0.6,\\;R=1000\\;R_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("C/O (number ratio)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-o.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(column, height=1.25), constrained_layout=True
)

axs = axs.flatten()
ms = [1.8, 2.2, 2.6, 3.0]


df = AbundanceTables()

for i, (m, ax) in enumerate(zip(ms, axs)):

    for model in grid.filter(m=m, R=1000, q=0.6):
        print(model)
        star = get_star(m=m)

    methods = ["TP count", "$M_\\textrm{DUP}$", "TP + offset"]

    ab1 = Abundances(model, df, method="tp")
    ab2 = Abundances(model, df, method="m_dup")
    ab3 = Abundances(model, df, method="tp offset")

    ls = []
    for j, ab in enumerate([ab1, ab2, ab3]):
        c = ab.he.envelope[star.ntpagb :]
        c2 = ab.he.intershell[star.ntpagb :]

        (l1,) = ax.plot(ab.time[star.ntpagb :], c, label=methods[j], c=f"C{j}")
        ax.plot(ab.time[star.ntpagb :], c2, c=f"C{j}", alpha=0.5, linewidth=2)
        ls.append(l1)

    (l1,) = ax.plot(
        model.age,
        model.envelope_he3 + model.envelope_he4,
        c="C9",
        label="MESA",
    )
    ls.append(l1)
    ax.plot(
        star.age[star.ntpagb : ab.simple_end_idx],
        star.envelope_he3[star.ntpagb : ab.simple_end_idx]
        + star.envelope_he4[star.ntpagb : ab.simple_end_idx],
        c="C9",
    )

    ax.set_title(f"$M={m}\\;M_\\odot$")
    if ax.get_ylim()[-1] > 7.5:
        ax.set_ylim(ax.get_ylim()[0], 7.5)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=ls)

fig.supxlabel("Star age (yr)", fontsize=10)
fig.supylabel("$X(\\textrm{He})$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-he.pgf", format="pgf")
plt.show()
plt.close()


# %%

profile = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M1.5/LOGS/TPAGB/profile10.data"
)
# %%
profile.header_names

# %%
plt.plot(profile.mass, profile.o16)
plt.plot(profile.mass, profile.c12)
plt.plot(profile.mass, profile.z_mass_fraction_metals)
plt.axvline(profile.he_core_mass)
plt.axvline(profile.co_core_mass)
plt.axvline(profile.fe_core_mass)
plt.show()
# %%

l = mr.MesaLogDir(f"{MASTER}/single-stars/z0.00557/completed/M1.5/LOGS/TPAGB")

# %%
for profile in l.profile_numbers:
    profile = l.profile_data(profile_number=profile)
    print(profile.model_number)

print(profile.bulk_names)

# %%

profiles = list(l.profile_dict.values())
profiles[0].header_names
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
plt.plot(age, z_intershell, "o-")
plt.plot(age, z_envelope, "o-")

plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$Z$ (metal mass fraction)")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-z-inter+env.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

core_masses = []
lums = []
ages = []

for p in profiles:
    z = p.he_core_mass
    age = p.star_age
    core_masses.append(z)
    ages.append(age)
    lums.append(p.power_he_burn)

star = get_star(m=1.5)
axs[0].plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    star.m_core[star.ntpagb :],
    c="k",
    linewidth=1,
)
axs[0].scatter(ages, core_masses, marker=".", s=75, zorder=100, color="k")
axs[0].scatter(ages, core_masses, marker=".", s=250, zorder=10, color="white")

axs[0].annotate(
    "profile data",
    xy=(ages[3], core_masses[3]),
    xycoords="data",
    xytext=(-80, 15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-.2", color="C9", linewidth=0.75
    ),
    zorder=1000,
)

axs[0].annotate(
    "history data",
    xy=(
        star.age[star.ntpagb + 10500] - star.age[star.ntpagb],
        star.m_core[star.ntpagb + 10500],
    ),
    xycoords="data",
    xytext=(25, -20),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-.2", color="C9", linewidth=0.75
    ),
    zorder=1000,
)


axs[1].plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    star.log_LHe[star.ntpagb :],
    c="k",
    linewidth=1,
)
axs[1].scatter(ages, np.log10(lums), marker=".", s=75, zorder=100, color="k")
axs[1].scatter(ages, np.log10(lums), marker=".", s=250, zorder=10, color="white")


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("TPAGB age (yr)")
axs[1].set_ylabel("$\log(L_\\textrm{He} / L_\\odot)$")
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-profiles-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
z_intershell = []
z_envelope = []
age = []
m_env = []
for p in profiles:
    z = p.c12
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


plt.scatter(ages, z_envelope, marker=".", s=75, zorder=100, color="k")
plt.scatter(ages, z_envelope, marker=".", s=250, zorder=10, color="white")

star = get_star(m=1.5)
plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    star.envelope_c12[star.ntpagb :],
    c="k",
    linewidth=1,
)

plt.annotate(
    "profile data",
    xy=(ages[6], z_envelope[6]),
    xycoords="data",
    xytext=(30, -15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-.2", color="C9", linewidth=0.75
    ),
    zorder=1000,
)

plt.annotate(
    "history data",
    xy=(
        star.age[star.ntpagb + 15500] - star.age[star.ntpagb],
        star.envelope_c12[star.ntpagb + 15500],
    ),
    xycoords="data",
    xytext=(-80, 15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-.2", color="C9", linewidth=0.75
    ),
    zorder=1000,
)


# plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$X(\\textrm{C}_{12})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-profiles-envelope-check.pgf", format="pgf"
)
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)
intershell_he = []
intershell_c = []
intershell_n = []
intershell_o = []
intershell_ne = []
intershell_mg = []

age = []
m_env = []
for p in profiles:
    he = p.he3 + p.he4
    c = p.c12 + p.c13
    n = p.n14
    o = p.o16
    ne = p.ne20
    mg = p.mg24

    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    intershell_he.append(
        np.average(
            he[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_c.append(
        np.average(
            c[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_n.append(
        np.average(
            n[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_o.append(
        np.average(
            o[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_ne.append(
        np.average(
            ne[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_mg.append(
        np.average(
            mg[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    m_env.append(m[0] - m[ind])
    age.append(p.star_age)


plt.step(ages, intershell_he, zorder=100, label="He")
plt.step(ages, intershell_c, zorder=100, label="C")
plt.step(ages, intershell_n, zorder=100, label="N")
plt.step(ages, intershell_o, zorder=100, label="O")
plt.step(ages, intershell_ne, zorder=100, label="Ne")
plt.step(ages, intershell_mg, zorder=100, label="Mg")

star = get_star(m=1.5)

v_offset = [5, -5, 1000, 0, 0, 0]
for i, line in enumerate(axs.lines):
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-1],
        xytext=(5, v_offset[i] - 3),
        textcoords="offset points",
        color=line.get_color(),
    )

for i, line in enumerate(axs.lines):
    if i != 2:
        continue
    print(line.get_label())
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-2],
        xytext=(18, -3),
        textcoords="offset points",
        color=line.get_color(),
    )


plt.ylim(1e-5, 1)
plt.yscale("log")

# plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$X$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-profiles-intershell.pgf", format="pgf")

plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


norm = plt.Normalize(0, len(profiles) + 1)
cmap = plt.cm.inferno
# color = cmap(norm(x))


for p_i, p in enumerate(profiles):
    plt.plot(p.mass, p.n14, color=cmap(norm(p_i)), zorder=-1)
    # plt.axvline(p.he_core_mass)

    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    val = np.average(
        p.n14[ind + 100 : ind + 110],
        weights=np.diff(m[ind + 100 : ind + 111]),
    )
    plt.scatter(m[ind + 105], val, color=cmap(norm(p_i)), zorder=3, s=75, marker=".")
    plt.scatter(m[ind + 105], val, color="w", zorder=2, s=250, marker=".")

plt.yscale("log")
plt.ylim(1e-25, 1e1)
plt.xlim(0.51, 0.61)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$X(\\textrm{N})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-profiles-nitrogen.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m in np.arange(1, 3.05, 0.1):

    c = cmap(norm(m))
    profiles = profiles_dict[m]

    intershell_n = []

    age = []
    m_env = []
    for p in profiles:
        n = p.n14

        m = p.mass
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5

        intershell_n.append(
            np.average(
                n[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        m_env.append(m[0] - m[ind])
        age.append(p.star_age)

    axs[0].plot(
        np.array(range(len(intershell_n))) / np.max(np.array(range(len(intershell_n)))),
        intershell_n,
        zorder=100,
        label="N",
        color=c,
    )
    axs[1].plot(
        np.array(range(len(intershell_n))) / np.max(np.array(range(len(intershell_n)))),
        intershell_n,
        zorder=100,
        label="N",
        color=c,
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs)
cbar.set_label(r"$M_\textrm{TPAGB}$ ($M_\odot$)")


axs[1].set_yscale("log")

# plt.yscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Normalized thermal pulse count")
fig.supylabel("$X(\\textrm{N})$", fontsize=10)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-profiles-intershell-nitrogen.pgf", format="pgf"
)

plt.show()
plt.close()


# %%

with open("data/single-star-profiles.pkl", "rb") as f:
    profiles_dict = pickle.load(f)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)
intershell_he = []
intershell_c = []
intershell_n = []
intershell_o = []
intershell_ne = []
intershell_mg = []

age = []
m_env = []
for m in np.arange(1, 3.05, 0.1):
    ms = m
for p in profiles_dict[ms]:
    he = p.he3 + p.he4
    c = p.c12 + p.c13
    n = p.n14
    o = p.o16
    ne = p.ne20
    mg = p.mg24

    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    intershell_he.append(
        np.average(
            he[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_c.append(
        np.average(
            c[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_n.append(
        np.average(
            n[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_o.append(
        np.average(
            o[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_ne.append(
        np.average(
            ne[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_mg.append(
        np.average(
            mg[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    m_env.append(m[0] - m[ind])
    age.append(p.star_age)


plt.step(age, intershell_he, zorder=100, label="He")
plt.step(age, intershell_c, zorder=100, label="C")
plt.step(age, intershell_n, zorder=100, label="N")
plt.step(age, intershell_o, zorder=100, label="O")
plt.step(age, intershell_ne, zorder=100, label="Ne")
plt.step(age, intershell_mg, zorder=100, label="Mg")

star = get_star(m=1.5)

v_offset = [0, 0, 0, 0, 0, 0]
for i, line in enumerate(axs.lines):
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-1],
        xytext=(5, v_offset[i] - 3),
        textcoords="offset points",
        color=line.get_color(),
    )

for i, line in enumerate(axs.lines):
    if i != 2:
        continue
    print(line.get_label())
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-2],
        xytext=(8, -3),
        textcoords="offset points",
        color=line.get_color(),
    )


plt.ylim(1e-4, 1)
plt.yscale("log")

# plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$X$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-profiles-intershell-m3.pgf", format="pgf")

plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

wanted_mass = 2.8

intershell_he = []
intershell_c = []
intershell_n = []
intershell_o = []
intershell_ne = []
intershell_mg = []

age = []
m_env = []
for ms in np.arange(1, 3.05, 0.1):
    if np.round(ms, 2) == wanted_mass:
        break


for p in profiles_dict[ms]:
    he = p.he3 + p.he4
    c = p.c12 + p.c13
    n = p.n14
    o = p.o16
    ne = p.ne20
    mg = p.mg24

    m = p.mass
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    intershell_he.append(
        np.average(
            he[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_c.append(
        np.average(
            c[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_n.append(
        np.average(
            n[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_o.append(
        np.average(
            o[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_ne.append(
        np.average(
            ne[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    intershell_mg.append(
        np.average(
            mg[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    m_env.append(m[0] - m[ind])
    age.append(p.star_age)


plt.step(age, intershell_he, zorder=100, label="He")
plt.step(age, intershell_c, zorder=100, label="C")
plt.step(age, intershell_n, zorder=100, label="N")
plt.step(age, intershell_o, zorder=100, label="O")
plt.step(age, intershell_ne, zorder=100, label="Ne")
plt.step(age, intershell_mg, zorder=100, label="Mg")

v_offset = [0, 0, 0, 0, 3, 0]
for i, line in enumerate(axs.lines):
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-1],
        xytext=(5, v_offset[i] - 3),
        textcoords="offset points",
        color=line.get_color(),
    )

for i, line in enumerate(axs.lines):
    if i != 2:
        continue
    axs.annotate(
        line.get_label(),
        xy=line.get_xydata()[-2],
        xytext=(8, -3),
        textcoords="offset points",
        color=line.get_color(),
    )


ab = Abundances(model=None, df=df, mass=wanted_mass)
star = get_star(m=wanted_mass)

for i, el in enumerate(["he", "c", "n", "o", "ne", "mg"]):
    plt.plot(
        ab.time[star.ntpagb :] - ab.time[star.ntpagb],
        ab.__getattr__(el).intershell[star.ntpagb :],
        c=f"C{i}",
        linewidth=3,
        alpha=0.5,
    )


plt.ylim(2e-8, 1)
plt.yscale("log")

# plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$X$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-profiles-intershell-m2.5-monash.pgf", format="pgf"
)

plt.show()
plt.close()


# %%

df = AbundanceTables()
# %%
