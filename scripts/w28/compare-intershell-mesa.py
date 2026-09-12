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
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)

plt.cplot = cplot
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

df = AbundanceTables()

with open("data/single-star-profiles.pkl", "rb") as f:
    profiles_dict = pickle.load(f)

# %%


fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.he.intershell[star.ntpagb :],
        c="k",
        linewidth=1,
        label="Monash abundances",
    )
    ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.he.envelope[star.ntpagb :],
        c="k",
        linewidth=1,
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.he3 + p.he4
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l1,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        c="C9",
        linewidth=1,
        label="MESA abundances",
    )
    ax.step(np.array(age) / age[-1], envelope, c="C9", linewidth=1)
    ax.set_yscale("log")


axs[8].annotate(
    "intershell",
    xy=(0.75, 7e-1),
    xycoords="data",
    xytext=(-17, -22.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

axs[8].annotate(
    "envelope",
    xy=(0.75, 3e-1),
    xycoords="data",
    xytext=(-16, 17.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2])

plt.yscale("log")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{He})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-he-all.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.intershell[star.ntpagb :],
        c="k",
        linewidth=1,
        label="Monash abundances",
    )
    ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.envelope[star.ntpagb :],
        c="k",
        linewidth=1,
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.c12 + p.c13
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l1,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        c="C9",
        linewidth=1,
        label="MESA abundances",
    )
    ax.step(np.array(age) / age[-1], envelope, c="C9", linewidth=1)
    ax.set_yscale("log")


axs[4].annotate(
    "intershell",
    xy=(0.25, 2e-1),
    xycoords="data",
    xytext=(-17, -22.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

axs[4].annotate(
    "envelope",
    xy=(0.25, 8e-4),
    xycoords="data",
    xytext=(-16, 17.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2])

plt.yscale("log")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{C})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-c-all.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l1,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.n.intershell[star.ntpagb :],
        linewidth=1,
        label="Monash abundances intershell",
    )
    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.n.envelope[star.ntpagb :],
        linewidth=1,
        label="Monash abundances envelope",
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.n14
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l3,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        linewidth=1,
        label="MESA abundances intershell",
    )
    (l4,) = ax.step(
        np.array(age) / age[-1], envelope, linewidth=1, label="MESA abundances envelope"
    )
    ax.set_yscale("log")


# axs[4].annotate(
#     "intershell",
#     xy=(0.25, 2e-1),
#     xycoords="data",
#     xytext=(-17, -22.5),
#     textcoords="offset points",
#     arrowprops=dict(
#         arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
#     ),
#     zorder=1000,
#     fontsize=8,
# )
#
# axs[4].annotate(
#     "envelope",
#     xy=(0.25, 8e-4),
#     xycoords="data",
#     xytext=(-16, 17.5),
#     textcoords="offset points",
#     arrowprops=dict(
#         arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
#     ),
#     zorder=1000,
#     fontsize=8,
# )

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2, l3, l4])

axs[8].set_ylim(1e-22, 1e-2)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{N})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-n-all.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.o.intershell[star.ntpagb :],
        c="k",
        linewidth=1,
        label="Monash abundances",
    )
    ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.o.envelope[star.ntpagb :],
        c="k",
        linewidth=1,
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.o16
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l1,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        c="C9",
        linewidth=1,
        label="MESA abundances",
    )
    ax.step(np.array(age) / age[-1], envelope, c="C9", linewidth=1)
    ax.set_yscale("log")


axs[4].annotate(
    "intershell",
    xy=(0.47, 2e-2),
    xycoords="data",
    xytext=(-17, -15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

axs[4].annotate(
    "envelope",
    xy=(0.3, 2.5e-3),
    xycoords="data",
    xytext=(-16, 10),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2])

plt.yscale("log")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{O})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-o-all.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.ne.intershell[star.ntpagb :],
        c="k",
        linewidth=1,
        label="Monash abundances",
    )
    ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.ne.envelope[star.ntpagb :],
        c="k",
        linewidth=1,
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.ne20
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l1,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        c="C9",
        linewidth=1,
        label="MESA abundances",
    )
    ax.step(np.array(age) / age[-1], envelope, c="C9", linewidth=1)
    ax.set_yscale("log")


axs[4].annotate(
    "intershell",
    xy=(0.6, 6e-3),
    xycoords="data",
    xytext=(-17, -15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

axs[4].annotate(
    "envelope",
    xy=(0.3, 5e-4),
    xycoords="data",
    xytext=(-16, 10),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2])

plt.yscale("log")

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{Ne})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-ne-all.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]
print(ms)

for ax, m in zip(axs, ms):
    if m < 1.4:
        continue
    star = get_star(m=m)
    ab = Abundances(model=None, df=df, mass=m)

    (l1,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.mg.intershell[star.ntpagb :],
        linewidth=1,
        label="Monash abundances intershell",
    )
    (l2,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.mg.envelope[star.ntpagb :],
        linewidth=1,
        label="Monash abundances envelope",
    )

    intershell = []
    envelope = []
    age = []
    for p in profiles_dict[m]:
        m = p.mass
        el = p.mg24
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(m[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
        )
        age.append(p.star_age)

    (l3,) = ax.step(
        np.array(age) / age[-1],
        intershell,
        linewidth=1,
        label="MESA abundances intershell",
    )
    (l4,) = ax.step(
        np.array(age) / age[-1], envelope, linewidth=1, label="MESA abundances envelope"
    )
    ax.set_yscale("log")


# axs[4].annotate(
#     "intershell",
#     xy=(0.25, 2e-1),
#     xycoords="data",
#     xytext=(-17, -22.5),
#     textcoords="offset points",
#     arrowprops=dict(
#         arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
#     ),
#     zorder=1000,
#     fontsize=8,
# )
#
# axs[4].annotate(
#     "envelope",
#     xy=(0.25, 8e-4),
#     xycoords="data",
#     xytext=(-16, 17.5),
#     textcoords="offset points",
#     arrowprops=dict(
#         arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
#     ),
#     zorder=1000,
#     fontsize=8,
# )

fig.legend(loc="outside upper center", ncols=2, handles=[l1, l2, l3, l4])

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("Normalized TPAGB age", size=10)
fig.supylabel("$X(\\textrm{Mg})$", size=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-compare-mg-all.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

intershell = []
envelope = []
age = []

ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]

print(ms[2])

for p in profiles_dict[ms[2]]:
    m = p.mass
    el = p.c12
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    intershell.append(
        np.average(
            el[ind + 100 : ind + 110],
            weights=np.diff(m[ind + 100 : ind + 111]),
        )
    )

    envelope.append(
        np.average(el[: ind - 100], weights=np.diff(m[: ind - 100], prepend=m[0]))
    )
    age.append(p.star_age)

plt.scatter(age, envelope, marker=".", s=100, color="white", zorder=4)
plt.scatter(age, envelope, marker=".", s=20, color="k", zorder=5)


plt.scatter(age, intershell, marker=".", s=100, color="white", zorder=4)
plt.scatter(age, intershell, marker=".", s=20, color="k", zorder=5, label="profile")

star = get_star(m=ms[2])

plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    star.envelope_c12[star.ntpagb :],
    c="C9",
    alpha=0.5,
    linewidth=4,
    label="MESA",
)

star_age = np.asarray(star.age) - star.age[star.ntpagb]

envelope = np.asarray(envelope)
intershell = np.asarray(intershell)
idx = np.searchsorted(age, star_age, side="right")

el_long = np.full_like(star_age, np.nan, dtype=float)
int_long = np.full_like(star_age, np.nan, dtype=float)

valid = idx >= 0
print(envelope[idx[valid]])
print(envelope)
el_long[valid] = envelope[idx[valid]]
int_long[valid] = intershell[idx[valid]]

plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    el_long[star.ntpagb :],
    c="k",
    linewidth=1,
)

plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    int_long[star.ntpagb :],
    c="k",
    linewidth=1,
    label="profile reconstruction",
)

fig.legend(loc="outside upper center", ncols=3)

plt.annotate(
    "intershell",
    xy=(1.3e6, 4e-1),
    xycoords="data",
    xytext=(-17, -32.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

plt.annotate(
    "envelope",
    xy=(1.3e6, 1e-2),
    xycoords="data",
    xytext=(-16, 27.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    fontsize=8,
    zorder=1000,
)

plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("TPAGB age (yr)")
plt.ylabel("$X(\\textrm{C})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-reconstruct-intershell-c.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    3,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
)

axs = axs.flatten()


ms = np.arange(1.0, 3.01, 0.1)
ms = ms[::2]
ms = ms[2:]

for ax, m in zip(axs, ms):

    intershell = []
    envelope = []
    age = []

    for p in profiles_dict[m]:
        ma = p.mass
        el = p.c12
        i = np.argmax(
            np.abs(
                np.log10(p.z_mass_fraction_metals[10:])
                - np.log10(p.z_mass_fraction_metals[:-10])
            )
        )
        ind = i + 5
        intershell.append(
            np.average(
                el[ind + 100 : ind + 110],
                weights=np.diff(ma[ind + 100 : ind + 111]),
            )
        )

        envelope.append(
            np.average(el[: ind - 100], weights=np.diff(ma[: ind - 100], prepend=ma[0]))
        )
        age.append(p.star_age)

    star = get_star(m=m)
    star_age = np.asarray(star.age) - star.age[star.ntpagb]

    envelope = np.asarray(envelope)
    intershell = np.asarray(intershell)
    idx = np.searchsorted(age, star_age, side="right")

    el_long = np.full_like(star_age, np.nan, dtype=float)
    int_long = np.full_like(star_age, np.nan, dtype=float)

    valid = idx >= 0
    el_long[valid] = envelope[idx[valid]]
    int_long[valid] = intershell[idx[valid]]

    ax.plot(
        (star.age[star.ntpagb :] - star.age[star.ntpagb])
        / (star.age[-1] - star.age[star.ntpagb]),
        el_long[star.ntpagb :],
        c="k",
        linewidth=1,
    )

    (l2,) = ax.plot(
        (star.age[star.ntpagb :] - star.age[star.ntpagb])
        / (star.age[-1] - star.age[star.ntpagb]),
        int_long[star.ntpagb :],
        c="k",
        linewidth=1,
        label="profile reconstruction",
    )

    ab = Abundances(
        model=None, df=df, mass=m, intershell=int_long, initial_abundance=el_long[0]
    )

    (l1,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.intershell[star.ntpagb :],
        linewidth=4,
        c="C9",
        alpha=0.75,
        zorder=-10,
    )

    (l1,) = ax.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.envelope[star.ntpagb :],
        linewidth=4,
        c="C9",
        alpha=0.75,
        label="Post-processed reconstruction",
        zorder=-10,
    )
    ax.set_yscale("log")

axs[4].annotate(
    "intershell",
    xy=(0.25, 4e-1),
    xycoords="data",
    xytext=(-17, -27.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)

axs[4].annotate(
    "envelope",
    xy=(0.25, 6e-4),
    xycoords="data",
    xytext=(-16, 22.5),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0", color="C9", linewidth=0.75
    ),
    zorder=1000,
    fontsize=8,
)


fig.legend(loc="outside upper center", ncols=3, handles=[l1, l2])
plt.yscale("log")

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)

axs[7].set_xlabel("Normalized TPAGB age")
axs[3].set_ylabel("$X(\\textrm{C})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-test-envelope.pgf", format="pgf")
plt.show()
plt.close()


# %%
