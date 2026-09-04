import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle
import re

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
# %%

with open("data/tp_info_pd_df.pkl", "rb") as f:
    tp_info = pickle.load(f)

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)
# %%

i1 = intershell[intershell["M1tp"] == 1.5]
i1 = i1[i1["Z"] == 0.014]
i1 = i1[i1["pmz"] == 2e-3]
i1["age"]
# %%
m = 2.75
z = 0.007

i1 = intershell.query(f"M1tp == {m} and Z == {z} and pmz == 2e-3")
tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
print(tp1.columns)
print(i1.columns)

plt.plot(tp1.pulse, tp1.Mcore)
plt.plot(i1.ntp, i1.Mcore)
plt.show()
# %%

z = 0.014
for m_i, m in enumerate([1.5, 1.75, 2, 2.25, 2.5, 2.75, 3][::-1]):

    i1 = intershell.query(f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1")
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    plt.plot(tp1.pulse, tp1.Ddredge, label=f"M = {m}", c=f"C{m_i}")
    plt.plot(i1.ntp, i1.ba138, c=f"C{m_i}", linewidth=4, alpha=0.5)

plt.legend()
plt.yscale("log")
plt.show()
# %%

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

z = 0.007
for m_i, m in enumerate([1.5, 1.75, 1.9, 2.1, 2.25, 2.5, 2.75, 3][::-1]):

    i1 = intershell.query(f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1")
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    axs[1].plot(
        tp1.pulse[: len(i1.ntp)],
        np.array(tp1.Mcore[: len(i1.Mcore)]) - np.array(i1.Mcore[: len(tp1.Mcore)]),
        c=f"C{m_i}",
    )
    axs[0].plot(tp1.pulse, tp1.Mcore, label=f"M = {m}", c=f"C{m_i}")
    axs[0].plot(i1.ntp, i1.Mcore, c=f"C{m_i}", linewidth=4, alpha=0.5)

fig.legend(loc="outside upper center", ncols=4)
# plt.yscale("log")

axs[0].text(0.05, 0.95, "$Z=0.007$", transform=axs[0].transAxes)
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
axs[1].set_ylabel("Residual ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-check-masses.pgf", format="pgf")
plt.show()
plt.close()


# %%
def abundance_distance(x_old, x_new, floor=1e-12):
    mask = (x_old > floor) | (x_new > floor)

    log_ratio = np.log10((x_new[mask] + floor) / (x_old[mask] + floor))

    return np.sqrt(np.mean(log_ratio**2))


# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


z = 0.007

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()

    # abundance-pattern change between consecutive TPs
    distances = np.array(
        [
            abundance_distance(abundances[j - 1], abundances[j])
            for j in range(1, len(abundances))
        ]
    )

    # distance belongs to the second TP in each pair
    axs[0].plot(
        i1.ntp.iloc[1:],
        distances + m_i,
        label=f"M = {m}",
        c=f"C{m_i}",
    )

    # dredge-up events
    axs[1].plot(
        tp1.pulse,
        tp1.Ddredge,
        c=f"C{m_i}",
    )

for i in range(int(axs[0].get_xlim()[-1])):
    axs[0].axvline(i, c="C9", zorder=-1, linewidth=0.75)
    axs[1].axvline(i, c="C9", zorder=-1, linewidth=0.75)

plt.yscale("log")
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("Abundance-pattern change")
axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w26-abundance-pattern.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


z = 0.0028

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()

    # abundance-pattern change between consecutive TPs
    distances = np.array(
        [
            abundance_distance(abundances[j - 1], abundances[j])
            for j in range(1, len(abundances))
        ]
    )

    # distance belongs to the second TP in each pair
    axs[0].plot(
        i1.ntp.iloc[1:],
        distances + m_i,
        label=f"M = {m}",
        c=f"C{m_i}",
    )

    # dredge-up events
    axs[1].plot(
        tp1.pulse,
        tp1.Ddredge,
        c=f"C{m_i}",
    )

for i in range(int(axs[0].get_xlim()[-1])):
    axs[0].axvline(i, c="C9", zorder=-1, linewidth=0.75)
    axs[1].axvline(i, c="C9", zorder=-1, linewidth=0.75)

plt.yscale("log")
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("Abundance-pattern change")
axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w26-abundance-pattern-z0028.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


z = 0.014

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()

    # abundance-pattern change between consecutive TPs
    distances = np.array(
        [
            abundance_distance(abundances[j - 1], abundances[j])
            for j in range(1, len(abundances))
        ]
    )

    # distance belongs to the second TP in each pair
    axs[0].plot(
        i1.ntp.iloc[1:],
        distances + m_i,
        label=f"M = {m}",
        c=f"C{m_i}",
    )

    # dredge-up events
    axs[1].plot(
        tp1.pulse,
        tp1.Ddredge,
        c=f"C{m_i}",
    )

for i in range(int(axs[0].get_xlim()[-1])):
    axs[0].axvline(i, c="C9", zorder=-1, linewidth=0.75)
    axs[1].axvline(i, c="C9", zorder=-1, linewidth=0.75)

plt.yscale("log")
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("Abundance-pattern change")
axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w26-abundance-pattern-z014.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

z = 0.0028

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    if m == 1.15:
        axs[1].plot(
            range(2, 17),
            np.abs(np.array(tp1.Mcore[1:16]) - np.array(i1.Mcore[:16])),
            c=f"C{m_i}",
        )
    else:
        axs[1].plot(
            tp1.pulse[: len(i1.ntp)],
            np.abs(
                np.array(tp1.Mcore[: len(i1.Mcore)])
                - np.array(i1.Mcore[: len(tp1.Mcore)])
            ),
            c=f"C{m_i}",
        )
    axs[0].plot(tp1.pulse, tp1.Mcore, label=f"M = {m}", c=f"C{m_i}")
    axs[0].plot(i1.ntp, i1.Mcore, c=f"C{m_i}", linewidth=4, alpha=0.5)

fig.legend(loc="outside upper center", ncols=4)
# plt.yscale("log")

axs[0].text(0.05, 0.95, "$Z=0.0028$", transform=axs[0].transAxes)
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
axs[1].set_ylabel("Residual ($M_\\odot$)")
axs[1].set_yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-check-masses-0.0028.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1),
    constrained_layout=True,
    height_ratios=[1, 0.5],
)

z = 0.014

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    axs[1].plot(
        tp1.pulse[: len(i1.ntp)],
        np.abs(
            np.array(tp1.Mcore[: len(i1.Mcore)]) - np.array(i1.Mcore[: len(tp1.Mcore)])
        ),
        c=f"C{m_i}",
    )
    axs[0].plot(tp1.pulse, tp1.Mcore, label=f"M = {m}", c=f"C{m_i}")
    axs[0].plot(i1.ntp, i1.Mcore, c=f"C{m_i}", linewidth=4, alpha=0.5)

fig.legend(loc="outside upper center", ncols=4)
# plt.yscale("log")

axs[0].text(0.05, 0.95, "$Z=0.014$", transform=axs[0].transAxes)
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
axs[1].set_ylabel("Residual ($M_\\odot$)")
axs[1].set_yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-check-masses-0.014.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


z = 0.007

intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

    # distance belongs to the second TP in each pair
    plt.plot(
        np.cumsum(tp1.Ddredge[: len(i1.ba138)]),
        138 * i1.ba138[: len(tp1.Ddredge)],
        label=f"M = {m}",
        c=f"C{m_i}",
    )


plt.yscale("log")
plt.xscale("log")
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.ylabel("$X(\\textrm{Ba138})$")
# axs[0].set_ylabel("Abundance-pattern change")
# axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w26-barium-mdredge-z007.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


norm = plt.Normalize(1.15, 3)
cmap = plt.cm.Spectral
# color = cmap(norm(x))


zs = [0.0028, 0.007, 0.014]
for ax, z in zip(axs, zs):

    print(z)
    intershell_metal = intershell.query(
        f"Z == {z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)

    for m_i, m in enumerate(ms[::-1]):
        print(m)

        i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

        tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

        # distance belongs to the second TP in each pair

        M_dredge = []
        ba138 = []
        for tp in np.unique(tp1.pulse):
            i_data = i1[i1.ntp == tp]
            tp_data = tp1[tp1.pulse == tp]
            if len(i_data) != 0:
                M_dredge.append(np.array(tp_data.Ddredge)[0])
                ba138.append(138 * np.array(i_data.ba138)[0])

        ax.plot(
            np.cumsum(M_dredge),
            ba138,
            label=f"M = {m}",
            c=cmap(norm(m)),
        )

        ax.plot(
            np.cumsum(M_dredge), ba138, label=f"M = {m}", c="k", linewidth=2, zorder=-1
        )
    ax.text(0.05, 0.95, f"$Z={z:.4f}$", ha="left", va="top", transform=ax.transAxes)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs)
cbar.set_label(r"Initial mass ($M_\odot$)")

# plt.yscale("log")
plt.xscale("log")
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
axs[1].set_ylabel("$X(\\textrm{Ba138})$")
# axs[0].set_ylabel("Abundance-pattern change")
# axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
# fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    # ax.set_facecolor("C9")
    ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w26-barium-mdredge.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


norm = plt.Normalize(1.15, 3)
cmap = plt.cm.Spectral
# color = cmap(norm(x))


zs = [0.0028, 0.007, 0.014]
for ax, z in zip(axs, zs):

    print(z)
    intershell_metal = intershell.query(
        f"Z == {z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)

    for m_i, m in enumerate(ms[::-1]):
        print(m)

        i1 = intershell_metal.query(f"M1tp == {m}").sort_values("ntp")

        tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")

        # distance belongs to the second TP in each pair

        M_dredge = []
        ba138 = []
        for tp in np.unique(tp1.pulse):
            i_data = i1[i1.ntp == tp]
            tp_data = tp1[tp1.pulse == tp]
            if len(i_data) != 0:
                M_dredge.append(np.array(tp_data.Ddredge)[0])
                ba138.append(138 * np.array(i_data.ba138)[0])

        ax.plot(
            np.cumsum(M_dredge),
            ba138,
            label=f"M = {m}",
            c=cmap(norm(m)),
        )

        ax.plot(
            np.cumsum(M_dredge), ba138, label=f"M = {m}", c="k", linewidth=2, zorder=-1
        )
    ax.text(0.05, 0.95, f"$Z={z:.4f}$", ha="left", va="top", transform=ax.transAxes)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs)
cbar.set_label(r"Initial mass ($M_\odot$)")

# plt.yscale("log")
plt.xscale("log")
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
axs[1].set_ylabel("$X(\\textrm{Ba138})$")
# axs[0].set_ylabel("Abundance-pattern change")
# axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
# fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    # ax.set_facecolor("C9")
    ax.spines[["right", "top"]].set_visible(False)
# plt.savefig("/home/koen/LaTeX-setup/plots/w26-barium-mdredge.pgf", format="pgf")
plt.show()
plt.close()
# %%


def get_iso_and_Mdredge(isotope, intershell, tp_info):

    isotope_list = []
    Mdredge = []

    for tp in np.unique(tp_info.pulse):
        i_data = intershell[intershell.ntp == tp]
        tp_data = tp_info[tp_info.pulse == tp]
        if len(i_data) != 0:
            Mdredge.append(np.array(tp_data.Ddredge)[0] + 1e-12)
            isotope_list.append(isotope.mass * np.array(i_data[isotope.key])[0])

    isotope_list = np.log10(np.array(isotope_list))
    Mdredge = np.log10(np.cumsum(Mdredge))
    return isotope_list, Mdredge


def get_iso_and_Mdredge(isotope, intershell, tp_info):

    isotope_list = []
    Mdredge = []

    for tp in np.unique(tp_info.pulse):
        i_data = intershell[intershell.ntp == tp]
        tp_data = tp_info[tp_info.pulse == tp]
        isotope_keys = i_data.columns[8:]
        print(isotope_keys)
        if len(i_data) != 0:
            Mdredge.append(np.array(tp_data.Ddredge)[0] + 1e-12)
            isotope_list.append(isotope.mass * np.array(i_data[isotope.key])[0])
            abundances = i_data.to_numpy()[0][8:]
            print(abundances)

    isotope_list = np.log10(np.array(isotope_list))
    Mdredge = np.log10(np.cumsum(Mdredge))
    return isotope_list, Mdredge


def get_iso_and_Mdredge(isotope, intershell, tp_info):
    # keep only pulses that exist in both datasets
    intershell = intershell[intershell.ntp.isin(tp_info.pulse)]

    # one intershell abundance per pulse
    i_data = intershell.drop_duplicates("ntp").set_index("ntp")

    # one dredge-up mass per pulse
    tp_data = tp_info.drop_duplicates("pulse").set_index("pulse")

    # common pulses, in ascending order
    pulses = i_data.index.intersection(tp_data.index).sort_values()

    Mdredge = tp_data.loc[pulses, "Ddredge"].to_numpy()
    isotope_abundance = i_data.loc[pulses, isotope.key].to_numpy()

    Mdredge = np.log10(np.cumsum(Mdredge) + 1e-12)
    isotope_abundance = np.log10(isotope.mass * isotope_abundance)

    return isotope_abundance, Mdredge


#
# from pathlib import Path
# import pandas as pd
# import pickle
#
# CACHE_DIR = Path("/home/koen/master-internship/data/tp+intershell/")
#
#
# def get_iso_and_Mdredge(isotope, intershell, tp_info, mass, z):
#     cache_file = CACHE_DIR / f"m{mass:g}_z{z:g}.pkl"
#
#     # load cached data if it exists
#     if cache_file.exists():
#         print(f"reading {cache_file}")
#         with open(cache_file, "rb") as f:
#             data = pickle.load(f)
#
#         return data.Mdredge, isotope.mass * data[isotope.key]
#
#     # keep only pulses that exist in both datasets
#     intershell = intershell[intershell.ntp.isin(tp_info.pulse)]
#
#     # one intershell abundance per pulse
#     i_data = intershell.drop_duplicates("ntp").set_index("ntp")
#     isotope_keys = i_data.columns[7:]
#
#     # one dredge-up mass per pulse
#     tp_data = tp_info.drop_duplicates("pulse").set_index("pulse")
#
#     # common pulses, in ascending order
#     pulses = i_data.index.intersection(tp_data.index).sort_values()
#
#     Mdredge = tp_data.loc[pulses, "Ddredge"].to_numpy()
#     isotope_abundance = i_data.loc[pulses, isotope_keys].to_numpy()
#
#     # cumulative dredged-up mass
#     Mdredge = np.log10(np.cumsum(Mdredge) + 1e-12)
#
#     # convert mass fractions to isotope masses
#     isotope_abundance = np.log10(isotope_abundance)
#
#     # construct dataframe
#     data = pd.DataFrame(
#         isotope_abundance,
#         index=pulses,
#         columns=isotope_keys,
#     )
#
#     data.insert(0, "Mdredge", Mdredge)
#     data.insert(0, "ntp", pulses.to_numpy())
#
#     # create cache directory and save
#     CACHE_DIR.mkdir(parents=True, exist_ok=True)
#
#     with open(cache_file, "wb") as f:
#         print(f"writing {cache_file}")
#         pickle.dump(data, f)
#
#     print(data[isotope.key])
#     return data.Mdredge, isotope.mass * data[isotope.key]
#

# %%


def get_abundance(isotope, M, Z, Mdredge_interp, drop=None):

    if Z in [0.0028, 0.007, 0.014]:
        return 10 ** get_abundance_Z(isotope, M, Z, Mdredge_interp, drop)

    if Z <= 0.0028:
        return 10 ** get_abundance_Z(isotope, M, 0.0028, Mdredge_interp, drop)
    if Z >= 0.014:
        return 10 ** get_abundance_Z(isotope, M, 0.014, Mdredge_interp, drop)

    if Z <= 0.007:
        z_min = 0.0028
        z_max = 0.007
    else:
        z_min = 0.007
        z_max = 0.014

    abundance_min = get_abundance_Z(isotope, M, z_min, Mdredge_interp, drop)
    abundance_max = get_abundance_Z(isotope, M, z_max, Mdredge_interp, drop)
    weight = (np.log10(Z) - np.log10(z_min)) / (np.log10(z_max) - np.log10(z_min))

    return 10 ** (abundance_min + weight * (abundance_max - abundance_min))


def get_abundance_Z(isotope, M, Z, Mdredge_interp, drop=None):
    intershell_metal = intershell.query(
        f"Z == {Z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)
    ms.sort()
    if drop != None:
        for i, m in enumerate(ms):
            if m == drop:
                ms = np.delete(ms, i)
    print(ms)
    try:
        arg_max = np.where(ms >= M)[0][0]
        i1_max = intershell_metal.query(f"M1tp == {ms[arg_max]}")
        tp1_max = tp_info.query(f"initial_mass == {ms[arg_max]} and z == {Z}")
        iso, Mdredge_real = get_iso_and_Mdredge(isotope, i1_max, tp1_max)
        abundance_max = np.interp(np.log10(Mdredge_interp), Mdredge_real, iso)
    except IndexError:
        abundance_max = None
    try:
        arg_min = np.where(ms < M)[0][-1]
        i1_min = intershell_metal.query(f"M1tp == {ms[arg_min]}")
        tp1_min = tp_info.query(f"initial_mass == {ms[arg_min]} and z == {Z}")
        iso, Mdredge_real = get_iso_and_Mdredge(isotope, i1_min, tp1_min)
        abundance_min = np.interp(np.log10(Mdredge_interp), Mdredge_real, iso)
    except IndexError:
        abundance_min = None

    if type(abundance_min) == type(None):
        return abundance_max
    if type(abundance_max) == type(None):
        return abundance_min
    weight = (M - ms[arg_min]) / (ms[arg_max] - ms[arg_min])

    return abundance_min + weight * (abundance_max - abundance_min)


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mdredge = np.logspace(-5, 0, 100)
m = 2.5
ab = get_abundance(Isotope("ba138"), m, 0.0028, mdredge)
(k,) = plt.plot(mdredge, ab, c="C0", zorder=10, label="$Z=0.0028$")

for z in np.linspace(0.0028, 0.007, 10)[1:]:
    ab = get_abundance(Isotope("ba138"), m, z, mdredge)
    plt.plot(mdredge, ab, c="C9", linewidth=0.75)

ab = get_abundance(Isotope("ba138"), m, 0.007, mdredge)
(n,) = plt.plot(mdredge, ab, c="C1", zorder=10, label="$Z=0.007$")

for z in np.linspace(0.007, 0.014, 10)[1:]:
    ab = get_abundance(Isotope("ba138"), m, z, mdredge)
    (l,) = plt.plot(mdredge, ab, c="C9", label="interpolated", linewidth=0.75)


ab = get_abundance(Isotope("ba138"), m, 0.014, mdredge)
(m,) = plt.plot(mdredge, ab, c="C2", zorder=10, label="$Z=0.014$")

plt.xscale("log")
plt.yscale("log")

fig.legend(loc="outside upper center", ncols=4, handles=[k, n, m, l])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.ylabel("$X(\\textrm{Ba}_{138})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-interp-z.pgf", format="pgf")
plt.show()
plt.close()

# %%


class Isotope:
    def __init__(self, isotope):
        self.key = isotope

        match isotope:
            case "n":
                self.mass = 1
                self.name = f"$\\textrm{{{isotope}}}$"
                self.short_name = "neutron"
            case "p":
                self.mass = 1
                self.name = f"$\\textrm{{{isotope}}}^{{+}}$"
                self.short_name = "proton"
            case "d":
                self.mass = 2
                self.name = f"$\\textrm{{{isotope}}}$"
                self.short_name = "deuterium"
            case _:
                match = re.match(r"([a-zA-Z]+)(\d+)$", isotope)
                if match:
                    self.short_name = match.group(1)
                    self.mass = int(match.group(2))
                    self.name = (
                        f"$\\textrm{{{self.short_name.capitalize()}}}_{{{self.mass}}}$"
                    )
                else:
                    self.short_name = None
                    self.name = None
                    self.mass = None

    def __str__(self):
        return f"{self.key} with mass {self.mass}"

    def __repr__(self):
        return f"{self.key}"


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mdredge = np.logspace(-5, 0, 100)
z = 0.007
ab = get_abundance(Isotope("ba138"), 2.1, z, mdredge)
(k,) = plt.plot(mdredge, ab, c="C0", zorder=10, label="$M=2.1\\;M_\\odot$")

for m in np.linspace(2.1, 2.25, 10)[1:]:
    ab = get_abundance(Isotope("ba138"), m, z, mdredge)
    plt.plot(mdredge, ab, c="C9", linewidth=0.75)

ab = get_abundance(Isotope("ba138"), 2.25, z, mdredge)
(n,) = plt.plot(mdredge, ab, c="C1", zorder=10, label="$M=2.25\\;M_\\odot$")

for m in np.linspace(2.25, 2.5, 10)[1:]:
    ab = get_abundance(Isotope("ba138"), m, z, mdredge)
    (l,) = plt.plot(mdredge, ab, c="C9", label="interpolated", linewidth=0.75)


ab = get_abundance(Isotope("ba138"), 2.5, z, mdredge)
(m,) = plt.plot(mdredge, ab, c="C2", zorder=10, label="$M=2.5\\;M_\\odot$")

plt.xscale("log")
plt.yscale("log")

fig.legend(loc="outside upper center", ncols=2, handles=[k, n, m, l])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.ylabel("$X(\\textrm{Ba}_{138})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-interp-m.pgf", format="pgf")
plt.show()
plt.close()


# %%

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mdredge = np.logspace(-5, 0, 100)
z = 0.007

for i, m in enumerate([1.75, 1.9, 2.1, 2.25, 2.5, 2.75]):

    ab = get_abundance(Isotope("ba138"), m, z, mdredge)
    (k,) = plt.plot(mdredge, ab, c=f"C{i}", zorder=10, label=f"$M={m:.2f}\\;M_\\odot$")

    ab = get_abundance(Isotope("ba138"), m, z, mdredge, drop=m)
    (k,) = plt.plot(mdredge, ab, c=f"C{i}", zorder=10, linewidth=4, alpha=0.5)

plt.xscale("log")
# plt.yscale("log")

fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.ylabel("$X(\\textrm{Ba}_{138})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w26-interp-m-check.pgf", format="pgf")
plt.show()
plt.close()
