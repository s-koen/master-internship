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


def get_iso_and_Mdredge(isotope, intershell, tp_info):
    # keep only pulses that exist in both datasets
    intershell = intershell[intershell.ntp.isin(tp_info.pulse)]

    # one intershell abundance per pulse
    i_data = intershell.drop_duplicates("ntp").set_index("ntp")

    # one dredge-up mass per pulse
    tp_data = tp_info.drop_duplicates("pulse").set_index("pulse")

    # common pulses, in ascending order
    pulses = i_data.index.intersection(tp_data.index).sort_values()

    Mcore_tp_data = tp_data.loc[pulses, "Mcore"].to_numpy()
    Mcore_i_data = i_data.loc[pulses, "Mcore"].to_numpy()

    Mcore_tp_data = np.log10(np.cumsum(Mcore_tp_data) + 1e-12)
    Mcore_i_data = np.log10(np.cumsum(Mcore_i_data) + 1e-12)

    return pulses, Mcore_i_data, Mcore_tp_data


# %%
def abundance_distance(x_old, x_new, floor=1e-12):
    mask = (x_old > floor) | (x_new > floor)

    log_ratio = np.log10((x_new[mask] + floor) / (x_old[mask] + floor))

    return np.sqrt(np.mean(log_ratio**2))


# %%
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter()
formatter.set_powerlimits((-4, 4))

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
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

    # # abundance-pattern change between consecutive TPs
    # distances = np.array(
    #     [
    #         abundance_distance(abundances[j - 1], abundances[j])
    #         for j in range(1, len(abundances))
    #     ]
    # )
    ab = np.zeros(len(i1.ntp))
    for abundance in abundance_cols[96:]:

        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            mass = int(match.group(2))
        ab += mass * i1[abundance]

    # distance belongs to the second TP in each pair
    axs[1].plot(i1.ntp.iloc[:], ab, "-", c=f"C{m_i}", linewidth=1, zorder=-m_i)

    axs[1].scatter(i1.ntp.iloc[:], ab, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i)

    axs[1].scatter(i1.ntp.iloc[:], ab, marker=".", s=75, c="white", zorder=10 - m_i)

    # dredge-up events
    axs[2].plot(tp1.pulse, tp1.Ddredge, c=f"C{m_i}", linewidth=1, zorder=-m_i)

    axs[2].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i
    )

    axs[2].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=75, c=f"white", zorder=10 - m_i
    )

    ab = np.zeros(len(i1.ntp))
    for abundance in abundance_cols[4:96]:

        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            mass = int(match.group(2))
        ab += mass * i1[abundance]

    # distance belongs to the second TP in each pair
    axs[0].plot(
        i1.ntp.iloc[:], ab, "-", label=f"M = {m}", c=f"C{m_i}", linewidth=1, zorder=-m_i
    )

    axs[0].scatter(i1.ntp.iloc[:], ab, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i)

    axs[0].scatter(i1.ntp.iloc[:], ab, marker=".", s=75, c="white", zorder=10 - m_i)

    # dredge-up events
    axs[2].plot(tp1.pulse, tp1.Ddredge, c=f"C{m_i}", linewidth=1, zorder=-m_i)

    axs[2].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i
    )

    axs[2].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=75, c=f"white", zorder=10 - m_i
    )


for i in range(int(axs[0].get_xlim()[-1])):
    axs[0].axvline(i, c="C9", zorder=-1, linewidth=0.75 / 2)
    axs[1].axvline(i, c="C9", zorder=-1, linewidth=0.75 / 2)
    axs[2].axvline(i, c="C9", zorder=-1, linewidth=0.75 / 2)


axs[0].yaxis.set_major_formatter(formatter)

plt.yscale("log")
plt.xlabel("Thermal pulse count")
plt.xticks(np.arange(0, int(plt.gca().get_xlim()[-1]), 2), minor=False)
plt.xticks(np.arange(1, int(plt.gca().get_xlim()[-1]) + 1, 2), minor=True)
axs[0].set_ylabel(r"$\displaystyle\sum_{Z\leq 26}X_i$")
axs[1].set_ylabel(r"$\displaystyle\sum_{Z>26}X_i$")
axs[2].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlim(8)
plt.show()
plt.close()
# %%


df = AbundanceTables()


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


z = 0.014

intershell_metal = intershell.query(
    f"M1tp == {2.5} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
z = []
els = []
final.append(ab.h.intershell[-1] / ab.h.intershell[star.ntpagb])
final.append(ab.he.intershell[-1] / ab.he.intershell[star.ntpagb])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).intershell
    e = e[-1] / e[star.ntpagb]
    final.append(e)
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

plt.plot(
    z,
    final,
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)


for m_i, m in enumerate([0.007, 0.0028]):

    i1 = intershell_metal.query(f"Z == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {2.5} and z == {m}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()
    print(abundance_cols)

    # # abundance-pattern change between consecutive TPs
    # distances = np.array(
    #     [
    #         abundance_distance(abundances[j - 1], abundances[j])
    #         for j in range(1, len(abundances))
    #     ]
    # )
    ab = {}
    last_el = None
    el = None
    for i, abundance in enumerate(abundance_cols):
        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            el = match.group(1)
            mass = int(match.group(2))
        else:
            continue

        if el == last_el:
            ab[el] += mass * np.array(i1[abundance])

        else:
            ab[el] = mass * np.array(i1[abundance])

        last_el = el

    ab_class = Abundances(None, df, method="tp offset", mass=2.5)

    els = []
    el_names = []
    z = []

    els.append(np.array(i1["p"])[-1] / np.array(i1["p"])[0])
    el_names.append("h")
    z.append(1)

    for el in ab:
        if el in ["li", "be", "b", "tc", "pm", "po"]:
            continue
        el_names.append(el)
        els.append(ab[el][-1] / ab[el][0])
        data = ab_class.df.envelope[ab_class.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))

    # distance belongs to the second TP in each pair
    print(z)
    print(el_names)
    axs.plot(
        z,
        els,
        linestyle="-",
        c=f"C{m_i+1}",
        linewidth=1,
        zorder=-m_i,
        label=f"Monash $M=2.5\\;M_\\odot,\\;Z={m}$",
    )

labels = el_names
for i, _ in enumerate(labels):
    labels[i] = labels[i].capitalize()

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f} / X_\\textrm{i}$")
plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-intershell-change.pgf", format="pgf")
plt.show()
plt.close()

# %%

df = AbundanceTables()


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


z = 0.014

intershell_metal = intershell.query(
    f"M1tp == {2.5} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)

ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
z = []
els = []
final.append(ab.h.envelope[-1] / ab.h.envelope[star.ntpagb])
final.append(ab.he.envelope[-1] / ab.he.envelope[star.ntpagb])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).envelope
    e = e[-1] / e[star.ntpagb]
    final.append(e)
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

plt.plot(
    z,
    final,
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)


for m_i, m in enumerate([0.007, 0.0028]):

    i1 = intershell_metal.query(f"Z == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {2.5} and z == {m}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()
    print(abundance_cols)

    # # abundance-pattern change between consecutive TPs
    # distances = np.array(
    #     [
    #         abundance_distance(abundances[j - 1], abundances[j])
    #         for j in range(1, len(abundances))
    #     ]
    # )
    ab = {}
    last_el = None
    el = None
    for i, abundance in enumerate(abundance_cols):
        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            el = match.group(1)
            mass = int(match.group(2))
        else:
            continue

        if el == last_el:
            ab[el] += mass * np.array(i1[abundance])

        else:
            ab[el] = mass * np.array(i1[abundance])

        last_el = el

    ab_class = Abundances(None, df, method="tp offset", mass=2.5)

    els = []
    el_names = []
    z = []

    # els.append(np.array(i1["p"])[-1] / np.array(i1["p"])[0])
    # el_names.append("h")
    # z.append(1)
    data = ab_class.df.envelope[ab_class.df.envelope["element"] == "h"]
    data = data.query(f"Z == {m} and M_init == 2.5")
    z.append(int(np.array(data["elemental_mass"])[0]))
    x = np.array(data["massfrac"])
    els.append(x[-1] / x[0])
    el_names.append("h")

    for el in ab:
        if el in ["li", "be", "b", "tc", "pm", "po"]:
            continue
        el_names.append(el)
        data = ab_class.df.envelope[ab_class.df.envelope["element"] == el]
        data = data.query(f"Z == {m} and M_init == 2.5")
        z.append(int(np.array(data["elemental_mass"])[0]))
        x = np.array(data["massfrac"])
        els.append(x[-1] / x[0])

    # distance belongs to the second TP in each pair
    print(z)
    print(el_names)
    axs.plot(
        z,
        els,
        linestyle="-",
        c=f"C{m_i+1}",
        linewidth=1,
        zorder=-m_i,
        label=f"Monash $M=2.5\\;M_\\odot,\\;Z={m}$",
    )

labels = el_names
for i, _ in enumerate(labels):
    labels[i] = labels[i].capitalize()

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

plt.ylim(1e-4, 1e4)

plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f} / X_\\textrm{i}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-change.pgf", format="pgf")
plt.show()
plt.close()

# %%

df = AbundanceTables()


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


zz = []
ms = [0.007, 0.0028]
for m_i, m in enumerate(ms):

    i1 = intershell_metal.query(f"Z == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {2.5} and z == {m}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()

    # # abundance-pattern change between consecutive TPs
    # distances = np.array(
    #     [
    #         abundance_distance(abundances[j - 1], abundances[j])
    #         for j in range(1, len(abundances))
    #     ]
    # )
    ab = {}
    last_el = None
    el = None
    for i, abundance in enumerate(abundance_cols):
        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            el = match.group(1)
            mass = int(match.group(2))
        else:
            continue

        if el == last_el:
            ab[el] += mass * np.array(i1[abundance])

        else:
            ab[el] = mass * np.array(i1[abundance])

        last_el = el

    ab_class = Abundances(None, df, method="tp offset", mass=2.5)

    els = []
    el_names = []
    z = []

    el_names.append("h")
    data = ab_class.df.envelope[ab_class.df.envelope["element"] == "h"]
    data = data.query(f"Z == {m} and M_init == 2.5")
    z.append(int(np.array(data["elemental_mass"])[0]))
    x = np.array(data["massfrac"])
    els.append(x[-1] / x[0])

    for el in ab:
        if el in ["li", "be", "b", "tc", "pm", "po"]:
            continue
        el_names.append(el)
        data = ab_class.df.envelope[ab_class.df.envelope["element"] == el]
        data = data.query(f"Z == {m} and M_init == 2.5")
        z.append(int(np.array(data["elemental_mass"])[0]))
        x = np.array(data["massfrac"])
        els.append(x[-1] / x[0])

    zz.append(els)

labels = el_names
for i, _ in enumerate(labels):
    labels[i] = labels[i].capitalize()

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


z = 0.014

intershell_metal = intershell.query(
    f"M1tp == {2.5} and pmz == 2e-3 and last == 1"
).sort_values("ntp")


ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
z = []
els = []
final.append(ab.h.envelope[-1] / ab.h.envelope[star.ntpagb])
final.append(ab.he.envelope[-1] / ab.he.envelope[star.ntpagb])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).envelope
    e = e[-1] / e[0]
    final.append(e)
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

print(els)

plt.plot(
    z,
    np.array(final) / np.array(zz[0]),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

for i, els in enumerate(zz):
    # distance belongs to the second TP in each pair
    axs.plot(
        z,
        np.array(els) / np.array(zz[0]),
        linestyle="-",
        c=f"C{i+1}",
        linewidth=1,
        label=f"Monash $M=2.5\\;M_\\odot,\\;Z={ms[i]}$",
    )

fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel(
    "$(X_\\textrm{f} / X_\\textrm{i}) / (X_\\textrm{f} / X_\\textrm{i})_\\textrm{Monash $M=2.5\\;M_\\odot,\\;Z=0.007$}$"
)
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-change-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%

df = AbundanceTables()


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


zz = []
ms = [0.007, 0.0028]
for m_i, m in enumerate(ms):

    i1 = intershell_metal.query(f"Z == {m}").sort_values("ntp")

    tp1 = tp_info.query(f"initial_mass == {2.5} and z == {m}")

    # all abundance columns
    abundance_cols = [
        c
        for c in i1.columns
        if c not in ["M1tp", "Z", "pmz", "last", "ntp", "snapshot", "age", "Mcore"]
    ]
    abundances = i1[abundance_cols].to_numpy()

    # # abundance-pattern change between consecutive TPs
    # distances = np.array(
    #     [
    #         abundance_distance(abundances[j - 1], abundances[j])
    #         for j in range(1, len(abundances))
    #     ]
    # )
    ab = {}
    last_el = None
    el = None
    for i, abundance in enumerate(abundance_cols):
        match = re.match(r"([a-zA-Z]+)(\d+)$", abundance)
        if match:
            el = match.group(1)
            mass = int(match.group(2))
        else:
            continue

        if el == last_el:
            ab[el] += mass * np.array(i1[abundance])

        else:
            ab[el] = mass * np.array(i1[abundance])

        last_el = el

    ab_class = Abundances(None, df, method="tp offset", mass=2.5)

    els = []
    el_names = []
    z = []

    el_names.append("h")
    data = ab_class.df.envelope[ab_class.df.envelope["element"] == "h"]
    data = data.query(f"Z == {m} and M_init == 2.5")
    z.append(int(np.array(data["elemental_mass"])[0]))
    x = np.array(data["massfrac"])
    els.append(x[-1])

    for el in ab:
        if el in ["li", "be", "b", "tc", "pm", "po"]:
            continue
        el_names.append(el)
        data = ab_class.df.envelope[ab_class.df.envelope["element"] == el]
        data = data.query(f"Z == {m} and M_init == 2.5")
        z.append(int(np.array(data["elemental_mass"])[0]))
        x = np.array(data["massfrac"])
        els.append(x[-1])

    zz.append(els)

labels = el_names
for i, _ in enumerate(labels):
    labels[i] = labels[i].capitalize()

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


z = 0.014

intershell_metal = intershell.query(
    f"M1tp == {2.5} and pmz == 2e-3 and last == 1"
).sort_values("ntp")


ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
z = []
els = []
final.append(ab.h.envelope[-1])
final.append(ab.he.envelope[-1])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).envelope
    e = e[-1]
    final.append(e)
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

print(els)

plt.plot(
    z,
    np.array(final) / np.array(zz[0]),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)

for i, els in enumerate(zz):
    # distance belongs to the second TP in each pair
    axs.plot(
        z,
        np.array(els) / np.array(zz[0]),
        linestyle="-",
        c=f"C{i+1}",
        linewidth=1,
        label=f"Monash $M=2.5\\;M_\\odot,\\;Z={ms[i]}$",
    )

fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f} / X_\\textrm{f, Monash $M=2.5\\;M_\\odot,\\;Z=0.007$}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%
