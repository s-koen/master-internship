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
import re

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

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1.2),
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
plt.xticks(np.arange(0, 22, 2), minor=False)
plt.xticks(np.arange(1, 23, 2), minor=True)
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
axs[1].set_ylabel("Residual ($M_\\odot$)")
axs[1].set_yscale("log")
plt.savefig(
    "/home/koen/master-internship/scripts/w27/amanda-mass.png", format="png", dpi=600
)
plt.show()
plt.close()


# %%

print(intershell.columns)
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

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=set_size(column, height=1.2),
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
    pulses, Mcore_i_data, Mcore_tp_data = get_iso_and_Mdredge(None, i1, tp1)

    axs[1].plot(
        pulses,
        np.abs(Mcore_i_data - Mcore_tp_data),
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
plt.xticks(np.arange(0, int(plt.gca().get_xlim()[-1]), 2), minor=False)
plt.xticks(np.arange(1, int(plt.gca().get_xlim()[-1]) + 1, 2), minor=True)
axs[0].set_ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
axs[1].set_ylabel("Residual ($M_\\odot$)")
axs[1].set_yscale("log")
plt.savefig(
    "/home/koen/master-internship/scripts/w27/amanda-mass-2.png", format="png", dpi=600
)
plt.show()
plt.close()


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
    2, 1, sharex=True, figsize=set_size(column, height=1.2), constrained_layout=True
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
    axs[0].plot(
        i1.ntp.iloc[:], ab, "-", label=f"M = {m}", c=f"C{m_i}", linewidth=1, zorder=-m_i
    )

    axs[0].scatter(i1.ntp.iloc[:], ab, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i)

    axs[0].scatter(i1.ntp.iloc[:], ab, marker=".", s=75, c="white", zorder=10 - m_i)

    # dredge-up events
    axs[1].plot(tp1.pulse, tp1.Ddredge, c=f"C{m_i}", linewidth=1, zorder=-m_i)

    axs[1].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=25, c=f"C{m_i}", zorder=11 - m_i
    )

    axs[1].scatter(
        tp1.pulse, tp1.Ddredge, marker=".", s=75, c=f"white", zorder=10 - m_i
    )


for i in range(int(axs[0].get_xlim()[-1])):
    axs[0].axvline(i, c="C9", zorder=-1, linewidth=0.75 / 2)
    axs[1].axvline(i, c="C9", zorder=-1, linewidth=0.75 / 2)


axs[0].yaxis.set_major_formatter(formatter)

plt.yscale("log")
plt.xlabel("Thermal pulse count")
plt.xticks(np.arange(0, int(plt.gca().get_xlim()[-1]), 2), minor=False)
plt.xticks(np.arange(1, int(plt.gca().get_xlim()[-1]) + 1, 2), minor=True)
axs[0].set_ylabel(r"$\displaystyle\sum_{Z>26}X_i$")
axs[1].set_ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
fig.legend(loc="outside upper center", ncols=4)

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

plt.xlim(8)
plt.savefig(
    "/home/koen/master-internship/scripts/w27/amanda-abundance.png",
    format="png",
    dpi=600,
)
plt.show()
plt.close()
# %%
