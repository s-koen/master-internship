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

# %%

with open("data/tp_info_pd_df.pkl", "rb") as f:
    tp_info = pickle.load(f)

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.5, 3.0)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    (l1,) = plt.plot(tp1.pulse, tp1.Ddredge, c=cmap(norm(m)), label="Monash")


for m in np.arange(1.5, 3.1, 0.2)[::-1]:
    star = get_star(m=m)
    (l2,) = plt.plot(
        range(1, len(star.m_DUP) + 1),
        star.m_DUP,
        c=cmap(norm(m)),
        linestyle="--",
        zorder=-10,
        label="MESA",
    )
    # plt.plot(star.TP_count, star.lambda_DUP,linewidth=3, alpha=0.5, c=cmap(norm(m)), linestyle="--", zorder=-10)

fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial star mass ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-m-dup-per-pulse.pgf", format="pgf"
)
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.5, 3.0)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    (l1,) = plt.plot(
        tp1.pulse - np.array(tp1.pulse)[-1],
        np.cumsum(tp1.Ddredge),
        c=cmap(norm(m)),
        label="Monash",
    )


for m in np.arange(1.5, 3.1, 0.2)[::-1]:
    star = get_star(m=m)
    (l2,) = plt.plot(
        np.array(range(1, len(star.m_DUP) + 1)) - len(star.m_DUP),
        np.cumsum(star.m_DUP),
        c=cmap(norm(m)),
        linestyle="--",
        zorder=-10,
        label="MESA",
    )
    # plt.plot(star.TP_count, star.lambda_DUP,linewidth=3, alpha=0.5, c=cmap(norm(m)), linestyle="--", zorder=-10)

fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial star mass ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-m-dup-per-pulse-cum.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007

zs = [0.0028, 0.007, 0.014]
offset = [0.005, -0.005, 0]

for z_i, z in enumerate(zs):

    intershell_metal = intershell.query(
        f"Z == {z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)
    print(ms)

    norm = plt.Normalize(1.5, 3.0)
    cmap = plt.cm.viridis
    # color = cmap(norm(x))

    m_dup_monash = []
    for m_i, m in enumerate(ms[::-1]):

        i1 = intershell_metal.query(
            f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
        )
        tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
        m_dup_monash.append(np.sum(tp1.Ddredge))

    plt.plot(ms[::-1], m_dup_monash)
    plt.text(
        ms[::-1][0] + 0.05,
        m_dup_monash[0] + offset[z_i],
        f"Monash $Z={z:.4f}$",
        va="center",
        ha="left",
        size=8,
    )


m_dup_mesa = []
ms = np.arange(1, 3.1, 0.1)[::-1]
for m in ms:
    star = get_star(m=m)
    m_dup_mesa.append(star.m_DUP_time[0][-1])

plt.plot(ms, m_dup_mesa)
plt.text(
    ms[0] + 0.05, m_dup_mesa[0], f"MESA $Z=0.00557$", va="center", ha="left", size=8
)
# fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

plt.xlim(0.9, 4)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star mass ($M_\\odot$)")
plt.ylabel("$\sum M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-m-dup-per-mass.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.0028
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.1, 2.75)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    (l1,) = plt.plot(tp1.pulse, tp1.Ddredge, c=cmap(norm(m)), label="Monash")


for m in np.arange(1.1, 2.8, 0.2)[::-1]:
    star = get_star(m=m)
    (l2,) = plt.plot(
        range(1, len(star.m_DUP) + 1),
        star.m_DUP,
        c=cmap(norm(m)),
        linestyle="--",
        zorder=-10,
        label="MESA",
    )
    # plt.plot(star.TP_count, star.lambda_DUP,linewidth=3, alpha=0.5, c=cmap(norm(m)), linestyle="--", zorder=-10)

fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial star mass ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-m-dup-per-pulse-low-Z.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007

zs = [0.0028, 0.007, 0.014]
offset = [0.002, 0.002, -0.002]

for z_i, z in enumerate(zs):

    intershell_metal = intershell.query(
        f"Z == {z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)
    print(ms)

    norm = plt.Normalize(1.5, 3.0)
    cmap = plt.cm.viridis
    # color = cmap(norm(x))

    m_dup_monash = []
    for m_i, m in enumerate(ms[::-1]):

        i1 = intershell_metal.query(
            f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
        )
        tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
        # print(tp1.columns)
        m_dup_monash.append(list(tp1.Mcore)[-1])

    plt.plot(ms[::-1], m_dup_monash)
    plt.text(
        ms[::-1][0] + 0.05,
        m_dup_monash[0] + offset[z_i],
        f"Monash $Z={z:.4f}$",
        va="center",
        ha="left",
        size=8,
    )


m_dup_mesa = []
ms = np.arange(1, 3.1, 0.1)[::-1]
for m in ms:
    star = get_star(m=m)
    m_dup_mesa.append(star.m_core[-1])

plt.plot(ms, m_dup_mesa)
plt.text(
    ms[0] + 0.05, m_dup_mesa[0], f"MESA $Z=0.00557$", va="center", ha="left", size=8
)
# fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

plt.xlim(0.9, 4)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star mass ($M_\\odot$)")
plt.ylabel("$M_\\textrm{core,f}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-mcore.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007

zs = [0.0028, 0.007, 0.014]
offset = [0.002, 0.002, -0.002]

for z_i, z in enumerate(zs):

    intershell_metal = intershell.query(
        f"Z == {z} and pmz == 2e-3 and last == 1"
    ).sort_values("ntp")

    ms = np.unique(intershell_metal.M1tp)
    print(ms)

    norm = plt.Normalize(1.5, 3.0)
    cmap = plt.cm.viridis
    # color = cmap(norm(x))

    m_dup_monash = []
    for m_i, m in enumerate(ms[::-1]):

        i1 = intershell_metal.query(
            f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
        )
        tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
        # print(tp1.columns)
        m_dup_monash.append(list(tp1.Mcore)[0])

    plt.plot(ms[::-1], m_dup_monash)
    plt.text(
        ms[::-1][0] + 0.05,
        m_dup_monash[0] + offset[z_i],
        f"Monash $Z={z:.4f}$",
        va="center",
        ha="left",
        size=8,
    )


m_dup_mesa = []
ms = np.arange(1, 3.1, 0.1)[::-1]
for m in ms:
    star = get_star(m=m)
    m_dup_mesa.append(star.m_core[star.ntpagb])

plt.plot(ms, m_dup_mesa)
plt.text(
    ms[0] + 0.05, m_dup_mesa[0], f"MESA $Z=0.00557$", va="center", ha="left", size=8
)
# fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

plt.xlim(0.9, 4)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star mass ($M_\\odot$)")
plt.ylabel("$M_\\textrm{core,i}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-compare-initial-mcore.pgf", format="pgf")
plt.show()
plt.close()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.007
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.4, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    (l1,) = plt.plot(tp1.pulse, tp1.Mcore, c=cmap(norm(m)), label="Monash")


for m in np.arange(1.4, 3.1, 0.2)[::-1]:
    star = get_star(m=m)

    c_masses = []

    tps = range(1, int(star.TP_count[-1]) + 1)
    for tp in tps:
        core_mass = np.where(np.abs(star.TP_count - tp) < 0.001, star.m_core, np.nan)
        max_core_mass = np.nanmax(core_mass)
        c_masses.append(max_core_mass)

    (l2,) = plt.plot(
        tps,
        c_masses,
        c=cmap(norm(m)),
        linestyle="--",
        zorder=-10,
        label="MESA",
    )
    # plt.plot(star.TP_count, star.lambda_DUP,linewidth=3, alpha=0.5, c=cmap(norm(m)), linestyle="--", zorder=-10)

fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial star mass ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$M_\\textrm{core}$ ($M_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-core-mass-evolution.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

z = 0.0028
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.4, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    print(tp1.columns)
    (l1,) = plt.plot(
        tp1.pulse,
        tp1["lambda"],
        c=cmap(norm(m)),
        label="Monash ($Z=0.0028$)",
        linestyle="-.",
    )

z = 0.007
intershell_metal = intershell.query(
    f"Z == {z} and pmz == 2e-3 and last == 1"
).sort_values("ntp")

ms = np.unique(intershell_metal.M1tp)
print(ms)

norm = plt.Normalize(1.4, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m_i, m in enumerate(ms[::-1]):

    i1 = intershell_metal.query(
        f"M1tp == {m} and Z == {z} and pmz == 2e-3 and last == 1"
    )
    tp1 = tp_info.query(f"initial_mass == {m} and z == {z}")
    print(tp1.columns)
    (l1,) = plt.plot(tp1.pulse, tp1["lambda"], c=cmap(norm(m)), label="Monash")


for m in np.arange(1.4, 3.1, 0.2)[::-1]:
    star = get_star(m=m)

    c_masses = []

    tps = range(1, int(star.TP_count[-1]) + 1)
    for tp in tps:
        core_mass = np.where(
            np.abs(star.TP_count - tp) < 0.001, star.lambda_DUP, np.nan
        )
        max_core_mass = np.nanmax(core_mass)
        c_masses.append(max_core_mass)

    (l2,) = plt.plot(
        tps,
        c_masses,
        c=cmap(norm(m)),
        linestyle="--",
        zorder=-10,
        label="MESA",
    )
    # plt.plot(star.TP_count, star.lambda_DUP,linewidth=3, alpha=0.5, c=cmap(norm(m)), linestyle="--", zorder=-10)

fig.legend(loc="outside upper center", handles=[l1, l2], ncols=2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial star mass ($M_\odot$)")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$M_\\textrm{DUP}$ ($M_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w27-compare-lambda-evolution.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
