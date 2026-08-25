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
profiles[0].bulk_names
# %%
for profile in profiles:
    print(profile.header_names)

    plt.plot(
        (profile.mass - profile.co_core_mass)
        / (profile.he_core_mass - profile.co_core_mass),
        profile.z_mass_fraction_metals,
    )
plt.xlim(-0.5, 1.5)
plt.show()

# %%
for profile in profiles[::100]:
    plt.plot(profile.logR, profile.pp)
    plt.plot(profile.logR, profile.tri_alpha)
plt.show()


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, profile in enumerate(profiles[100::500]):
    print(profile.header_names)

    plt.plot(
        profile.mass,
        profile.z_mass_fraction_metals,
        c=f"C{i}",
        linewidth=2,
        label=f"{profile.star_age:.1e} yr",
    )
    plt.axvline(profile.he_core_mass, c=f"C{i}", linewidth=0.75)
    plt.axvline(profile.co_core_mass, c=f"C{i}", linewidth=0.75)
fig.legend(loc="outside upper center", ncols=3)
plt.xlim(0.4, 0.7)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("internal mass $m$ ($M_\\odot$)")
plt.ylabel("$Z$ (metal mass fraction)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-show-profiles-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
(line,) = ax.plot([], [])
(line2,) = ax.plot([], [], c="C1", linewidth=3)


def update(i):
    p = profiles[i]
    x = (p.mass - p.co_core_mass) / (p.he_core_mass - p.co_core_mass)
    line.set_data(x, p.z_mass_fraction_metals)
    i = np.argmax(
        np.abs(
            np.log10(p.z_mass_fraction_metals[10:])
            - np.log10(p.z_mass_fraction_metals[:-10])
        )
    )
    ind = i + 5
    # ind = np.argmax(np.diff(p.z_mass_fraction_metals))
    line2.set_data(
        x[ind + 80 : ind + 130], p.z_mass_fraction_metals[ind + 80 : ind + 130]
    )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1e-3, 1)
    return (line, line2)


ax.set_yscale("log")

ani = FuncAnimation(fig, update, frames=len(profiles), interval=20)
plt.show()
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
plt.plot(age, z_intershell)
plt.plot(age, z_envelope)

plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$Z$ (metal mass fraction)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-z-inter+env.pgf", format="pgf")
plt.show()
plt.close()
# %%

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

z_intershell = np.array(z_intershell)
z_envelope = np.array(z_envelope)
age = np.array(age)
m_env = np.array(m_env)

# plt.plot(age,m_env)
# plt.plot(age, z_intershell)
plt.plot(age, z_envelope)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel("$Z$ (metal mass fraction)")
# plt.savefig("/home/koen/LaTeX-setup/plots/w25-z-inter+env.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


star = get_star(
    full_path="/home/koen/master-internship/mesa-models/single-stars/z0.00453/M2.0"
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

z_intershell = np.array(z_intershell)
z_envelope = np.array(z_envelope)
age = np.array(age)
m_env = np.array(m_env)

# plt.plot(age,m_env)
# plt.plot(age, z_intershell)
plt.plot(
    age[1:],
    np.cumsum(
        np.clip(np.diff(z_envelope * m_env), a_min=0, a_max=np.inf) / z_intershell[1:]
    ),
    label="Actual",
)
plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    star.m_DUP_time[0][star.ntpagb :],
    label="Post-Processing",
)


fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Age (yr)")
plt.ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-compare-dup.pgf", format="pgf")
plt.show()
plt.close()

# %%


# z_metal_env = M_Z / M_env
# z_metal_env_next = z_metal_env * M_env / (M_env + M_dup)

dup = np.interp(star.age[star.ntpagb :] - star.age[star.ntpagb], age, z_intershell)
z_metal_env = [z_envelope[1]]
for i in range(len(star.m_DUP_time) - 1):
    z_metal_env = [z_metal_env[-1]]
    # plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], star.m_DUP_time[i+1][star.ntpagb:])
    #
    #
    #
    # WHAT THE FLIP, WHY DOES 6 TIMES WORK
    #
    #
    delta_MZ = dup[1:] * np.diff(star.m_DUP_time[i + 1][star.ntpagb :])
    dup_delta = np.diff(star.m_DUP_time[i + 1][star.ntpagb :])
    for i, mz in enumerate(delta_MZ):
        if mz == 0:
            z_metal_env.append(z_metal_env[-1])
            continue
        z_metal_env.append(
            (z_metal_env[-1] * star.m_env[star.ntpagb + i] + mz)
            / (star.m_env[star.ntpagb + i] + dup_delta[i])
        )

    plt.plot(star.age[star.ntpagb :] - star.age[star.ntpagb], z_metal_env)
# plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], z_env)
# plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], )
plt.plot(age, z_envelope)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

# z_metal_env = M_Z / M_env
# z_metal_env_next = z_metal_env * M_env / (M_env + M_dup)

dup = np.interp(star.age[star.ntpagb :] - star.age[star.ntpagb], age, z_intershell)
z_metal_env = [z_envelope[1]]
# z_metal_env = [z_metal_env[-1]]
# plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], star.m_DUP_time[i+1][star.ntpagb:])

delta_MZ = dup[1:] * np.diff(star.m_DUP_time[0][star.ntpagb :])
dup_delta = np.diff(star.m_DUP_time[0][star.ntpagb :])
for i, mz in enumerate(delta_MZ):
    if mz == 0:
        z_metal_env.append(z_metal_env[-1])
        continue
    z_metal_env.append(
        (z_metal_env[-1] * star.m_env[star.ntpagb + i] + mz)
        / (star.m_env[star.ntpagb + i] + dup_delta[i])
    )

# plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], z_env)
# plt.plot(star.age[star.ntpagb:] - star.age[star.ntpagb], )
plt.plot(age, z_envelope, label="Actual")

plt.plot(
    star.age[star.ntpagb :] - star.age[star.ntpagb],
    z_metal_env,
    label="Post-Processing",
)

fig.legend(loc="outside upper center", ncols=2)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel(r"$Z_\textrm{env}$ (mass ratio of metals in envelope)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w25-compare-real-articifial.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


axs[0].plot(star.age[star.ntpagb :], star.m_env[star.ntpagb :])
axs[1].plot(star.age[star.ntpagb + 1 :], np.diff(star.m_DUP_time[0][star.ntpagb :]))

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("age")
axs[0].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[1].set_ylabel(r"$M_\textrm{DUP}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w25-m_env.pgf", format="pgf")
plt.show()
plt.close()
# %%
