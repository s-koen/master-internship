import numpy as np
from numpy.typing import NDArray
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

for m in np.arange(1, 3, 0.1):
    star = get_star(m=m)
    plt.plot(star.age, star.mu)

plt.show()
# %%

data = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M1.0/LOGS/MS/profile1.data"
)
data.bulk_names
# %%

print(np.logspace(-4, np.log10(0.7), 50))
# %%

profiles = []
for i in range(1, 41):
    profiles.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-ms-stars/M1.9/LOGS/MS/profile{i}.data"
        )
    )
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


print(profile.header_names)

for profile in profiles:
    axs.plot(
        -1 * (profile.mass - profile.mass[0]),
        profile.mu,
        c=cmap(norm(profile.center_h1)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")
axs.text(0.05, 0.95, "$M=1.9\\;M_\\odot$", transform=axs.transAxes)

axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{star} - m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-env-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


print(profile.header_names)

for profile in profiles:
    axs.plot(profile.mass, profile.mu, c=cmap(norm(profile.center_h1)))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")
axs.text(0.95, 0.95, "$M=1.9\\;M_\\odot$", transform=axs.transAxes, ha="right")

# axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-core-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
profiles_profiles = []

for i, _ in enumerate(range(37)):
    print(i)
    profiles = []
    for j in range(1, 41):
        profiles.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/single-ms-stars/M{0.8+0.1*i:.1f}/LOGS/MS/profile{j}.data"
            )
        )
    profiles_profiles.append(profiles)
# %%

fig, axs = plt.subplots(
    6,
    6,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))

axs = axs.flatten()
for i, ax in enumerate(axs):

    for profile in profiles_profiles[i]:
        ax.plot(
            profile.mass, profile.mu, c=cmap(norm(profile.center_h1)), rasterized=True
        )
    ax.set_title(f"$M={0.8+0.1*i:.1f}\\;M_\\odot$")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs, aspect=50)
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")

ax.set_ylim(0.598, 0.64)
ax.set_xlim(0, 1.5)

# axs.set_xscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("$M_\\textrm{star} - m$ ($M_\\odot$)", fontsize=10)
fig.supylabel("$\\mu$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-core-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

import periodictable as pt

# %%
for Z in range(1, 10):
    element = pt.elements[Z]
    print(element.symbol, element.mass)

# %%
with open("data/tp_info_pd_df.pkl", "rb") as f:
    tp_info = pickle.load(f)

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)
# %%

df = AbundanceTables()

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
initial = []
z = []
els = []
final.append(ab.h.envelope[-1])
initial.append(ab.h.envelope[0])
final.append(ab.he.envelope[-1])
initial.append(ab.he.envelope[0])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).envelope
    final.append(e[-1])
    initial.append(e[0])
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

final = final / np.sum(final)
initial = initial / np.sum(initial)

plt.plot(
    z,
    np.array(final),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)

plt.plot(
    z,
    np.array(initial),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)

ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)


fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f} / X_\\textrm{f, Monash $M=2.5\\;M_\\odot,\\;Z=0.007$}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mu_inv = []

for z_i, x_i in zip(z, final):
    element = pt.elements[z_i]
    mu_inv.append(x_i * (1 + z_i) / element.mass)

plt.plot(1 / np.cumsum(mu_inv), label="Final envelope abundances")

mu_inv = []
for z_i, x_i in zip(z, initial):
    element = pt.elements[z_i]
    mu_inv.append(x_i * (1 + z_i) / element.mass)

plt.plot(1 / np.cumsum(mu_inv), label="Initial envelope abundances")

fig.legend(loc="outside upper center", ncols=2)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$Z$")
plt.ylabel("$\sum_{i}1/\mu_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-mu-approx-1.pgf", format="pgf")
plt.show()
plt.close()


# %%


def effective_mu_vs_depth(profile, M_acc, mu_acc):
    ms = -np.diff(profile.mass)
    mus = (profile.mu[1:] + profile.mu[:-1]) / 2
    arg = np.argmin(mus)
    mus[:arg] = mus[arg]
    mu_current = mu_acc
    m_current = M_acc

    mu_profile = [mu_acc]

    crossing_index = None
    for i, (m, mu) in enumerate(zip(ms, mus)):
        mu_last = mu_current
        mu_current = (m + m_current) / ((m / mu) + (m_current / mu_current))
        m_current = m + m_current
        mu_profile.append(mu_current)
        if mu_current < mu_last:
            crossing_index = i + 3

    mus = np.concatenate([[mus[0], mus[0]], mus])
    mu_profile_original = mus
    mu_profile = np.concatenate([[mu_acc], mu_profile])
    params = {}
    params["index"] = crossing_index
    mass = np.concatenate([[profile.mass[0] + M_acc], profile.mass])

    return mass, mu_profile, mu_profile_original, params


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


profile = profiles[-10]

for profile in profiles:
    mass, mu_profile, mu_profile_original, params = effective_mu_vs_depth(
        profile, 0.5, 0.64
    )
    plt.plot(
        mass[: params["index"]],
        mu_profile[: params["index"]],
        color=cmap(norm(profile.center_h1)),
        linewidth=1,
    )
    plt.plot(
        mass[1:],
        mu_profile_original[1:],
        c="C9",
        alpha=0.5,
        zorder=-10,
        linewidth=1,
    )

axs.axvspan(mass[0], mass[1], alpha=0.2, color="C9")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")

plt.text(1.975, 0.6175, "Accreted\nmaterial", fontsize=8)
plt.annotate(
    "Mixing",
    (1.25, 0.5975),
    (1.55, 0.597),
    arrowprops=dict(arrowstyle="-|>", linewidth=0.75, color="k"),
    fontsize=8
)
plt.ylim(0.595, 0.645)
plt.xlim(plt.gca().get_xlim()[0], mass[0])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-test-mu-mixing.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


profile = profiles[-10]

for profile in profiles:
    mass, mu_profile, mu_profile_original, params = effective_mu_vs_depth(
        profile, 0.025, 0.64
    )
    plt.plot(
        mass[: params["index"]],
        mu_profile[: params["index"]],
        color=cmap(norm(profile.center_h1)),
        linewidth=1,
    )
    plt.plot(
        mass[1:],
        mu_profile_original[1:],
        c="C9",
        alpha=0.5,
        zorder=-10,
        linewidth=1,
    )

axs.axvspan(mass[0], mass[1], alpha=0.2, color="C9")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")

plt.ylim(0.595, 0.645)
plt.xlim(plt.gca().get_xlim()[0], mass[0])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-test-mu-mixing-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

mixing mass = np.linspace(0,1,100)


# %%

for profile in profiles_profiles[0]:
    print(profile.star_age, profile.center_h1)
# %%

class AccretionResult:
    def __init__(self, index, mixing_mass, final_mu) -> None:
        self.index = index
        self.mixing_mass = mixing_mass
        self.final_mu = final_mu
        self.mass_profile: None | NDArray[np.float64] = None
        self.mu_profile_mix:None | NDArray[np.float64] = None
        self.mu_profile_original: None | NDArray[np.float64] = None

    def add_integration_result(self, mass: NDArray[np.float64], mu_profile: NDArray[np.float64], mu_profile_original: NDArray[np.float64]) -> None:
        self.mass_profile = mass
        self.mu_profile_mix = mu_profile
        self.mu_profile_original = mu_profile_original


class AccretorProfile:
    def __init__(self, profile: mr.MesaData) -> None:
        self.center_h1 = float(profile.center_h1)
        self.age = float(profile.star_age)
        self.profile = profile
        self.mass = np.array(profile.mass,dtype=np.float64)
        self.mu = np.array(profile.mu,dtype=np.float64)
        arg = np.argmin(self.mu)
        self.mu[:arg] = self.mu[arg]


class AccretorProfiles:
    def __init__(self) -> None:
        self.profiles = self.__get_profiles()

    def __get_profiles(self, fresh = False) -> dict[float, list[AccretorProfile]]:
        if fresh:
            print("In Accretor.__get_profiles:\n\tloading profiles")
            profiles_dict: dict[float, list[AccretorProfile]] = {} 
            for i, _ in enumerate(range(37)):
                profiles: list[AccretorProfile] = []
                mass = np.round(0.8+0.1*i,1)
                print(f"\tloading mass {mass}")
                for j in range(1, 41):
                    profile = mr.MesaData(
                            f"/home/koen/master-internship/mesa-models/single-ms-stars/M{mass}/LOGS/MS/profile{j}.data"
                        )
                    profiles.append(AccretorProfile(profile))
                profiles_dict[mass] = profiles

            with open(
                f"/home/koen/master-internship/data/accretor-cache/profiles.pkl",
                "wb",
            ) as f:
                pickle.dump(profiles_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            return profiles_dict


        else:
            try:
                with open(
                f"/home/koen/master-internship/data/accretor-cache/profiles.pkl",
                "rb",
            ) as f:
                    return pickle.load(f)

            except FileNotFoundError:
                return self.__get_profiles(fresh = True)


class Accretor:
    def __init__(self, profiles:AccretorProfiles,age: float, mass: float) -> None:
        self.mass = mass
        self.age  = age

        self.profiles = profiles.profiles
        self.masses = self.__determine_profile_masses()
        self.central_h1 = self.__get_center_h1()
        self.mu_curve, self.mass_curve = self.__get_mu_mass_curve()
    
    def __get_center_h1(self) -> float:
        interp_ages: list[float] = []
        for mass in self.masses:
            mass_profiles: list[AccretorProfile] = self.profiles[mass]
            h1s: list[float] = []
            ages: list[float] = []
            for profile in mass_profiles:
                h1s.append(profile.center_h1)
                ages.append(profile.age)

            interp_ages.append(np.interp(self.age, ages, h1s))

        return np.interp(self.mass, self.masses, interp_ages)

    def __determine_profile_masses(self) -> list[float]:
        masses: list[float] = []

        profile_masses: list[float] = list(self.profiles.keys())

        x: float = self.mass
        i: np.intp = np.searchsorted(profile_masses, x)

        lower: None | float
        upper: None | float

        if i == 0:
            lower = None
            upper = profile_masses[0]
        elif i == len(profile_masses):
            lower = profile_masses[-1]
            upper = None
        else:
            lower = profile_masses[i - 1]
            upper = profile_masses[i]

        if lower != None:
            masses.append(lower)
        if upper != None:
            masses.append(upper)

        return masses

    def __get_mu_mass_curve(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] :

        if len(self.masses) == 1:
            mu, mass = self.__get_mu_mass_curve_per_mass(self.masses[0])
            mu = mu[::-1]
            # arg = np.argmin(mu)
            # mu[:arg] = mu[arg]
            return mu , mass[::-1]

        mu_lower, mass_lower = self.__get_mu_mass_curve_per_mass(self.masses[0])
        mu_upper, mass_upper = self.__get_mu_mass_curve_per_mass(self.masses[1])

        mass = np.unique(np.concatenate([mass_lower, mass_upper]))
        mu_lower = np.interp(mass, mass_lower, mu_lower)
        mu_upper = np.interp(mass, mass_upper, mu_upper)

        # Interpolation fraction in central H
        f = (self.mass - self.masses[0]) / (self.masses[1] - self.masses[0])

        # Interpolate evolutionary state
        mu = (1 - f) * mu_lower + f * mu_upper
        mu = mu[::-1]
        # arg = np.argmin(mu)
        # mu[:arg] = mu[arg]
        return mu, mass[::-1]
    
    def __get_mu_mass_curve_per_mass(self, mass) -> tuple[NDArray[np.float64], NDArray[np.float64]] :
        surrounding_profiles: list[AccretorProfile] = []

        profiles = self.profiles[mass][::-1]
        all_center_h1s: list[float] = []
        for profile in profiles:
            all_center_h1s.append(profile.center_h1)

        x: float = self.central_h1
        i: np.intp = np.searchsorted(all_center_h1s, x)

        lower: None | AccretorProfile
        upper: None | AccretorProfile

        if i == 0:
            lower = None
            upper = profiles[0]
        elif i == len(all_center_h1s):
            lower = profiles[-1]
            upper = None
        else:
            lower = profiles[i - 1]
            upper = profiles[i]

        if lower != None:
            surrounding_profiles.append(lower)
        if upper != None:
            surrounding_profiles.append(upper)

        if len(surrounding_profiles) == 1:
            profile = surrounding_profiles[0]
            return np.array(profile.mu[::-1]), np.array(profile.mass[::-1])


        mass_lower = np.array(surrounding_profiles[0].mass)[::-1]
        mu_lower = np.array(surrounding_profiles[0].mu)[::-1]
        mass_upper = np.array(surrounding_profiles[1].mass)[::-1]
        mu_upper = np.array(surrounding_profiles[1].mu)[::-1]
        mass = np.unique(np.concatenate([mass_lower, mass_upper]))

        mu_lower = np.interp(mass, mass_lower, mu_lower)
        mu_upper = np.interp(mass, mass_upper, mu_upper)

        # Interpolation fraction in central H
        f = (self.central_h1 - surrounding_profiles[0].center_h1) / (surrounding_profiles[1].center_h1 - surrounding_profiles[0].center_h1)

        # Interpolate evolutionary state
        mu = (1 - f) * mu_lower + f * mu_upper

        return mu, mass

    def effective_mu_vs_depth(self, M_acc: float, mu_acc: float, save_profile: bool = False):
        ms = self.mass_curve

        ms = -np.diff(self.mass_curve)
        mus = (self.mu_curve[1:] + self.mu_curve[:-1]) / 2
        mu_current = mu_acc
        m_current = M_acc

        mu_profile = [mu_acc]

        crossing_index: int = 0
        for i, (m, mu) in enumerate(zip(ms, mus)):
            mu_last = mu_current
            mu_current = (m + m_current) / ((m / mu) + (m_current / mu_current))
            m_current = m + m_current
            mu_profile.append(mu_current)
            if mu < mu_current:
                crossing_index = np.min([i + 3, len(mus)])
        mu_profile = np.array(mu_profile, dtype=np.float64)

        mu_start = np.array([mus[0], mus[0]], dtype=np.float64)
        mu_profile_original = np.concatenate([mu_start, mus])
        mu_profile = np.concatenate([[mu_acc], mu_profile])
        mass = np.concatenate([[self.mass_curve[0] + M_acc], self.mass_curve])
        result = AccretionResult(crossing_index, self.mass_curve[0] -mass[crossing_index], mu_profile[crossing_index])

        if save_profile:
            result.add_integration_result(mass, mu_profile, mu_profile_original)

        return result




# %%

profiles = AccretorProfiles()
# %%
fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

norm = plt.Normalize(np.log10(5e7), np.log10(1.1e9))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i, m in enumerate(np.logspace(np.log10(1e6),np.log10(1.1e9),5000)):
    ac = Accretor(profiles = profiles, age = m, mass = 1.8)
    mu, mass = ac.get_mu_mass_curve()
    plt.plot(mass, mu, c=cmap(norm(np.log10(m))),linewidth=1.5, rasterized=True)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Main Sequence age log$_{10}$(yr)")

plt.text(0.95,0.95,r"$M=1.8\;M_\odot$", transform = axs.transAxes, ha="right",
         va="top")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-interpolated-age.pgf", format="pgf",
            dpi=600)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

norm = plt.Normalize(np.log10(5e7), np.log10(1.1e9))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i, m in enumerate(np.logspace(np.log10(1e6),np.log10(1.1e9),5000)):
    ac = Accretor(profiles = profiles, age = m, mass = 1.8)
    mu, mass = ac.get_mu_mass_curve()
    plt.plot(mass, mu, c=cmap(norm(np.log10(m))),linewidth=1.5, rasterized=True)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Main Sequence age log$_{10}$(yr)")

plt.text(0.95,0.95,r"$M=1.8\;M_\odot$", transform = axs.transAxes, ha="right",
         va="top")

plt.ylim(0.597, 0.65)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-interpolated-age-zoom.pgf", format="pgf",
            dpi=600)
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

norm = plt.Normalize(0.8, 4.4)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for i, m in enumerate(np.linspace(0.8,4.4,2000)):
    ac = Accretor(profiles = profiles, age = 6e8, mass = m)
    mu, mass = ac.get_mu_mass_curve()
    plt.plot(mass, mu, c=cmap(norm(m)),linewidth=3, rasterized=True)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$M$ ($M_\odot$)")

plt.text(0.95,0.95,r"$t=600\;\textrm{Myr}$", transform = axs.transAxes, ha="right",
         va="top")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-interpolated-mass.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

norm = plt.Normalize(0.8, 4.4)
cmap = plt.cm.viridis
# color = cmap(norm(x))

ms = np.linspace(0.8,4.4,100)
ages = np.logspace(6.5,10,100)

res = np.zeros((len(ms), len(ages)))

for i, m in enumerate(ms):
    print(i)
    for j, a in enumerate(ages):
        ac = Accretor(profiles = profiles, age = a, mass = m)
        mix = ac.effective_mu_vs_depth(0.5,0.64).mixing_mass
        res[i,j] = mix
# %%

fig, axs = plt.subplots(1, 1, sharex=True, figsize=set_size(column), constrained_layout=True)

plt.pcolormesh(ms, np.log10(ages), res.T, cmap="viridis", rasterized=True)

plt.colorbar(label="$M_\\textrm{mix}$ ($M_\\odot$)")

plt.xlabel("$M$ ($M_\\odot$)")
plt.ylabel("Main Sequence age log$_{10}$(yr)")
plt.title("$M_\\textrm{acc} = 0.5\\;M_\\odot,\\;\\mu_\\textrm{acc} = 0.64$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-m_mix-grid.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

ress = []
norm = plt.Normalize(0.8, 4.4)
cmap = plt.cm.viridis
# color = cmap(norm(x))

ms = np.linspace(0.8,4.4,50)
ages = np.logspace(6.5,10,50)

res = np.zeros((len(ms), len(ages)))

m_mixs = [0.05, 0.1,0.25,0.5]
mu_mixs = [0.61, 0.65, 0.70]

for m_mix in m_mixs:
    for mu_mix in mu_mixs:
        for i, m in enumerate(ms):
            print(i)
            for j, a in enumerate(ages):
                ac = Accretor(profiles = profiles, age = a, mass = m)
                mix = ac.effective_mu_vs_depth(m_mix,mu_mix).mixing_mass
                res[i,j] = mix
        ress.append(res)
# %%

fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=set_size(full, height=1), constrained_layout=True)

axs = axs.flatten()

vmin= 1e99
vmax = -1e99

for i, res in enumerate(ress):
    x = np.log10(m_mixs[i//3] / res)
    if np.min(x) < vmin:
        vmin = np.min(x)
    if np.max(x) > vmax:
        vmax = np.max(x)

for i, (ax, res) in enumerate(zip(axs, ress)):
    c = ax.pcolormesh(ms, np.log10(ages), np.log10(m_mixs[i//3] / res.T) , cmap="viridis", rasterized=True, vmin=vmin,
                   vmax=vmax)
    ax.set_title(f"$M_\\textrm{{acc}} = {m_mixs[i//3]}\\;M_\\odot,\\;\\mu_\\textrm{{acc}} = {mu_mixs[i%3]}$")

plt.colorbar(c, ax=axs, label="$M_\\textrm{acc} / M_\\textrm{mix}$", aspect=50)
fig.supxlabel("$M$ ($M_\\odot$)", fontsize=10)
fig.supylabel("Main Sequence age log$_{10}$(yr)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w29-effects-of-mu-and-m_acc.pgf", format="pgf")
plt.show()
plt.close()

# %%

ms = np.linspace(0.8,4.4,200)
ages = np.logspace(6.5,10,200)

res = np.zeros((len(ms), len(ages)))

for i, m in enumerate(ms):
    print(i)
    for j, a in enumerate(ages):
        ac = Accretor(profiles = profiles, age = a, mass = m)
        res[i,j] = ac.central_h1

fig, axs = plt.subplots(2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True)

c = axs[0].pcolormesh(ms, np.log10(ages), res.T , cmap="viridis", rasterized=True)

plt.colorbar(c, ax=axs[0], label="$X(\\textrm{H})_\\textrm{center}$", aspect=20)

axs[0].set_ylim(axs[0].get_ylim())
axs[0].plot(ms, np.log10(10**9.75*ms**-2.8), c="k", linewidth=1)

ms = np.linspace(0.8,4.4,200)
ages = np.logspace(6.5,10,200)

res = np.zeros((len(ms), len(ages)))

for i, m in enumerate(ms):
    print(i)
    for j, a in enumerate(ages):
        ac = Accretor(profiles = profiles, age = a, mass = m)
        res[i,j] = ac.central_h1

res = np.max(res) - res + 1e-20

c = axs[1].pcolormesh(ms, np.log10(ages), np.log10(res.T) , cmap="viridis", rasterized=True, vmin=-4)
axs[1].set_ylim(axs[1].get_ylim())
axs[1].plot(ms, np.log10(10**9.75*ms**-2.8), c="k", linewidth=1)

plt.colorbar(c, ax=axs[1], label="$\\textrm{log}_{10}[X(H)_\\textrm{center,i} - X(H)_\\textrm{center}]$", aspect=20)

fig.supxlabel("$M$ ($M_\\odot$)", fontsize=10)
axs[0].set_ylabel("Main Sequence age log$_{10}$(yr)", fontsize=10)
axs[1].set_ylabel("Main Sequence age log$_{10}$(yr)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w29-central-h1.pgf", format="pgf")
plt.show()
plt.close()
# %%

