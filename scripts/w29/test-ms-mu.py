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

    def get_mu_mass_curve(self) -> tuple[np.ndarray, np.ndarray] :

        if len(self.masses) == 1:
            mu, mass = self.get_mu_mass_curve_per_mass(self.masses[0])
            mu = mu[::-1]
            # arg = np.argmin(mu)
            # mu[:arg] = mu[arg]
            return mu[::-1] , mass[::-1]

        mu_lower, mass_lower = self.get_mu_mass_curve_per_mass(self.masses[0])
        mu_upper, mass_upper = self.get_mu_mass_curve_per_mass(self.masses[1])

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
    

    def get_mu_mass_curve_per_mass(self, mass) -> tuple[np.ndarray, np.ndarray] :
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

plt.text(0.95,0.95,r"$M=1.95\;M_\odot$", transform = axs.transAxes, ha="right",)

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

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-interpolated-mass.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

