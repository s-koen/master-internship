import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.plot_helpers import *
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.accretor import *
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

df = AbundanceTables()
ab = Abundances(None, df, mass=2, m_acc=1, mass_transfer_efficiency=1)
print(np.sum(ab.MS_massfrac))
# %%

print(ab.initial_envelope_abundances)
# %%
asplund = Asplund(he_method="karakas")
print(asplund.elements)


# %%
ms = ab.initial_envelope_abundances.copy()


def get_massfrac(element):
    try:
        return asplund.elements[element].massfrac
    except:
        return 1e-99


ms["massfrac"] = ms["elemental_mass"].map(get_massfrac)

print(ms)
print(ab.initial_envelope_abundances)
# %%
fig, axs = plt.subplots(
    2, 1, sharex=False, figsize=set_size(full, height=1), constrained_layout=True
)

axs[0].plot(ab.elements_mass, ms["massfrac"], linewidth=1, label="Asplund (2009)")
axs[0].plot(
    ab.elements_mass,
    ab.initial_envelope_abundances["massfrac"],
    linewidth=1,
    label="Karakas (2016) TPAGB",
)
axs[0].set_yscale("log")
axs[0].set_ylim(1e-11, 1e0)

axs[1].plot(
    ab.elements_mass,
    (ms["massfrac"] - ab.initial_envelope_abundances["massfrac"])
    / ab.initial_envelope_abundances["massfrac"],
    linewidth=1,
)
axs[1].axhline(0, c="C9", zorder=-10, linewidth=0.75)
axs[0].set_yscale("log")
axs[0].set_ylim(1e-11, 1e0)

element_labels(fig, ab.elements_mass, ab.elements_name, axs=axs[0])
element_labels(fig, ab.elements_mass, ab.elements_name, axs=axs[1])

axs[0].set_ylabel("$X$")
axs[1].set_ylabel("$(X_\\textrm{Asplund} - X_\\textrm{TPAGB,i})/X_\\textrm{TPAGB,i}$")
axs[1].set_xlabel("Elements")

axs[0].legend()

plt.savefig("/home/koen/LaTeX-setup/plots/w30-ms-vs-tpagb-ab.pgf", format="pgf")
plt.show()
plt.close()
# %%

print(np.sum(ms["massfrac"]))
print(np.sum(ab.initial_envelope_abundances["massfrac"]))
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

profiles_profiles_2 = []

for i, _ in enumerate(range(37)):
    print(i)
    profiles = []
    for j in range(1, 41):
        profiles.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/single-ms-stars-3/M{0.8+0.1*i:.1f}/LOGS/MS/profile{j}.data"
            )
        )
    profiles_profiles_2.append(profiles)
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
            profile.mass,
            profile.mu,
            c="C9",
            rasterized=True,
            linewidth=0.75 / 2,
            zorder=-10,
        )
    ax.set_title(f"$M={0.8+0.1*i:.1f}\\;M_\\odot$")

    for profile in profiles_profiles_2[i]:
        ax.plot(
            profile.mass, profile.mu, c=cmap(norm(profile.center_h1)), rasterized=True
        )
    ax.set_title(f"$M={0.8+0.1*i:.1f}\\;M_\\odot$")


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs, aspect=50)
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")

ax.set_ylim(0.598, 0.61)
ax.set_xlim(0, 1.5)

# axs.set_xscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("$m$ ($M_\\odot$)", fontsize=10)
fig.supylabel("$\\mu$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w30-mu-core-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
profiles = AccretorProfiles()

# %%
norm = plt.Normalize(0.8, 4.4)
cmap = plt.cm.viridis
# color = cmap(norm(x))

ms = np.linspace(0.8, 4.4, 100)
ages = np.logspace(6.5, 10, 100)

a_res = np.zeros((len(ms), len(ages)))

for i, m in enumerate(ms):
    print(i)
    for j, a in enumerate(ages):
        ac = Accretor(profiles=profiles, age=a, mass=m)
        mix = ac.effective_mu_vs_depth(0.5, 0.64).mixing_mass

        ac = Accretor(profiles=profiles, age=a, mass=m, new=False)
        mix2 = ac.effective_mu_vs_depth(0.5, 0.64).mixing_mass
        a_res[i, j] = mix / mix2

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(ms, np.log10(ages), a_res.T, cmap="viridis", rasterized=True)

plt.colorbar(label="$M_\\textrm{mix, Karakas (2016)} / M_\\textrm{mix, MESA}$")

plt.xlabel("$M$ ($M_\\odot$)")
plt.ylabel("Main Sequence age log$_{10}$(yr)")
plt.title("$M_\\textrm{acc} = 0.5\\;M_\\odot,\\;\\mu_\\textrm{acc} = 0.64$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w30-m_mix-grid.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.pcolormesh(ms, np.log10(ages), a_res.T, cmap="viridis", rasterized=True, vmin=0.94)

plt.colorbar(label="$M_\\textrm{mix, Karakas (2016)} / M_\\textrm{mix, MESA}$")

mline = np.logspace(np.log10(0.6), np.log10(4.4), 100)
mline_look = np.array([1, 1.45, 1.7, 2, 3])
mline_look_2 = np.array([1.5, 1.58, 1.7, 1.8, 1.9])

plt.ylim(plt.gca().get_ylim())
plt.xlim(plt.gca().get_xlim())
plt.plot(mline, 9.75 + np.log10(mline**-2.8), c="w", linewidth=1.5)
plt.plot(mline, 9.75 + np.log10(mline**-2.8), c="k", linewidth=1)
plt.plot(mline, 7 + np.log10(mline**-2.5), c="w", linewidth=1.5)
plt.plot(mline, 7 + np.log10(mline**-2.5), c="k", linewidth=1)


plt.scatter(mline_look, 9.1 + np.log10(mline_look**-2.8), c="w", linewidth=1, s=15)
plt.scatter(mline_look, 9.1 + np.log10(mline_look**-2.8), c="k", linewidth=1, s=10)
plt.scatter(
    mline_look_2, 4.41714387588 + np.log10(mline_look_2**17.5), c="w", linewidth=1, s=15
)
plt.scatter(
    mline_look_2, 4.41714387588 + np.log10(mline_look_2**17.5), c="k", linewidth=1, s=10
)

plt.annotate(
    "PMS age",
    xy=(0.9, 7.15),
    xycoords="data",
    xytext=(0, 15),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

plt.annotate(
    "MS age",
    xy=(2.22, 8.75),
    xycoords="data",
    xytext=(0, 15),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

plt.xscale("log")

plt.xlabel("$M$ ($M_\\odot$)")
plt.ylabel("Main Sequence age log$_{10}$(yr)")
plt.title("$M_\\textrm{acc} = 0.5\\;M_\\odot,\\;\\mu_\\textrm{acc} = 0.64$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-m_mix-grid-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    5,
    5,
    sharex="col",
    sharey="row",
    figsize=set_size(full, height=1),
    constrained_layout=True,
)
axs = axs.flatten()


mline_look = np.array([1, 1.45, 1.7, 2, 3])
age_look = 10 ** (9.1 + np.log10(mline_look**-2.8))

for i, (m, age) in enumerate(zip(mline_look, age_look)):

    ac = Accretor(profiles=profiles, age=age, mass=m)
    mix = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)

    axs[i + 10].plot(mix.mass_profile, mix.mu_profile_original, linewidth=1, c="C0")
    axs[i + 10].plot(mix.mass_profile, mix.mu_profile_mix, linewidth=1, c="C2")

    ac = Accretor(profiles=profiles, age=age, mass=m, new=False)
    mix = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)

    axs[i + 10].plot(mix.mass_profile, mix.mu_profile_original, linewidth=1, c="C1")
    axs[i + 10].plot(mix.mass_profile, mix.mu_profile_mix, linewidth=1, c="C3")


mline_look = np.array([1.5, 1.58, 1.7, 1.8, 1.9])[::-1]
age_look = 10 ** (4.41714387588 + np.log10(mline_look**17.5))

for i, (m, age) in enumerate(zip(mline_look, age_look)):
    if i == 2:
        continue

    ac = Accretor(profiles=profiles, age=age, mass=m)
    mix = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)

    (l1,) = axs[2 + 5 * i].plot(
        mix.mass_profile,
        mix.mu_profile_original,
        linewidth=1,
        label="$\\mu$-profile Karakas (2016)",
        c="C0",
    )
    (l2,) = axs[2 + 5 * i].plot(
        mix.mass_profile,
        mix.mu_profile_mix,
        linewidth=1,
        label="$\\mu$-mix Karakas (2016)",
        c="C2",
    )

    ac = Accretor(profiles=profiles, age=age, mass=m, new=False)
    mix = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)

    (l3,) = axs[2 + 5 * i].plot(
        mix.mass_profile,
        mix.mu_profile_original,
        linewidth=1,
        label="$\\mu$-profile MESA",
        c="C1",
    )
    (l4,) = axs[2 + 5 * i].plot(
        mix.mass_profile,
        mix.mu_profile_mix,
        linewidth=1,
        label="$\\mu$-mix MESA",
        c="C3",
    )

for i in [0, 1, 3, 4, 5, 6, 8, 9, 15, 16, 18, 19, 20, 21, 23, 24]:
    axs[i].axis("off")

fig.legend(handles=[l1, l2, l3, l4], ncols=1, loc="upper left")


from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import LinearLocator

# show y tick labels on the left-most horizontal panels
for i in range(25):
    if i not in [2, 7, 10, 17, 22]:
        axs[i].tick_params(labelbottom=False, labelleft=False)
        continue

    if i == 10:
        continue

    ax = axs[i]
    fake_ax = axs[i - 1]
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ticks = ax.get_yticks()

    ax.tick_params(labelleft=False)

    formatter = ax.yaxis.get_major_formatter()

    fake_ax.text(0.6, 0.5, "$\\mu$", transform=fake_ax.transAxes)

    for tick in ticks[1:-1]:
        label = formatter.format_data(tick)
        print(label)

        fake_ax.text(
            1.0,
            tick,
            rf"${label}$",
            transform=fake_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
        )
    # show x tick labels on the bottom-most vertical panel

for i in [10, 11, 22, 13, 14]:
    axs[i].tick_params(labelbottom=True)
    if i == 22:
        continue

    axs[i + 5].text(
        0.5,
        1,
        "$m$ ($M_\\odot$)",
        transform=axs[i + 5].transAxes,
        va="top",
        ha="center",
    )

for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-show-differences-mu-profiles.pgf", format="pgf"
)
plt.show()
plt.close()

# %%
ac = Accretor(profiles=profiles, age=10**9.5, mass=1)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

ac = Accretor(profiles=profiles, age=10**9.5, mass=1, new=False)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

plt.show()
# %%

ac = Accretor(profiles=profiles, age=10**8.5, mass=2)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

ac = Accretor(profiles=profiles, age=10**8.5, mass=2, new=False)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

plt.show()
# %%

ac = Accretor(profiles=profiles, age=10**8, mass=2)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

ac = Accretor(profiles=profiles, age=10**8, mass=2, new=False)
accres = ac.effective_mu_vs_depth(0.5, 0.64, save_profile=True)
plt.plot(accres.mass_profile, accres.mu_profile_original)
plt.plot(accres.mass_profile, accres.mu_profile_mix)

plt.show()
# %%


ress = []
norm = plt.Normalize(0.8, 4.4)
cmap = plt.cm.viridis
# color = cmap(norm(x))

size = 50
ms = np.linspace(0.8, 4.4, size)
ages = np.logspace(6.5, 10, size)

res = np.zeros((len(ms), len(ages)))

m_mixs = [0.05, 0.1, 0.25, 0.5]
mu_mixs = [0.61, 0.65, 0.70]

for m_mix in m_mixs:
    for mu_mix in mu_mixs:
        res = np.zeros((len(ms), len(ages)))
        for i, m in enumerate(ms):
            print(i)
            for j, a in enumerate(ages):
                ac = Accretor(profiles=profiles, age=a, mass=m)
                mix = ac.effective_mu_vs_depth(m_mix, mu_mix).mixing_mass

                ac = Accretor(profiles=profiles, age=a, mass=m, new=False)
                mix2 = ac.effective_mu_vs_depth(m_mix, mu_mix).mixing_mass
                res[i, j] = mix / mix2
        ress.append(res)
# %%

fig, axs = plt.subplots(
    4,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

vmin = 1e99
vmax = -1e99

for i, res in enumerate(ress):
    x = res
    if np.min(x) < vmin:
        vmin = np.min(x)
    if np.max(x) > vmax:
        vmax = np.max(x)

for i, (ax, res) in enumerate(zip(axs, ress)):
    c = ax.pcolormesh(
        ms,
        np.log10(ages),
        res.T,
        cmap="viridis",
        rasterized=True,
        vmin=vmin,
        vmax=1.01,
    )
    ax.set_title(
        f"$M_\\textrm{{acc}} = {m_mixs[i//3]}\\;M_\\odot,\\;\\mu_\\textrm{{acc}} = {mu_mixs[i%3]}$"
    )

plt.colorbar(c, ax=axs, label="$M_\\textrm{acc} / M_\\textrm{mix}$", aspect=50)
fig.supxlabel("$M$ ($M_\\odot$)", fontsize=10)
fig.supylabel("Main Sequence age log$_{10}$(yr)", fontsize=10)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-effects-of-mu-and-m_acc.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    4,
    3,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

axs = axs.flatten()

vmin = 1e99
vmax = -1e99

for i, res in enumerate(ress):
    x = res
    if np.min(x) < vmin:
        vmin = np.min(x)
    if np.max(x) > vmax:
        vmax = np.max(x)

for i, (ax, res) in enumerate(zip(axs, ress)):
    c = ax.pcolormesh(
        ms,
        np.log10(ages),
        res.T,
        cmap="viridis",
        rasterized=True,
        vmin=0.96,
        vmax=1.001,
    )
    ax.set_title(
        f"$M_\\textrm{{acc}} = {m_mixs[i//3]}\\;M_\\odot,\\;\\mu_\\textrm{{acc}} = {mu_mixs[i%3]}$"
    )

plt.colorbar(
    c, ax=axs, label="$M_\\textrm{acc} / M_\\textrm{mix}$", aspect=50, extend="both"
)
fig.supxlabel("$M$ ($M_\\odot$)", fontsize=10)
fig.supylabel("Main Sequence age log$_{10}$(yr)", fontsize=10)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-effects-of-mu-and-m_acc-colorzoom.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%
