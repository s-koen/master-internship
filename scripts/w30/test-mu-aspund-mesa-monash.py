import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import periodictable as pt
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

# %%

Z_sun = 0.0142
ZS = np.logspace(np.log10(1 / 100), np.log10(3), 100)
mus = []
for Z in ZS:

    asplund = Asplund(z=Z * Z_sun)

    mu_inv = 0
    for i, (key, ab) in enumerate(asplund.elements.items()):
        X_i = ab.massfrac
        Z_i = ab.Z
        A_i = ab.atomic_mass
        mu_inv += X_i * (1 + Z_i) / A_i
    mu = 1 / mu_inv
    mus.append(mu)

plt.plot(ZS, mus)
plt.show()


# %%


class MonashModel:
    def __init__(self, M, Z, pulses, m_dup, intershell_isos, envelope_abundance):
        self.M = M
        self.Z = Z
        self.pulses = pulses
        self.m_dup = m_dup
        self.intershell_isos = intershell_isos
        self.envelope_abundance = envelope_abundance

        index = np.where(self.m_dup > -4.5)[0][0]
        self.pulses_offset = self.pulses - self.pulses[index]


def _get_initial_envelope_abundance(Z, M, monash_models):

    if Z in [0.0028, 0.007, 0.014]:
        return _prepare_initial_envelope_abundance_Z(Z, M, monash_models)

    if Z <= 0.0028:
        return _prepare_initial_envelope_abundance_Z(0.0028, M, monash_models)

    if Z >= 0.014:
        return _prepare_initial_envelope_abundance_Z(0.014, M, monash_models)

    if Z <= 0.007:
        z_min = 0.0028
        z_max = 0.007
    else:
        z_min = 0.007
        z_max = 0.014

    abundance_min = _prepare_initial_envelope_abundance_Z(z_min, M, monash_models)
    abundance_max = _prepare_initial_envelope_abundance_Z(z_max, M, monash_models)

    abundance_min = abundance_min.set_index("element")
    abundance_max = abundance_max.set_index("element")

    abundance = abundance_min.copy()

    weight = (Z - z_min) / (z_max - z_min)
    abundance["massfrac"] = abundance_min["massfrac"] + weight * (
        abundance_max["massfrac"] - abundance_min["massfrac"]
    )
    abundance["massfrac"] = abundance["massfrac"] / sum(abundance["massfrac"])
    abundance = abundance.reset_index()
    return abundance


def _prepare_initial_envelope_abundance_Z(Z, M, monash_models):
    if len(monash_models[Z]) == 1:
        abundance = monash_models[Z][0].envelope_abundance
        return abundance

    abundance_min = monash_models[Z][0].envelope_abundance
    mass_min = monash_models[Z][0].M
    abundance_max = monash_models[Z][1].envelope_abundance
    mass_max = monash_models[Z][1].M

    abundance_min = abundance_min.set_index("element")
    abundance_max = abundance_max.set_index("element")

    abundance = abundance_min.copy()

    weight = (M - mass_min) / (mass_max - mass_min)
    abundance["massfrac"] = abundance_min["massfrac"] + weight * (
        abundance_max["massfrac"] - abundance_min["massfrac"]
    )
    abundance["massfrac"] = abundance["massfrac"] / sum(abundance["massfrac"])
    abundance = abundance.reset_index()
    return abundance


def _get_monash_masses_per_metallicity() -> dict:
    try:
        with open(
            f"/home/koen/master-internship/data/intershell-cache/MZ.pkl",
            "rb",
        ) as f:
            mass_Z_dict = pickle.load(f)
            return mass_Z_dict
    except FileNotFoundError:
        raise Exception("oof")


def _prepare_monash_models(Z, mass, monash_models):
    """
    this method combines the Monash Isotope dataset and tp-info dataset and saves it as a collection of simple dicts.
    it caches the result on disk using pickle.

    the dicts contain:
        M:     (float)
        Z:     (float)
        TPs:   numpy 1D array(floats)
        M_dup: numpy 1D array(floats)
        isos:  328 x numpy 1D array(floats)

    """
    M = mass
    mass_Z_dict = _get_monash_masses_per_metallicity()

    if Z in [0.0028, 0.007, 0.014]:
        masses = mass_Z_dict[Z]

        x = M
        i = np.searchsorted(masses, x)

        if i == 0:
            lower = None
            upper = masses[0]
        elif i == len(masses):
            lower = masses[-1]
            upper = None
        else:
            lower = masses[i - 1]
            upper = masses[i]

        if lower != None:
            monash_models[Z].append(_prepare_single_monash_model(lower, Z))
        if upper != None:
            monash_models[Z].append(_prepare_single_monash_model(upper, Z))

    elif Z < 0.007:
        _prepare_monash_models(mass=mass, Z=0.007, monash_models=monash_models)
        _prepare_monash_models(mass=mass, Z=0.0028, monash_models=monash_models)
    else:
        _prepare_monash_models(mass=mass, Z=0.014, monash_models=monash_models)
        _prepare_monash_models(Z=0.007, mass=mass, monash_models=monash_models)

    return monash_models


def _prepare_single_monash_model(M, Z, fresh=False):

    if not fresh:
        try:
            with open(
                f"/home/koen/master-internship/data/intershell-cache/M{M:.3f}Z{Z:.4f}.pkl",
                "rb",
            ) as f:
                monash_model = pickle.load(f)
                return monash_model
        except FileNotFoundError:
            return _prepare_single_monash_model(M, Z, True)

    else:
        intershell = df.intershell.query(
            f"Z == {Z} and pmz == 2e-3 and last == 1 and M1tp == {M}"
        ).sort_values("ntp")

        envelope = df.envelope.query(
            f"Z == {Z} and pmz == 2e-3 and N_ov != 0.0 and M_init == {M} and ntp == 1"
        )

        tp_info = df.tp.query(f"initial_mass == {M} and z == {Z}")

        intershell = intershell[intershell.ntp.isin(tp_info.pulse)]

        # one intershell abundance per pulse
        i_data = intershell.drop_duplicates("ntp").set_index("ntp")

        # one dredge-up mass per pulse
        tp_data = tp_info.drop_duplicates("pulse").set_index("pulse")

        # common pulses, in ascending order
        pulses = i_data.index.intersection(tp_data.index).sort_values()

        Mdredge = tp_data.loc[pulses, "Ddredge"].to_numpy()
        Mdredge = np.log10(np.cumsum(Mdredge) + 1e-12)

        intershell_abundance = np.log10(i_data.iloc[:, 7:].clip(lower=1e-99))
        envelope_abundance = envelope[["element", "massfrac", "elemental_mass"]]

        monash_model = MonashModel(
            M, Z, pulses, Mdredge, intershell_abundance, envelope_abundance
        )

        with open(
            f"/home/koen/master-internship/data/intershell-cache/M{M:.3f}Z{Z:.4f}.pkl",
            "wb",
        ) as f:
            pickle.dump(monash_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return monash_model


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

df = AbundanceTables()
mus = []

for mass in [2]:
    Z_sun = 0.014
    ZS = np.logspace(np.log10(0.0028 / Z_sun), np.log10(1), 100)
    mus = []

    for Z in ZS:

        monash_models = defaultdict(list)
        monash_models = _prepare_monash_models(
            Z=Z * Z_sun, mass=mass, monash_models=monash_models
        )
        ia = _get_initial_envelope_abundance(Z * Z_sun, mass, monash_models)

        mu_inv = 0
        for elemental_mass in ia["elemental_mass"]:
            row = ia[ia["elemental_mass"] == elemental_mass]
            X_i = row["massfrac"].iloc[0]
            Z_i = row["elemental_mass"].iloc[0]
            A_i = pt.elements[Z_i].mass
            mu_inv += X_i * (1 + Z_i) / A_i
        mu = 1 / mu_inv
        mus.append(mu)

    plt.plot(ZS, mus, label="Karakas (2016) interpolation")


mus = []
Z_real = [0.0028, 0.007, 0.014]
for Z in Z_real:

    monash_models = defaultdict(list)
    monash_models = _prepare_monash_models(Z=Z, mass=mass, monash_models=monash_models)
    ia = _get_initial_envelope_abundance(Z, mass, monash_models)

    mu_inv = 0
    for elemental_mass in ia["elemental_mass"]:
        row = ia[ia["elemental_mass"] == elemental_mass]
        X_i = row["massfrac"].iloc[0]
        Z_i = row["elemental_mass"].iloc[0]
        A_i = pt.elements[Z_i].mass
        mu_inv += X_i * (1 + Z_i) / A_i
    mu = 1 / mu_inv
    mus.append(mu)

plt.scatter(np.array(Z_real) / Z_sun, mus, marker="^", label="Karakas (2016) models")

mus = []
for Z in ZS:

    asplund = Asplund(z=Z * Z_sun)

    mu_inv = 0
    for i, (key, ab) in enumerate(asplund.elements.items()):
        X_i = ab.massfrac
        Z_i = ab.Z
        A_i = ab.atomic_mass
        mu_inv += X_i * (1 + Z_i) / A_i
    mu = 1 / mu_inv
    mus.append(mu)

plt.plot(ZS, mus, label="Asplund (2009) with MESA $Y$")


mus = []
for Z in ZS:

    asplund = Asplund(z=Z * Z_sun, he_method="karakas")

    mu_inv = 0
    for i, (key, ab) in enumerate(asplund.elements.items()):
        X_i = ab.massfrac
        Z_i = ab.Z
        A_i = ab.atomic_mass
        mu_inv += X_i * (1 + Z_i) / A_i
    mu = 1 / mu_inv
    mus.append(mu)

plt.plot(ZS, mus, label="Asplund (2009) with Karakas (2016) $Y$")

ab = Abundances(
    None,
    df,
    mass=1,
    m_acc=1,
    mass_transfer_efficiency=1,
    save_accretor=True,
)


prof_heavy = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/MS/profile1.data"
)

plt.scatter(ab.Z / Z_sun, np.nanmin(ab.accretor_res.mu_profile_original), s=20)
plt.annotate(
    "MESA MS standard $Y$",
    xy=(ab.Z / Z_sun, np.nanmin(ab.accretor_res.mu_profile_original)),
    xycoords="data",
    xytext=(70, -15),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    ha="left",
    fontsize=8,
)
plt.scatter(ab.Z / Z_sun, np.nanmin(prof_heavy.mu), s=20)
plt.annotate(
    "MESA MS Karakas $Y$",
    xy=(ab.Z / Z_sun, np.nanmin(prof_heavy.mu)),
    xycoords="data",
    xytext=(70, -15),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    ha="left",
    fontsize=8,
)


prof_heavy_2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/TPAGB/profile1.data"
)

nanmin = np.nanmin(prof_heavy_2.mu)
plt.scatter(ab.Z / Z_sun, nanmin, c="C0", s=20)
ind = np.searchsorted(prof_heavy_2.logR[::-1], 0, "right")
nanmax = prof_heavy_2.mu[::-1][ind]
plt.scatter(ab.Z / Z_sun, nanmax, c="C0", s=20)
plt.plot([ab.Z / Z_sun, ab.Z / Z_sun], [nanmin, nanmax], c="C0", linewidth=1)
plt.annotate(
    "Average MESA TPAGB Karakas $Y$",
    xy=(ab.Z / Z_sun, nanmax),
    xycoords="data",
    xytext=(-100, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    ha="left",
    fontsize=8,
)
plt.annotate(
    "Minimum MESA TPAGB$\\phantom{\\textrm{tandard }Y}$",
    xy=(ab.Z / Z_sun, nanmin),
    xycoords="data",
    xytext=(-100, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="C9", linewidth=0.75),
    ha="left",
    fontsize=8,
)


fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
axs.set_xscale("log")
axs.set_xlabel("$Z$")
axs.set_ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-re.pgf", format="pgf")
plt.show()
plt.close()

# %%
df = AbundanceTables()
ab = Abundances(None, df, mass=2, m_acc=1, mass_transfer_efficiency=1)
print(ab.initial_envelope_abundances)

mu_inv = 0
for elemental_mass in ab.initial_envelope_abundances["elemental_mass"]:
    row = ab.initial_envelope_abundances[
        ab.initial_envelope_abundances["elemental_mass"] == elemental_mass
    ]
    X_i = row["massfrac"].iloc[0]
    Z_i = row["elemental_mass"].iloc[0]
    A_i = pt.elements[Z_i].mass
    mu_inv += X_i * (1 + Z_i) / A_i
mu = 1 / mu_inv
print(mu)

# %%

norm = plt.Normalize(1, 13)
cmap = plt.cm.viridis
# color = cmap(norm(x))

print(prof_heavy.bulk_names)

for i in range(1, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )
    plt.plot(prof_heavy.mass, prof_heavy.mu, c=cmap(norm(i)))


norm = plt.Normalize(1, 4)
cmap = plt.cm.jet
for i in range(1, 6):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/TPAGB/profile{i}.data"
    )
    plt.plot(prof_heavy.mass, prof_heavy.mu, c=cmap(norm(i)))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")
plt.show()
# %%

hist1 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M2.0/LOGS/TPAGB/history.data"
)
# %%
hist2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/TPAGB/history.data"
)
# %%
hist1eagb = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M2.0/LOGS/EAGB/history.data"
)
hist2eagb = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/EAGB/history.data"
)
# %%

plt.plot(hist1.star_age + hist1eagb.star_age[-1], hist1.log_LHe)
plt.plot(hist1eagb.star_age, hist1eagb.log_LHe)
plt.plot(hist2.star_age + hist2eagb.star_age[-1], hist2.log_LHe)
plt.plot(hist2eagb.star_age, hist2eagb.log_LHe)
plt.show()
# %%

histss = []
for path in ["z0.00557/completed/", "other-he/"]:
    hists = []
    for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
        hists.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/single-stars/"
                + path
                + "M2.0/LOGS/"
                + phase
                + "/history.data"
            )
        )
    histss.append(hists)
# %%

for hists in histss:
    print(hists)
    for hist in hists:
        plt.plot(hist.log_Teff, hist.log_L)

plt.gca().invert_xaxis()
plt.show()

# %%

plt.plot(hist1.star_age + hist1eagb.star_age[-1], hist1.log_LHe)
plt.plot(hist1eagb.star_age, hist1eagb.log_LHe)
plt.plot(hist2.star_age + hist2eagb.star_age[-1], hist2.log_LHe)
plt.plot(hist2eagb.star_age, hist2eagb.log_LHe)


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 13)
cmap = plt.cm.Greens
# color = cmap(norm(x))

print(prof_heavy.bulk_names)

h1 = []
for i in range(1, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )
    (l1,) = plt.plot(
        prof_heavy.logR, prof_heavy.mu, c=cmap(norm(i)), label="GB", linewidth=1
    )
    h1.append(l1)


norm = plt.Normalize(1, 5)
cmap = plt.cm.Purples
h2 = []
for i in range(1, 6):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/TPAGB/profile{i}.data"
    )
    (l2,) = plt.plot(
        prof_heavy.logR, prof_heavy.mu, c=cmap(norm(i)), label="TPAGB", linewidth=1
    )
    h2.append(l2)

plt.ylim(0.60, 0.64)
plt.xlim(-2, 2)

fig.legend(handles=[h1[-4], h2[-2]], loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(R/ R_\\odot)$")
plt.ylabel("$\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-mu-profiles.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 13)
cmap = plt.cm.Greens
# color = cmap(norm(x))

print(prof_heavy.bulk_names)

h1 = []
for i in range(1, 14):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/GB/profile{i}.data"
    )
    (l1,) = plt.plot(
        prof_heavy.mass, prof_heavy.mu, c=cmap(norm(i)), label="GB", linewidth=1
    )
    h1.append(l1)


norm = plt.Normalize(1, 5)
cmap = plt.cm.Purples
h2 = []
for i in range(1, 6):
    prof_heavy = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/single-stars/other-he/M2.0/LOGS/TPAGB/profile{i}.data"
    )
    (l2,) = plt.plot(
        prof_heavy.mass, prof_heavy.mu, c=cmap(norm(i)), label="TPAGB", linewidth=1
    )
    h2.append(l2)

plt.ylim(0.60, 0.64)
plt.xlim(0.25, 2.01)

fig.legend(handles=[h1[-4], h2[-2]], loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-mu-profiles-mass.pgf", format="pgf")
plt.show()
plt.close()
