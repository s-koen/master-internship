import time
import sys
import numpy as np
import pickle
import re
from collections import defaultdict
import periodictable as pt
from scripts.general_utils.accretor import *

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")

from scripts.general_utils.cache import get_star


def compute_m_DUP(model, combined=None, sampling=1):
    """
    Compute the dredged-up mass and the timestep at which it is injected.

    Returns
    -------
    dup : dict
        Dictionary keyed by global/local TP number.

        dup[tp]["mass"]  = dredged-up mass for this TP
        dup[tp]["index"] = timestep at which the material is injected
        dup[tp]["time"]  = corresponding model time
    """

    lambda_DUP = np.asarray(model.lambda_DUP)[::sampling]
    try:
        he_core_mass = np.asarray(model.he_core_mass)[::sampling]
    except:
        he_core_mass = np.asarray(model.m_core)[::sampling]
    TP_count = np.asarray(model.TP_count)[::sampling]

    dup = {}

    unique_tps = np.unique(TP_count)

    for tp in unique_tps:

        tp = int(tp)

        if tp <= 1:
            if combined != None:
                pulse_idx = np.where(TP_count == tp)[0]
                local_min = np.nanargmin(he_core_mass[pulse_idx])

                dup_index = pulse_idx[local_min]

                dup[tp] = {
                    "mass": combined[model.params["TP"]]["mass"],
                    "index": dup_index,
                    "time": model.age[::sampling][dup_index],
                }
            continue

        # ----------------------------------------------------------
        # all timesteps belonging to this TP
        # ----------------------------------------------------------

        pulse_idx = np.where(TP_count == tp)[0]

        if len(pulse_idx) == 0:
            continue

        # previous TP
        previous_idx = np.where(TP_count == tp - 1)[0]

        if len(previous_idx) == 0:
            continue

        # ----------------------------------------------------------
        # interpulse core growth
        #
        # minimum core mass during previous pulse
        # -> core mass at beginning of current pulse
        # ----------------------------------------------------------

        core_previous = np.nanmin(he_core_mass[previous_idx])

        core_current = he_core_mass[pulse_idx[0]]

        delta_core = core_current - core_previous

        if not np.isfinite(delta_core) or delta_core <= 0:
            continue

        # ----------------------------------------------------------
        # lambda for this pulse
        # ----------------------------------------------------------

        lambda_max = np.nanmax(lambda_DUP[pulse_idx])

        if not np.isfinite(lambda_max):
            continue

        # ----------------------------------------------------------
        # dredged-up mass
        # ----------------------------------------------------------

        M_DUP = lambda_max * delta_core

        # ----------------------------------------------------------
        # find the post-pulse minimum core mass
        #
        # this is our estimate of the time at which TDU occurs
        # ----------------------------------------------------------

        local_min = np.nanargmin(he_core_mass[pulse_idx])

        dup_index = pulse_idx[local_min]

        dup[tp] = {
            "mass": M_DUP,
            "index": dup_index,
            "time": model.age[::sampling][dup_index],
        }

    return dup


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


class Isotope:
    def __init__(self, isotope, i):
        self.key = isotope
        self.index = i

        match isotope:
            case "n":
                self.mass = 1
                self.name = f"$\\textrm{{{isotope}}}$"
                self.short_name = "neutron"
            case "p":
                self.mass = 1
                self.name = f"$\\textrm{{{isotope}}}^{{+}}$"
                self.short_name = "h"
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


class Element:
    def __init__(self, name):
        self.key = name
        self.name = name.capitalize()
        self.isotopes = {}

    def add_isotope(self, isotope):
        self.isotopes[isotope.key] = isotope

    def __getitem__(self, isotope):
        return self.isotopes[isotope]

    def __iter__(self):
        return iter(self.isotopes.values())

    def __repr__(self):
        return self.key

    def __getattr__(self, name):
        if name in self.isotopes:
            return self.isotopes[name]


class AbundanceTables:
    def __init__(self):

        with open("data/intershell_pd_df.pkl", "rb") as f:
            self.intershell = pickle.load(f)

        with open("data/env_pd_df.pkl", "rb") as f:
            self.envelope = pickle.load(f)

        with open("data/tp_info_pd_df.pkl", "rb") as f:
            self.tp = pickle.load(f)

        species = self.intershell.columns[8:]

        self.isotopes = {}
        for i, spec in enumerate(species):
            self.isotopes[spec] = Isotope(spec, i)

        self.elements = {}

        for isotope in self.isotopes.values():
            if isotope.short_name is None:
                continue

            element = self.elements.setdefault(
                isotope.short_name, Element(isotope.short_name)
            )

            element.add_isotope(isotope)

        self.envelope = self.envelope[self.envelope["pmz"] == 2e-3]
        self.envelope = self.envelope[self.envelope["N_ov"] != 0]
        self.envelope_filtered = self.envelope[
            self.envelope["Z"].astype(np.float64) == 0.014
        ]
        self.envelope_filtered = self.envelope_filtered[
            self.envelope_filtered["M_init"].astype(np.float64) == 2
        ]

        self.accretor_profiles = AccretorProfiles()

    def __getattr__(self, name):
        try:
            return self.isotopes[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def get_initial_envelope_abundance(self, element, metallicity):
        df = self.envelope_filtered[self.envelope_filtered["element"] == element]
        return float(df["massfrac"].iloc[0]) * metallicity / 0.014

    def get_pulse_info_specific_model(self, mass, metallicity):
        df = self.tp[self.tp["initial_mass"] == mass]
        df = df[df["z"] == metallicity]
        return df


class Abundances:
    """
    this class computes and contains the abundances of the envelope of a
    binary run. it uses the MESA binary simulation, combined with the
    simple binary to determine
    """

    def __init__(
        self,
        model,
        df,
        method="tp offset",
        mass=None,
        intershell=None,
        initial_abundance=None,
        sampling=100,
        m_acc=None,
        mass_transfer_efficiency=None,
        full_mixing=False,
    ):
        self.model = model
        df_mix = df.intershell[df.intershell["pmz"] == "2e-3"]
        self.df = df

        self.method = method

        if mass != None:
            self.mass = mass
            simple = get_star(m=self.mass)
            # NOTE: CHANGE THIS TO THE ACTUAL METALLICITY
            self.Z = simple.Z_init

        else:
            self.mass = self.model.params["m"]
            self.Z = self.model.params["z"]
            simple = get_star(full_path=self.model.params["single_star"])

        self.dup_simple = compute_m_DUP(simple, sampling=sampling)

        if mass == None:
            self.dup_detailed = compute_m_DUP(model, self.dup_simple, sampling=sampling)

            binary_start_age = model.age[0]

            simple_age = np.asarray(simple.age)[::sampling]

            self.simple_end_idx = (
                np.searchsorted(
                    simple_age,
                    binary_start_age,
                    side="right",
                )
                - 1
            )

            self.total_length = self.simple_end_idx + len(self.model.age[::sampling])

        else:

            self.simple_end_idx = -1
            self.total_length = len(simple.age[::sampling])

        self.m_dup = np.zeros(self.total_length)
        self.tp_count = np.zeros(self.total_length)

        if mass == None:
            self.m_env = np.concatenate(
                [
                    simple.m_env[::sampling][: self.simple_end_idx],
                    self.model.envelope_mass[::sampling],
                ]
            )
            self.time = np.concatenate(
                [
                    simple.age[::sampling][: self.simple_end_idx],
                    self.model.age[::sampling],
                ]
            )

            m2 = np.concatenate(
                [
                    self.model.sb.m2[::sampling][: self.simple_end_idx],
                    self.model.star_2_mass[::sampling],
                ]
            )

            valid = ~np.isnan(m2)

            m2_filled = np.nan_to_num(m2, nan=0)

            dm = np.diff(m2_filled)
            dm[~valid[:-1] | ~valid[1:]] = 0

            self.dm_acc = np.concatenate([[0], np.clip(dm, 0, np.inf)])
            self.total_mass_accreted = m2[-1]

            m1 = np.concatenate(
                [
                    -1 * simple.mass[::sampling][: self.simple_end_idx],
                    -1 * self.model.star_1_mass[::sampling],
                ]
            )

            valid = ~np.isnan(m1)

            m1_filled = np.nan_to_num(m1, nan=0)

            dm = np.diff(m1_filled)
            dm[~valid[:-1] | ~valid[1:]] = 0

            self.dm = np.concatenate([[0], np.clip(dm, 0, np.inf)])

            for key, value in self.dup_simple.items():
                if value["index"] > self.simple_end_idx:
                    break
                self.m_dup[value["index"]] = value["mass"]

            for key, value in self.dup_detailed.items():
                self.m_dup[self.simple_end_idx + value["index"]] = value["mass"]

            self.tp_count[: self.simple_end_idx] = simple.TP_count[::sampling][
                : self.simple_end_idx
            ]

            self.tp_count[self.simple_end_idx :] = (
                self.model.TP_count[::sampling] + simple.TP_count[self.simple_end_idx]
            )

        else:
            self.m_env = simple.m_env[::sampling]
            self.time = simple.age[::sampling]
            self.dm = np.concatenate([[0], -1 * np.diff(simple.mass[::sampling])])
            for key, value in self.dup_simple.items():
                self.m_dup[value["index"]] = value["mass"]

            self.tp_count = simple.TP_count[::sampling]

        self.total_mass_expelled = simple.mass[0] - simple.mass[-1]
        self.monash_models = defaultdict(list)
        self._get_monash_masses_per_metallicity()
        self._prepare_monash_models()
        self.initial_envelope_abundances = self._get_initial_envelope_abundance(
            self.Z, self.mass
        )
        self.elements_mass = []
        for element in self.initial_envelope_abundances["elemental_mass"]:
            self.elements_mass.append(element)

        self.elements_name = []
        for element in self.initial_envelope_abundances["element"]:
            self.elements_name.append(element)

        self.intershell = intershell
        self.initial_abundance = initial_abundance

        intershell = self.compute_all_intershell()
        self.all_intershell = np.max(intershell, axis=0)
        envelope = self.compute_all_envelope_abundances(intershell)

        if mass == None:
            yields = np.cumsum(envelope * self.dm_acc[:, None], axis=0)[-1, :]
            self.accreted_abundances = yields / self.total_mass_accreted

        else:
            yields = np.cumsum(envelope * self.dm[:, None], axis=0)[-1, :]
            self.accreted_abundances = yields / self.total_mass_expelled

        self.accreted_abundances /= np.sum(self.accreted_abundances)

        mu_inv = 0
        for i, ab in enumerate(self.accreted_abundances):
            X_i = ab
            Z_i = self.elements_mass[i]
            A_i = pt.elements[Z_i].mass
            mu_inv += X_i * (1 + Z_i) / A_i
        self.mu = 1 / mu_inv

        if mass == None:

            # accretor mass is simply q * TPAGB mass because this is the mass BEFORE accretion
            m_acc = self.model.params["q"] * self.mass

            # accretor age is less obvious, but i think taking the time where R_RL < R_star is fine.
            # TPAGB mass transfer is a really short duration event when compared to MS lifetimes.
            age_arg = np.argmax(self.dm)
            age = self.time[age_arg]
            accretor = Accretor(
                profiles=self.df.accretor_profiles,
                age=self.time[-1],
                mass=m_acc,
            )
            self.mixing_mass = accretor.effective_mu_vs_depth(
                M_acc=self.total_mass_accreted, mu_acc=self.mu
            ).mixing_mass

        else:
            if m_acc != None:
                accretor = Accretor(
                    profiles=self.df.accretor_profiles, age=self.time[-1], mass=m_acc
                )
                self.mixing_mass = accretor.effective_mu_vs_depth(
                    M_acc=self.total_mass_expelled, mu_acc=self.mu
                ).mixing_mass
            else:
                raise Exception(
                    "need to provide m_acc when not providing a binary model"
                )

        # clip mixing mass
        self.mixing_mass = np.clip(self.mixing_mass, 0, m_acc)

        # TODO: implement proper low mass MESA models
        if m_acc < 0.8:
            self.mixing_mass = m_acc

        if full_mixing:
            self.mixing_mass = m_acc

        self.MS_abundances = self.__compute_MS_abundances(mass_transfer_efficiency)

    def __getattr__(self, name):

        if name in self.df.elements:

            # compute the intershell elemental abundance

            # quick hack to test MESA abundances
            if type(self.intershell) == type(None):
                intershell = self.compute_intershell(name)
            else:
                intershell = self.intershell

            # compute the initial envelope abundance
            envelope = self.compute_envelope_abundance(name, intershell)

            self.df.elements[name].envelope = envelope
            self.df.elements[name].intershell = intershell
            if self.model != None:
                self.df.elements[name].m_accreted = np.cumsum(envelope * self.dm_acc)
            self.df.elements[name].m_yield = np.cumsum(envelope * self.dm)

            return self.df.elements[name]

        if name in self.df.isotopes:

            return self.df.isotopes[name]

    def __compute_MS_abundances(
        self, mass_transfer_efficiency=None
    ) -> NDArray[np.float64]:
        initial_ms_abundances = self.initial_envelope_abundances["massfrac"]
        initial_ms_masses = initial_ms_abundances * self.mixing_mass
        accreted_abundances = self.accreted_abundances

        if self.model != None:
            accreted_masses = self.total_mass_accreted * self.accreted_abundances
            m_acc = self.total_mass_accreted
        else:
            accreted_masses = (
                self.total_mass_expelled
                * mass_transfer_efficiency
                * self.accreted_abundances
            )
            m_acc = self.total_mass_expelled * mass_transfer_efficiency

        # computing the final mass fractions is simple. just add the total masses and
        # the masses of the individual elements and divide :)
        ms_abundances = (initial_ms_masses + accreted_masses) / (
            self.mixing_mass + m_acc
        )

        return ms_abundances

    def _get_initial_envelope_abundance(self, Z, M):

        if self.Z in [0.0028, 0.007, 0.014]:
            return self._prepare_initial_envelope_abundance_Z(Z, M)

        if self.Z <= 0.0028:
            return self._prepare_initial_envelope_abundance_Z(0.0028, M)

        if self.Z >= 0.014:
            return self._prepare_initial_envelope_abundance_Z(0.014, M)

        if self.Z <= 0.007:
            z_min = 0.0028
            z_max = 0.007
        else:
            z_min = 0.007
            z_max = 0.014

        abundance_min = self._prepare_initial_envelope_abundance_Z(z_min, M)
        abundance_max = self._prepare_initial_envelope_abundance_Z(z_max, M)

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

    def _prepare_initial_envelope_abundance_Z(self, Z, M):
        if len(self.monash_models[Z]) == 1:
            abundance = self.monash_models[Z][0].envelope_abundance
            return abundance

        abundance_min = self.monash_models[Z][0].envelope_abundance
        mass_min = self.monash_models[Z][0].M
        abundance_max = self.monash_models[Z][1].envelope_abundance
        mass_max = self.monash_models[Z][1].M

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

    def _prepare_single_monash_model(self, M, Z, fresh=False):

        if not fresh:
            try:
                with open(
                    f"/home/koen/master-internship/data/intershell-cache/M{M:.3f}Z{Z:.4f}.pkl",
                    "rb",
                ) as f:
                    monash_model = pickle.load(f)
                    return monash_model
            except FileNotFoundError:
                return self._prepare_single_monash_model(M, Z, True)

        else:
            intershell = self.df.intershell.query(
                f"Z == {Z} and pmz == 2e-3 and last == 1 and M1tp == {M}"
            ).sort_values("ntp")

            envelope = self.df.envelope.query(
                f"Z == {Z} and pmz == 2e-3 and N_ov != 0.0 and M_init == {M} and ntp == 1"
            )

            tp_info = self.df.tp.query(f"initial_mass == {M} and z == {Z}")

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

    def _get_monash_masses_per_metallicity(self):
        try:
            with open(
                f"/home/koen/master-internship/data/intershell-cache/MZ.pkl",
                "rb",
            ) as f:
                self._mass_Z_dict = pickle.load(f)
        except FileNotFoundError:
            self._mass_Z_dict = {}
            for Z in np.unique(self.df.intershell["Z"]):
                i = self.df.intershell.query(
                    f"Z == {Z} and pmz == 2e-3 and last == 1"
                ).sort_values("ntp")
                self._mass_Z_dict[Z] = np.unique(i["M1tp"])

    def _prepare_monash_models(self, Z=None):
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
        if Z == None:
            Z = self.Z
        M = self.mass

        if Z in [0.0028, 0.007, 0.014]:
            masses = self._mass_Z_dict[Z]

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
                self.monash_models[Z].append(
                    self._prepare_single_monash_model(lower, Z)
                )
            if upper != None:
                self.monash_models[Z].append(
                    self._prepare_single_monash_model(upper, Z)
                )

        elif Z < 0.007:
            self._prepare_monash_models(Z=0.007)
            self._prepare_monash_models(Z=0.0028)
        else:
            self._prepare_monash_models(Z=0.014)
            self._prepare_monash_models(Z=0.007)

        # self._prepare_single_monash_model(M, Z)

    def compute_intershell(self, name, isotope=False):
        # TODO: this needs to be changed to the ACTUAL abundances
        # keep in mind that we need to sum the abundances for the
        # different isotopes separately.
        # it is also important to keep in mind that the abundances
        # here should be MASS RATIOS.

        intershell = np.zeros(self.total_length)

        match self.method:
            case "m_dup":
                interp_x = np.log10(np.cumsum(self.m_dup) + 1e-12)

            case "tp":
                interp_x = self.tp_count

            case "tp offset":
                index = np.where(self.m_dup > 10.0**-4.5)[0][0]
                interp_x = self.tp_count - self.tp_count[index]

        for isotope in self.df.elements[name].isotopes:

            intershell += self.df.elements[name].isotopes[
                isotope
            ].mass * self.get_abundance(
                self.df.isotopes[isotope],
                self.mass,
                self.Z,
                interp_x,
            )

        if name == "nb":
            isos = self.df.elements["zr"].isotopes
            intershell += 93 * self.get_abundance(
                self.df.isotopes["zr93"], self.mass, self.Z, interp_x
            )

        if name == "zr":
            isos = self.df.elements["zr"].isotopes
            intershell -= 93 * self.get_abundance(
                self.df.isotopes["zr93"], self.mass, self.Z, interp_x
            )

        return intershell

    def get_abundance(self, isotope, M, Z, interp, drop=None):

        if Z in [0.0028, 0.007, 0.014]:
            return 10 ** self.get_abundance_Z(isotope, M, Z, interp, drop)

        if Z <= 0.0028:
            return 10 ** self.get_abundance_Z(isotope, M, 0.0028, interp, drop)
        if Z >= 0.014:
            return 10 ** self.get_abundance_Z(isotope, M, 0.014, interp, drop)

        if Z <= 0.007:
            z_min = 0.0028
            z_max = 0.007
        else:
            z_min = 0.007
            z_max = 0.014

        abundance_min = self.get_abundance_Z(isotope, M, z_min, interp, drop)
        abundance_max = self.get_abundance_Z(isotope, M, z_max, interp, drop)
        weight = (np.log10(Z) - np.log10(z_min)) / (np.log10(z_max) - np.log10(z_min))

        return 10 ** (abundance_min + weight * (abundance_max - abundance_min))

    def get_all_abundances(self, M, Z, interp):

        if Z in [0.0028, 0.007, 0.014]:
            return 10 ** self.get_all_abundances_Z(M, Z, interp)

        if Z <= 0.0028:
            return 10 ** self.get_all_abundances_Z(M, 0.0028, interp)
        if Z >= 0.014:
            return 10 ** self.get_all_abundances_Z(M, 0.014, interp)

        if Z <= 0.007:
            z_min = 0.0028
            z_max = 0.007
        else:
            z_min = 0.007
            z_max = 0.014

        abundance_min = self.get_all_abundances_Z(M, z_min, interp)
        abundance_max = self.get_all_abundances_Z(M, z_max, interp)
        weight = (np.log10(Z) - np.log10(z_min)) / (np.log10(z_max) - np.log10(z_min))

        return 10 ** (abundance_min + weight * (abundance_max - abundance_min))

    def __interp2d(
        self,
        x_interp: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        y = np.asarray(y)
        idx = np.searchsorted(x, x_interp, side="right") - 1
        idx = np.clip(idx, 0, len(x) - 2)

        weight = np.array(((x_interp - x[idx]) / (x[idx + 1] - x[idx])))[:, None]
        weight = np.clip(weight, 0, 1)

        result = y[idx] + weight * (y[idx + 1] - y[idx])
        return result

    def get_abundance_Z(self, isotope, M, Z, interp_x, drop=None):

        match self.method:
            case "m_dup":
                attr = "m_dup"
            case "tp":
                attr = "pulses"
            case "tp offset":
                attr = "pulses_offset"

        if len(self.monash_models[Z]) == 1:
            abundance = self.monash_models[Z][0].intershell_isos[isotope.key]
            x = getattr(self.monash_models[Z][0], attr)
            return np.interp(interp_x, x, abundance)

        abundance_min = self.monash_models[Z][0].intershell_isos[isotope.key]
        x_min = getattr(self.monash_models[Z][0], attr)
        abundance_min = np.interp(interp_x, x_min, abundance_min)
        mass_min = self.monash_models[Z][0].M
        abundance_max = self.monash_models[Z][1].intershell_isos[isotope.key]
        x_max = getattr(self.monash_models[Z][1], attr)
        abundance_max = np.interp(interp_x, x_max, abundance_max)
        mass_max = self.monash_models[Z][1].M

        weight = (M - mass_min) / (mass_max - mass_min)

        return abundance_min + weight * (abundance_max - abundance_min)

    def get_all_abundances_Z(self, M, Z, interp_x):
        match self.method:
            case "m_dup":
                attr = "m_dup"
            case "tp":
                attr = "pulses"
            case "tp offset":
                attr = "pulses_offset"

        if len(self.monash_models[Z]) == 1:
            abundances = self.monash_models[Z][0].intershell_isos
            x = getattr(self.monash_models[Z][0], attr)
            return self.__interp2d(interp_x, x, abundances)

        abundances_min = self.monash_models[Z][0].intershell_isos
        x_min = getattr(self.monash_models[Z][0], attr)
        abundances_min = self.__interp2d(interp_x, x_min, abundances_min)
        mass_min = self.monash_models[Z][0].M
        abundances_max = self.monash_models[Z][1].intershell_isos
        x_max = getattr(self.monash_models[Z][1], attr)
        abundances_max = self.__interp2d(interp_x, x_max, abundances_max)
        mass_max = self.monash_models[Z][1].M

        weight = (M - mass_min) / (mass_max - mass_min)
        return abundances_min + weight * (abundances_max - abundances_min)

    def compute_envelope_abundance(self, name, intershell, initial=None):
        # INFO: gets the initial envelope abundance of the element
        # scaled by the metallicity of the model.

        # INFO: THIS is the naive method that just uses a scaled metallicity

        # initial_envelope_abundance = self.df.get_initial_envelope_abundance(
        #     element=name,
        #     metallicity=self.Z,
        # )

        # INFO: THIS is the linearly interpolated method
        if initial == None:
            if type(self.initial_abundance) == type(None):
                initial_envelope_abundance = self.initial_envelope_abundances[
                    self.initial_envelope_abundances["element"] == name
                ]["massfrac"]
            else:
                initial_envelope_abundance = self.initial_abundance
        else:
            initial_envelope_abundance = initial

        # INFO: computes the elemental abundance in the envelope by
        # enriching it with intershell abundances.
        envelope = np.zeros(self.total_length)
        delta_M_element = intershell * self.m_dup
        for i in range(self.total_length):
            if i == 0:
                envelope[i] = initial_envelope_abundance
                continue

            # INFO: this is WRONG because the envelope mass is taken AFTER dredge-up
            # already occurred.

            # envelope[i] = (envelope[i - 1] * self.m_env[i] + delta_M_element[i]) / (
            #     self.m_env[i] + self.m_dup[i]
            # )

            envelope[i] = (
                envelope[i - 1] * (self.m_env[i] - self.m_dup[i]) + delta_M_element[i]
            ) / self.m_env[i]

        return envelope

    def compute_all_envelope_abundances(self, intershell):
        initial_envelope_abundances = self.initial_envelope_abundances

        envelope = np.zeros((self.total_length, len(self.initial_envelope_abundances)))
        delta_M_element = intershell * self.m_dup[:, None]

        for i in range(self.total_length):
            if i == 0:
                envelope[i, :] = initial_envelope_abundances.massfrac
                continue

            # INFO: this is WRONG because the envelope mass is taken AFTER dredge-up
            # already occurred.

            # envelope[i] = (envelope[i - 1] * self.m_env[i] + delta_M_element[i]) / (
            #     self.m_env[i] + self.m_dup[i]
            # )

            envelope[i, :] = (
                envelope[i - 1, :] * (self.m_env[i] - self.m_dup[i])
                + delta_M_element[i, :]
            ) / self.m_env[i]

        return envelope

    def compute_all_intershell(self):

        intershell = np.zeros(
            (self.total_length, len(self.initial_envelope_abundances))
        )

        match self.method:
            case "m_dup":
                interp_x = np.log10(np.cumsum(self.m_dup) + 1e-12)

            case "tp":
                interp_x = self.tp_count

            case "tp offset":
                index = np.where(self.m_dup > 10.0**-4.5)[0][0]
                interp_x = self.tp_count - self.tp_count[index]

        isotopic_abundances = self.get_all_abundances(self.mass, self.Z, interp_x)

        for i, name in enumerate(self.initial_envelope_abundances["element"]):
            for isotope in self.df.elements[name].isotopes:
                intershell[:, i] += (
                    self.df.elements[name].isotopes[isotope].mass
                    * isotopic_abundances[
                        :, self.df.elements[name].isotopes[isotope].index
                    ]
                )

            # logic for zr93 decay to nb93
            if name == "nb":
                for isotope in self.df.elements["zr"].isotopes:
                    if self.df.elements["zr"].isotopes[isotope].mass == 93:
                        intershell[:, i] += (
                            93
                            * isotopic_abundances[
                                :, self.df.elements["zr"].isotopes[isotope].index
                            ]
                        )

            if name == "zr":
                for isotope in self.df.elements["zr"].isotopes:
                    if self.df.elements["zr"].isotopes[isotope].mass == 93:
                        intershell[:, i] -= (
                            93
                            * isotopic_abundances[
                                :, self.df.elements["zr"].isotopes[isotope].index
                            ]
                        )

        return intershell


# %%
