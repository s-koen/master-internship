import sys
import numpy as np
import pickle
import re
from collections import defaultdict

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")

from scripts.general_utils.cache import get_star


def compute_m_DUP(model, combined=None):
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

    lambda_DUP = np.asarray(model.lambda_DUP)
    try:
        he_core_mass = np.asarray(model.he_core_mass)
    except:
        he_core_mass = np.asarray(model.m_core)
    TP_count = np.asarray(model.TP_count)

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
                    "time": model.age[dup_index],
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
            "time": model.age[dup_index],
        }

    return dup


def track_DUP(model):
    """
    Track material dredged up during the TP-AGB of a combined star
    and subsequently during the binary evolution.

    The combined star is obtained from:

        get_star(m=model.params["m"])

    The combined-star evolution is followed until the start of the
    binary model. Existing tracers are then carried continuously into
    the binary calculation.

    Returns
    -------
    result : dict
        Full tracer histories and mass-budget information.
    """

    # ==============================================================
    # 1. get combined star
    # ==============================================================

    combined = get_star(m=model.params["m"])

    # ==============================================================
    # 2. compute dredge-up events
    # ==============================================================

    dup_combined = compute_m_DUP(combined)
    dup_binary = compute_m_DUP(model, dup_combined)

    # ==============================================================
    # 3. determine when binary starts in combined-star evolution
    # ==============================================================

    binary_start_age = model.age[0]

    combined_age = np.asarray(combined.age)

    combined_start_idx = (
        np.searchsorted(
            combined_age,
            binary_start_age,
            side="right",
        )
        - 1
    )

    if combined_start_idx < 0:
        raise ValueError("binary model starts before the combined-star track")

    if combined_start_idx >= len(combined_age):
        raise ValueError("binary model starts after the combined-star track")

    # ==============================================================
    # 4. truncate combined star at binary starting point
    # ==============================================================

    time_c = combined.age[: combined_start_idx + 1]

    env_c = combined.m_env[: combined_start_idx + 1]

    core_c = combined.m_core[: combined_start_idx + 1]

    tp_c = combined.TP_count[: combined_start_idx + 1]

    wind_c = combined.Mdot[: combined_start_idx + 1]

    transfer_c = np.zeros_like(wind_c)

    # ==============================================================
    # 5. binary arrays
    # ==============================================================

    time_b = np.asarray(model.age)

    env_b = np.asarray(model.envelope_mass)

    core_b = np.asarray(model.he_core_mass)

    tp_b_local = np.asarray(model.TP_count)

    wind_b = 10 ** np.asarray(model.lg_wind_mdot_1)

    total_b = 10 ** np.asarray(model.lg_mstar_dot_1)

    transfer_b = total_b - wind_b

    # ==============================================================
    # 6. determine TP-number offset
    #
    # The binary TP_count may restart at 1.
    #
    # Example:
    #
    # combined ends at TP 15
    # binary starts at local TP 1
    #
    # therefore:
    #
    # local TP 1 -> global TP 15
    # local TP 2 -> global TP 16
    # ...
    # ==============================================================

    combined_start_tp = int(tp_c[-1])
    binary_start_tp = int(tp_b_local[0])

    tp_offset = combined_start_tp - binary_start_tp

    def global_binary_tp(local_tp):
        return int(local_tp) + tp_offset

    # ==============================================================
    # 7. construct global list of dredge-up events
    # ==============================================================

    dup = {}

    # combined-star TPs
    for tp, event in dup_combined.items():

        # only events that occur before binary starts
        if event["time"] <= time_c[-1]:

            dup[tp] = {
                "mass": event["mass"],
                "combined_index": event["index"],
                "combined_time": event["time"],
                "phase": "combined",
            }

    # binary TPs
    for local_tp, event in dup_binary.items():

        global_tp = global_binary_tp(local_tp)

        dup[global_tp] = {
            "mass": event["mass"],
            "binary_index": event["index"],
            "binary_time": event["time"],
            "phase": "binary",
        }

    # ==============================================================
    # 8. allocate tracer arrays
    # ==============================================================

    global_tps = np.array(
        sorted(dup.keys()),
        dtype=int,
    )

    tp_to_index = {tp: i for i, tp in enumerate(global_tps)}

    n_tp = len(global_tps)

    # ==============================================================
    # 9. storage
    # ==============================================================

    n_c = len(time_c)
    n_b = len(time_b)

    tracer = np.zeros(n_tp)

    core_lost = np.zeros(n_tp)
    wind_lost = np.zeros(n_tp)
    accreted = np.zeros(n_tp)

    tracer_history_c = np.zeros((n_tp, n_c))

    tracer_history_b = np.zeros((n_tp, n_b))

    core_history_c = np.zeros((n_tp, n_c))
    wind_history_c = np.zeros((n_tp, n_c))

    core_history_b = np.zeros((n_tp, n_b))
    wind_history_b = np.zeros((n_tp, n_b))
    accreted_history_b = np.zeros((n_tp, n_b))

    # ==============================================================
    # 10. helper for injecting dredge-up material
    # ==============================================================

    def inject_combined(j):

        for tp, event in dup.items():

            if event["phase"] != "combined":
                continue

            if event["combined_index"] != j:
                continue

            i = tp_to_index[tp]

            tracer[i] += event["mass"]

    def inject_binary(j):

        local_tp = int(tp_b_local[j])

        # only inject when this timestep is the beginning of
        # a new dredge-up event
        global_tp = global_binary_tp(local_tp)

        if global_tp not in dup:
            return

        event = dup[global_tp]

        if event["binary_index"] != j:
            return

        i = tp_to_index[global_tp]

        tracer[i] += event["mass"]

    # ==============================================================
    # 11. combined-star evolution
    # ==============================================================

    for j in range(1, n_c):

        dt = time_c[j] - time_c[j - 1]

        # ----------------------------------------------------------
        # inject dredged-up material
        # ----------------------------------------------------------

        inject_combined(j)

        # ----------------------------------------------------------
        # physical mass changes
        # ----------------------------------------------------------

        dm_wind = wind_c[j] * dt

        dm_transfer = transfer_c[j] * dt

        dm_core = max(
            core_c[j] - core_c[j - 1],
            0.0,
        )

        # ----------------------------------------------------------
        # tracer composition
        # ----------------------------------------------------------

        if env_c[j - 1] <= 0:
            raise ValueError("combined-star envelope mass became non-positive")

        fraction = tracer / env_c[j - 1]

        # ----------------------------------------------------------
        # tracer losses
        # ----------------------------------------------------------

        dcore = fraction * dm_core
        dwind = fraction * dm_wind
        dtransfer = fraction * dm_transfer

        tracer -= dcore + dwind + dtransfer

        tracer = np.maximum(tracer, 0.0)

        core_lost += dcore
        wind_lost += dwind
        accreted += dtransfer

        # ----------------------------------------------------------
        # history
        # ----------------------------------------------------------

        tracer_history_c[:, j] = tracer
        core_history_c[:, j] = core_lost
        wind_history_c[:, j] = wind_lost

    # ==============================================================
    # 12. save handover state
    # ==============================================================

    tracer_at_binary_start = tracer.copy()

    core_at_binary_start = core_lost.copy()

    wind_at_binary_start = wind_lost.copy()

    accreted_at_binary_start = accreted.copy()

    # ==============================================================
    # 13. binary evolution
    # ==============================================================

    for j in range(1, n_b):

        dt = time_b[j] - time_b[j - 1]

        # ----------------------------------------------------------
        # inject new binary dredge-up material
        # ----------------------------------------------------------

        inject_binary(j)

        # ----------------------------------------------------------
        # mass changes
        # ----------------------------------------------------------

        dm_wind = wind_b[j] * dt

        dm_transfer = transfer_b[j] * dt

        dm_core = max(
            core_b[j] - core_b[j - 1],
            0.0,
        )

        # ----------------------------------------------------------
        # tracer fractions
        # ----------------------------------------------------------

        if env_b[j - 1] <= 0:
            raise ValueError("binary envelope mass became non-positive")

        fraction = tracer / env_b[j - 1]

        # ----------------------------------------------------------
        # losses
        # ----------------------------------------------------------

        dcore = fraction * dm_core

        dwind = fraction * dm_wind

        dtransfer = fraction * dm_transfer

        tracer -= dcore + dwind + dtransfer

        tracer = np.maximum(tracer, 0.0)

        # ----------------------------------------------------------
        # accumulate
        # ----------------------------------------------------------

        core_lost += dcore

        wind_lost += dwind

        accreted += dtransfer

        # ----------------------------------------------------------
        # history
        # ----------------------------------------------------------

        tracer_history_b[:, j] = tracer

        core_history_b[:, j] = core_lost

        wind_history_b[:, j] = wind_lost

        accreted_history_b[:, j] = accreted

    # ==============================================================
    # 14. mass conservation
    # ==============================================================

    M_DUP = np.array([dup[tp]["mass"] for tp in global_tps])

    final_mass = tracer + core_lost + wind_lost + accreted

    conservation_error = final_mass - M_DUP

    # ==============================================================
    # 15. return everything
    # ==============================================================

    return {
        # TP information
        "TP": global_tps,
        "M_DUP": M_DUP,
        "dup": dup,
        # final fate
        "tracer_final": tracer,
        "core_lost": core_lost,
        "wind_lost": wind_lost,
        "accreted": accreted,
        # histories
        "tracer_history_combined": tracer_history_c,
        "tracer_history_binary": tracer_history_b,
        "core_history_combined": core_history_c,
        "core_history_binary": core_history_b,
        "wind_history_combined": wind_history_c,
        "wind_history_binary": wind_history_b,
        "accreted_history_binary": accreted_history_b,
        # handover
        "tracer_at_binary_start": tracer_at_binary_start,
        "core_at_binary_start": core_at_binary_start,
        "wind_at_binary_start": wind_at_binary_start,
        "accreted_at_binary_start": accreted_at_binary_start,
        # times
        "time_combined": time_c,
        "time_binary": time_b,
        "combined_start_age": time_c[-1],
        "binary_start_age": time_b[0],
        # TP bookkeeping
        "tp_offset": tp_offset,
        # conservation
        "conservation_error": conservation_error,
    }


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

        species = self.intershell.columns[4:]

        self.isotopes = {spec: Isotope(spec) for spec in species}
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
        self.envelope = self.envelope[self.envelope["Z"].astype(np.float64) == 0.014]
        self.envelope = self.envelope[self.envelope["M_init"].astype(np.float64) == 2]

    def __getattr__(self, name):
        try:
            return self.isotopes[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def get_initial_envelope_abundance(self, element, metallicity):
        df = self.envelope[self.envelope["element"] == element]
        return float(df["massfrac"].iloc[-1]) * metallicity

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

    def __init__(self, model, df):
        self.model = model
        df_mix = df.intershell[df.intershell["pmz"] == "2e-3"]
        self.df = df

        simple = get_star(full_path=self.model.params["single_star"])

        self.dup_simple = compute_m_DUP(simple)
        self.dup_detailed = compute_m_DUP(model, self.dup_simple)

        binary_start_age = model.age[0]

        simple_age = np.asarray(simple.age)

        self.simple_end_idx = (
            np.searchsorted(
                simple_age,
                binary_start_age,
                side="right",
            )
            - 1
        )

        self.total_length = self.simple_end_idx + len(self.model.age)
        self.m_dup = np.zeros(self.total_length)
        self.m_env = np.concatenate(
            [simple.m_env[: self.simple_end_idx], self.model.envelope_mass]
        )
        self.time = np.concatenate([simple.age[: self.simple_end_idx], self.model.age])

        m2 = np.concatenate(
            [
                self.model.sb.m2[: self.simple_end_idx],
                self.model.star_2_mass,
            ]
        )

        valid = ~np.isnan(m2)

        m2_filled = np.nan_to_num(m2, nan=0)

        dm = np.diff(m2_filled)
        dm[~valid[:-1] | ~valid[1:]] = 0

        self.dm_acc = np.concatenate([[0], np.clip(dm, 0, np.inf)])

        for key, value in self.dup_simple.items():
            if value["index"] > self.simple_end_idx:
                break
            self.m_dup[value["index"]] = value["mass"]

        for key, value in self.dup_detailed.items():
            self.m_dup[self.simple_end_idx + value["index"]] = value["mass"]

    def __getattr__(self, name):
        if name in self.df.elements:

            # compute the intershell elemental abundance
            intershell = self.compute_intershell(name)

            # compute the initial envelope abundance
            envelope = self.compute_envelope_abundance(name, intershell)

            self.df.elements[name].envelope = envelope
            self.df.elements[name].intershell = intershell
            self.df.elements[name].m_accreted = np.cumsum(envelope * self.dm_acc)

            return self.df.elements[name]

        if name in self.df.isotopes:

            return self.df.isotopes[name]

    def _prepare_single_monash_model(self, M, Z):

        intershell = self.df.intershell.query(
            f"Z == {Z} and pmz == 2e-3 and last == 1 and M1tp == {M}"
        ).sort_values("ntp")

        tp_info = self.df.tp.query(f"initial_mass == {M} and z == {Z}")

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
        isotope_abundance = np.log10(isotope.mass * isotope_abundance + 1e-20)

        pass

    def _prepare_monash_models(self):
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

        self._prepare_single_monash_model(M, Z)

    def compute_intershell(self, name):
        # TODO: this needs to be changed to the ACTUAL abundances
        # keep in mind that we need to sum the abundances for the
        # different isotopes separately.
        # it is also important to keep in mind that the abundances
        # here should be MASS RATIOS.

        self.monash_models = self._prepare_monash_models()

        intershell = np.zeros(self.total_length)
        for isotope in self.df.elements[name].isotopes:

            intershell += self.get_abundance(
                self.df.isotopes[isotope],
                self.model.params["m"],
                self.model.params["z"],
                np.cumsum(self.m_dup + 1e-12),
            )
        return intershell

    def get_iso_and_Mdredge(self, isotope, intershell, tp_info):
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
        isotope_abundance = np.log10(isotope.mass * isotope_abundance + 1e-20)

        return isotope_abundance, Mdredge

    def get_abundance(self, isotope, M, Z, Mdredge_interp, drop=None):

        if Z in [0.0028, 0.007, 0.014]:
            return 10 ** self.get_abundance_Z(isotope, M, Z, Mdredge_interp, drop)

        if Z <= 0.0028:
            return 10 ** self.get_abundance_Z(isotope, M, 0.0028, Mdredge_interp, drop)
        if Z >= 0.014:
            return 10 ** self.get_abundance_Z(isotope, M, 0.014, Mdredge_interp, drop)

        if Z <= 0.007:
            z_min = 0.0028
            z_max = 0.007
        else:
            z_min = 0.007
            z_max = 0.014

        abundance_min = self.get_abundance_Z(isotope, M, z_min, Mdredge_interp, drop)
        abundance_max = self.get_abundance_Z(isotope, M, z_max, Mdredge_interp, drop)
        weight = (np.log10(Z) - np.log10(z_min)) / (np.log10(z_max) - np.log10(z_min))

        return 10 ** (abundance_min + weight * (abundance_max - abundance_min))

    def get_abundance_Z(self, isotope, M, Z, Mdredge_interp, drop=None):
        intershell_metal = self.df.intershell.query(
            f"Z == {Z} and pmz == 2e-3 and last == 1"
        ).sort_values("ntp")

        ms = np.unique(intershell_metal.M1tp)
        ms.sort()
        if drop != None:
            for i, m in enumerate(ms):
                if m == drop:
                    ms = np.delete(ms, i)
        try:
            arg_max = np.where(ms >= M)[0][0]
            i1_max = intershell_metal.query(f"M1tp == {ms[arg_max]}")
            tp1_max = self.df.tp.query(f"initial_mass == {ms[arg_max]} and z == {Z}")
            iso, Mdredge_real = self.get_iso_and_Mdredge(isotope, i1_max, tp1_max)
            abundance_max = np.interp(np.log10(Mdredge_interp), Mdredge_real, iso)
        except IndexError:
            abundance_max = None
        try:
            arg_min = np.where(ms < M)[0][-1]
            i1_min = intershell_metal.query(f"M1tp == {ms[arg_min]}")
            tp1_min = self.df.tp.query(f"initial_mass == {ms[arg_min]} and z == {Z}")
            iso, Mdredge_real = self.get_iso_and_Mdredge(isotope, i1_min, tp1_min)
            abundance_min = np.interp(np.log10(Mdredge_interp), Mdredge_real, iso)
        except IndexError:
            abundance_min = None

        if type(abundance_min) == type(None):
            return abundance_max
        if type(abundance_max) == type(None):
            return abundance_min
        weight = (M - ms[arg_min]) / (ms[arg_max] - ms[arg_min])

        return abundance_min + weight * (abundance_max - abundance_min)

    def compute_envelope_abundance(self, name, intershell):
        # INFO: gets the initial envelope abundance of the element
        # scaled by the metallicity of the model.
        initial_envelope_abundance = self.df.get_initial_envelope_abundance(
            element=name,
            metallicity=self.model.params["z"],
        )

        # INFO: computes the elemental abundance in the envelope by
        # enriching it with intershell abundances.
        envelope = np.zeros(self.total_length)
        delta_M_element = intershell * self.m_dup
        for i in range(self.total_length):
            if i == 0:
                envelope[i] = initial_envelope_abundance
                continue
            envelope[i] = (envelope[i - 1] * self.m_env[i] + delta_M_element[i]) / (
                self.m_env[i] + self.m_dup[i]
            )

        return envelope


# %%

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

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

grid = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid3 = MesaGrid(f"{MASTER}/grid-masses-3-2026-08-24")
grid4 = MesaGrid(f"{MASTER}/grid-masses-4-2026-08-25")
grid.merge(grid2)
grid.merge(grid3, overwrite=True)
grid.merge(grid4, overwrite=True)

# %%
m = grid.models[30]

df = AbundanceTables()
ab = Abundances(model=m, df=df)


# %%

plt.plot(ab.time, ab.ba.envelope)
plt.show()
# %%
