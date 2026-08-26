import sys
import numpy as np

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


# %%
