import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
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

grid = MesaGrid(f"{MASTER}grid-masses-2026-08-14")
grid2 = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16")

grid.merge(grid2)


# %%

star = get_star(m=2)

# %%

# input arrays from your mesa/model data
time = star.age[star.ntpagb :]
M_env = star.m_env[star.ntpagb :]
M_core = star.m_core[star.ntpagb :]
mdot_transfer = 0 * star.Mdot[star.ntpagb :]
mdot_wind = star.Mdot[star.ntpagb :]

TP_count = star.TP_count[star.ntpagb :]
M_DUP = 0 + star.m_DUP

# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------


n_times = len(time)
n_TP = len(M_DUP)

# mass from each TP currently remaining in the donor envelope
tracer = np.zeros(n_TP)

# cumulative fate of the material from each TP
accreted = np.zeros(n_TP)
wind_lost = np.zeros(n_TP)
core_lost = np.zeros(n_TP)

# optionally store the full history
tracer_history = np.zeros((n_TP, n_times))
accreted_history = np.zeros((n_TP, n_times))
wind_history = np.zeros((n_TP, n_times))
core_history = np.zeros((n_TP, n_times))

# %%

# ----------------------------------------------------------------------
# evolve through the stellar track
# ----------------------------------------------------------------------


def do_m_DUP_calculation():
    for j in range(1, n_times):

        dt = time[j] - time[j - 1]

        # mass-loss rates should be positive here
        dm_transfer = mdot_transfer[j] * dt
        dm_wind = mdot_wind[j] * dt

        # core growth
        dm_core = max(M_core[j] - M_core[j - 1], 0)
        # dm_core = M_core[j] - M_core[j - 1]

        # --------------------------------------------------------------
        # check whether a new thermal pulse / dredge-up occurred
        # --------------------------------------------------------------

        if TP_count[j] > TP_count[j - 1]:

            tp = int(TP_count[j])

            # add the newly dredged-up material to this TP's tracer
            try:
                tracer[tp - 1] += M_DUP[tp - 1]
            except:
                pass

        # --------------------------------------------------------------
        # composition of the envelope
        # --------------------------------------------------------------

        # fraction of the envelope belonging to each TP
        fraction = tracer / M_env[j - 1]

        # --------------------------------------------------------------
        # remove the corresponding fraction through each channel
        # --------------------------------------------------------------

        dm_core_i = fraction * dm_core
        dm_wind_i = fraction * dm_wind
        dm_transfer_i = fraction * dm_transfer

        # update the tracer masses
        tracer -= dm_core_i
        tracer -= dm_wind_i
        tracer -= dm_transfer_i

        # accumulate where the material went
        core_lost += dm_core_i
        wind_lost += dm_wind_i
        accreted += dm_transfer_i

        # --------------------------------------------------------------
        # save history
        # --------------------------------------------------------------

        tracer_history[:, j] = tracer
        core_history[:, j] = core_lost
        wind_history[:, j] = wind_lost
        accreted_history[:, j] = accreted
    return


# %%

winds = np.zeros(len(time))
cores = np.zeros(len(time))
traces = np.zeros(len(time))

for wind, core, trace in zip(wind_history, core_history, tracer_history):
    winds += wind
    cores += core
    traces += trace


plt.plot(star.age[star.ntpagb :], traces)
plt.plot(star.age[star.ntpagb :], cores)
plt.plot(star.age[star.ntpagb :], winds)
plt.show()


# %%

grid = MesaGrid(f"{MASTER}grid-masses-2026-08-14")
grid2 = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16")

grid.merge(grid2)

# %%

rng = np.random.default_rng(seed=00)
models = rng.permutation(grid.models)
for model in models:

    if model.envelope_mass[-1] < 0.01:
        break

print(model.bulk_names)
plt.plot(model.envelope_mass, model.lg_mstar_dot_1)
plt.plot(model.envelope_mass, model.lg_wind_mdot_1)
plt.show()
# %%

# input arrays from your mesa/model data
time = model.age
M_env = model.envelope_mass
M_core = model.he_core_mass
mdot_transfer = 10**model.lg_mstar_dot_1 - 10**model.lg_wind_mdot_1
mdot_wind = 10**model.lg_wind_mdot_1

TP_count = model.TP_count
M_DUP = [0] + compute_m_DUP(model)
print(M_DUP)

# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------


n_times = len(time)
n_TP = len(M_DUP)

# mass from each TP currently remaining in the donor envelope
tracer = np.zeros(n_TP)

# cumulative fate of the material from each TP
accreted = np.zeros(n_TP)
wind_lost = np.zeros(n_TP)
core_lost = np.zeros(n_TP)

# optionally store the full history
tracer_history = np.zeros((n_TP, n_times))
accreted_history = np.zeros((n_TP, n_times))
wind_history = np.zeros((n_TP, n_times))
core_history = np.zeros((n_TP, n_times))


for j in range(1, n_times):

    dt = time[j] - time[j - 1]

    # mass-loss rates should be positive here
    dm_transfer = mdot_transfer[j] * dt
    dm_wind = mdot_wind[j] * dt

    # core growth
    dm_core = max(M_core[j] - M_core[j - 1], 0)
    # dm_core = M_core[j] - M_core[j - 1]

    # --------------------------------------------------------------
    # check whether a new thermal pulse / dredge-up occurred
    # --------------------------------------------------------------

    if TP_count[j] > TP_count[j - 1]:

        tp = int(TP_count[j])

        # add the newly dredged-up material to this TP's tracer
        try:
            tracer[tp - 1] += M_DUP[tp - 1]
        except:
            pass

    # --------------------------------------------------------------
    # composition of the envelope
    # --------------------------------------------------------------

    # fraction of the envelope belonging to each TP
    fraction = tracer / M_env[j - 1]

    # --------------------------------------------------------------
    # remove the corresponding fraction through each channel
    # --------------------------------------------------------------

    dm_core_i = fraction * dm_core
    dm_wind_i = fraction * dm_wind
    dm_transfer_i = fraction * dm_transfer

    # update the tracer masses
    tracer -= dm_core_i
    tracer -= dm_wind_i
    tracer -= dm_transfer_i

    # accumulate where the material went
    core_lost += dm_core_i
    wind_lost += dm_wind_i
    accreted += dm_transfer_i

    # --------------------------------------------------------------
    # save history
    # --------------------------------------------------------------

    tracer_history[:, j] = tracer
    core_history[:, j] = core_lost
    wind_history[:, j] = wind_lost
    accreted_history[:, j] = accreted


# %%

print(tracer)

winds = np.zeros(len(time))
cores = np.zeros(len(time))
traces = np.zeros(len(time))
accreted = np.zeros(len(time))

for wind, core, trace, accrete in zip(
    wind_history, core_history, tracer_history, accreted_history
):
    winds += wind
    accreted += accrete
    cores += core
    traces += trace


plt.plot(model.age, traces)
plt.plot(model.age, cores)
plt.plot(model.age, winds)
plt.plot(model.age, accreted)
plt.show()


# %%
import numpy as np


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
    print(dup_binary)

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
        print(local_tp, event)

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

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(full), constrained_layout=True
)

seeds = [0, 2, 100, 1000]

for i, ax in enumerate(axs.flatten()):
    rng = np.random.default_rng(seed=seeds[i])
    models = rng.permutation(grid.models)
    for model in models:

        if model.envelope_mass[-1] < 0.01:
            break

    dup = track_DUP(model)

    size = np.shape(dup["tracer_history_combined"])[1]
    print(size)

    winds_s = np.zeros(size)
    cores_s = np.zeros(size)
    traces_s = np.zeros(size)
    accreted_s = np.zeros(size)

    for dup_count in range(len(dup["tracer_history_combined"])):
        traces_s += dup["tracer_history_combined"][dup_count]
        winds_s += dup["wind_history_combined"][dup_count]
        cores_s += dup["core_history_combined"][dup_count]

    dup = track_DUP(model)

    winds = np.zeros(len(model.model_number))
    cores = np.zeros(len(model.model_number))
    traces = np.zeros(len(model.model_number))
    accreted = np.zeros(len(model.model_number))

    for dup_count in range(len(dup["tracer_history_binary"])):
        traces += dup["tracer_history_binary"][dup_count]
        winds += dup["wind_history_binary"][dup_count]
        cores += dup["core_history_binary"][dup_count]
        accreted += dup["accreted_history_binary"][dup_count]

    star = get_star(m=model.params["m"])

    ax.plot(
        star.age[:size][star.ntpagb :], traces_s[star.ntpagb :], c="C0", linestyle=":"
    )
    ax.plot(
        star.age[:size][star.ntpagb :], winds_s[star.ntpagb :], c="C1", linestyle=":"
    )
    ax.plot(
        star.age[:size][star.ntpagb :], cores_s[star.ntpagb :], c="C2", linestyle=":"
    )
    ax.plot(
        star.age[:size][star.ntpagb :],
        0 * winds_s[star.ntpagb :],
        c="C3",
        linestyle=":",
    )
    (l1,) = ax.plot(model.age[1:], traces[1:], c="C0", label="envelope")
    (l2,) = ax.plot(model.age[1:], winds[1:], c="C1", label="wind")
    (l3,) = ax.plot(model.age[1:], cores[1:], c="C2", label="core")
    (l4,) = ax.plot(model.age[1:], accreted[1:], c="C3", label="RLOF")


for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=[l1, l2, l3, l4])
fig.supxlabel("Age (yr)", fontsize=10)
fig.supylabel("$M_\\textrm{DUP}$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w24-dup-combined-age.pgf", format="pgf")
plt.show()
plt.close()

plt.show()

# %%

rng = np.random.default_rng(seed=0)
models = rng.permutation(grid.models)
for model in models:

    if model.envelope_mass[-1] < 0.01:
        break


dup = track_DUP(model)

size = np.shape(dup["tracer_history_combined"])[1]
print(size)

winds_s = np.zeros(size)
cores_s = np.zeros(size)
traces_s = np.zeros(size)
accreted_s = np.zeros(size)


for dup_count in range(len(dup["tracer_history_combined"])):
    traces_s += dup["tracer_history_combined"][dup_count]
    winds_s += dup["wind_history_combined"][dup_count]
    cores_s += dup["core_history_combined"][dup_count]

dup = track_DUP(model)

winds = np.zeros(len(model.model_number))
cores = np.zeros(len(model.model_number))
traces = np.zeros(len(model.model_number))
accreted = np.zeros(len(model.model_number))


for dup_count in range(len(dup["tracer_history_binary"])):
    traces += dup["tracer_history_binary"][dup_count]
    winds += dup["wind_history_binary"][dup_count]
    cores += dup["core_history_binary"][dup_count]
    accreted += dup["accreted_history_binary"][dup_count]

star = get_star(m=model.params["m"])

plt.plot(
    star.m_env[:size][star.ntpagb :], traces_s[star.ntpagb :], c="C0", linestyle=":"
)
plt.plot(
    star.m_env[:size][star.ntpagb :], winds_s[star.ntpagb :], c="C1", linestyle=":"
)
plt.plot(
    star.m_env[:size][star.ntpagb :], cores_s[star.ntpagb :], c="C2", linestyle=":"
)
plt.plot(
    star.m_env[:size][star.ntpagb :], 0 * winds_s[star.ntpagb :], c="C3", linestyle=":"
)
plt.plot(model.envelope_mass[1:], traces[1:], c="C0")
plt.plot(model.envelope_mass[1:], winds[1:], c="C1")
plt.plot(model.envelope_mass[1:], cores[1:], c="C2")
plt.plot(model.envelope_mass[1:], accreted[1:], c="C3")
plt.show()


# %%

fig, axs = plt.subplots(
    2, 2, sharex=False, figsize=set_size(full), constrained_layout=True
)

seeds = [0, 2, 100, 1000]

for i, ax in enumerate(axs.flatten()):
    rng = np.random.default_rng(seed=seeds[i])
    models = rng.permutation(grid.models)
    for model in models:

        if model.envelope_mass[-1] < 0.01:
            break

    dup = track_DUP(model)

    size = np.shape(dup["tracer_history_combined"])[1]
    print(size)

    winds_s = np.zeros(size)
    cores_s = np.zeros(size)
    traces_s = np.zeros(size)
    accreted_s = np.zeros(size)

    for dup_count in range(len(dup["tracer_history_combined"])):
        traces_s += dup["tracer_history_combined"][dup_count]
        winds_s += dup["wind_history_combined"][dup_count]
        cores_s += dup["core_history_combined"][dup_count]

    dup = track_DUP(model)

    winds = np.zeros(len(model.model_number))
    cores = np.zeros(len(model.model_number))
    traces = np.zeros(len(model.model_number))
    accreted = np.zeros(len(model.model_number))

    for dup_count in range(len(dup["tracer_history_binary"])):
        traces += dup["tracer_history_binary"][dup_count]
        winds += dup["wind_history_binary"][dup_count]
        cores += dup["core_history_binary"][dup_count]
        accreted += dup["accreted_history_binary"][dup_count]

    star = get_star(m=model.params["m"])

    ax.plot(
        star.m_env[:size][star.ntpagb :], traces_s[star.ntpagb :], c="C0", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :], winds_s[star.ntpagb :], c="C1", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :], cores_s[star.ntpagb :], c="C2", linestyle=":"
    )
    ax.plot(
        star.m_env[:size][star.ntpagb :],
        0 * winds_s[star.ntpagb :],
        c="C3",
        linestyle=":",
    )
    (l1,) = ax.plot(model.envelope_mass[1:], traces[1:], c="C0", label="envelope")
    (l2,) = ax.plot(model.envelope_mass[1:], winds[1:], c="C1", label="wind")
    (l3,) = ax.plot(model.envelope_mass[1:], cores[1:], c="C2", label="core")
    (l4,) = ax.plot(model.envelope_mass[1:], accreted[1:], c="C3", label="RLOF")
    ax.set_xscale("log")


for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)

fig.legend(loc="outside upper center", ncols=4, handles=[l1, l2, l3, l4])
fig.supxlabel(r"$M_\textrm{env}$ ($M_\odot$)", fontsize=10)
fig.supylabel(r"$M_\textrm{DUP}$ ($M_\odot$)", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w24-dup-combined-m_env.pgf", format="pgf")
plt.show()
plt.close()

plt.show()
