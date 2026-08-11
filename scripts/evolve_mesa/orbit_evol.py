from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint

from constants import Constants as const
from bin_input import *


class Empty:
    """Empty class to bundle together some parameter values."""

    pass


def evolve_orbit(
    Star,
    q_init,
    a_init,
    e_init,
    EvPars,
    verbosity=1,
    start_at_zams=True,
    solve_method="Radau",
):
    """
    Evolve binary orbit and stellar spin for a precomuted stellar evolution
    sequence, until either the Roche lobe is filled, or the last stellar model
    is reached.

    Args:
        Star: object containing the stellar model data (assumed to come
              from MIST)
        q_init: initial mass ratio, M_companion/M_star
        a_init: initial semimajor axis [Rsun]
        e_init: initial eccentricity
        EvPars: set of options and parameters, with the following attributes:

            zams_spin_factor: fraction of breakup spin velocity at ZAMS (default: 0.3)
            tide_model: option for tidal interaction:
                - None -> ignore tides
                - 'BSE' -> tidal disspation following Hurley+2002 (default)
                - 'Preece' -> tidal disspation in convective envelopes following
                              Preece+2022
            tidal_freq_scaling: option for frequency scaling of dissipation rate in
                                equilibrium tide model:
                - 'BSE' -> quadratic 'fast tide' correction, following BSE (default)
                - 'Zahn' -> linear 'fast tide' correction, following Zahn 1966
                - 'Duguid' -> frequency scaling from the simulatins of Duguid+2020
            tide_max_fconv: maximum on allowed multiplier for dissipation rate in
                            equiulibrium tide model (default: 1e6, i.e. not capped)
            tide_scale_factor: overall scaling factor for tidal dissipation rate
                               (default: 1.0)
            wind_model: option for wind interaction:
                - 'fast' (or None) -> fast isotropic wind without accretion
                - 'BHL' -> fast isotropic wind with BHL accretion (default)
                - 'Saladino' -> use fit to hydro simulations from Saladino+2019
            MB_scale_factor: scaling factor for the strength of magnetic braking,
                             relative to the model of Rappaport+1983 (default: 1.0)

        start_at_zams: if True (default), start evolution at ZAMS model,
                       else start at first model in Star.

    Returns:
        Bin: object containing the evolved binary and orbital parameters.
    """
    # Set default options and parameters.
    if not hasattr(EvPars, "zams_spin_factor"):
        EvPars.zams_spin_factor = 0.3
    if not hasattr(EvPars, "tide_model"):
        EvPars.tide_model = "BSE"
    if not hasattr(EvPars, "wind_model"):
        EvPars.wind_model = "Saladino"
    if not hasattr(EvPars, "MB_scale_factor"):
        EvPars.MB_scale_factor = 1.0
    if not hasattr(EvPars, "tide_scale_factor"):
        EvPars.tide_scale_factor = 1.0
    if not hasattr(EvPars, "tidal_freq_scaling"):
        EvPars.tidal_freq_scaling = "BSE"
    if not hasattr(EvPars, "tide_max_fconv"):
        EvPars.tide_max_fconv = 1e6

    # Set initial values of semi-major axis, eccentricity, companion mass,
    # and stellar spin.
    sma = a_init
    ecc = e_init
    mcomp = q_init * Star.mass[0]
    # Spin at ZAMS.
    nzams = Star.model_ZAMS
    spin = EvPars.zams_spin_factor * omega_Kep(Star.mass[nzams], Star.radius[nzams])
    if start_at_zams:
        nstart = nzams
    else:
        nstart = 0
        # Correct initial spin for pre-MS contraction, such that it has the
        # desired value at ZAMS.
        spin = (
            spin
            * (Star.radius[nzams] / Star.radius[0]) ** 2
            * Star.rg2[nzams]
            / Star.rg2[0]
        )
    # Keep track of total angular momentum lost.
    amloss = 0.0

    if verbosity:
        print(
            f"\nEvolving:    M1, M2, a, e = {Star.mass[0]:.6f} {mcomp:.6f}"
            f" {a_init:.6f} {e_init:.6f}"
        )
    if verbosity > 1:
        print(f"initial spin period: {2*np.pi/spin/const.day:.6f} d")

    # Extract some data from the stellar model, for later use.
    # Time steps and mass steps:
    dt = np.diff(Star.age)
    dM = np.diff(Star.mass)
    # Functions to interpolate M and R:
    M_func = Star.M_func
    logR_func = Star.logR_func
    # Time derivative of the log of specific moment of intertia:
    # (This is used to account for evolutionare expansion in the spin evolution.)
    Rg = np.sqrt(Star.rg2) * Star.radius
    dln_Rg2_dt = 2 * np.diff(np.log(Rg)) / dt
    # the version below is more accurate for large time steps
    dRg_over_Rg = np.diff(Rg) / Rg[:-1]
    dln_Rg2_step_dt = (1 - (1 + dRg_over_Rg) ** (-2)) / dt

    # Initialize object to collect the computed orbital evolution data.
    Bin = Empty()
    Bin.age = [Star.age[nstart]]
    Bin.a = [sma]
    Bin.e = [ecc]
    Bin.spin1 = [spin]
    Bin.type1 = [Star.type[nstart]]
    Bin.m1 = [Star.mass[nstart]]
    Bin.m2 = [mcomp]
    Bin.amloss = [amloss]
    Bin.beta = [0]
    Bin.eta = [0]
    Bin.vw_over_vorb = [0]
    Bin.fconv = []

    RL_0 = roche_lobe(Star.mass[nstart] / mcomp) * sma * (1 - ecc)
    if Star.radius[nstart] > RL_0:
        Bin.status = "collide"
    else:
        Bin.status = "detached"

    # Initialize object to collect the current stellar parameters.
    SP = Empty()

    # Keep track of steps taken by each integration method.
    track_int_simple = []
    track_int_solve = []
    solve_steps = 0

    # Start looping over the stellar model sequence index.
    i = nstart

    while (Bin.status == "detached") & (i < len(dM)):

        # Zero-duration stellar timestep: nothing evolves.
        if dt[i] == 0:
            Bin.age.append(Star.age[i + 1])
            Bin.a.append(sma)
            Bin.e.append(ecc)
            Bin.spin1.append(spin)
            Bin.type1.append(Star.type[i + 1])
            Bin.m1.append(Star.mass[i + 1])
            Bin.m2.append(mcomp)
            Bin.amloss.append(amloss)
            Bin.beta.append(0)
            Bin.eta.append(0)
            Bin.vw_over_vorb.append(0)
            Bin.fconv.append(0)

            track_int_simple.append(i)
            i += 1
            continue

        # Define stellar parameters required for integration.
        SP.mass = Star.mass[i]
        SP.radius = Star.radius[i]
        SP.dM_dt = Star.dM_dt[i]
        SP.dln_Rg2_dt = dln_Rg2_step_dt[i]
        # Take average values over stellar timestep for some quantities.
        SP.tconv = max(1e-6, 0.5 * (Star.tconv[i] + Star.tconv[i + 1]))
        SP.k_over_T_conv = 0.5 * (Star.k_over_T_conv[i] + Star.k_over_T_conv[i + 1])
        SP.m_cenv = 0.5 * (Star.m_cenv[i] + Star.m_cenv[i + 1])
        SP.rg2 = 0.5 * (Star.rg2[i] + Star.rg2[i + 1])
        SP.vwind = 0.5 * (Star.vwind[i] + Star.vwind[i + 1])

        # Compute approximate changes over stellar model timestep.
        dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt, beta, eta, vw_over_vorb, f_conv = (
            orbit_evol_eqs(sma, ecc, spin, mcomp, SP, EvPars)
        )
        sma_next = max(1e-30, sma * (1 + dln_a_dt * dt[i]))
        ecc_next = ecc + de_dt * dt[i]
        spin_next = spin + dspin_dt * dt[i]
        mcomp_next = mcomp + dmcomp_dt * dt[i]
        if np.isnan(mcomp_next) and not np.isnan(mcomp):
            print(i, mcomp, dmcomp_dt, dt[i])
        amloss_next = amloss + dj_dt * dt[i]

        # Expected relative changes in orbit, mass, spin and tidal
        # synchronization.
        del_orbit = abs(dt[i] * dln_a_dt)
        del_mass = 10 * abs(dM[i] / Star.mass[i])
        if EvPars.tide_model == None:
            del_tide = 0
        else:
            tsync, tcirc, f_conv = tidal_timescale(sma, ecc, spin, mcomp, SP, EvPars)
            del_tide = dt[i] / min(tsync, tcirc)

        # Decide whether approximate changes are small enough.
        slowly_changing = (
            max(del_mass, del_orbit, del_tide) < EvPars.eps or EvPars.simple_only
        )

        # Check for Roche-lobe filling at periastron at end of timestep.
        # If so, then do proper integration.
        log_RL_next = np.log10(
            roche_lobe(Star.mass[i + 1] / mcomp_next) * sma_next * (1 - ecc_next)
        )
        if Star.log_R[i + 1] > log_RL_next:
            slowly_changing = False
            if verbosity > 1:
                print(
                    f"\n[!] RL overfilled at model {i+1} by "
                    f"d(log R) = {Star.log_R[i+1] - log_RL_next:.6f}",
                    end="",
                )

        if (verbosity == 1) & (i % 50 == 0):
            print(".", end="", flush=True)
        if (verbosity > 1) & (i % 20 == 0):
            print(
                f"{Star.model[i]} {del_orbit:.6f} {del_mass:.6f} {del_tide:.6f}"
                f" {slowly_changing}"
            )

        if slowly_changing:
            #
            # Adopt approximate changes over current timestep.
            #
            sma = sma_next
            ecc = ecc_next
            spin = spin_next
            mcomp = mcomp_next
            amloss = amloss_next

            Bin.age.append(Star.age[i + 1])
            Bin.a.append(sma)
            Bin.e.append(ecc)
            Bin.spin1.append(spin)
            Bin.type1.append(Star.type[i + 1])
            Bin.m1.append(Star.mass[i + 1])
            Bin.m2.append(mcomp)
            Bin.amloss.append(amloss)
            Bin.beta.append(beta)
            Bin.eta.append(eta)
            Bin.vw_over_vorb.append(vw_over_vorb)
            Bin.fconv.append(f_conv)

            track_int_simple.append(i)

        else:
            #
            # Integrate over current timestep using ODE solver.
            #
            SP.dln_Rg2_dt = dln_Rg2_dt[i]

            t_interval = np.array([Star.age[i], Star.age[i + 1]])
            y0 = [np.log(sma), ecc, spin, mcomp, amloss]

            sol = solve_ivp(
                derivs,
                t_interval,
                y0,
                events=fill_RL,
                method=solve_method,
                args=(M_func, logR_func, SP, EvPars),
            )

            sma = np.exp(sol.y[0, -1])
            ecc = sol.y[1, -1]
            spin = sol.y[2, -1]
            mcomp = sol.y[3, -1]
            amloss = sol.y[4, -1]

            if len(sol.t_events[0]) > 0:
                Bin.status = "RLOF"

            Bin.age.extend(sol.t)
            Bin.a.extend(np.exp(sol.y[0,]))
            Bin.e.extend(sol.y[1,])
            Bin.spin1.extend(sol.y[2,])
            Bin.type1.extend(len(sol.t) * [Star.type[i + 1]])
            Bin.m1.extend(M_func(sol.t))
            Bin.m2.extend(sol.y[3,])
            Bin.amloss.extend(sol.y[4,])
            Bin.beta.append(0)
            Bin.eta.append(0)
            Bin.vw_over_vorb.append(0)
            Bin.fconv.append(0)

            track_int_solve.append(i)
            solve_steps += len(sol.t)

        i += 1

    Bin.age = np.array(Bin.age)
    Bin.a = np.array(Bin.a)
    Bin.e = np.array(Bin.e)
    Bin.spin1 = np.array(Bin.spin1)
    Bin.type1 = np.array(Bin.type1)
    Bin.m1 = np.array(Bin.m1)
    Bin.m2 = np.array(Bin.m2)
    Bin.amloss = np.array(Bin.amloss)

    if verbosity:
        print(
            f"\n{Bin.status:8s} {Bin.type1[-1]:2d}  M1, M2, a, e = "
            f"{Bin.m1[-1]:.6f} {Bin.m2[-1]:.6f} {Bin.a[-1]:.6f} {Bin.e[-1]:.6f}"
        )
        print(
            f"steps taken: {len(track_int_simple)} by simple intergration, "
            f"{len(track_int_solve)} ({solve_steps}) using solve_ivp"
        )

    # Check how well angular momentum is conserved.
    am_orb = (
        (Bin.m1 * Bin.m2)
        / (Bin.m1 + Bin.m2)
        * Bin.a**2
        * omega_Kep(Bin.m1 + Bin.m2, Bin.a)
        * np.sqrt(1 - Bin.e**2)
    )
    Rg2 = Star.rg2_func(Bin.age) * 10 ** (2 * Star.logR_func(Bin.age))
    am_rot = Bin.m1 * Rg2 * Bin.spin1
    am_sum = am_orb + am_rot - Bin.amloss
    am_cons = abs(am_sum / am_sum[0] - 1.0).max()

    if verbosity:
        print(f"AM conserved up to {100*am_cons:.4f}%")

    Bin.steps_simple = len(track_int_simple)
    Bin.steps_solver = len(track_int_solve)
    Bin.steps_ivp = solve_steps
    Bin.am_cons = am_cons

    return Bin


def fill_RL(t, y, M_func, logR_func, SP, EvPars):
    """
    Trigger a solve_ivp event when Roche-lobe is filled at periastron.
    """
    ln_a = max(-30, min(30, y[0]))
    a = np.exp(ln_a)
    e = y[1]
    mcomp = y[3]

    mass = M_func(t)
    log_R = logR_func(t)
    # For now simply use RL*(1-e), could improve this following Sepinsky+ 2007.
    log_RL = np.log10(roche_lobe(mass / mcomp) * a * (1 - e))

    return log_R - log_RL


fill_RL.terminal = True
fill_RL.direction = 1


def derivs(t, y, M_func, logR_func, SP, EvPars):
    """
    Wrapper to pass the orbital evolution DEs to solve_ivp.
    """
    ln_a = max(-30, min(30, y[0]))
    a = np.exp(ln_a)
    e = y[1]
    spin = y[2]
    mcomp = y[3]

    # Use interpolated values for current stellar mass and radius.
    SP.mass = M_func(t)
    SP.radius = 10 ** logR_func(t)

    dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt, beta, eta, vw_over_vorb, f_conv = (
        orbit_evol_eqs(a, e, spin, mcomp, SP, EvPars)
    )

    return [dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt]


def orbit_evol_eqs(a, e, spin, mcomp, SP, EvPars):
    """
    Differential equations for orbital evolution.
    """
    # Effects of wind mass loss.
    dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt, beta, eta, vw_over_vorb = wind_loss(
        a, e, spin, mcomp, SP, EvPars
    )

    # Effects of evolutionary expansion.
    dspin_dt -= SP.dln_Rg2_dt * spin

    f_conv = 1
    if EvPars.tide_model != None:
        # Effects of tides.
        dln_a_dt_tid, de_dt_tid, dspin_dt_tid, f_conv = tides(
            a, e, spin, mcomp, SP, EvPars
        )
        dln_a_dt += dln_a_dt_tid
        de_dt += de_dt_tid
        dspin_dt += dspin_dt_tid

    return dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt, beta, eta, vw_over_vorb, f_conv


def wind_loss(a, e, spin, mcomp, SP, EvPars):
    """
    Orbital changes and stellar spin-down due to wind mass loss and accretion,
    also allowing for magnetic braking.
    Changes in orbital angular momentum and companion mass are parameterised in
    terms of the dimensionless parameters eta and beta, following Saladino et al
    (2018, 2019).

    Args:
        a: semimajor axis [Rsun]
        e: eccentricity
        spin: stellar spin frequency [1/s]
        mcomp: companion mass [Msun]
        SP: object containing the current stellar parameters
        EvPars: object containing model choices and parameters.

    Returns:
        dlna_dt: rate of change of ln(semimajor axis) [1/yr]
        de_dt: rate of change of eccentricity [1/yr]
        dspin_dt: rate of change of spin frequency [1/(s*yr)]
        dmcomp_dt: rate of change of companion mass [Msun/yr]
        dj_dt: rate of overall angular momentum loss [(Msun*Rsun^2)/(s*yr)]
    """
    Q = SP.mass / mcomp
    vw_over_vorb = SP.vwind / v_orbit(SP.mass + mcomp, a)
    if EvPars.wind_model == "Saladino":
        # Accretion and AM loss from SPH simulations by Saladino et al (2019).
        # (N.B. model results and fit are only for circular orbits...)
        eta = eta_Sal(Q, vw_over_vorb)
        beta = beta_Sal(Q, vw_over_vorb)
    elif EvPars.wind_model == "BHL":
        # Bondi-Hoyle-Lyttleton accretion, with AM loss from fast winds.
        eta = (1 + Q) ** (-2)
        beta = beta_BHL(Q, vw_over_vorb)
    else:
        # Only fast-wind AM loss, no accretion.
        eta = (1 + Q) ** (-2)
        beta = 0.0
    # Change in companion mass, with accretion efficiency modified for eccentric
    # orbits (this is only strictly valid in the fast-wind BHL regime).
    beta = beta / np.sqrt(1 - e**2)
    dmcomp_dt = -beta * SP.dM_dt

    # Change in eccentricity following Dosopoulou & Kalogera (2016), eq. 82.
    # This assumes BHL mass transfer in the fast wind regime!
    de_dt = e * beta * SP.dM_dt / SP.mass * (2 * Q - 2 + Q / (1 + Q))

    # Change in semi-major axis, from angular momentum conservation.
    dln_a_dt = (
        SP.dM_dt
        / SP.mass
        * (
            2 * (1 - beta) ** 1 * eta * (1 + Q)
            - 2
            + 2 * beta * Q
            + (Q * (1 - beta)) / (1 + Q)
        )
        + 2 * e / (1 - e**2) * de_dt
    )

    # Change in spin frequency due to mass loss and/or magnetic braking.
    mom_inerta = SP.rg2 * SP.mass * SP.radius**2
    dj_dt_rot = SP.dM_dt * SP.radius**2 * spin / 1.5
    if EvPars.MB_scale_factor > 0:
        # Include AM loss by magnetic braking, based on Rappaport et al (1983)
        # with gamma=3, and reducced for convective envelope mass fraction < 0.02
        # (modified from Podsiadlowski et al 2002).
        dj_dt_mb = -7.64 * EvPars.MB_scale_factor * mom_inerta * SP.radius * spin**3
        if SP.m_cenv / SP.mass < 0.02:
            dj_dt_mb *= np.sqrt(50 * SP.m_cenv / SP.mass)
        dj_dt_rot = min(dj_dt_rot, dj_dt_mb)
    dspin_dt = dj_dt_rot / mom_inerta - SP.dM_dt / SP.mass * spin

    # Change in orbital and total angular momentum.
    dj_dt_orb = (
        (1 - beta)
        * eta
        * a**2
        * np.sqrt(1 - e**2)
        * omega_Kep(SP.mass + mcomp, a)
        * SP.dM_dt
    )
    dj_dt = dj_dt_rot + dj_dt_orb

    return dln_a_dt, de_dt, dspin_dt, dmcomp_dt, dj_dt, beta, eta, vw_over_vorb


def beta_BHL(q, v, alpha=0.75):
    """
    Bondi-Hoyle-Lyttleton accretion efficiency, for a circular orbit.
        q = M_loser/M_companion
        v = ratio of outflow velocity to orbital velocity
        alpha = BHL efficiency parameter (0.75 by default)
    """
    vrel = np.sqrt(1 + v**2)
    beta = alpha / ((1 + q) ** 2 * v * vrel**3)
    return beta


def beta_Sal(q, v):
    """
    Fit to the accretion efficieny from the SPH simulation results of
    Saladino et al (2019) [eq.11 in the paper].
        q = M_loser/M_companion
        v = ratio of outflow velocity to orbital velocity
    """
    c1 = 1.7 + 0.3 * q
    c2 = 0.5 + 0.2 * q
    alpha = 1.0 / (c1 + (c2 * v) ** 5) + 0.75
    beta_max = min(0.3, 1.4 / q**2)
    beta = min(beta_BHL(q, v, alpha), beta_max)
    return beta


def eta_Sal(q, v):
    """
    Specific orbital angular momentum loss (in units of J/mu = a^2 omega) fitted
    to the SPH simulation results of Saladino et al (2019) [eq.7 in the paper].
        q = M_loser/M_companion
        v = ratio of outflow velocity to orbital velocity
    """
    eta_iso = (1 + q) ** (-2)
    c1 = max(q, 0.6 * q**1.7)
    c2 = 1.5 + 0.3 * q
    eta = min(1.0 / (c1 + (c2 * v) ** 3) + eta_iso, 0.6)
    return eta


def tides(a, e, spin, mcomp, SP, EvPars):
    """
    Tidal evolution equations for the equilibrium tide, following Hut (1981).

    Args:
        a: semimajor axis [Rsun]
        e: eccentricity
        spin: stellar spin frequency [1/s]
        mcomp: companion mass [Msun]
        SP: object containing some relevant stellar parameters:
            mass, radius: stellar mass and radius [Msun, Rsun]
            rg2: dimensionless moment of inertia, I/(M*R^2)
        EvPars: object containing model choices and parameters.

    Returns:
        dlna_dt: rate of change of ln(semimajor axis) [1/yr]
        de_dt: rate of change of eccentricity [1/yr]
        dspin_dt: rate of change of spin frequency [1/(s*yr)]
    """
    # For non-circular orbits, compute eccentricity-dependent factors.
    ge = 1.0 * np.ones(6)
    if e > 1e-7:
        # fe[i] are the factors f_i(e^2), where element 0 is redundant.
        # pe[i] are the powers of (1 - e^2) by which fe[i] is to be divided
        # in the evolution equations, yielding ge[i].
        e2 = e**2
        e4 = e**4
        e6 = e**6
        e8 = e**8
        fe = np.array(
            [
                1,
                1 + 31 * e2 / 2 + 255 * e4 / 8 + 185 * e6 / 16 + 25 * e8 / 64,
                1 + 15 * e2 / 2 + 45 * e4 / 8 + 5 * e6 / 16,
                1 + 15 * e2 / 4 + 15 * e4 / 8 + 5 * e6 / 64,
                1 + 3 * e2 / 2 + e4 / 8,
                1 + 3 * e2 + 3 * e4 / 8,
            ]
        )
        pe = np.array([0, 7.5, 6, 6.5, 5, 4.5])
        ge = fe / (1 - e2) ** pe

    q = mcomp / SP.mass
    R_over_a = SP.radius / a
    # mean orbital frequency (2*pi/P_orb):
    o_orb = omega_Kep(SP.mass + mcomp, a)

    # The factor k/T, describing the rate of dissipation:
    k_over_T, f_conv = dissipation_rate(a, q, spin, o_orb, SP, EvPars)
    tide_factor = k_over_T * (1 + q) * q * (R_over_a) ** 8

    # Change in semi-major axis, Hut's eq. (9)
    dln_a_dt = -6 * tide_factor * (ge[1] - ge[2] * spin / o_orb)

    # Change in eccentricity, Hut's eq. (10)
    de_dt = 0
    if e > 1e-7:
        de_dt = -27 * tide_factor * e * (ge[3] - (11.0 / 18) * ge[4] * spin / o_orb)

    # Change in spin frequency, Hut's eq. (11)
    dspin_dt = (
        3 * k_over_T * q**2 / SP.rg2 * (R_over_a) ** 6 * (ge[2] * o_orb - ge[5] * spin)
    )

    return dln_a_dt, de_dt, dspin_dt, f_conv


def dissipation_rate(a, q, spin, o_orb, SP, EvPars):
    """
    Tidal dissipation rate factor (k/T) used in Hut's equations.

    Args:
        a: semimajor axis [Rsun]
        q: mass ratio (M_comp/M_star)
        spin: stellar spin frequency [1/s]
        o_orb: mean orbital frequency, 2*pi/P_orb [1/s]
        SP: object containing some relevant stellar parameters:
            mass, radius: stellar mass and radius [Msun, Rsun]
            m_cenv: convective envelope mass [Msun]
            tconv: convective turnover timescale [yr]
            k_over_T_conv: viscous dissipation rate in convective envelope [1/yr]
        EvPars: object containing model choices and parameters.

    Returns: k/T in [1/yr].
    """
    # Convective damping of equilibrium tide.
    if EvPars.tide_model == "Preece":
        # Viscous dissipation rate following Preece et al (2022).
        k_over_T_conv = SP.k_over_T_conv
    elif EvPars.tide_model == "BSE":
        # Use the BSE prescription from Hurley et al (2002).
        k_over_T_conv = (2.0 / 21.0) * (SP.m_cenv / SP.mass) / SP.tconv
    else:
        k_over_T_conv = 0

    # Frequency scaling (correction for fast tides).
    # f_conv is a multiplier for the effective viscosity implied by MLT,
    #     nu_eff = f_conv * (w_conv*l_mix)/3
    o_tid = max(1e-10, const.yr * abs(o_orb - spin))
    p_tid = 2 * np.pi / o_tid
    if EvPars.tidal_freq_scaling == "Duguid":
        # Frequency scaling of effectve viscosity from Duguid et al (2020).
        o_tid_over_conv = SP.tconv * o_tid
        f_visc = min(5, 0.5 / np.sqrt(o_tid_over_conv), 5.59 / o_tid_over_conv**2)
        f_conv = 3 * f_visc  # ... since f_visc = 1/3 is assumed by default
    elif EvPars.tidal_freq_scaling == "Zahn":
        # Fast tide reduction from Zahn (1966).
        f_conv = min(1.0, 0.5 * p_tid / SP.tconv)

    elif EvPars.tidal_freq_scaling == "None":
        f_conv = 1
    else:
        # BSE method, following Goldreich & Nicholson (1977).
        f_conv = min(1.0, (0.5 * p_tid / SP.tconv) ** 2)
    f_conv = min(f_conv, EvPars.tide_max_fconv)

    k_over_T_conv *= f_conv

    # Radiative damping of dynamical tide, for stars with radiative envelopes,
    # following  Hurley et al (2002; based on Zahn 1975, 1977).
    if SP.m_cenv < 1e-2:
        E2 = 1.592e-9 * SP.mass**2.84
        k_over_T_rad = (
            1.98e4
            * E2
            * np.sqrt(SP.mass / SP.radius**3)
            * (1 + q) ** (5.0 / 6.0)
            * (SP.radius / a) ** 2.5
        )
    else:
        k_over_T_rad = 0

    # Use whichever is strongest. In practice this means radiative damping
    # only when convective envelope mass (SP.m_cenv) equals zero.
    k_over_T = max(k_over_T_conv, k_over_T_rad)

    # Scale dissipation rate by an (ad hoc) overall factor
    k_over_T *= EvPars.tide_scale_factor

    return k_over_T, f_conv


def tidal_timescale(a, e, spin, mcomp, SP, EvPars):
    """
    Tidal synchronisation and circularisation timescales [yr].
    """
    dln_a_dt, de_dt, dspin_dt, f_conv = tides(a, e, spin, mcomp, SP, EvPars)
    o_orb = omega_Kep(SP.mass + mcomp, a)
    tsync = abs((spin - o_orb) / dspin_dt)
    tcirc = 1e12
    if e > 1e-7:
        tcirc = abs(e / de_dt)

    return tsync, tcirc, f_conv
