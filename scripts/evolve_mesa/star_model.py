import read_mist_models
import glob
import numpy as np
from scipy.interpolate import interp1d

from constants import Constants as const
from rgbf import *
from mrenv import *
import mesa_reader as mr

#
# def read_stellar_models(path):
#     """
#     using the same input arguments as before, I simply just insert the MESA history data.
#     """
#     print(f"loading {path}...")
#     n = [0]
#     for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
#         print(f"\t{phase}...")
#         data = mr.MesaData(f"{path}/LOGS/{phase}/history.data")
#         n.append(n[-1] + len(data.model_number[1:]))
#
#     combined_star = {}
#     for bulk_name in data.bulk_names:
#         combined_star[bulk_name] = np.empty(n[-1])
#     combined_star["phase"] = np.empty(n[-1])
#
#     for i, phase in enumerate(["MS", "GB", "CHeB", "EAGB", "TPAGB"]):
#         data = mr.MesaData(f"{path}/LOGS/{phase}/history.data")
#         n.append(n[-1] + len(data.model_number[1:]))
#         for bulk_name in data.bulk_names:
#             if bulk_name in ["model_number", "star_age"] and i != 0:
#                 combined_star[bulk_name][n[i] : n[i + 1]] = (
#                     data.data(bulk_name)[1:] + combined_star[bulk_name][n[i] - 1]
#                 )
#             else:
#                 combined_star[bulk_name][n[i] : n[i + 1]] = data.data(bulk_name)[1:]
#         combined_star["phase"][n[i] : n[i + 1]] = i * np.ones(n[i + 1] - n[i])
#
#     Stars = [StellarModel(combined_star, n)]
#     return Stars


def read_stellar_models(path):
    """
    using the same input arguments as before, I simply just insert the MESA history data.
    """
    phases = ["MS", "GB", "CHeB", "EAGB", "TPAGB"]

    print(f"loading {path}...")

    histories = []
    n = [0]

    # read each file only once
    for phase in phases:
        print(f"  - {phase}...")
        data = mr.MesaData(f"{path}/LOGS/{phase}/history.data")
        histories.append(data)
        n.append(n[-1] + len(data.model_number[1:]))

    combined_star = {}

    # use the TPAGB history to determine available fields
    for bulk_name in histories[-1].bulk_names:
        combined_star[bulk_name] = np.empty(n[-1])

    combined_star["phase"] = np.empty(n[-1])

    print(f"")
    print(f"combining data into shared StarObject")
    for i, (phase, data) in enumerate(zip(phases, histories)):
        for bulk_name in data.bulk_names:
            values = data.data(bulk_name)[1:]

            if bulk_name in ["model_number", "star_age"] and i != 0:
                values = values + combined_star[bulk_name][n[i] - 1]

            combined_star[bulk_name][n[i] : n[i + 1]] = values

        combined_star["phase"][n[i] : n[i + 1]] = i

    return [StellarModel(combined_star, n)]


class StellarModel:

    global XHc_exh, XHec_exh
    XHc_exh, XHec_exh = 1.0e-4, 1.0e-4

    def __init__(self, track, n):
        self.nzams = np.where(track["center_h1"] / track["center_h1"][0] < 0.997)[0][0]
        self.ntams = n[1]
        self.nzacheb = n[2]
        self.ntacheb = n[3]
        self.ntpagb = n[4]
        self.npagb = n[5] - 1
        self.nwd = n[5] - 1

        self.m_init = track["star_mass"][0]
        self.Z_init = 0.014
        self.Y_init = 0.24 + 2 * self.Z_init

        self.age = track["star_age"]
        self.n_models = len(self.age)
        self.model = np.array(range(self.n_models))
        self.phase = track["phase"]
        self.varcontrol = track["varcontrol"]

        self.mass = track["star_mass"]
        self.m_core = track["he_core_mass"]
        self.m_core_CO = track["co_core_mass"]
        self.log_R = track["log_R"]
        self.log_L = track["log_L"]
        self.log_Teff = track["log_Teff"]
        self.XH_center = track["center_h1"]
        self.XHe_center = track["center_he4"]
        self.log_LH = track["log_LH"]
        self.log_LHe = track["log_LHe"]
        self.log_rhoc = track["log_cntr_Rho"]
        self.log_Tc = track["log_cntr_T"]
        self.Mdot = 10 ** track["log_abs_mdot"]
        self.log_Mdot_crit = track["quasi_adiabatic_Mdot"]
        self.surf_o16 = track["surface_o16"]
        self.surf_c12 = track["surface_c12"]
        try:
            self.envelope_o16 = track["envelope_o16"]
            self.envelope_c12 = track["envelope_c12"]
            self.envelope_c13 = track["envelope_c13"]
        except:
            pass

        self.lambda_DUP = np.pad(
            track["lambda_DUP"][-(self.n_models - self.ntpagb) :],
            (
                self.n_models
                - len(track["lambda_DUP"][-(self.n_models - self.ntpagb) :]),
                0,
            ),
            constant_values=0,
        )

        self.TP_count = np.pad(
            track["TP_count"][-(self.n_models - self.ntpagb) :],
            (
                self.n_models
                - len(track["TP_count"][-(self.n_models - self.ntpagb) :]),
                0,
            ),
            constant_values=0,
        )

        self.m_DUP = self.compute_m_DUP()

        self.m_env = self.mass - self.m_core
        self.radius = 10**self.log_R
        self.lum = 10**self.log_L

        # find critical points along track:
        self.model_ZAMS = self.nzams
        self.model_Rmin_ZAMS = self.find_Rmin_ZAMS()
        self.model_TAMS = self.ntams
        self.model_Rmax_MS = self.find_Rmax_MS()
        self.model_Rmax_HSB = (
            np.argmax(self.log_R[self.ntams : self.nzacheb + 1]) + self.ntams
        )
        self.model_Rmax = np.argmax(self.log_R[self.nzams :]) + self.nzams
        self.model_BGB = self.find_BGB()
        self.model_first_TP = self.find_first_TP()
        self.model_Rmax_EAGB = self.find_Rmax_EAGB()

        # assign stellar type, analogous to SSE (Hurley et al 2000)
        self.type = np.array(self.n_models * [0])
        self.type[self.model_ZAMS : self.model_TAMS] = 1
        self.type[self.model_TAMS : self.model_BGB] = 2
        self.type[self.model_BGB : self.nzacheb + 1] = 3
        self.type[self.nzacheb + 1 : self.ntacheb] = 4
        self.type[self.ntacheb : self.model_first_TP] = 5
        self.type[self.model_first_TP : self.npagb] = 6
        self.type[self.npagb : self.nwd] = 9
        self.type[self.nwd :] = 10

        """ quantities needed for winds and tidal interaction """
        self.vwind = wind_velocity(self.mass, self.radius, self.log_Teff, self.phase)
        self.r_core = track["he_core_radius"]
        self.m_cenv = track["M_conv"]
        self.r_cenv = track["R_conv"]
        self.tconv = conv_turnover_time(self.m_cenv, self.r_cenv, self.radius, self.lum)
        # gyration radius; k2g and k2r below are not needed but useful for testing
        self.rg2 = track["rg2"]
        # viscous tidal dissipation rate for convective envelopes
        self.k_over_T_conv = conv_dissipation_rate(
            self.m_cenv, self.r_cenv, self.mass, self.radius, self.lum, self.Z_init
        )

        # mass-loss rate
        dt = np.diff(self.age)
        dM = np.diff(self.mass)
        self.dM_dt = dM / dt

        # functions to interpolate M and log(R):
        self.M_func = interp1d(self.age, self.mass, fill_value="extrapolate")
        self.logR_func = interp1d(self.age, self.log_R, fill_value="extrapolate")
        self.rg2_func = interp1d(self.age, self.rg2, fill_value="extrapolate")

    def find_Rmin_ZAMS(self):
        nmax = np.where(self.XH_center < self.XH_center[0] - 0.1)[0][0]
        mod_Rmin_ZAMS = np.argmin(self.log_R[0 : nmax + 1])
        return mod_Rmin_ZAMS

    #    def find_TAMS (self):
    #        mod_TAMS = np.where(self.XH_center < XHc_exh)[0][0]
    #        return mod_TAMS

    def compute_m_DUP(self):

        lambda_DUP = np.asarray(self.lambda_DUP)[self.ntpagb :]
        he_core_mass = np.asarray(self.m_core)[self.ntpagb :]
        TP_count = np.asarray(self.TP_count)[self.ntpagb :]
        self.m_DUP_time = {}

        M_DUP = []
        TP_numbers = []

        unique_TPs = np.unique(TP_count)

        self.m_DUP_time[0] = np.zeros(self.n_models)

        for tp in unique_TPs:  # skip before first pulse
            self.m_DUP_time[tp] = np.zeros(self.n_models)

            # indices during this thermal pulse
            pulse_idx = np.where(TP_count == tp)[0]

            if tp == 1:
                M_DUP.append(
                    0
                )  # negligible and prevents difficulties with trying to compute the core mass growth during previous phases.
                continue

            if len(pulse_idx) == 0:
                continue

            # maximum lambda during pulse
            lambda_max = np.nanmax(lambda_DUP[pulse_idx])

            previous_tp = tp - 1
            previous_idx = np.where(TP_count == previous_tp)[0]

            if len(previous_idx) == 0:
                continue

            core_previous = np.min(he_core_mass[previous_idx])
            core_current = he_core_mass[pulse_idx[0]]
            delta_core = core_current - core_previous

            if delta_core <= 0:
                continue

            M_DUP.append(lambda_max * delta_core)

            min_index = (
                self.ntpagb
                + np.argwhere(he_core_mass == np.nanmin(he_core_mass[pulse_idx]))[0][0]
            )
            self.m_DUP_time[tp][min_index:] = lambda_max * delta_core
            self.m_DUP_time[0][min_index:] += lambda_max * delta_core

            TP_numbers.append(tp)

        return np.array(M_DUP)

    def find_Rmax_MS(self):
        mod_TAMS = self.model_TAMS
        mod_Rmax_MS = np.argmax(self.log_R[self.nzams : mod_TAMS + 1]) + self.nzams
        return mod_Rmax_MS

    #    def find_Rmax_HSB (self):
    #        mod_Rmax_HSB = np.argmax(self.log_R[self.ntams:self.nzacheb+1]) + ntams
    #        return mod_Rmax_HSB

    def find_BGB(self):
        """
        Find base of the GB, as first post-MS model with Teff < Teff_BGB
        (the latter based on SSE fit formula).
        """
        log_Teff_BGB = np.log10(Teff_at_BGB(self.mass[0], self.Z_init))
        log_Teff_min = self.log_Teff[
            np.argmin(self.log_Teff[self.nzams :]) + self.nzams
        ]
        if log_Teff_min < log_Teff_BGB:
            mod_BGB = (
                np.where(self.log_Teff[self.ntams :] < log_Teff_BGB)[0][0] + self.ntams
            )
        else:
            mod_BGB = self.model_Rmax
        # print( self.m_init, log_Teff_min, log_Teff_BGB, mod_BGB )
        return mod_BGB

    def find_first_TP(self):
        if self.n_models > self.ntpagb + 1:
            dblsh = np.where(
                (self.age >= self.age[self.ntpagb]) & (self.log_LH > self.log_LHe + 0.5)
            )[0]
            if len(dblsh) > 0:
                mod_dblsh = dblsh[0]
                tpone = np.where(
                    (self.age >= self.age[mod_dblsh])
                    & (self.log_LHe > self.log_LHe[mod_dblsh] + 0.2)
                )[0]
                if len(tpone) > 0:
                    mod_TPone = tpone[0] - 1
                else:
                    mod_TPone = self.model_Rmax
            else:
                mod_TPone = self.model_Rmax
        else:
            mod_TPone = self.model_Rmax
        return mod_TPone

    def find_Rmax_EAGB(self):
        #        mod_Rmax_EAGB = np.argmax(self.log_R[self.ntacheb:mod_TP1+1]) + ntacheb
        if self.n_models > self.ntpagb + 1:
            mod_TP1 = self.model_first_TP
            mod_Rmax_EAGB = (
                np.argmax(self.log_R[self.ntpagb : mod_TP1 + 1]) + self.ntpagb
            )
        else:
            mod_Rmax_EAGB = self.model_Rmax
        return mod_Rmax_EAGB

    def core_radius_polytropic(self):
        # mean core density for an n=1.5 polytrope:
        rho_avg = 10.0**self.log_rhoc / 5.99
        vol = self.m_core / rho_avg
        r_core = (3 * vol / (4 * np.pi)) ** (1.0 / 3)
        return r_core


def gas_pressure_scale_height(mass, radius, log_Teff, mmw):
    """
    Gas pressure scale height in the photosphere [Rsun], assuming ideal gas law holds.
        mass = stellar mass [Msun]
        radius = stellar radius [Rsun]
        log_Teff = log10(T_eff [K])
        mmw = mean molecular weight in photosphere
    """
    H_P = (
        (const.R_gas * 10**log_Teff / mmw)
        * (radius * const.Rsun) ** 2
        / (const.G * (mass * const.Msun))
    )
    return H_P / const.Rsun


def wind_velocity(mass, radius, log_Teff, phase):
    """
    Stellar wind terminal velocity [km/s], based loosely on Lamers et al 1995.
        mass = stellar mass [Msun]
        radius = stellar radius [Rsun]
        log_Teff = log10(T_eff [K]); must be numpy array
    """
    v_esc = np.sqrt(2 * const.G * (mass * const.Msun) / (radius * const.Rsun))
    vw_factor = 0.7 + 0.7 * (log_Teff - 4.0)
    vw_factor[np.where(log_Teff < 3.5)] = 0.35
    vw_factor[np.where(log_Teff > 4.0)] = 1.3
    vw_factor[np.where(log_Teff > 4.35)] = 2.6
    v_wind = vw_factor * v_esc
    v_wind_kms = v_wind / const.km

    v_wind_kms[np.argwhere(phase == 4)] = 15
    return v_wind_kms


def conv_turnover_time(m_env, r_env, radius, lum):
    """
    Convective turnover timescale [yr], following Hurley et al 2002.
        m_env = conv. envelope mass [Msun]
        r_env = conv. envelope radius [Rsun]
        radius = stellar radius [Rsun]
        lum = stellar luminosity [Lsun]
    """
    t_conv_yr = 0.4311 * (3 * m_env * r_env**2 / lum) ** (1.0 / 3.0)
    return t_conv_yr


def conv_dissipation_rate(m_cenv, r_cenv, mass, radius, lum, Z):
    """
    Viscous dissipation rate for convectve envelopes [1/yr], following the
    prescription by Preece et al 2022.
        m_cenv = conv. envelope mass [Msun]
        r_cenv = conv. envelope radius [Rsun]
        mass = stellar mass [Rsun]
        radius = stellar radius [Rsun]
        lum = stellar luminosity [Lsun]
        Z = initial metallicity
    """
    # Coefficients from Table 1.
    # (with a corrected typo in the expression for C; H. Preece, priv. comm.).
    logz = np.log10(Z)
    A = 2.72 + 0.63 * logz
    B = 0.68 - 0.219 * logz
    C = 0.12 - 0.023 * logz
    # Use alternative definition of convective timescale, eq. 22 in arXiv version of
    # Preece et al (2022).
    t_conv = 0.4311 * (3 * m_cenv * r_cenv**2 / lum) ** (1.0 / 3.0)
    t_conv = np.where(t_conv < 1e-6, 1e-6, t_conv)
    k_over_T_conv = (r_cenv / radius) ** A * (m_cenv / mass) ** B * (C / t_conv)
    return k_over_T_conv


def pulsation_period(mass, radius):
    """
    Mira pulsation period [days], from Vassilisadis & Wood 1993, Wood 1990.
        mass = stellar mass [Msun]
        radius = stellar radius [Rsun]
    """
    P = 10 ** (-2.07) * radius**1.94 * mass ** (-0.9)
    return P


def Teff_at_BGB(mass, Z):
    """
    Effective temperature at base of giant branch, obtained from SSE fit formulae.
    (Hurley et al 2000)
    """
    gbp = rgb_coeffs(Z)
    L_bgb = lbgbf(mass, gbp)
    R_bgb = rgbf(mass, L_bgb, gbp)
    Teff_bgb = (
        (L_bgb * const.Lsun) / (4 * np.pi * const.sigma_SB * (R_bgb * const.Rsun) ** 2)
    ) ** 0.25
    return Teff_bgb


def gyration_radius(mass, m_core, radius, r_core, lum, log_Teff, age, Z):
    """
    Fit formula for the dimensionless gyration radius, rg^2,
    based on the BSE implementation (Hurley et al 2002, de Mink et al 2013).
        mass = stellar mass [Msun]
        m_core = core mass [Msun]
        radius = stellar radius [Rsun]
        r_core = core radius [Rsun]
        lum = stellar luminosity [Lsun]
        log_Teff = log10(T_eff [K])
        age = stellar age [yr]
        Z = initial metallicity
    """
    log_m = np.log10(mass)
    log_m0 = log_m[0]
    Teff = 10.0**log_Teff
    # L and Teff at base of giant branch, could use lum[mod_BGB] for latter?
    Teff_bgb = Teff_at_BGB(mass[0], Z)
    L_bgb = lbgbf(mass[0], rgb_coeffs(Z))

    # Some constants needed later
    A = min(0.81, max(0.68, 0.68 + 0.4 * log_m0))
    C = max(-2.5, min(-1.5, -2.5 + 5 * log_m0))
    D = -0.1
    E = 0.025
    G = 2.0 + 8 * max(0.0, min(1.0, (0.1 - log_m0) / 0.55))

    # ZAMS and BGB values of rg^2
    k2z = min(0.21, max(0.09 - 0.27 * log_m0, 0.037 + 0.033 * log_m0))
    if log_m0 > 1.3:
        k2z = k2z - 0.055 * (log_m0 - 1.3) ** 2
    k2bgb = min(0.15, min(0.147 + 0.03 * log_m0, 0.162 - 0.04 * log_m0))
    # rg^2 of the core, assumed to be n=1.5 polytrope
    k2c = 0.21

    # below could be improved...
    self.nzams, self.ntams = 201, 453
    Rzams = radius[self.nzams]
    MS_age = age[self.ntams]

    # Envelope rg^2 for giants with convective envelopes
    F = 0.208 + 0.125 * log_m - 0.035 * log_m**2
    B = 1.0e4 * mass ** (3.0 / 2.0) / (1.0 + 0.1 * mass ** (3.0 / 2.0))
    x = ((lum - L_bgb) / B) ** 2
    y = (F - 0.033 * np.log10(L_bgb)) / k2bgb - 1.0
    k2g = (F - 0.033 * np.log10(lum) + 0.4 * x) / (1.0 + y * (L_bgb / lum) + x)

    # Envelope rg^2 for radiative-envelope stars (MS, HG, CHeB)
    k2r = (k2z - E) * (radius / Rzams) ** C + E * (radius / Rzams) ** D

    # Smooth transition when approaching the giant branch.
    # tau_env measures proximity to the Hayashi track in terms of Teff
    tau_env = (Teff_bgb / Teff - A) / (1.0 - A)
    tau_env[np.where(tau_env < 0.0)] = 0.0
    tau_env[np.where(tau_env > 1.0)] = 1.0
    k2add = tau_env**3 * (k2g - k2r)
    tau = age / MS_age
    on_MS = np.where(tau < 1.0)
    k2add[on_MS] *= tau[on_MS] ** G
    k2e = k2r + k2add

    is_giant = np.where((Teff < Teff_bgb) & (age > MS_age))
    k2e[is_giant] = k2g[is_giant]

    # Add the envelope and (very approximate) core contributions
    rg2 = (
        k2e * ((mass - m_core) / mass) + k2c * (m_core / mass) * (r_core / radius) ** 2
    )

    return rg2, k2g, k2r


# %%
