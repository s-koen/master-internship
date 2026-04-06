import numpy as np

def conv_envelope_size (mass, m_core, radius, r_core, lum, log_Teff, age, mod_TAMS, mod_BGB, mod_HeI):
    """
    Fit formula for the convective envelope size (mass and radial extent) 
    based on the BSE implementation (Hurley et al 2002).
        mass = stellar mass [Msun]
        m_core = core mass [Msun]
        radius = stellar radius [Rsun]
        r_core = core radius [Rsun]
        lum = stellar luminosity [Lsun]
        log_Teff = log10(T_eff [K])
        age = stellar age [yr]
        mod_TAMS, mod_BGB_ mod_HeI = model numbers at TAMS, BGB and He-ignition
    """
    log_m = np.log10(mass)
    log_m0 = log_m[0]
    Teff = 10.0**log_Teff
    # L and Teff at base of giant branch
    Teff_bgb = Teff[mod_BGB]
    L_bgb = lum[mod_BGB]
    L_HeI = lum[mod_HeI]

    # Some constants needed later
    A = min(0.81, max(0.68, 0.68 + 0.4*log_m0))

    MS_age = age[mod_TAMS]
    pre_FGB = np.where(age < age[mod_BGB])
    on_FGB = np.where((age >= age[mod_BGB]) & (age <= age[mod_HeI]))

    # Relative CE mass and radius for giant-like stars, equal to 1 for stars beyond the FGB
    menvg = np.ones(len(mass))
    renvg = np.ones(len(mass))
    # modified values for FGB stars close to the BGB: 
    x = min(3.0, L_HeI/L_bgb)
    tau = (x - lum/L_bgb)/(x - 1.0)
    tau[np.where(tau < 0.0)] = 0.0
    tau[np.where(tau > 1.0)] = 1.0
    menvg[on_FGB] = 1.0 - 0.5*tau[on_FGB]**4 # 2
    renvg[on_FGB] = 1.0 - 0.35*tau[on_FGB]**4 # 2
    # proxy values for stars not yet on the FGB:
    menvg[pre_FGB] = 0.5
    renvg[pre_FGB] = 0.65

    # Relative CE mass and radius for stars hotter than the giant branch
    # (essentially, HG and CHeB; will be modified below for MS stars).
    # tau_env measures proximity to the Hayashi track in terms of Teff.
    tau_env = (Teff_bgb/Teff - A)/(1.0 - A)
    tau_env[np.where(tau_env < 0.0)] = 0.0
    tau_env[np.where(tau_env > 1.0)] = 1.0
    menv = menvg * tau_env**5
    renv = renvg * tau_env**1.25
    # zero-age values, depending on iniital mass:
    x = max(0.0, min(1.0, (0.10 - log_m0)/0.55))
    menvz = 0.18*x + 0.82*x**5
    renvz = 0.4*x**0.25 + 0.6*x**10
    y = 2.0 + 8.0*x
    # values at end of the MS:
    menvt = menv[mod_TAMS] + 1e-14
    renvt = renv[mod_TAMS] + 1e-14
    # Modified CE mass and radius during MS evolution
    tau = age/MS_age
    on_MS = np.where(tau < 1.0)
    menv[on_MS] = menvz + tau[on_MS]**y * menv[on_MS]/menvt * (menvt - menvz)
    renv[on_MS] = renvz + tau[on_MS]**y * renv[on_MS]/renvt * (renvt - renvz)
    
    # Relative CE mass and radius for true giants
    on_GB = np.where((Teff < Teff_bgb) & (age > MS_age))
    menv[on_GB] = menvg[on_GB]
    renv[on_GB] = renvg[on_GB]

    # Absolte values for CE mass and radius
    menv = menv*(mass - m_core)
    renv = renv*(radius - r_core)

    return menv, renv
