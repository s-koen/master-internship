import numpy as np
from constants import Constants as const


def omega_Kep (Mtot, a):
    """
    Orbital (circular) angular velocity [1/s].

	Args:
        Mtot = binary mass [Msun]
        a = semi-major axis [Rsun]
    """
    omega = np.sqrt(const.G*(Mtot*const.Msun)/(a*const.Rsun)**3)
    return omega
    
def orbital_period (Mtot, a):
    """
    Orbital period [d] from Kepler's 3rd law.

	Args:
        Mtot = binary mass [Msun]
        a = semi-major axis [Rsun]
    """
    P = 2*np.pi/omega_Kep(Mtot, a)
    P_days = P/const.day
    return P_days

def v_orbit (Mtot, a):
    """
    Relative orbital (circular) velocity [km/s].

	Args:
        Mtot = binary mass [Msun]
        a = semi-major axis [Rsun]
    """
    vorb = np.sqrt(const.G*(Mtot*const.Msun)/(a*const.Rsun))
    vorb_kms = vorb*1e-5
    return vorb_kms
    
def roche_lobe (q):
    """
    Eggleton's (1983) formula for the relative roche-lobe radius, R_L/a.

	Args:
        q = M_star/M_companion
    """
    q13 = q**(1./3.)
    q23 = q13**2
    rl = 0.49*q23/(0.6*q23 + np.log(1 + q13))
    return rl


