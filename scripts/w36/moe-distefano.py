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
plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad

# %%

"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                                                        ▌
▐  ████████╗██╗    ██╗██╗███╗   ██╗    ███████╗██╗  ██╗ ██████╗███████╗███████╗███████╗  ▌
▐  ╚══██╔══╝██║    ██║██║████╗  ██║    ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██╔════╝██╔════╝  ▌
▐     ██║   ██║ █╗ ██║██║██╔██╗ ██║    █████╗   ╚███╔╝ ██║     █████╗  ███████╗███████╗  ▌
▐     ██║   ██║███╗██║██║██║╚██╗██║    ██╔══╝   ██╔██╗ ██║     ██╔══╝  ╚════██║╚════██║  ▌
▐     ██║   ╚███╔███╔╝██║██║ ╚████║    ███████╗██╔╝ ██╗╚██████╗███████╗███████║███████║  ▌
▐     ╚═╝    ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝╚══════╝  ▌
▐                                                                                        ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def __get_F_twin_logP_lt_1(m1: float) -> float:
    """
    computes the excess twin fraction at short orbital periods.
    Equation (6) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.

    --- returns ---
        F_twin_logP_lt_1: float
            the excess twin fraction for log P < 1
    """
    return 0.3 - 0.15 * np.log10(m1)


def __get_log_P_twin(m1: float) -> float:
    """
    computes the maximum orbital period of the observed excess twin population.
    Equation (7) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.

    --- returns ---
        log_P_twin: float
            log10 of the maximum orbital period of the observed excess twin population.
    """
    if m1 < 6.5:
        return 8.0 - m1
    return 1.5


def get_F_twin(m1: float, P: float) -> float:
    """
    computes the excess twin fraction.
    Equation (5) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        F_twin: float
            the excess twin fraction
    """

    if np.log10(P) < 1:
        return __get_F_twin_logP_lt_1(m1)
    else:
        log_P_twin = __get_log_P_twin(m1)
        if np.log10(P) < log_P_twin:
            return __get_F_twin_logP_lt_1(m1) * (
                1 - (np.log10(P) - 1) / (log_P_twin - 1)
            )
        else:
            return 0


"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                     ▌
▐   ██████╗  █████╗ ███╗   ███╗███╗   ███╗ █████╗     ▌
▐  ██╔════╝ ██╔══██╗████╗ ████║████╗ ████║██╔══██╗    ▌
▐  ██║  ███╗███████║██╔████╔██║██╔████╔██║███████║    ▌
▐  ██║   ██║██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║    ▌
▐  ╚██████╔╝██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║    ▌
▐   ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝    ▌
▐                                                     ▌
▐  ███████╗██╗      ██████╗ ██████╗ ███████╗███████╗  ▌
▐  ██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝██╔════╝  ▌
▐  ███████╗██║     ██║   ██║██████╔╝█████╗  ███████╗  ▌
▐  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝  ╚════██║  ▌
▐  ███████║███████╗╚██████╔╝██║     ███████╗███████║  ▌
▐  ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚══════╝  ▌
▐                                                     ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def __get_gamma_large_q_solar(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for solar type primaries.
    Equation (9) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_large_q: float
            the power law slope
    """
    if m1 < 0.8 or m1 > 1.2:
        raise Exception(
            f"function __get_gamma_large_q_solar() can only be called with m1 between 0.8 and 1.2.\nsupplied m1 = {m1}"
        )
    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_large_q_solar() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    if log_P < 5:
        return -0.5
    else:
        return -0.5 - 0.3 * (log_P - 5)


def __get_gamma_large_q_mid(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for the midpoint of A/ late-B primaries.
    Equation (10) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_large_q: float
            the power law slope
    """
    if m1 != 3.5:
        raise Exception(
            f"function __get_gamma_large_q_mid() can only be called with m1 = 3.5, not {m1}"
        )

    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_large_q_mid() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    if log_P < 1.0:
        return -0.5
    elif log_P < 4.5:
        return -0.5 - 0.2 * (log_P - 1.0)
    elif log_P < 6.5:
        return -1.2 - 0.4 * (log_P - 4.5)
    return -2.0


def __get_gamma_large_q_heavy(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for mid-B, early-B, and O-type stars.
    Equation (11) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_large_q: float
            the power law slope
    """
    if m1 < 6.0:
        raise Exception(
            f"function __get_gamma_large_q_heavy() can only be called with m1 > 6.0.\nsupplied m1 = {m1}"
        )

    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_large_q_heavy() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    if log_P < 1.0:
        return -0.5
    elif log_P < 2.0:
        return -0.5 - 0.9 * (log_P - 1)
    elif log_P < 4.0:
        return -1.4 - 0.3 * (log_P - 2)
    return -2.0


def get_gamma_large_q(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for a given primary mass and orbital period.
    Equation (9, 10 and 11) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_large_q: float
            the power law slope
    """

    if m1 < 1.2:
        return __get_gamma_large_q_solar(m1, P)
    elif m1 < 3.5:
        low = __get_gamma_large_q_solar(1.2, P)
        high = __get_gamma_large_q_mid(3.5, P)
        frac = (m1 - 1.2) / (3.5 - 1.2)
        if frac > 1 or frac < 0:
            raise Exception(
                f"function get_gamma_large_q not meant for extrapolation.\nfrac should be between 0 and 1 but is {frac:.3f}"
            )
        return frac * high + (1 - frac) * low
    elif m1 < 6:
        low = __get_gamma_large_q_mid(3.5, P)
        high = __get_gamma_large_q_heavy(6, P)
        frac = (m1 - 3.5) / (6.0 - 3.5)
        if frac > 1 or frac < 0:
            raise Exception(
                f"function get_gamma_large_q not meant for extrapolation.\nfrac should be between 0 and 1 but is {frac:.3f}"
            )
        return frac * high + (1 - frac) * low
    return __get_gamma_large_q_heavy(m1, P)


def __get_gamma_small_q_solar(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for solar type primaries.
    Equation (13) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_small_q: float
            the power law slope
    """
    if m1 < 0.8 or m1 > 1.2:
        raise Exception(
            f"function __get_gamma_small_q_solar() can only be called with m1 between 0.8 and 1.2.\nsupplied m1 = {m1}"
        )
    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_small_q_solar() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    return 0.3


def __get_gamma_small_q_mid(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for the midpoint of A/ late-B primaries.
    Equation (14) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_small_q: float
            the power law slope
    """
    if m1 != 3.5:
        raise Exception(
            f"function __get_gamma_small_q_mid() can only be called with m1 = 3.5, not {m1}"
        )

    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_small_q_mid() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    if log_P < 2.5:
        return 0.2
    elif log_P < 5.5:
        return 0.2 - 0.3 * (log_P - 2.5)
    return -0.7 - 0.2 * (log_P - 5.5)


def __get_gamma_small_q_heavy(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for mid-B, early-B, and O-type stars.
    Equation (15) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_small_q: float
            the power law slope
    """
    if m1 < 6.0:
        raise Exception(
            f"function __get_gamma_small_q_heavy() can only be called with m1 > 6.0.\nsupplied m1 = {m1}"
        )

    log_P = np.log10(P)
    if log_P < 0.2 or log_P > 8.0:
        raise Exception(
            f"function __get_gamma_small_q_heavy() can only be called with log_P between 0.2 and 8.0. \nsupplied log_P = {log_P}"
        )

    if log_P < 1.0:
        return 0.1
    elif log_P < 3.0:
        return 0.1 - 0.15 * (log_P - 1)
    elif log_P < 5.6:
        return -0.2 - 0.50 * (log_P - 3)
    return -1.5


def get_gamma_small_q(m1: float, P: float) -> float:
    """
    computes the power law slope for the mass distribution
    for a given primary mass and orbital period.
    Equation (9, 10 and 11) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        gamma_small_q: float
            the power law slope
    """

    if m1 < 1.2:
        return __get_gamma_small_q_solar(m1, P)
    elif m1 < 3.5:
        low = __get_gamma_small_q_solar(1.2, P)
        high = __get_gamma_small_q_mid(3.5, P)
        frac = (m1 - 1.2) / (3.5 - 1.2)
        if frac > 1 or frac < 0:
            raise Exception(
                f"function get_gamma_small_q not meant for extrapolation.\nfrac should be between 0 and 1 but is {frac:.3f}"
            )
        return frac * high + (1 - frac) * low
    elif m1 < 6:
        low = __get_gamma_small_q_mid(3.5, P)
        high = __get_gamma_small_q_heavy(6, P)
        frac = (m1 - 3.5) / (6.0 - 3.5)
        if frac > 1 or frac < 0:
            raise Exception(
                f"function get_gamma_small_q not meant for extrapolation.\nfrac should be between 0 and 1 but is {frac:.3f}"
            )
        return frac * high + (1 - frac) * low
    return __get_gamma_small_q_heavy(m1, P)


"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                                                           ▌
▐      ███╗   ███╗ █████╗ ███████╗███████╗      ██████╗  █████╗ ████████╗██╗ ██████╗        ▌
▐      ████╗ ████║██╔══██╗██╔════╝██╔════╝      ██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗       ▌
▐      ██╔████╔██║███████║███████╗███████╗█████╗██████╔╝███████║   ██║   ██║██║   ██║       ▌
▐      ██║╚██╔╝██║██╔══██║╚════██║╚════██║╚════╝██╔══██╗██╔══██║   ██║   ██║██║   ██║       ▌
▐      ██║ ╚═╝ ██║██║  ██║███████║███████║      ██║  ██║██║  ██║   ██║   ██║╚██████╔╝       ▌
▐      ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝        ▌
▐                                                                                           ▌
▐  ██████╗ ██╗███████╗████████╗██████╗ ██╗██████╗ ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗  ▌
▐  ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗██║██╔══██╗██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║  ▌
▐  ██║  ██║██║███████╗   ██║   ██████╔╝██║██████╔╝██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║  ▌
▐  ██║  ██║██║╚════██║   ██║   ██╔══██╗██║██╔══██╗██║   ██║   ██║   ██║██║   ██║██║╚██╗██║  ▌
▐  ██████╔╝██║███████║   ██║   ██║  ██║██║██████╔╝╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║  ▌
▐  ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  ▌
▐                                                                                           ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def get_q_distribution(m1, P, q_min=0.1, n=10000):
    """
    Moe & Di Stefano (2017) mass-ratio distribution.

    The distribution is normalized over q = 0.3 - 1.0,
    with the small-q component extended continuously
    down to q_min.

    Returns
    -------
    q : ndarray
        Mass-ratio grid.
    pdf : ndarray
        Probability density, normalized over q >= 0.3.
    """

    q_break = 0.3
    q_twin = 0.95

    gamma_small = get_gamma_small_q(m1, P)
    gamma_large = get_gamma_large_q(m1, P)
    F_twin = get_F_twin(m1, P)

    # Normalize underlying large-q power law
    C = 1 / quad(lambda q: q**gamma_large, q_break, 1.0)[0]

    # Underlying large-q distribution
    def g(q):
        return C * q**gamma_large

    # Complete large-q distribution
    def p_large(q):
        twin = 1 / (1 - q_twin) if q >= q_twin else 0

        return (1 - F_twin) * g(q) + F_twin * twin

    # Extend continuously to small q
    p_break = p_large(q_break)

    def p(q):
        if q < q_break:
            return p_break * (q / q_break) ** gamma_small

        return p_large(q)

    q = np.linspace(q_min, 1.0, n)
    pdf = np.array([p(qi) for qi in q])

    return q, pdf


def sample_q(m1, P, size=1, q_min=0.1, n=10000):
    """
    Sample mass ratios from the Moe & Di Stefano (2017)
    mass-ratio distribution.
    """

    q, pdf = get_q_distribution(m1, P, q_min=q_min, n=n)

    # Normalize over the complete interval
    pdf /= trapezoid(pdf, q)

    # Construct normalized CDF
    cdf = cumulative_trapezoid(pdf, q, initial=0)
    cdf /= cdf[-1]

    # Draw random numbers and invert CDF
    u = np.random.uniform(0, 1, size)

    return np.interp(u, cdf, q)


"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                                                  ▌
▐  ███████╗████████╗ █████╗     ███████╗██╗      ██████╗ ██████╗ ███████╗███████╗  ▌
▐  ██╔════╝╚══██╔══╝██╔══██╗    ██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝██╔════╝  ▌
▐  █████╗     ██║   ███████║    ███████╗██║     ██║   ██║██████╔╝█████╗  ███████╗  ▌
▐  ██╔══╝     ██║   ██╔══██║    ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝  ╚════██║  ▌
▐  ███████╗   ██║   ██║  ██║    ███████║███████╗╚██████╔╝██║     ███████╗███████║  ▌
▐  ╚══════╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚══════╝  ▌
▐                                                                                  ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def __get_eta_low(m1: float, P: float) -> float:
    """
    computes the power law slope for the eccentricity distribution
    for low mass stars.
    Equation (17) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        eta_small: float
            the power law slope
    """
    if m1 > 3 or m1 < 0.8:
        raise Exception(
            f"function __get_eta_low() can only compute eta for m between 0.8 and 3.\nsupplied m = {m1}"
        )
    log_P = np.log10(P)
    if log_P > 6.0 or log_P < 0.5:
        raise Exception(
            f"function __get_eta_low() can only compute log period between 0.5 and 6.\nsupplied log period = {log_P}"
        )
    return 0.6 - 0.7 / (log_P - 0.5)


def __get_eta_high(m1: float, P: float) -> float:
    """
    computes the power law slope for the eccentricity distribution
    for high mass stars.
    Equation (18) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        eta_high: float
            the power law slope
    """
    if m1 < 7:
        raise Exception(
            f"function __get_eta_low() can only compute eta for m above 7.\nsupplied m = {m1}"
        )
    log_P = np.log10(P)
    if log_P > 5.0 or log_P < 0.5:
        raise Exception(
            f"function __get_eta_low() can only compute log period between 0.5 and 5.\nsupplied log period = {log_P}"
        )
    return 0.9 - 0.2 / (log_P - 0.5)


def get_eta(m1: float, P: float) -> float:
    """
    computes the power law slope for the eccentricity distribution.
    Equation (17 and 18) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the binary period in days

    --- returns ---
        eta: float
            the power law slope
    """
    if m1 < 3:
        return __get_eta_low(m1, P)
    elif m1 < 7:
        low = __get_eta_low(3, P)
        high = __get_eta_high(7, P)
        frac = (m1 - 3.0) / (7.0 - 3.0)
        if frac > 1 or frac < 0:
            raise Exception(
                f"function get_eta not meant for extrapolation.\nfrac should be between 0 and 1 but is {frac:.3f}"
            )
        return frac * high + (1 - frac) * low
    return __get_eta_high(m1, P)


"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                                                               ▌
▐  ███████╗ ██████╗ ██████╗███████╗███╗   ██╗████████╗██████╗ ██╗ ██████╗██╗████████╗██╗   ██╗  ▌
▐  ██╔════╝██╔════╝██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██║██╔════╝██║╚══██╔══╝╚██╗ ██╔╝  ▌
▐  █████╗  ██║     ██║     █████╗  ██╔██╗ ██║   ██║   ██████╔╝██║██║     ██║   ██║    ╚████╔╝   ▌
▐  ██╔══╝  ██║     ██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██║██║     ██║   ██║     ╚██╔╝    ▌
▐  ███████╗╚██████╗╚██████╗███████╗██║ ╚████║   ██║   ██║  ██║██║╚██████╗██║   ██║      ██║     ▌
▐  ╚══════╝ ╚═════╝ ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚═╝   ╚═╝      ╚═╝     ▌
▐                                                                                               ▌
▐  ██████╗ ██╗███████╗████████╗██████╗ ██╗██████╗ ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗      ▌
▐  ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗██║██╔══██╗██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║      ▌
▐  ██║  ██║██║███████╗   ██║   ██████╔╝██║██████╔╝██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║      ▌
▐  ██║  ██║██║╚════██║   ██║   ██╔══██╗██║██╔══██╗██║   ██║   ██║   ██║██║   ██║██║╚██╗██║      ▌
▐  ██████╔╝██║███████║   ██║   ██║  ██║██║██████╔╝╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║      ▌
▐  ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝      ▌
▐                                                                                               ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def get_eccentricity_distribution(m1, P, n=10000):
    """
    Moe & Di Stefano (2017) mass-ratio distribution.

    The distribution is normalized over eccentricity = 0.0 - e_max,

    Returns
    -------
    eccentricity: ndarray
        eccentricity grid.
    pdf : ndarray
        Probability density, normalized over eccentricity from 0.0 to e_max.
    """

    eta = get_eta(m1, P)
    if P < 2:

        return np.zeros(n), np.zeros(n)
    e_max = 1 - (P / 2) ** (-2 / 3)

    print(e_max)
    # Normalize underlying large-q power law
    C = 1 / quad(lambda eps: eps**eta, 0, e_max)[0]

    def p(eps):
        return C * eps**eta

    eps = np.linspace(1e-10, e_max, n)
    pdf = np.array([p(epsi) for epsi in eps])

    return eps, pdf


def sample_eccentricity(m1, P, size=1, n=10000):
    """
    Sample mass ratios from the Moe & Di Stefano (2017)
    mass-ratio distribution.
    """

    eps, pdf = get_eccentricity_distribution(m1, P, n=n)

    # Normalize over the complete interval
    pdf /= trapezoid(pdf, eps)

    # Construct normalized CDF
    cdf = cumulative_trapezoid(pdf, eps, initial=0)
    cdf /= cdf[-1]

    # Draw random numbers and invert CDF
    u = np.random.uniform(0, 1, size)

    return np.interp(u, cdf, eps)


"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                                           ▌
▐              ██████╗ ███████╗██████╗ ██╗ ██████╗ ██████╗                  ▌
▐              ██╔══██╗██╔════╝██╔══██╗██║██╔═══██╗██╔══██╗                 ▌
▐              ██████╔╝█████╗  ██████╔╝██║██║   ██║██║  ██║                 ▌
▐              ██╔═══╝ ██╔══╝  ██╔══██╗██║██║   ██║██║  ██║                 ▌
▐              ██║     ███████╗██║  ██║██║╚██████╔╝██████╔╝                 ▌
▐              ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝                  ▌
▐                                                                           ▌
▐  ███████╗██████╗  █████╗  ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗  ▌
▐  ██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝  ▌
▐  █████╗  ██████╔╝███████║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗  ▌
▐  ██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║  ▌
▐  ██║     ██║  ██║██║  ██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║  ▌
▐  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝  ▌
▐                                                                           ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
"""


def __get_f_log_P_lt_1(m1: float) -> float:
    """
    computes the frequency of short-period binaries.
    Equation (20) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.

    --- returns ---
        f_log_P_lt_1: float
            frequency
    """
    log_m1 = np.log10(m1)
    return 0.020 + 0.04 * log_m1 + 0.07 * (log_m1) ** 2


def __get_f_log_P_lt_2_7(m1: float) -> float:
    """
    computes the frequency at log P = 2.7.
    Equation (21) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.

    --- returns ---
        f_log_P_lt_2_7: float
            frequency
    """
    log_m1 = np.log10(m1)
    return 0.039 + 0.07 * log_m1 + 0.01 * (log_m1) ** 2


def __get_f_log_P_lt_5_5(m1: float) -> float:
    """
    computes the frequency at log P = 2.7.
    Equation (22) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.

    --- returns ---
        f_log_P_lt_5_5: float
            frequency
    """
    log_m1 = np.log10(m1)
    return 0.078 - 0.05 * log_m1 + 0.04 * (log_m1) ** 2


def get_f_log_P(m1: float, P: float) -> float:
    """
    computes the frequency of log periods.
    Equation (23) in Moe & Di Stefano (2017)

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the period of the binary

    --- returns ---
        f_log_P_: float
            frequency
    """
    alpha = 0.018
    delta_log_P = 0.7
    log_P = np.log10(P)

    if log_P < 0.2:
        raise Exception(
            f"function get_f_log_P() must have log_P between 0.2 and 8.\nsupplied log P = {log_P}"
        )
    if log_P < 1:
        return __get_f_log_P_lt_1(m1)
    elif log_P < 2.7 - delta_log_P:
        return __get_f_log_P_lt_1(m1) + (log_P - 1) / (1.7 - delta_log_P) * (
            __get_f_log_P_lt_2_7(m1) - __get_f_log_P_lt_1(m1) - alpha * delta_log_P
        )
    elif log_P < 2.7 + delta_log_P:
        return __get_f_log_P_lt_2_7(m1) + alpha * (log_P - 2.7)
    elif log_P < 5.5:
        return (
            __get_f_log_P_lt_2_7(m1)
            + alpha * delta_log_P
            + (log_P - 2.7 - delta_log_P)
            / (2.8 - delta_log_P)
            * (
                __get_f_log_P_lt_5_5(m1)
                - __get_f_log_P_lt_2_7(m1)
                - alpha * delta_log_P
            )
        )
    return __get_f_log_P_lt_5_5(m1) * np.exp(-0.3 * (log_P - 5.5))


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

Ps = np.logspace(0.2, 8, 1000)

plt.plot(Ps, [get_F_twin(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_F_twin(2, p) for p in Ps], label="$2\\;M_\\odot$")
plt.plot(Ps, [get_F_twin(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_F_twin(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(Ps, [get_F_twin(16, p) for p in Ps], label="$16\\;M_\\odot$")

plt.xscale("log")

fig.legend(loc="outside upper center", ncols=5)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("$\\mathcal{F}_\\textrm{twin}(M_\\textrm{1}, P)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-F_twin.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

Ps = np.logspace(0.2, 8, 1000)

plt.plot(Ps, [get_gamma_large_q(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_gamma_large_q(2, p) for p in Ps], label="$2\\;M_\\odot$")
plt.plot(Ps, [get_gamma_large_q(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_gamma_large_q(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(
    Ps, [get_gamma_large_q(16, p) for p in Ps], label="$16\\;M_\\odot$", linestyle="-."
)

plt.axhline(0, c="C9", linewidth=0.75)
plt.text(10**4, -0.05, "Uniform", fontsize=8, ha="center", va="top")

plt.axhline(-2.35, c="C9", linewidth=0.75)
plt.text(10**4, -2.3, "Random pairing from IMF", fontsize=8, ha="center", va="bottom")

plt.xscale("log")

fig.legend(loc="outside upper center", ncols=5)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("$\\gamma_\\textrm{largeq}(M_\\textrm{1}, P)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-gamma-large.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

Ps = np.logspace(0.2, 8, 1000)

plt.plot(Ps, [get_gamma_small_q(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_gamma_small_q(2, p) for p in Ps], label="$2\\;M_\\odot$")
plt.plot(Ps, [get_gamma_small_q(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_gamma_small_q(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(
    Ps, [get_gamma_small_q(16, p) for p in Ps], label="$16\\;M_\\odot$", linestyle="-."
)

plt.xscale("log")

plt.axhline(0, c="C9", linewidth=0.75)
plt.text(10**0.5, -0.05, "Uniform", fontsize=8, ha="left", va="top")

plt.axhline(-2.35, c="C9", linewidth=0.75)
plt.text(10**0.5, -2.3, "Random pairing from IMF", fontsize=8, ha="left", va="bottom")


fig.legend(loc="outside upper center", ncols=5)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("$\\gamma_\\textrm{smallq}(M_\\textrm{1}, P)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-gamma-small.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

m1 = 2.2
P = 10000.0

qs = sample_q(m1, P, size=100_000)
q, pdf = get_q_distribution(m1, P)
pdf /= trapezoid(pdf, q)
plt.plot(q, pdf, label="theoretical PDF", linewidth=1, zorder=100)
plt.hist(
    qs, bins=50, density=True, histtype="step", label="$10^{5}$ samples", linewidth=1
)

qs = sample_q(m1, P, size=1_000_000)
q, pdf = get_q_distribution(m1, P)
pdf /= trapezoid(pdf, q)
# plt.plot(q, pdf, label="theoretical PDF")
plt.hist(
    qs, bins=50, density=True, histtype="step", label="$10^{6}$ samples", linewidth=1
)

plt.xlabel("$q$")
plt.ylabel("$p(q)$")
fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-q-sample.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

Ps = np.logspace(0.5, 5, 1000)

plt.plot(Ps, [get_eta(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_eta(2, p) for p in Ps], label="$2\\;M_\\odot$", linestyle="-.")
plt.plot(Ps, [get_eta(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_eta(6, p) for p in Ps], label="$6\\;M_\\odot$")
plt.plot(Ps, [get_eta(8, p) for p in Ps], zorder=-10)
plt.plot(Ps, [get_eta(8, p) for p in Ps], zorder=-10)
plt.plot(Ps, [get_eta(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(Ps, [get_eta(8, p) for p in Ps], zorder=-10)
plt.plot(Ps, [get_eta(16, p) for p in Ps], label="$16\\;M_\\odot$", linestyle="-.")

plt.axhline(1, c="C9", linewidth=0.75)
plt.text(10**4, 1.05, "Thermal", fontsize=8, ha="center", va="bottom")

plt.axhline(0, c="C9", linewidth=0.75)
plt.text(10**4, -0.05, "Uniform", fontsize=8, ha="center", va="top")

plt.xscale("log")
plt.ylim(-0.75, 1.25)

fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("$\\gamma_\\textrm{largeq}(M_\\textrm{1}, P)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-eta.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

m1 = 1.2
P = 14.0

qs = sample_eccentricity(m1, P, size=100_000)
q, pdf = get_eccentricity_distribution(m1, P)
pdf /= trapezoid(pdf, q)
plt.plot(
    q,
    pdf,
    label="theoretical PDF\n$M_1 = 1.2\\;M_\\odot$\n$P = 14$ days",
    linewidth=1,
    zorder=100,
)
plt.hist(
    qs, bins=50, density=True, histtype="step", label="$10^{5}$ samples", linewidth=1
)

m1 = 1.6
P = 100.0

qs = sample_eccentricity(m1, P, size=100_000)
q, pdf = get_eccentricity_distribution(m1, P)
pdf /= trapezoid(pdf, q)
plt.plot(
    q,
    pdf,
    label="theoretical PDF\n$M_1 = 1.6\\;M_\\odot$\n$P = 100$ days",
    linewidth=1,
    zorder=100,
)
plt.hist(
    qs, bins=50, density=True, histtype="step", label="$10^{5}$ samples", linewidth=1
)


m1 = 4
P = 10000.0

qs = sample_eccentricity(m1, P, size=100_000)
q, pdf = get_eccentricity_distribution(m1, P)
pdf /= trapezoid(pdf, q)
plt.plot(
    q,
    pdf,
    label="theoretical PDF\n$M_1 = 4.0\\;M_\\odot$\n$P = 10\,000$ days",
    linewidth=1,
    zorder=100,
)
plt.hist(
    qs, bins=50, density=True, histtype="step", label="$10^{5}$ samples", linewidth=1
)

plt.ylim(-0.2, 2)

plt.xlabel("$e$")
plt.ylabel("$p(e)$")
fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-ecc-sample.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

Ps = np.logspace(0.5, 8, 1000)

plt.plot(Ps, [get_f_log_P(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P(2, p) for p in Ps], label="$2\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P(6, p) for p in Ps], label="$6\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P(16, p) for p in Ps], label="$16\\;M_\\odot$")

plt.xscale("log")
# plt.ylim(-0.75, 1.25)

fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("$f_{\\log P; q>0.3}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-log_P.pgf", format="pgf")
plt.show()
plt.close()


# %%
