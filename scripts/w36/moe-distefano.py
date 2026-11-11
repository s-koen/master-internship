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
from scipy.optimize import brentq
from scipy.special import factorial

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

    logP = np.log10(P)

    F_short = 0.3 - 0.15 * np.log10(m1)
    logP_twin = 8.0 - m1 if m1 < 6.5 else 1.5

    return np.where(
        logP < 1,
        F_short,
        np.where(
            logP < logP_twin,
            F_short * (1 - (logP - 1) / (logP_twin - 1)),
            0.0,
        ),
    )


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
    if not 0.8 <= m1 <= 1.2:
        raise ValueError(
            f"function can only be called with m1 between 0.8 and 1.2. "
            f"supplied m1 = {m1}"
        )

    P = np.asarray(P)
    log_P = np.log10(P)

    if np.any((log_P < 0.2) | (log_P > 8.0)):
        raise ValueError("function can only be called with log_P between 0.2 and 8.0.")

    return np.where(
        log_P < 5,
        -0.5,
        -0.5 - 0.3 * (log_P - 5),
    )


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

    logP = np.log10(P)

    # Solar-type primary (1.2 Msun)
    solar = np.where(
        logP < 5,
        -0.5,
        -0.5 - 0.3 * (logP - 5),
    )

    # Midpoint primary (3.5 Msun)
    mid = np.select(
        [logP < 1, logP < 4.5, logP < 6.5],
        [
            -0.5,
            -0.5 - 0.2 * (logP - 1),
            -1.2 - 0.4 * (logP - 4.5),
        ],
        default=-2.0,
    )

    # Heavy primary (6 Msun and above)
    heavy = np.select(
        [logP < 1, logP < 2, logP < 4],
        [
            -0.5,
            -0.5 - 0.9 * (logP - 1),
            -1.4 - 0.3 * (logP - 2),
        ],
        default=-2.0,
    )

    if m1 < 1.2:
        return __get_gamma_large_q_solar(m1, P)

    if m1 < 3.5:
        frac = (m1 - 1.2) / (3.5 - 1.2)
        return (1 - frac) * solar + frac * mid

    if m1 < 6:
        frac = (m1 - 3.5) / (6 - 3.5)
        return (1 - frac) * mid + frac * heavy

    return heavy


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

    logP = np.log10(P)

    # Solar-type primary (1.2 Msun)
    solar = np.full_like(logP, 0.3, dtype=float)

    # Midpoint primary (3.5 Msun)
    mid = np.select(
        [logP < 2.5, logP < 5.5],
        [
            0.2,
            0.2 - 0.3 * (logP - 2.5),
        ],
        default=-0.7 - 0.2 * (logP - 5.5),
    )

    # Heavy primary (6 Msun and above)
    heavy = np.select(
        [logP < 1, logP < 3, logP < 5.6],
        [
            0.1,
            0.1 - 0.15 * (logP - 1),
            -0.2 - 0.50 * (logP - 3),
        ],
        default=-1.5,
    )

    if m1 < 1.2:
        return np.full_like(logP, 0.3, dtype=float)

    if m1 < 3.5:
        frac = (m1 - 1.2) / (3.5 - 1.2)
        return (1 - frac) * solar + frac * mid

    if m1 < 6:
        frac = (m1 - 3.5) / (6 - 3.5)
        return (1 - frac) * mid + frac * heavy

    return heavy


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


def get_f_log_P_q_gt_0_3(m1: float, P: float) -> float:
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
    delta = 0.7

    logP = np.log10(P)

    f1 = __get_f_log_P_lt_1(m1)
    f27 = __get_f_log_P_lt_2_7(m1)
    f55 = __get_f_log_P_lt_5_5(m1)

    return np.select(
        [
            logP < 1,
            logP < 2.7 - delta,
            logP < 2.7 + delta,
            logP < 5.5,
        ],
        [
            f1,
            f1 + (logP - 1) / (1.7 - delta) * (f27 - f1 - alpha * delta),
            f27 + alpha * (logP - 2.7),
            f27
            + alpha * delta
            + (logP - 2.7 - delta) / (2.8 - delta) * (f55 - f27 - alpha * delta),
        ],
        default=f55 * np.exp(-0.3 * (logP - 5.5)),
    )


def get_q_correction(m1, P, q_min=0.1, n=10000):
    """
    vectorized correction for mass ratio distribution where i integrate the distribution
    from 0.1 to 1 and 0.3 to 1 (which is equal to 1).
    """
    q_break = 0.3

    gamma_small = get_gamma_small_q(m1, P)
    gamma_large = get_gamma_large_q(m1, P)
    F_twin = get_F_twin(m1, P)

    # Analytic normalization of the large-q power law
    a = gamma_large + 1
    if np.isscalar(a):
        integral = np.log(1 / q_break) if np.isclose(a, 0) else (1 - q_break**a) / a
    else:
        integral = np.where(
            np.isclose(a, 0),
            np.log(1 / q_break),
            (1 - q_break**a) / np.where(np.isclose(a, 0), 1, a),
        )

    C = 1 / integral

    # Distribution evaluated at q = 0.3
    p_break = (1 - F_twin) * C * q_break**gamma_large

    # Integrate the small-q power law
    b = gamma_small + 1
    x = q_min / q_break

    small_integral = np.where(
        np.isclose(b, 0),
        p_break * q_break * np.log(1 / x),
        p_break * q_break * (1 - x**b) / np.where(np.isclose(b, 0), 1, b),
    )

    return 1 + small_integral


def get_f_log_P(m1: float, P: float) -> float:
    """
    computes the frequency of log periods across the entire q range.

    --- parameters ---
        m1: float
            primary mass in solar masses.
        P: float
            the period of the binary

    --- returns ---
        f_log_P_: float
            frequency
    """
    return get_f_log_P_q_gt_0_3(m1, P) * get_q_correction(m1, P)


"""
█████████████████████████████████████████████████████████████████████████████████████████████
█▌                                                                                         ▐█
█▌  ███╗   ███╗██╗   ██╗██╗  ████████╗██╗██████╗ ██╗     ██╗ ██████╗██╗████████╗██╗   ██╗  ▐█
█▌  ████╗ ████║██║   ██║██║  ╚══██╔══╝██║██╔══██╗██║     ██║██╔════╝██║╚══██╔══╝╚██╗ ██╔╝  ▐█
█▌  ██╔████╔██║██║   ██║██║     ██║   ██║██████╔╝██║     ██║██║     ██║   ██║    ╚████╔╝   ▐█
█▌  ██║╚██╔╝██║██║   ██║██║     ██║   ██║██╔═══╝ ██║     ██║██║     ██║   ██║     ╚██╔╝    ▐█
█▌  ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║██║     ███████╗██║╚██████╗██║   ██║      ██║     ▐█
█▌  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚═╝   ╚═╝      ╚═╝     ▐█
█▌                                                                                         ▐█
█████████████████████████████████████████████████████████████████████████████████████████████
"""


def get_period_distribution(m1, n=2000, q_min=0.1):
    """
    get period distribution using the period frequencies and normalization.

    --- parameters ---
        m1: float
            primary mass is solar masses

    --- returns ---
        log period: array
        pdf
        cdf
        mu: float
    """
    logP = np.linspace(0.2, 8.0, n)
    P = 10**logP

    rate = get_f_log_P_q_gt_0_3(m1, P) * get_q_correction(m1, P, q_min=q_min)

    mu = trapezoid(rate, logP)

    pdf = rate / mu
    cdf = cumulative_trapezoid(pdf, logP, initial=0)
    cdf /= cdf[-1]

    return logP, pdf, cdf, mu


def get_multiplicity_probabilities(mu):
    """
    Truncated Poisson distribution for n = 0, 1, 2, 3 companions.
    """

    if not 0 <= mu < 3:
        raise ValueError("mu must be between 0 and 3.")

    n = np.arange(4)

    def mean_multiplicity(lam):
        weights = lam**n / factorial(n)
        probabilities = weights / weights.sum()
        return np.sum(n * probabilities)

    lam = brentq(
        lambda lam: mean_multiplicity(lam) - mu,
        1e-12,
        1e6,
    )

    weights = lam**n / factorial(n)
    return weights / weights.sum()


def sample_companion_periods(m1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    logP, pdf, cdf, mu = get_period_distribution(m1)

    probabilities = get_multiplicity_probabilities(mu)
    n_comp = rng.choice(4, p=probabilities)

    u = rng.random(n_comp)
    sampled_logP = np.interp(u, cdf, logP)

    return 10**sampled_logP


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
plt.xlabel("$P$ (days)")
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
plt.xlabel("$P$ (days)")
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
plt.xlabel("$P$ (days)")
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
plt.xlabel("$P$ (days)")
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

plt.plot(Ps, [get_f_log_P_q_gt_0_3(1, p) for p in Ps], label="$1\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P_q_gt_0_3(2, p) for p in Ps], label="$2\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P_q_gt_0_3(4, p) for p in Ps], label="$4\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P_q_gt_0_3(6, p) for p in Ps], label="$6\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P_q_gt_0_3(8, p) for p in Ps], label="$8\\;M_\\odot$")
plt.plot(Ps, [get_f_log_P_q_gt_0_3(16, p) for p in Ps], label="$16\\;M_\\odot$")

plt.xscale("log")
# plt.ylim(-0.75, 1.25)

fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$P$ (days)")
plt.ylabel("$f_{\\log P; q>0.3}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-log_P_q_gt_0_3.pgf", format="pgf")
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
plt.xlabel("$P$ (days)")
plt.ylabel("$f_{\\log P; q>0.1}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-log_P.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

N = 100_000
Ps = np.logspace(0.2, 8, 1000)

psss = []
for m in [2, 6, 14]:
    pss = []
    for n in range(N):
        if n % 1000 == 0:
            print(n)
        periods = sample_companion_periods(m)
        pss.append(periods)

    ps = [x for xs in pss for x in xs]

    logP_edges = np.linspace(0.2, 8, 51)
    bin_width = logP_edges[1] - logP_edges[0]

    plt.plot(
        np.log10(Ps),
        N * bin_width * get_f_log_P(m, Ps),
        label=f"theoretical PDF\n$M_1 = {m:.0f}\\;M_\\odot$",
        zorder=1000,
        linewidth=1,
    )

    plt.hist(
        np.log10(ps),
        bins=logP_edges,
        histtype="step",
        label="$10^{5}$ samples",
        linewidth=1,
    )

    psss.append(pss)

fig.legend(loc="outside upper center", ncols=3)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\log(P / \\textrm{days})$")
plt.ylabel("counts")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-test-period-sampling.pgf", format="pgf")
plt.show()
plt.close()

# %%

s = []
b = []
t = []
q = []

N = 100000
ms = np.logspace(np.log10(1.2), np.log10(40), 50)
for m in ms:
    print(m)
    aaa = []
    for n in range(N):
        periods = sample_companion_periods(m)
        aaa.append(periods)

    singles = 0
    binaries = 0
    triples = 0
    quadruples = 0

    for aa in aaa:
        if len(aa) == 0:
            singles += 1

        if len(aa) == 1:
            binaries += 1

        if len(aa) == 2:
            triples += 1

        if len(aa) == 3:
            quadruples += 1

    s.append(singles / N)
    b.append(binaries / N)
    t.append(triples / N)
    q.append(quadruples / N)

plt.plot(ms, s)
plt.plot(ms, b)
plt.plot(ms, t)
plt.plot(ms, q)
plt.show()
# %%
from concurrent.futures import ProcessPoolExecutor
from numpy.random import SeedSequence, default_rng


def calculate_multiplicity(args):
    m, N, seed = args
    rng = default_rng(seed)

    # only calculate the period distribution once per primary mass
    _, _, _, mu = get_period_distribution(m)
    probabilities = get_multiplicity_probabilities(mu)

    # draw N multiplicities simultaneously
    counts = rng.multinomial(N, probabilities)

    return counts / N


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

N = 1000_000_000
ms = np.logspace(np.log10(0.8), np.log10(40), 50)

seeds = SeedSequence(42).spawn(len(ms))
args = [(m, N, seed) for m, seed in zip(ms, seeds)]

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(calculate_multiplicity, args))

results = np.array(results)

s, b, t, q = results.T

plt.plot(ms, s, label="singles")
plt.plot(ms, b, label="binaries")
plt.plot(ms, t, label="triples")
plt.plot(ms, q, label="quadruples")

plt.xscale("log")
plt.xlabel("Primary Mass $M_1$ ($M_\\odot$)")
plt.ylabel("$f_{n;q>0.1}$")


fig.legend(loc="outside upper center", ncols=4)

axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w36-multiplicity.pgf", format="pgf")
plt.show()
plt.close()
# %%

import numpy as np
from scipy.special import ndtr, ndtri


def _sample_power_law(n, m_min, m_max, alpha, rng):
    """Sample from p(m) proportional to m**(-alpha)."""
    u = rng.random(n)

    if np.isclose(alpha, 1.0):
        return m_min * (m_max / m_min) ** u

    exponent = 1.0 - alpha
    return (m_min**exponent + u * (m_max**exponent - m_min**exponent)) ** (
        1.0 / exponent
    )


def _sample_broken_power_law(n, m_min, m_max, breaks, slopes, rng):
    """Sample from a continuous, piecewise power-law IMF."""

    edges = [m_min] + [x for x in breaks if m_min < x < m_max] + [m_max]

    # Continuity coefficients at each break
    coefficients = [1.0]

    for i, slope in enumerate(slopes[:-1]):
        break_mass = breaks[i]
        coefficients.append(coefficients[-1] * break_mass ** (slopes[i + 1] - slope))

    # Calculate the relative probability of each segment
    weights = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        midpoint = np.sqrt(lo * hi)
        segment = np.searchsorted(breaks, midpoint)

        alpha = slopes[segment]
        coefficient = coefficients[segment]

        if np.isclose(alpha, 1.0):
            integral = np.log(hi / lo)
        else:
            integral = (hi ** (1 - alpha) - lo ** (1 - alpha)) / (1 - alpha)

        weights.append(coefficient * integral)

    probabilities = np.array(weights)
    probabilities /= probabilities.sum()

    # Select a segment for every sample
    segments = rng.choice(len(weights), size=n, p=probabilities)
    masses = np.empty(n)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = segments == i
        count = mask.sum()

        if count == 0:
            continue

        midpoint = np.sqrt(lo * hi)
        segment = np.searchsorted(breaks, midpoint)
        alpha = slopes[segment]

        masses[mask] = _sample_power_law(count, lo, hi, alpha, rng)

    return masses


def sample_salpeter(n, m_min: float = 0.08, m_max: float = 100, rng=None):
    """
    Sample stellar masses from the Salpeter (1955) IMF.
    """
    if rng is None:
        rng = np.random.default_rng()

    return _sample_power_law(n, m_min, m_max, alpha=2.35, rng=rng)


def sample_kroupa(n, m_min=0.01, m_max=120, rng=None):
    """
    Sample stellar masses from the Kroupa (2001) IMF.
    """
    if rng is None:
        rng = np.random.default_rng()

    return _sample_broken_power_law(
        n,
        m_min,
        m_max,
        breaks=[0.08, 0.5],
        slopes=[0.3, 1.3, 2.3],
        rng=rng,
    )


def sample_chabrier(n, m_min=0.01, m_max=100, rng=None):
    """
    Sample stellar masses from the Chabrier (2003) individual-star IMF.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.log10(0.079)
    sigma = 0.69

    # Only the log-normal part is needed
    if m_max <= 1:
        x_min = np.log10(m_min)
        x_max = np.log10(m_max)

        z_min = (x_min - mu) / sigma
        z_max = (x_max - mu) / sigma

        u = ndtr(z_min) + rng.random(n) * (ndtr(z_max) - ndtr(z_min))

        return 10 ** (mu + sigma * ndtri(u))

    # Only the power-law part is needed
    if m_min >= 1:
        return _sample_power_law(n, m_min, m_max, alpha=2.3, rng=rng)

    # Calculate relative weights of the two components
    x_min = np.log10(m_min)
    z_min = (x_min - mu) / sigma
    z_max = (0 - mu) / sigma

    low_weight = sigma * np.sqrt(2 * np.pi) * (ndtr(z_max) - ndtr(z_min))

    # Continuity at M = 1
    amplitude = np.exp(-0.5 * z_max**2)

    high_weight = amplitude * (1 - m_max ** (-1.3)) / (1.3 * np.log(10))

    p_low = low_weight / (low_weight + high_weight)

    # Draw which component each sample belongs to
    is_low = rng.random(n) < p_low

    masses = np.empty(n)

    # Log-normal component
    count_low = is_low.sum()

    if count_low > 0:
        u = ndtr(z_min) + rng.random(count_low) * (ndtr(z_max) - ndtr(z_min))

        masses[is_low] = 10 ** (mu + sigma * ndtri(u))

    # Power-law component
    count_high = n - count_low

    if count_high > 0:
        masses[~is_low] = _sample_power_law(count_high, 1.0, m_max, alpha=2.3, rng=rng)

    return masses


# %%

N = 1_000_000
rng = np.random.default_rng(42)

m_salpeter = sample_salpeter(30 * N, 0.01, 40, rng)
m_kroupa = sample_kroupa(N, 0.01, 40, rng)
m_chabrier = sample_chabrier(N, 0.01, 40, rng)

bins = np.logspace(np.log10(0.01), np.log10(40), 200)

plt.hist(m_salpeter, bins=bins, histtype="step", label="salpeter")
plt.hist(m_kroupa, bins=bins, histtype="step", label="kroupa")
plt.hist(m_chabrier, bins=bins, histtype="step", label="chabrier")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M_1\;[M_\odot]$")
plt.ylabel("probability density")
plt.legend()
plt.show()
# %%

N = 1_000_000
m_salpeter = sample_salpeter(N, 1, 3.2, rng)
p_bins = []
p_trips = []
p_quads = []

m_bins = []
m_trips = []
m_quads = []

for i, m in enumerate(m_salpeter):
    if i % 1000 == 0:
        print(i // 1000)

    periods = sample_companion_periods(m)
    if len(periods) == 1:
        if periods[0] < 30000:
            p_bins.append(periods[0])
            m_bins.append(m)
    elif len(periods) == 2:
        for per in periods:
            if per < 30000:
                p_trips.append(per)
                m_trips.append(m)
    elif len(periods) == 3:
        for per in periods:
            if per < 30000:
                p_quads.append(per)
                m_quads.append(m)
# %%
results = {
    "m_bins": m_bins,
    "p_bins": p_bins,
    "m_trips": m_trips,
    "p_trips": p_trips,
    "m_quads": m_quads,
    "p_quads": p_quads,
}

with open(
    f"/home/koen/master-internship/scripts/w36/cache/m-p-dump.pkl",
    "wb",
) as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%

plt.scatter(m_bins, p_bins)
plt.scatter(m_trips, p_trips)
plt.scatter(m_quads, p_quads)
plt.show()

# %%

plt.hist2d(np.log10(m_bins), np.log10(p_bins), bins=50, cmap="viridis")
plt.show()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.jointplot(
    x=m_bins,
    y=p_bins,
    kind="kde",
)
plt.show()
# %%
from scipy.ndimage import gaussian_filter

fig, axs = plt.subplots(1, 1, figsize=set_size(column), constrained_layout=True)


def plot_histogram(m, p, ax=None, bins=(50, 50), sigma=4, cmap="Blues"):
    if ax is None:
        fig, ax = plt.subplots()

    H, xedges, yedges = np.histogram2d(
        m,
        np.log10(p),
        bins=bins,
        density=True,
    )

    H = gaussian_filter(H, sigma=sigma)

    X, Y = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2,
    )

    # cf = ax.contourf(X, Y, H.T, levels=10, cmap="viridis")
    ax.contour(X, Y, H.T, levels=10, cmap=cmap)

    ax.set_xlabel(r"$M_1\ [M_\odot]$")
    ax.set_ylabel(r"$\log_{10}(P\ [\mathrm{days}])$")

    # return cf


plot_histogram(m_bins, p_bins, ax=axs, cmap="Blues", sigma=10)
plot_histogram(m_trips, p_trips, ax=axs, cmap="Oranges", sigma=10)
plot_histogram(m_quads, p_quads, ax=axs, cmap="Greens", sigma=10)

ax_histx = axs.inset_axes([0, 1.05, 1, 0.25], sharex=axs)
ax_histy = axs.inset_axes([1.05, 0, 0.25, 1], sharey=axs)
ax_histx.hist(m_bins, bins=50, histtype="step", density=True)
ax_histx.hist(m_trips, bins=50, histtype="step", density=True)
ax_histx.hist(m_quads, bins=50, histtype="step", density=True)
ax_histy.hist(
    np.log10(p_bins), orientation="horizontal", bins=50, histtype="step", density=True
)
ax_histy.hist(
    np.log10(p_trips), orientation="horizontal", bins=50, histtype="step", density=True
)
ax_histy.hist(
    np.log10(p_quads), orientation="horizontal", bins=50, histtype="step", density=True
)
# Draw the scatter plot and marginals.
# scatter_hist(x, y, ax, ax_histx, ax_histy)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/.pgf", format="pgf")
plt.show()
plt.close()
plt.show()
# %%

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-dump.pkl",
    "rb",
) as f:
    q_samples = pickle.load(f)

with open(
    f"/home/koen/master-internship/scripts/w36/cache/m-p-dump.pkl",
    "rb",
) as f:
    distr = pickle.load(f)

# %%

m1_samples = distr["m_bins"]
p_samples = distr["p_bins"]

# %%
q_samples = []
for i, (m1, p) in enumerate(zip(m1_samples, p_samples)):
    if i % 1000 == 0:
        print(i // 1000, " / ", len(m1_samples) // 1000)
    q_samples.append(sample_q(m1, p, n=1000, size=1000))

# %%
with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-dump.pkl",
    "wb",
) as f:
    pickle.dump(q_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
q_samples_flat = np.array(q_samples).flatten()
print(len(q_samples_flat))
# %%
print(np.size(m1_samples), np.size(q_samples_flat[::100][:153714]))
plt.hist2d(100 * m1_samples, q_samples_flat[:15371400], bins=200, cmap="viridis")
plt.show()
# %%

plt.hist2d(m1_samples, np.log10(p_samples), bins=50, cmap="viridis")
plt.show()

# %%


plt.hist2d(100 * list(np.log10(p_samples)), q_samples_flat[:15371400], bins=200)
plt.show()
# %%
import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("bids:fake_parula").to_mpl()
fig, axs = plt.subplots(
    3, 3, sharex="col", figsize=set_size(column, height=0.85), constrained_layout=True
)

print(len(m1_samples))
print(len(q_samples_flat))

axs[0, 0].hist(
    m1_samples,
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[1, 0].hist2d(
    100 * m1_samples,
    q_samples_flat[::10],
    bins=200,
    cmap=cmap,
    density=True,
    rasterized=True,
)
axs[2, 0].hist2d(
    m1_samples, np.log10(p_samples), bins=50, rasterized=True, cmap=cmap, density=True
)
axs[2, 1].hist2d(
    q_samples_flat[::10],
    100 * list(np.log10(p_samples)),
    bins=200,
    rasterized=True,
    cmap=cmap,
    density=True,
)
axs[1, 1].hist(
    q_samples_flat,
    bins=200,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[2, 2].hist(
    np.log10(p_samples),
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)

axs[0, 1].axis("off")
axs[0, 2].axis("off")
axs[1, 2].axis("off")


m_salpeter = sample_salpeter(1000000, 1, 3.2)
axs[0, 0].hist(m_salpeter, 100, density=True, color="k", histtype="step")
axs[0, 0].annotate(
    "Salpeter IMF",
    xy=(1.15, 1.5),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="left",
    va="center",
    fontsize=10,
)

axs[0, 0].tick_params(labelbottom=False)
axs[0, 0].set_yticks([])
axs[1, 0].tick_params(labelbottom=False)
axs[1, 1].tick_params(labelbottom=False)
axs[1, 1].set_yticks([])
axs[2, 1].tick_params(labelleft=False)
axs[2, 2].set_yticks([])

axs[1, 0].set_ylabel("$q$ ($M_2 / M_1$)")
axs[2, 0].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
axs[2, 0].set_xlabel("$M_1$ ($M_\\odot$)")
axs[2, 1].set_xlabel("$q$ ($M_2 / M_1$)")
for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-population-1.pgf", format="pgf")
plt.show()
plt.close()
# %%
m1_samples = distr["m_trips"]
p_samples = distr["p_trips"]

# %%
q_samples = []
for i, (m1, p) in enumerate(zip(m1_samples, p_samples)):
    if i % 1000 == 0:
        print(i // 1000, " / ", len(m1_samples) // 1000)
    q_samples.append(sample_q(m1, p, n=1000, size=1000))

q_samples_flat = np.array(q_samples).flatten()
# %%

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-trips-dump.pkl",
    "wb",
) as f:
    pickle.dump(q_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("bids:fake_parula").to_mpl()
fig, axs = plt.subplots(
    3, 3, sharex="col", figsize=set_size(column, height=0.85), constrained_layout=True
)

print(len(m1_samples))
print(len(q_samples_flat))

axs[0, 0].hist(
    m1_samples,
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[1, 0].hist2d(
    100 * m1_samples,
    q_samples_flat[::10],
    bins=200,
    cmap=cmap,
    density=True,
    rasterized=True,
)
axs[2, 0].hist2d(
    m1_samples, np.log10(p_samples), bins=50, rasterized=True, cmap=cmap, density=True
)
axs[2, 1].hist2d(
    q_samples_flat[::10],
    100 * list(np.log10(p_samples)),
    bins=200,
    rasterized=True,
    cmap=cmap,
    density=True,
)
axs[1, 1].hist(
    q_samples_flat,
    bins=200,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[2, 2].hist(
    np.log10(p_samples),
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)

axs[0, 1].axis("off")
axs[0, 2].axis("off")
axs[1, 2].axis("off")


m_salpeter = sample_salpeter(1000000, 1, 3.2)
axs[0, 0].hist(m_salpeter, 100, density=True, color="k", histtype="step")
axs[0, 0].annotate(
    "Salpeter IMF",
    xy=(1.15, 1.5),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="left",
    va="center",
    fontsize=10,
)

axs[0, 0].tick_params(labelbottom=False)
axs[0, 0].set_yticks([])
axs[1, 0].tick_params(labelbottom=False)
axs[1, 1].tick_params(labelbottom=False)
axs[1, 1].set_yticks([])
axs[2, 1].tick_params(labelleft=False)
axs[2, 2].set_yticks([])

axs[1, 0].set_ylabel("$q$ ($M_2 / M_1$)")
axs[2, 0].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
axs[2, 0].set_xlabel("$M_1$ ($M_\\odot$)")
axs[2, 1].set_xlabel("$q$ ($M_2 / M_1$)")
for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-population-trips.pgf", format="pgf")
plt.show()
plt.close()
# %%

m1_samples = distr["m_quads"]
p_samples = distr["p_quads"]

q_samples = []
for i, (m1, p) in enumerate(zip(m1_samples, p_samples)):
    if i % 1000 == 0:
        print(i // 1000, " / ", len(m1_samples) // 1000)
    q_samples.append(sample_q(m1, p, n=1000, size=1000))

q_samples_flat = np.array(q_samples).flatten()

with open(
    f"/home/koen/master-internship/scripts/w36/cache/q-quads-dump.pkl",
    "wb",
) as f:
    pickle.dump(q_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap("bids:fake_parula").to_mpl()
fig, axs = plt.subplots(
    3, 3, sharex="col", figsize=set_size(column, height=0.85), constrained_layout=True
)

print(len(m1_samples))
print(len(q_samples_flat))

axs[0, 0].hist(
    m1_samples,
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[1, 0].hist2d(
    100 * m1_samples,
    q_samples_flat[::10],
    bins=200,
    cmap=cmap,
    density=True,
    rasterized=True,
)
axs[2, 0].hist2d(
    m1_samples, np.log10(p_samples), bins=50, rasterized=True, cmap=cmap, density=True
)
axs[2, 1].hist2d(
    q_samples_flat[::10],
    100 * list(np.log10(p_samples)),
    bins=200,
    rasterized=True,
    cmap=cmap,
    density=True,
)
axs[1, 1].hist(
    q_samples_flat,
    bins=200,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)
axs[2, 2].hist(
    np.log10(p_samples),
    bins=50,
    histtype="stepfilled",
    density=True,
    edgecolor="k",
    alpha=0.5,
    color="C9",
)

axs[0, 1].axis("off")
axs[0, 2].axis("off")
axs[1, 2].axis("off")


m_salpeter = sample_salpeter(1000000, 1, 3.2)
axs[0, 0].hist(m_salpeter, 100, density=True, color="k", histtype="step")
axs[0, 0].annotate(
    "Salpeter IMF",
    xy=(1.15, 1.5),
    xycoords="data",
    xytext=(15, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="left",
    va="center",
    fontsize=10,
)

axs[0, 0].tick_params(labelbottom=False)
axs[0, 0].set_yticks([])
axs[1, 0].tick_params(labelbottom=False)
axs[1, 1].tick_params(labelbottom=False)
axs[1, 1].set_yticks([])
axs[2, 1].tick_params(labelleft=False)
axs[2, 2].set_yticks([])

axs[1, 0].set_ylabel("$q$ ($M_2 / M_1$)")
axs[2, 0].set_ylabel("$\\textrm{log}(P / \\textrm{days})$")
axs[2, 0].set_xlabel("$M_1$ ($M_\\odot$)")
axs[2, 1].set_xlabel("$q$ ($M_2 / M_1$)")
for ax in axs.flatten():
    ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-population-quads.pgf", format="pgf")
plt.show()
plt.close()
# %%
print(len(distr["m_bins"]))
# %%
