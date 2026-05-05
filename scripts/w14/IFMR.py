import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pandas as pd

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

import mesa_reader as mr

# %%

xs = [0.95, 2.75, 3.54, 5.21, 8]
ys = [0.5, 0.67, 0.81, 0.91, 1.37]


def IFMR(final_mass):

    ys = [0.95, 2.75, 3.54, 5.21, 8]
    xs = [0.5, 0.67, 0.81, 0.91, 1.37]
    return np.interp(final_mass, xs, ys)


x = np.linspace(0.5, 1.37, 100)
y = IFMR(x)
plt.plot(y, x, c="C9")
plt.show()
# %%
data = pd.read_csv("scripts/w14/Ba_star_orbits.csv")
# %%

es = []
ms = []
for i, (m, e_m) in enumerate(zip(data.M, data.e_M)):
    if i == 0 or type(m) == float or type(data.Mc[i]) == float:
        continue
    es.append(float(e_m))
    ms.append(float(m))

esc = []
msc = []
for i, (mc, e_mc) in enumerate(zip(data.Mc, data.e_Mc)):
    if i == 0 or type(mc) == float or type(data.M[i]) == float:
        continue
    esc.append(float(e_mc))
    msc.append(float(mc))

plt.errorbar(ms, msc, xerr=es, yerr=esc, fmt=".")
plt.show()
# %%

import numpy as np


def IFMR_with_error(final_mass, sigma_final_mass, sort=False):
    ys = np.array([0.95, 2.75, 3.54, 5.21, 8])
    xs = np.array([0.5, 0.67, 0.81, 0.91, 1.37])

    final_mass = np.asarray(final_mass)
    sigma_final_mass = np.asarray(sigma_final_mass)

    # Interpolated value
    initial_mass = np.interp(final_mass, xs, ys, np.nan, np.nan)

    # Find interval index
    idx = np.searchsorted(xs, final_mass) - 1
    idx = np.clip(idx, 0, len(xs) - 2)

    # Compute slope (derivative)
    slope = (ys[idx + 1] - ys[idx]) / (xs[idx + 1] - xs[idx])

    # Error propagation
    sigma_initial_mass = np.abs(slope) * sigma_final_mass

    order = None
    if sort:
        order = np.argsort(initial_mass)
        initial_mass = initial_mass[order]
        sigma_initial_mass = sigma_initial_mass[order]

    return initial_mass, sigma_initial_mass, order


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

initial_mass, err_initial_mass, order = IFMR_with_error(msc, esc, sort=True)

print(initial_mass)
# plt.fill_betweenx(np.array(msc)[order], initial_mass - err_initial_mass, initial_mass + err_initial_mass, alpha=0.4, color="C9")
plt.errorbar(
    initial_mass,
    np.array(msc)[order],
    xerr=err_initial_mass,
    ecolor="C9",
    elinewidth=3,
    alpha=0.5,
    fmt=".",
)
x = np.linspace(0.5, 1.37, 100)
y = IFMR(x)
plt.plot(y, x, c="C9")
plt.scatter(initial_mass, np.array(msc)[order], zorder=1000, s=5)


plt.xlabel("Initial mass ($M_\\odot$)")
plt.ylabel("Final mass ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w14-IFMR.pgf", format="pgf")
plt.show()
plt.close()
# %%
TPAGB1 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/TPAGB/history.data"
)
TPAGB2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/standard-2msun/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
for string, TPAGB in zip([r"$4\;M_\odot$", r"$2\;M_\odot$"], [TPAGB1, TPAGB2]):
    index = np.argwhere(TPAGB.surface_c12 / TPAGB.surface_o16 * 16 / 12 > 1)[0][0]
    (l1,) = plt.plot(
        TPAGB.star_age[:index] / TPAGB.star_age[-1],
        TPAGB.R[:index],
        c="C3",
        label="C/O-ratio $< 1$",
    )
    (l2,) = plt.plot(
        TPAGB.star_age[index:] / TPAGB.star_age[-1],
        TPAGB.R[index:],
        c="C2",
        label="C/O-ratio $> 1$",
    )
    plt.text(1.05, TPAGB.R[-1] - 20, string)
plt.xlim(-0.05, 1.2)


plt.legend(handles=[l1, l2])
plt.xlabel("Normalized TPAGB time")
plt.ylabel("Radius ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w14-radius.pgf", format="pgf")
plt.show()
plt.close()
# %%


def delta(R):
    R_RL = 3
    R_OL = 7
    delta = (R - R_OL) / R_OL
    return np.clip(delta, 0, 1)


rs = np.linspace(0, 10, 100)
plt.plot(rs, delta(rs))
plt.show()
# %%
