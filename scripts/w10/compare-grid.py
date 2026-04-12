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
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

from scipy.integrate import solve_ivp

# %%
Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


def get_period(a, M, q):
    return (
        np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * (1 + q) * M * (Msun))) / 3600 / 24
    )


def a_final(a_initial, M_initial, M_final, q):
    M_tot = M_initial + q * M_initial
    return (
        a_initial
        * ((M_initial * (M_tot - M_initial)) / (M_final * (M_tot - M_final))) ** 2
    )


def rl(a, q):
    return a * 0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log(1 + q ** (-1 / 3)))


def get_separation(R, q):
    return (
        (0.6 * (q) ** (-2 / 3) + np.log(1 + (q) ** (-1 / 3)))
        / (0.49 * (q) ** (-2 / 3))
        * R
    )


def evolve_q(q_initial, M1_initial, M1, beta=1):
    return (q_initial * M1_initial + beta * (M1_initial - M1)) / M1


def dlnadm(Md, ln_a, Md_i, Ma_i, beta):
    Ma = Ma_i + beta * (Md_i - Md)

    val = (
        -2
        * (1 - beta * (Md / Ma) - (1 - beta) * ((Md / Ma) + 0.5) * (Md / (Md + Ma)))
        / Md
    )

    return val


def integrate_a(a_i, Md_i, Ma_i, Md_f, beta):
    sol = solve_ivp(
        dlnadm,
        t_span=(Md_i, Md_f),  # integrate from initial to final Md
        y0=[np.log(a_i)],  # initial condition: ln(a)
        args=(Md_i, Ma_i, beta),
        rtol=1e-8,
        atol=1e-10,
    )

    return np.exp(sol.y[0, -1])  # %%


# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/binary-tpagb-grid-5/")

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = 0.6

R_initials = np.linspace(150, 500, 100)
for q in np.linspace(0.4, 1, 7):
    a_fs = []
    for R_i in R_initials:
        a_initial = get_separation(R_i, q)
        q_f = evolve_q(q, M_initial, M_final)
        a_f = a_final(a_initial, M_initial, M_final, q)
        a_fs.append(a_f)
    a_fs = np.array(a_fs)
    plt.plot(R_initials, get_period(a_fs, M_final, q_f), label=f"$q={q:.1f}$")


plt.xlim(150, 650)
plt.ylim(1800, 15000)
fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical4.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    periods = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            periods.append(model.star.period_days[0])
            Rs.append(R)
        print(
            qs,
            R,
            (model.star.star_2_mass[0] - model.star.star_2_mass[-1])
            / (model.star.star_1_mass[-1] - model.star.star_1_mass[0]),
        )
        print(np.max(model.star.eff_xfer_fraction))

    plt.plot(Rs, periods, ".", label=f"$q={qs}$")

M_initial = 2
M_final = 0.6

R_initials = np.linspace(150, 650, 100)
for i, q in enumerate(np.linspace(0.4, 1, 7)):
    a_fs = []
    a_is = []
    for R_i in R_initials:
        a_initial = get_separation(R_i, q)
        q_f = evolve_q(q, M_initial, M_final)
        a_f = a_final(a_initial, M_initial, M_final, q)
        a_fs.append(a_f)
        a_is.append(a_initial)
    a_is = np.array(a_is)
    a_fs = np.array(a_fs)
    plt.plot(R_initials, get_period(a_is, M_initial, q), c=f"C{i}", alpha=0.5)


fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-grid-1.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    periods = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            periods.append(model.star.period_days[-1])
            Rs.append(R)

    plt.plot(Rs, periods, ".-", label=f"$q={qs}$")


fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w8-grid-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

Md_i = 2.0
q_i = 0.5
Ma_i = q_i * Md_i

Md_f = 0.6
beta = 0.8

a_initial = 1000

a_f = integrate_a(a_initial, Md_i, Ma_i, Md_f, beta)
print(a_initial, a_f)
a_f = a_final(a_initial, Md_i, Md_f, q)
print(a_initial, a_f)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    periods = []
    integrated_periods = []
    computed_periods = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            periods.append(model.star.period_days[-1])
            Rs.append(R)

        Md_i = 2.0
        Ma_i = qs * Md_i

        Md_f = model.star.star_1_mass[-1]
        beta = (model.star.star_2_mass[-1] - model.star.star_2_mass[0]) / (
            model.star.star_1_mass[0] - model.star.star_1_mass[-1]
        )
        beta = 1
        a_initial = get_separation(R, qs)
        a_f = integrate_a(a_initial, Md_i, Ma_i, Md_f, beta)
        a_f2 = a_final(a_initial, Md_i, Md_f, qs)
        integrated_periods.append(get_period(a_f, Md_f, evolve_q(qs, Md_i, Md_f, beta)))
        computed_periods.append(get_period(a_f2, Md_f, evolve_q(qs, Md_i, Md_f)))

    plt.plot(Rs, periods, "o", c=f"C{i}")
    plt.plot(Rs, computed_periods, label=f"$q={qs}$")


fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-grid-3.pgf", format="pgf")
plt.show()
plt.close()
# %%


def integrate_a_hybrid(a_i, Md, Ma, Mdot_d, Mdot_a, t, Jdot_over_J):
    ln_a = np.log(a_i)

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]

        # midpoint values (important for stability)
        Md_mid = 0.5 * (Md[i] + Md[i + 1])
        Ma_mid = 0.5 * (Ma[i] + Ma[i + 1])
        Ma_mid = 0.5 * (Ma[i] + Ma[i + 1])
        Mtot_mid = Md_mid + Ma_mid
        Mdot_d_mid = 0.5 * (Mdot_d[i] + Mdot_d[i + 1])
        Mdot_a_mid = 0.5 * (Mdot_a[i] + Mdot_a[i + 1])
        Jdot_over_J_mid = 0.5 * (Jdot_over_J[i] + Jdot_over_J[i + 1])

        # --- full equation ---
        dlnadt = (
            2 * Jdot_over_J_mid
            - 2 * Mdot_d_mid / Md_mid
            - 2 * Mdot_a_mid / Ma_mid
            + (Mdot_d_mid + Mdot_a_mid) / Mtot_mid
        )

        ln_a += dlnadt * dt

    return np.exp(ln_a)


def integrate_mass(mdot, t):
    M = 0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]

        # midpoint values (important for stability)
        Mdot_mid = 0.5 * (mdot[i] + mdot[i + 1])
        M += Mdot_mid * dt

    return M


# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    periods = []
    integrated_periods = []
    computed_periods = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue
        else:
            periods.append(model.star.period_days[-1])
            Rs.append(R)

        Md_i = 2.0
        Ma_i = qs * Md_i

        Md_f = model.star.star_1_mass[-1]
        beta = (model.star.star_2_mass[-1] - model.star.star_2_mass[0]) / (
            model.star.star_1_mass[0] - model.star.star_1_mass[-1]
        )
        beta = 1
        a_initial = get_separation(R, qs)
        Md = model.star.star_1_mass
        Ma = model.star.star_2_mass
        Mdot_d = -(10**model.star.lg_mstar_dot_1)
        Mdot_a = 10**model.star.lg_mstar_dot_2
        beta = model.star.eff_xfer_fraction
        t = model.star.star_age
        Jdot_over_J = model.star.Jdot / model.star.J_orb * 3600 * 24 * 365.25
        a_f = integrate_a_hybrid(
            a_initial,
            Md,
            Ma,
            Mdot_d,
            Mdot_a,
            t,
            Jdot_over_J,
        )
        print(a_f)
        integrated_periods.append(get_period(a_f, Md[-1], Ma[-1] / Md[-1]))

    axs[0].plot(Rs, periods, "o", c=f"C{i}")
    axs[0].plot(Rs, integrated_periods, label=f"$q={qs}$", c=f"C{i}")
    axs[1].plot(Rs, np.array(periods) - np.array(integrated_periods), c=f"C{i}")

print(model.star.bulk_names)

fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
axs[1].set_ylabel(r"Residuals (days)")
axs[0].set_ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-grid-4.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

for i, qs in enumerate(np.round(np.linspace(0.4, 1, 7), 2)):
    betas = []
    Rs = []

    for R, model in grid.get_q_slice(qs):
        if model.env_mass[-1] > 0.01:
            continue

        Rs.append(R)
        dMstar2 = model.star.star_2_mass[-1] - model.star.star_2_mass[0]
        t = model.star.star_age
        mdot = 10**model.star.lg_mtransfer_rate
        dMstar1 = integrate_mass(mdot, t)

        beta = dMstar2 / dMstar1
        betas.append(beta)

    plt.plot(Rs, betas, label=f"$q={qs}$")
fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"$\langle \beta \rangle$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-efficiency.pgf", format="pgf")
plt.show()
plt.close()

# %%
model.star.bulk_names
# %%
