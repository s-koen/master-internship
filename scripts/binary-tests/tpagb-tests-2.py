import mesa_reader as mr

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
# %%


def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%
history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.375/LOGS/TPAGB/history.data"
)

binary_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/R225.00_q0.375/binary_history.data"
)
# %%
print(binary_history.bulk_names)
print(history.bulk_names)
plt.plot(history.star_age, np.log10(history.star_mass - history.he_core_mass))
plt.show()
# %%

plt.plot(history.star_age, history.log_R)
plt.plot(history.star_age, np.log10(history.rl_1))
plt.plot(history.age, np.log10(rol(history)))
plt.show()
# %%

plt.plot(history.star_age, history.R)
plt.plot(history.star_age, history.rl_1)
plt.plot(history.age, rol(history))
plt.show()
# %%

fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs[1].plot(history.star_age, history.log_R, label="Star radius")
axs[1].plot(history.star_age, np.log10(history.rl_1), label="Inner Roche lobe radius")
axs[1].plot(history.age, np.log10(rol(history)), label="Outer Roche lobe radius")

axs[0].plot(history.star_age, history.R, label="Star radius")
axs[0].plot(history.star_age, history.rl_1, label="Inner Roche lobe radius")
axs[0].plot(history.age, rol(history), label="Outer Roche lobe radius")


axs[1].legend()
axs[0].set_xlabel("Binary age (yr)")
axs[1].set_xlabel("Binary age (yr)")
axs[0].set_ylabel(r"$R / R_\odot$")
axs[1].set_ylabel(r"$\log(R / R_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/rocheouterlobe.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs = [axs]
env = history.star_mass - history.he_core_mass

axs[0].invert_xaxis()

axs[0].plot(env, history.R, label="Star radius")
axs[0].plot(env, history.rl_1, label="Inner Roche lobe radius")
axs[0].plot(env, rol(history), label="Outer Roche lobe radius")


axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R / R_\odot$")
plt.legend()
plt.savefig("/home/koen/LaTeX-setup/plots/rocheouterlobe-mass.pgf", format="pgf")
plt.show()
plt.close()


# %%
plt.plot(history.star_age, history.log_dt)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs = [axs]
env = history.star_mass - history.he_core_mass

axs[0].invert_xaxis()

axs[0].plot(env, history.log_abs_mdot, label="Star radius")
axs[0].plot(env, history.lg_quasi_adiabatic_Mdot, label="Inner Roche lobe radius")


axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R / R_\odot$")
plt.legend()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/massloss-vs-quasi-adiabatic-mass.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 2, sharey=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs[1].plot(history.star_age, history.log_abs_mdot, label="Star radius")
axs[1].plot(
    history.star_age, history.lg_quasi_adiabatic_Mdot, label="Inner Roche lobe radius"
)

axs[0].plot(env, history.log_abs_mdot, label="actual")
axs[0].plot(env, history.lg_quasi_adiabatic_Mdot, label="quasi-adiabatic")

axs[0].invert_xaxis()

axs[0].legend()
axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[1].set_xlabel("Binary age (yr)")
axs[0].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{ yr}))$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/mass-loss-vs-quasi-adiabatic.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs = [axs]
env = history.star_mass - history.he_core_mass

axs[0].invert_xaxis()

axs[0].plot(env, history.period_days, label="Star radius")


axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
axs[0].set_ylabel(r"$R / R_\odot$")
plt.savefig("/home/koen/LaTeX-setup/plots/w5-period.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

axs = [axs]
env = history.star_mass - history.he_core_mass

axs[0].invert_xaxis()

sc = axs[0].scatter(
    history.log_Teff,
    history.log_L,
    c=history.log_R,
    label="Star radius",
    s=2,
    cmap="viridis",
    vmin=0.5,
    rasterized=True,
)


plt.colorbar(sc, extend="min", label=r"$\log(R / R_\odot)$")
axs[0].set_xlabel(r"$\log(T_\textrm{eff} / K)$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
plt.savefig("/home/koen/LaTeX-setup/plots/w6-HR.pgf", format="pgf", dpi=900)
plt.show()
plt.close()
# %%
