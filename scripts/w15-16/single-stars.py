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

# %%

models = []
masses = [2.5, 3]
for mass in masses:
    models.append(
        mr.MesaData(f"{MASTER}single-stars/z0.014/M{mass:.1f}/LOGS/TPAGB/history.data")
    )

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in models:
    axs[0].plot(model.star_age, model.R)
    axs[1].plot(model.star_age, model.star_mass)
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-masses-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
masses = [3.5]
for mass in masses:
    models.append(
        mr.MesaData(f"{MASTER}single-stars/z0.014/M{mass:.1f}/LOGS/TPAGB/history.data")
    )


# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

masses = np.arange(2.5, 6.5, 0.5)
for i, model in enumerate(models):
    axs[0].plot(
        model.star_age,
        model.R,
        rasterized=True,
        linewidth=1.0,
        label=f"$M = {masses[i]:.1f} M_\\odot$",
    )
    axs[1].plot(
        model.star_age,
        model.surface_c12 / model.surface_o16 * 16 / 12,
        rasterized=True,
        linewidth=1,
    )
    axs[2].plot(
        model.star_age, model.envelope_fraction_left, rasterized=True, linewidth=1
    )

    axs[0].scatter(
        model.star_age[-1],
        model.R[-1],
        rasterized=True,
        marker="x" if i < 4 else ".",
    )
    axs[1].scatter(
        model.star_age[-1],
        model.surface_c12[-1] / model.surface_o16[-1] * 16 / 12,
        rasterized=True,
        marker="x" if i < 4 else ".",
    )
    axs[2].scatter(
        model.star_age[-1],
        model.envelope_fraction_left[-1],
        rasterized=True,
        marker="x" if i < 4 else ".",
    )


fig.legend(loc="outside upper center", ncols=4)
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel(r"C/O-ratio")
axs[2].set_ylabel(r"Envelope fraction left")
axs[2].set_xlabel(r"Age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-masses-1.pgf", format="pgf", dpi=600)
plt.show()
plt.close()

# %%

model.bulk_names
# %%
masses = [5, 5.5, 6]
for mass in masses:
    models.append(
        mr.MesaData(f"{MASTER}single-stars/z0.014/M{mass:.1f}/LOGS/TPAGB/history.data")
    )


# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

masses = np.arange(2.5, 6.5, 0.5)
for i, model in enumerate(models):
    if i > 3 or i < 1:
        continue
    axs[0].plot(
        model.star_age / model.star_age[-1],
        model.R,
        rasterized=True,
        linewidth=1.0,
        label=f"$M = {masses[i]:.1f} M_\\odot$",
    )
    axs[1].plot(
        model.star_age / model.star_age[-1],
        model.surface_c12 / model.surface_o16 * 16 / 12,
        rasterized=True,
        linewidth=1,
    )

    axs[0].scatter(
        model.star_age[-1] / model.star_age[-1],
        model.R[-1],
        rasterized=True,
        marker="x" if i < 4 else ".",
    )
    axs[1].scatter(
        model.star_age[-1] / model.star_age[-1],
        model.surface_c12[-1] / model.surface_o16[-1] * 16 / 12,
        rasterized=True,
        marker="x" if i < 4 else ".",
    )


fig.legend(loc="outside upper center", ncols=4)
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel(r"C/O-ratio")
axs[1].set_xlabel(r"Age (yr)")
axs[0].set_ylim(200, 550)
axs[0].set_xlim(0.3, 0.5)
axs[1].set_ylim(0, 2)
plt.savefig("/home/koen/LaTeX-setup/plots/w15-masses-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

profile = mr.MesaData(f"{MASTER}single-stars/z0.014/M4.0/LOGS/TPAGB/profile541.data")
profile2 = mr.MesaData(f"{MASTER}single-stars/z0.014/M4.0/LOGS/TPAGB/profile540.data")
profile3 = mr.MesaData(f"{MASTER}single-stars/z0.014/M4.0/LOGS/TPAGB/profile54.data")
# %%
print(profile.bulk_names)
plt.plot(profile.logR, profile.gradT)
plt.plot(profile2.logR, profile2.gradT)
plt.plot(profile3.logR, profile3.gradT)
plt.show()

# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

masses = np.arange(2.5, 6.5, 0.5)
for i, model in enumerate(models):
    if i > 3 or i < 1:
        continue
    axs[0].plot(
        model.star_age / model.star_age[-1],
        model.model_number,
        ".-",
        rasterized=True,
        linewidth=1.0,
        label=f"$M = {masses[i]:.1f} M_\\odot$",
    )
    axs[1].plot(
        model.star_age / model.star_age[-1],
        model.surface_c12 / model.surface_o16 * 16 / 12,
        rasterized=True,
        linewidth=1,
    )


fig.legend(loc="outside upper center", ncols=4)
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel(r"C/O-ratio")
axs[1].set_xlabel(r"Age (yr)")
# axs[0].set_ylim(200, 550)
axs[0].set_xlim(0.3, 0.5)
axs[1].set_ylim(0, 2)
plt.savefig("/home/koen/LaTeX-setup/plots/w15-masses-3.pgf", format="pgf", dpi=600)
plt.show()
plt.close()


# %%

model.bulk_names
# %%

new = mr.MesaData(f"{MASTER}single-stars/tests/M3.5/LOGS/TPAGB/history.data")
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(full, height=1), constrained_layout=True
)

axs[0].plot(
    new.star_age / new.star_age[-1],
    new.R,
    rasterized=True,
    linewidth=1.0,
)
axs[1].plot(
    new.star_age / new.star_age[-1],
    new.surface_c12 / new.surface_o16 * 16 / 12,
    rasterized=True,
    linewidth=1,
)

fig.legend(loc="outside upper center", ncols=4)
axs[0].set_ylabel(r"Radius ($R_\odot$)")
axs[1].set_ylabel(r"C/O-ratio")
axs[1].set_xlabel(r"Age (yr)")
axs[0].set_ylim(200, 550)
axs[0].set_xlim(0.3, 0.5)
axs[1].set_ylim(0, 2)
plt.savefig("/home/koen/LaTeX-setup/plots/w15-masses-4.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

model = models[1]

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(model.star_age, model.Teff)

plt.xlim(0, 2.5e6)
plt.ylim(10**3.35, 10**3.65)
plt.yscale("log")

plt.xlabel("Age (yr)")
plt.ylabel(r"$T_\textrm{eff}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w15-3M.pgf", format="pgf")
plt.show()
plt.close()
# %%

from PIL import Image

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(model.star_age - 76000, model.log_Teff, c="k", linewidth=1)

plt.xlim(0, 2.0e6)
plt.ylim(3.31, 3.60)


image = Image.open("scripts/w15-16/bitmap.png")
axs.imshow(image, cmap=plt.cm.Reds, interpolation="none", extent=[0, 2e6, 3.31, 3.60])
axs.set_aspect(
    6896551.72414 * 2 / 3
)  # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio
plt.xlabel("Star age (yr)")
plt.ylabel(r"$\log(T_\textrm{eff})$")

plt.savefig("/home/koen/LaTeX-setup/plots/w15-3M-2.pgf", format="pgf")
plt.close()

# %%

from PIL import Image

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(model.star_age - 76000, model.log_abs_mdot, c="k", linewidth=1)

plt.xlim(0, 2.0e6)
plt.ylim(-12, -4)


image = Image.open("scripts/w15-16/bitmap2.png")
axs.imshow(image, cmap=plt.cm.Reds, interpolation="none", extent=[0, 1.4e6, -12, -4])
axs.set_aspect(
    (1.4e6 / 8)
)  # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio
plt.xlabel("Star age (yr)")
plt.ylabel(r"$\log(T_\textrm{eff})$")

plt.savefig("/home/koen/LaTeX-setup/plots/w15-3M-3.pgf", format="pgf")
plt.show()
plt.close()


# %%

model = models[3]
from PIL import Image

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(model.star_age, model.log_R, c="k", linewidth=1)

plt.xlim(0, 400e3)


image = Image.open("scripts/w15-16/bitmap3.png")
axs.imshow(image, cmap=plt.cm.Reds, interpolation="none", extent=[0, 230000, 2.5, 3.02])
axs.set_aspect(
    (230000 / (0.52))
)  # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio
plt.xlabel("Star age (yr)")
plt.ylabel(r"$\log(T_\textrm{eff})$")

plt.savefig("/home/koen/LaTeX-setup/plots/w15-3M-4.pgf", format="pgf")
plt.show()
plt.close()
# %%
