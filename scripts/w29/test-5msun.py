import mesa_reader as mr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
sys.path.append("/home/koen/astro-codes/mkipp")
sys.path.append("/home/koen/astro-codes/read_mist/")

import read_mist_models
import mkipp
import kipp_data
import mesa_data
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

mkipp.kipp_plot(
    mkipp.Kipp_Args(
        logs_dirs=[
            "/home/koen/master-internship/mesa-models/compare-overshooting/rees/3msun/LOGS/MS/"
        ],
        core_masses=["He", "CO"],
        levels=[],
        log_levels=True,
        num_levels=10,
        xaxis="star_age",
    )
)
plt.show()
# %%

profile = mr.MesaData(
    "/home/koen/master-internship/mesa-models/compare-overshooting/rees/3msun/LOGS/MS/history.data"
)


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


kipp_args = mkipp.Kipp_Args(
    logs_dirs=[f"/home/koen/master-internship/mesa-models/{5}msun/LOGS/TPAGB/"],
    # xaxis="star_age",
    save_file=False,
    decorate_plot=False,
    # contour_colormap=plt.get_cmap("Greens"),
    # levels=np.linspace(-1, 5, 50),
    log10_on_data=True,
    levels=np.linspace(-1, 7, 50),
    identifier="tri_alpha",
)
mkipp.kipp_plot(kipp_args, axis=axs)
plt.show()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


kipp_args = mkipp.Kipp_Args(
    logs_dirs=[f"/home/koen/master-internship/mesa-models/{5}msun/LOGS/TPAGB/"],
    xaxis="star_age",
    save_file=False,
    decorate_plot=False,
    # contour_colormap=plt.get_cmap("Greens"),
    # levels=np.linspace(-1, 5, 50),
    log10_on_data=True,
    levels=np.arange(-5, 1.01, 0.1),
    identifier="z_mass_fraction_metals",
)
mkipp.kipp_plot(kipp_args, axis=axs)
plt.show()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


kipp_args = mkipp.Kipp_Args(
    logs_dirs=[f"/home/koen/master-internship/mesa-models/{5}msun/LOGS/TPAGB/"],
    xaxis="star_age",
    save_file=False,
    decorate_plot=False,
    # contour_colormap=plt.get_cmap("Greens"),
    # levels=np.linspace(-1, 5, 50),
    log10_on_data=True,
    levels=np.linspace(-100, 50, 50),
    identifier="tri_alpha",
)
mkipp.kipp_plot(kipp_args, axis=axs)
plt.show()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


kipp_args = mkipp.Kipp_Args(
    logs_dirs=[
        f"/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/TPAGB"
    ],
    xaxis="star_age",
    save_file=False,
    decorate_plot=False,
    # contour_colormap=plt.get_cmap("Greens"),
    # levels=np.linspace(-1, 5, 50),
    log10_on_data=True,
    levels=np.linspace(-100, 50, 50),
    identifier="tri_alpha",
)
mkipp.kipp_plot(kipp_args, axis=axs)
plt.show()


# %%

data = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/TPAGB/history.data"
)
# %%

data.bulk_names

# %%

datas = []

for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
    datas.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/tpagb-test-high-mass/LOGS/{phase}/history.data"
        )
    )


# %%

for data in datas:
    plt.plot(data.log_cntr_Rho, data.log_cntr_T)

plt.show()
# %%

plt.plot(data.star_age, data.log_LHe)
plt.plot(data.star_age, data.log_L)
plt.show()
# %%

print(data.model_number[-1])
# %%
