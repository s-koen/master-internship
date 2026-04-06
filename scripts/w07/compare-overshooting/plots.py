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
print(profile.bulk_names)
# %%
import matplotlib as mpl
from matplotlib.patches import PathPatch

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]

for mass in masses:
    fig, axss = plt.subplots(
        4, 4, sharex="col", figsize=set_size(full, height=1), constrained_layout=True
    )

    axs = []
    for i in range(4):
        axs.append(axss[i, :])

    for m, method in enumerate(methods):
        for i, axis in enumerate(axs[m]):
            phase = phases[i]
            kipp_args = mkipp.Kipp_Args(
                logs_dirs=[
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/"
                ],
                xaxis="star_age",
                save_file=False,
                decorate_plot=False,
                # contour_colormap=plt.get_cmap("Greens"),
                levels=np.linspace(-1, 5, 50),
            )
            mkipp.kipp_plot(kipp_args, axis=axis)
            axis.set_rasterization_zorder(-10)
            if m == 0:
                axis.set_xlabel(f"{phase} age (Myr)")
                axis.tick_params(
                    top=True, labeltop=True, bottom=False, labelbottom=False
                )
                axis.xaxis.set_label_position("top")
            if i == 0:
                axis.set_ylabel(f"mass ($M_\odot$)")

            elif i == 3:
                axis.set_ylabel(f"mass ($M_\odot$)")
                axis.tick_params(
                    right=True, labelright=True, left=False, labelleft=False
                )
                axis.yaxis.set_label_position("right")
            else:
                axis.tick_params(
                    right=True, labelright=False, left=True, labelleft=False
                )

    ymin = [0, 0, 0, 0]
    ymax = [0.4, 1, 1, 0.50]
    hatches = ["////", "||||", "\\\\\\\\"]
    for a, axis in enumerate(axs[-1]):

        for m, method in enumerate(methods):
            print(method, mass, i)
            phase = phases[a]
            kipp_args = mkipp.Kipp_Args(
                logs_dirs=[
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/"
                ],
                xaxis="star_age",
            )

            profile_paths = mesa_data.get_profile_paths(
                [
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/"
                ]
            )
            # if data is distributed among several history.data files, you can provide them
            history_paths = [
                f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
            ]
            xyz_data = kipp_data.get_xyz_data(profile_paths, kipp_args)
            mixing_zones = kipp_data.get_mixing_zones(
                history_paths, kipp_args, xlims=xyz_data.xlims
            )

            for i, zone in enumerate(mixing_zones.zones):
                color = ""
                if mixing_zones.mix_types[i] == 2:  # overshooting
                    color = f"C{m}"
                    hatch = hatches[m]
                else:
                    continue
                axis.add_patch(
                    PathPatch(
                        zone,
                        color=color,
                        fill=False,
                        alpha=1,
                        hatch=hatch,
                        rasterized=True,
                        linewidth=0.3,
                    )
                )
            # if xyz_data.Z.size > 0:
            #     CS = plt.contour(xyz_data.X, xyz_data.Y, xyz_data.Z, [0, 4, 8], colors="k")
            #     plt.clabel(CS, inline=1, fontsize=10)
        axis.plot(
            mixing_zones.x_coords,
            mixing_zones.y_coords,
            "w",
        )
        axis.set_xlabel(f"{phase} age (Myr)")
        axis.set_xlim(0)
        axis.set_ylim(ymin[a], ymax[a] * mass)
        if a == 0:
            axis.set_ylabel(f"mass ($M_\odot$)")

        if a == 3:
            axis.set_ylabel(f"mass ($M_\odot$)")
            axis.tick_params(right=True, labelright=True, left=True, labelleft=False)
            axis.yaxis.set_label_position("right")

    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=-1, vmax=5)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axss,
        orientation="horizontal",
        label=r"$\log(\epsilon_\textrm{nuc})$",
        location="top",
        fraction=0.05,
        extend="both",
        aspect=50,
    )

    plt.savefig(
        f"/home/koen/LaTeX-setup/plots/kippenhahn{mass}.pgf", format="pgf", dpi=600
    )
# %%

fig, axs = plt.subplots(
    1, 2, sharex=True, figsize=set_size(column), constrained_layout=True
)

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 4, 6, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
linestyle = [":", "-", "-"]

for mass in masses:
    for m, method in enumerate(methods):
        for i, phase in enumerate(phases):
            history = mr.MesaData(
                f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
            )
            try:
                index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[0][0]
            except:
                index = 0
            axs[0].plot(
                history.log_Teff[index:],
                history.log_L[index:],
                c=f"C{m}",
                linestyle=linestyle[m],
                linewidth=3,
                zorder=-m,
            )

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [3, 5, 7]
phases = ["MS", "GB", "CHeB", "EAGB"]
linestyle = [":", "-", "-"]

for mass in masses:
    for m, method in enumerate(methods):
        for i, phase in enumerate(phases):
            history = mr.MesaData(
                f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
            )
            try:
                index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[0][0]
            except:
                index = 0
            axs[1].plot(
                history.log_Teff[index:],
                history.log_L[index:],
                c=f"C{m}",
                linestyle=linestyle[m],
                linewidth=3,
                zorder=-m,
            )


plt.gca().invert_xaxis()
plt.xlabel(r"Effective temperature ($T_\textrm{eff}$)")
plt.ylabel(r"Luminosity ($L_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    3, 3, figsize=set_size(full, height=1.2), constrained_layout=True
)

axs = axs.flatten()

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
color = ["C0", "C9", "C1"]
linewidth = [1.5, 4, 1.5]
zorder = [3, 0, 1]
alpha = [1, 1, 1]
label = [
    "Rees et al. 2024",
    "Temmink et al. 2023",
    "Temmink et al. 2023,\nother $f_\\textrm{ov,0}$",
]

for i, ax in enumerate(axs):
    try:

        mass = masses[i]

        eep = read_mist_models.EEP(
            f"/home/koen/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00{mass}00M.track.eep"
        )

        ls = []
        for m, method in enumerate(methods):

            for i, phase in enumerate(phases):
                history = mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
                )
                try:
                    index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[
                        0
                    ][0]
                except:
                    index = 0
                if i == 0:
                    (l,) = ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        label=label[m],
                        alpha=alpha[m],
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        alpha=alpha[m],
                    )
        ax.invert_xaxis()

        ax.text(
            0.05,
            0.95,
            rf"$M={mass} M_\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        p = eep.eeps["phase"]
        for i_p, pp in enumerate([0, 1, 2, 3, 4]):
            p = eep.eeps["phase"]
            p_ind = np.where(p == pp)
            if len(p_ind) > 0:
                if i_p == 0:
                    (l,) = ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=-10,
                        alpha=1,
                        label="MIST",
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=2,
                        alpha=1,
                    )
    except:
        ax.axis("off")


axs[7].legend(loc="upper left", frameon=False, handles=ls)
axs[6].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[4].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
axs[3].set_ylabel(r"$\log(L / L_\odot)$")
axs[6].set_ylabel(r"$\log(L / L_\odot)$")
axs[5].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 3, figsize=set_size(full, height=1.2), constrained_layout=True
)

axs = axs.flatten()

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
phases = ["MS"]
color = ["C0", "C9", "C1"]
linewidth = [1.5, 4, 1.5]
zorder = [3, 0, 1]
alpha = [1, 1, 1]
label = [
    "Rees et al. 2024",
    "Temmink et al. 2023",
    "Temmink et al. 2023,\nother $f_\\textrm{ov,0}$",
]

for i, ax in enumerate(axs):
    try:

        mass = masses[i]

        eep = read_mist_models.EEP(
            f"/home/koen/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00{mass}00M.track.eep"
        )

        ls = []
        for m, method in enumerate(methods):

            for i, phase in enumerate(phases):
                history = mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
                )
                try:
                    index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[
                        0
                    ][0]
                except:
                    index = 0
                if i == 0:
                    (l,) = ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        label=label[m],
                        alpha=alpha[m],
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        alpha=alpha[m],
                    )
        ax.invert_xaxis()

        ax.text(
            0.05,
            0.95,
            rf"$M={mass} M_\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        p = eep.eeps["phase"]
        for i_p, pp in enumerate([0]):
            p = eep.eeps["phase"]
            p_ind = np.where(p == pp)
            if len(p_ind) > 0:
                if i_p == 0:
                    (l,) = ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=-10,
                        alpha=1,
                        label="MIST",
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=2,
                        alpha=1,
                    )
    except:
        ax.axis("off")

axs[8].text(
    0.05,
    0.95,
    rf"{phases[0]} phase",
    horizontalalignment="left",
    verticalalignment="top",
)
axs[7].legend(loc="upper left", frameon=False, handles=ls)
axs[6].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[4].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
axs[3].set_ylabel(r"$\log(L / L_\odot)$")
axs[6].set_ylabel(r"$\log(L / L_\odot)$")
axs[5].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting-MS.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    3, 3, figsize=set_size(full, height=1.2), constrained_layout=True
)

axs = axs.flatten()

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
phases = ["GB"]
color = ["C0", "C9", "C1"]
linewidth = [1.5, 4, 1.5]
zorder = [3, 0, 1]
alpha = [1, 1, 1]
label = [
    "Rees et al. 2024",
    "Temmink et al. 2023",
    "Temmink et al. 2023,\nother $f_\\textrm{ov,0}$",
]

for i, ax in enumerate(axs):
    try:

        mass = masses[i]

        eep = read_mist_models.EEP(
            f"/home/koen/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00{mass}00M.track.eep"
        )

        ls = []
        for m, method in enumerate(methods):

            for i, phase in enumerate(phases):
                history = mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
                )
                try:
                    index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[
                        0
                    ][0]
                except:
                    index = 0
                if i == 0:
                    (l,) = ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        label=label[m],
                        alpha=alpha[m],
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        alpha=alpha[m],
                    )
        ax.invert_xaxis()

        ax.text(
            0.05,
            0.95,
            rf"$M={mass} M_\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        p = eep.eeps["phase"]
        for i_p, pp in enumerate([1, 2]):
            p = eep.eeps["phase"]
            p_ind = np.where(p == pp)
            if len(p_ind) > 0:
                if i_p == 0:
                    (l,) = ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=-10,
                        alpha=1,
                        label="MIST",
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=2,
                        alpha=1,
                    )
    except:
        ax.axis("off")

axs[8].text(
    0.05,
    0.95,
    rf"{phases[0]} phase",
    horizontalalignment="left",
    verticalalignment="top",
)
axs[7].legend(loc="upper left", frameon=False, handles=ls)
axs[6].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[4].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
axs[3].set_ylabel(r"$\log(L / L_\odot)$")
axs[6].set_ylabel(r"$\log(L / L_\odot)$")
axs[5].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting-GB.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 3, figsize=set_size(full, height=1.2), constrained_layout=True
)

axs = axs.flatten()

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
phases = ["CHeB"]
color = ["C0", "C9", "C1"]
linewidth = [1.5, 4, 1.5]
zorder = [3, 0, 1]
alpha = [1, 1, 1]
label = [
    "Rees et al. 2024",
    "Temmink et al. 2023",
    "Temmink et al. 2023,\nother $f_\\textrm{ov,0}$",
]

for i, ax in enumerate(axs):
    try:

        mass = masses[i]

        eep = read_mist_models.EEP(
            f"/home/koen/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00{mass}00M.track.eep"
        )

        ls = []
        for m, method in enumerate(methods):

            for i, phase in enumerate(phases):
                history = mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
                )
                try:
                    index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[
                        0
                    ][0]
                except:
                    index = 0
                if i == 0:
                    (l,) = ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        label=label[m],
                        alpha=alpha[m],
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        alpha=alpha[m],
                    )
        ax.invert_xaxis()

        ax.text(
            0.05,
            0.95,
            rf"$M={mass} M_\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        p = eep.eeps["phase"]
        for i_p, pp in enumerate([3]):
            p = eep.eeps["phase"]
            p_ind = np.where(p == pp)
            if len(p_ind) > 0:
                if i_p == 0:
                    (l,) = ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=-10,
                        alpha=1,
                        label="MIST",
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=2,
                        alpha=1,
                    )
    except:
        ax.axis("off")

axs[8].text(
    0.05,
    0.95,
    rf"{phases[0]} phase",
    horizontalalignment="left",
    verticalalignment="top",
)
axs[7].legend(loc="upper left", frameon=False, handles=ls)
axs[6].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[4].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
axs[3].set_ylabel(r"$\log(L / L_\odot)$")
axs[6].set_ylabel(r"$\log(L / L_\odot)$")
axs[5].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting-CHeB.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 3, figsize=set_size(full, height=1.2), constrained_layout=True
)

axs = axs.flatten()

methods = ["rees", "temmink", "temmink-no-typo"]
masses = [2, 3, 4, 5, 6, 7, 8]
phases = ["MS", "GB", "CHeB", "EAGB"]
phases = ["EAGB"]
color = ["C0", "C9", "C1"]
linewidth = [1.5, 4, 1.5]
zorder = [3, 0, 1]
alpha = [1, 1, 1]
label = [
    "Rees et al. 2024",
    "Temmink et al. 2023",
    "Temmink et al. 2023,\nother $f_\\textrm{ov,0}$",
]

for i, ax in enumerate(axs):
    try:

        mass = masses[i]

        eep = read_mist_models.EEP(
            f"/home/koen/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00{mass}00M.track.eep"
        )

        ls = []
        for m, method in enumerate(methods):

            for i, phase in enumerate(phases):
                history = mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/compare-overshooting/{method}/{mass}msun/LOGS/{phase}/history.data"
                )
                try:
                    index = np.where(history.center_h1 / history.center_h1[0] < 0.99)[
                        0
                    ][0]
                except:
                    index = 0
                if i == 0:
                    (l,) = ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        label=label[m],
                        alpha=alpha[m],
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        history.log_Teff[index:],
                        history.log_L[index:],
                        c=color[m],
                        linewidth=linewidth[m],
                        zorder=zorder[m],
                        alpha=alpha[m],
                    )
        ax.invert_xaxis()

        ax.text(
            0.05,
            0.95,
            rf"$M={mass} M_\odot$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        p = eep.eeps["phase"]
        for i_p, pp in enumerate([4]):
            p = eep.eeps["phase"]
            p_ind = np.where(p == pp)
            if len(p_ind) > 0:
                if i_p == 0:
                    (l,) = ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=-10,
                        alpha=1,
                        label="MIST",
                    )
                    ls.append(l)
                else:
                    ax.plot(
                        eep.eeps["log_Teff"][p_ind],
                        eep.eeps["log_L"][p_ind],
                        c="C2",
                        linewidth=1.5,
                        zorder=2,
                        alpha=1,
                    )
    except:
        ax.axis("off")


axs[7].legend(loc="upper left", frameon=False, handles=ls)
axs[6].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[4].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
axs[0].set_ylabel(r"$\log(L / L_\odot)$")
axs[3].set_ylabel(r"$\log(L / L_\odot)$")
axs[6].set_ylabel(r"$\log(L / L_\odot)$")
axs[5].set_xlabel(r"$\log(T_\textrm{eff} / \textrm{K})$")
plt.savefig("/home/koen/LaTeX-setup/plots/hr-overshooting-EAGB.pgf", format="pgf")
plt.show()
plt.close()
# %%

print(eep.abun)
# %%
