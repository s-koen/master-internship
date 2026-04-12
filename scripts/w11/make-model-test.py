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
import mesa_reader as mr

# %%

TPAGB = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/standard-2msun/LOGS/TPAGB/history.data"
)
EAGB = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/standard-2msun/LOGS/EAGB/history.data"
)
# %%

TPAGB.bulk_names

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(TPAGB.star_age, TPAGB.R)
axs[0].plot(
    TPAGB.star_age[np.argwhere(TPAGB.LH / TPAGB.L > 0.7)],
    TPAGB.R[np.argwhere(TPAGB.LH / TPAGB.L > 0.7)],
)
axs[1].plot(TPAGB.star_age, TPAGB.LH / TPAGB.L)

axs[1].set_ylim(0, 10)
plt.show()
# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(EAGB.star_age, EAGB.R)
axs[0].plot(
    EAGB.star_age[np.argwhere(EAGB.LH / EAGB.L > 0.7)],
    TPAGB.R[np.argwhere(EAGB.LH / EAGB.L > 0.7)],
)
axs[1].plot(EAGB.star_age, EAGB.LH / EAGB.L)

axs[1].set_ylim(0, 10)
plt.show()

# %%
TPAGB2 = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/TPAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(TPAGB.star_age, TPAGB.R)
axs[0].plot(TPAGB2.star_age, TPAGB2.R)
axs[1].plot(TPAGB.star_age, TPAGB.LH / TPAGB.L)
axs[1].plot(TPAGB2.star_age, TPAGB2.LH / TPAGB2.L)

axs[1].set_ylim(0, 10)
plt.show()

# %%

TPAGB2.bulk_names
# %%
plt.plot(TPAGB2.star_age, TPAGB2.I_eff)
plt.show()
# %%
plt.plot(TPAGB2.star_age, TPAGB2.R_conv)
plt.show()
# %%
plt.plot(TPAGB2.star_age, TPAGB2.t_conv)
plt.show()
# %%

for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
    data = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/{phase}/history.data"
    )
    for bulk_name in data.bulk_names:
        print(data.data(bulk_name))
# %%
n = [0]
for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
    data = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/{phase}/history.data"
    )
    n.append(n[-1] + len(data.model_number[1:]))

combined_star = {}
for bulk_name in data.bulk_names:
    combined_star[bulk_name] = np.empty(n[-1])
combined_star["phase"] = np.empty(n[-1])

for i, phase in enumerate(["MS", "GB", "CHeB", "EAGB", "TPAGB"]):
    data = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/{phase}/history.data"
    )
    for bulk_name in data.bulk_names:
        if bulk_name in ["model_number", "star_age"] and i != 0:
            combined_star[bulk_name][n[i] : n[i + 1]] = (
                data.data(bulk_name)[1:] + combined_star[bulk_name][n[i] - 1]
            )
        else:
            combined_star[bulk_name][n[i] : n[i + 1]] = data.data(bulk_name)[1:]
    combined_star["phase"][n[i] : n[i + 1]] = i * np.ones(n[i + 1] - n[i])

# %%

plt.plot(combined_star["star_age"], combined_star["log_R"])
plt.show()
#
# %%

plt.plot(combined_star["star_age"], combined_star["I_eff"])
plt.show()

# %%

plt.plot(combined_star["star_age"], combined_star["log_R"])
index = np.where(combined_star["center_h1"] / combined_star["center_h1"][0] < 0.997)[0][
    0
]
plt.scatter(combined_star["star_age"][index], combined_star["log_R"][index])
plt.scatter(combined_star["star_age"][n[1]], combined_star["log_R"][n[1]])
plt.scatter(combined_star["star_age"][n[2]], combined_star["log_R"][n[2]])
plt.scatter(combined_star["star_age"][n[3]], combined_star["log_R"][n[3]])
plt.scatter(combined_star["star_age"][n[4]], combined_star["log_R"][n[4]])
plt.scatter(combined_star["star_age"][n[5] - 1], combined_star["log_R"][n[5] - 1])
plt.show()


# %%
class StellarModel:

    global nzams, ntams, nrgbt, nzacheb, ntacheb, ntpagb, npagb, nwd
    global XHc_exh, XHec_exh

    nzams, ntams, nrgbt, nzacheb, ntacheb, ntpagb, npagb, nwd = (
        201,
        453,
        604,
        630,
        706,
        807,
        1408,
        1709,
    )
    XHc_exh, XHec_exh = 1.0e-4, 1.0e-4

    def __init__(self, track):
        self.m_init = track.minit
        self.Y_init = track.abun["Yinit"]
        self.Z_init = track.abun["Zinit"]
        self.Fe_over_H = track.abun["[Fe/H]"]
        self.alpha_over_Fe = track.abun["[a/Fe]"]

        self.age = track.eeps["star_age"]
        self.n_models = len(self.age)
        self.model = np.array(range(self.n_models))
        self.phase = track.eeps["phase"]

        self.mass = track.eeps["star_mass"]
        self.m_core = track.eeps["he_core_mass"]
        self.m_core_CO = track.eeps["c_core_mass"]
        self.log_R = track.eeps["log_R"]
        self.log_L = track.eeps["log_L"]
        self.log_Teff = track.eeps["log_Teff"]
        self.XH_center = track.eeps["center_h1"]
        self.XHe_center = track.eeps["center_he4"]
        self.log_LH = track.eeps["log_LH"]
        self.log_LHe = track.eeps["log_LHe"]
        self.log_rhoc = track.eeps["log_center_Rho"]
        self.log_Tc = track.eeps["log_center_T"]
        self.Mdot = track.eeps["star_mdot"]

        self.m_env = self.mass - self.m_core
        self.radius = 10**self.log_R
        self.lum = 10**self.log_L

        # find critical points along track:
        self.model_ZAMS = nzams
        self.model_Rmin_ZAMS = self.find_Rmin_ZAMS()
        self.model_TAMS = np.where(self.XH_center < XHc_exh)[0][0]
        self.model_Rmax_MS = self.find_Rmax_MS()
        self.model_Rmax_HSB = np.argmax(self.log_R[ntams : nzacheb + 1]) + ntams
        self.model_Rmax = np.argmax(self.log_R[nzams:]) + nzams
        self.model_BGB = self.find_BGB()
        self.model_first_TP = self.find_first_TP()
        self.model_Rmax_EAGB = self.find_Rmax_EAGB()

        # assign stellar type, analogous to SSE (Hurley et al 2000)
        self.type = np.array(self.n_models * [0])
        self.type[self.model_ZAMS : self.model_TAMS] = 1
        self.type[self.model_TAMS : self.model_BGB] = 2
        self.type[self.model_BGB : nzacheb + 1] = 3
        self.type[nzacheb + 1 : ntacheb] = 4
        self.type[ntacheb : self.model_first_TP] = 5
        self.type[self.model_first_TP : npagb] = 6
        self.type[npagb:nwd] = 9
        self.type[nwd:] = 10

        """ quantities needed for winds and tidal interaction """
        self.vwind = wind_velocity(self.mass, self.radius, self.log_Teff)
        self.r_core = self.core_radius_polytropic()
        self.m_cenv, self.r_cenv = conv_envelope_size(
            self.mass,
            self.m_core,
            self.radius,
            self.r_core,
            self.lum,
            self.log_Teff,
            self.age,
            self.model_TAMS,
            self.model_BGB,
            self.model_Rmax_HSB,
        )
        self.tconv = conv_turnover_time(self.m_cenv, self.r_cenv, self.radius, self.lum)
        # gyration radius; k2g and k2r below are not needed but useful for testing
        self.rg2, self.k2g, self.k2r = gyration_radius(
            self.mass,
            self.m_core,
            self.radius,
            self.r_core,
            self.lum,
            self.log_Teff,
            self.age,
            self.Z_init,
        )
        # viscous tidal dissipation rate for convective envelopes
        self.k_over_T_conv = conv_dissipation_rate(
            self.m_cenv, self.r_cenv, self.mass, self.radius, self.lum, self.Z_init
        )

        # approximate surface mean molecular weight, for cool stars (neutral H and He)
        XH_surf = track.eeps["surface_h1"]
        XHe_surf = track.eeps["surface_he4"]
        Z_surf = 1.0 - XH_surf - XHe_surf
        mmw_surf = 1.0 / (XH_surf + 0.25 * XHe_surf + 0.5 * Z_surf)

        self.P_puls = pulsation_period(self.mass, self.radius)
        self.HPgas = gas_pressure_scale_height(
            self.mass, self.radius, self.log_Teff, mmw_surf
        )

        # mass-loss rate
        dt = np.diff(self.age)
        dM = np.diff(self.mass)
        self.dM_dt = dM / dt

        # functions to interpolate M and log(R):
        self.M_func = interp1d(self.age, self.mass, fill_value="extrapolate")
        self.logR_func = interp1d(self.age, self.log_R, fill_value="extrapolate")
        self.rg2_func = interp1d(self.age, self.rg2, fill_value="extrapolate")


# %%


class StellarModel:

    global nzams, ntams, nrgbt, nzacheb, ntacheb, ntpagb, npagb, nwd
    global XHc_exh, XHec_exh

    nzams, ntams, nrgbt, nzacheb, ntacheb, ntpagb, npagb, nwd = (
        201,
        453,
        604,
        630,
        706,
        807,
        1408,
        1709,
    )
    XHc_exh, XHec_exh = 1.0e-4, 1.0e-4

    def __init__(self, track):
        self.m_init = 0
        self.Z_init = 0.014
        self.Y_init = 0.24 + 2 * self.Z_init

        self.age = []
        self.n_models = 0
        self.model = []
        self.phase = []

        self.mass = []
        self.m_core = []
        self.m_core_CO = []
        self.log_R = []
        self.log_L = []
        self.log_Teff = []
        self.XH_center = []
        self.XHe_center = []
        self.log_LH = []
        self.log_LHe = []
        self.log_rhoc = []
        self.log_Tc = []
        self.Mdot = []

        # find critical points along track:
        self.model_ZAMS = 0
        self.model_TAMS = 0
        self.model_Rmax_HSB = 0
        self.model_Rmax = 0
        self.model_BGB = 0
        self.model_first_TP = 0
        self.model_Rmax_EAGB = 0

        # assign stellar type, analogous to SSE (Hurley et al 2000)
        self.type = []

    def add_phase(self, phase):
        self.age.append(phase.star_age)

    def compute(self):
        self.m_init = self.mass[0]

        self.m_env = self.mass - self.m_core
        self.radius = 10**self.log_R
        self.lum = 10**self.log_L

        self.type[self.model_ZAMS : self.model_TAMS] = 1
        self.type[self.model_TAMS : self.model_BGB] = 2
        self.type[self.model_BGB : nzacheb + 1] = 3
        self.type[nzacheb + 1 : ntacheb] = 4
        self.type[ntacheb : self.model_first_TP] = 5
        self.type[self.model_first_TP : npagb] = 6
        self.type[npagb:nwd] = 9
        self.type[nwd:] = 10

        """ quantities needed for winds and tidal interaction """
        self.vwind = wind_velocity(self.mass, self.radius, self.log_Teff)
        self.r_core = self.core_radius_polytropic()
        self.m_cenv, self.r_cenv = conv_envelope_size(
            self.mass,
            self.m_core,
            self.radius,
            self.r_core,
            self.lum,
            self.log_Teff,
            self.age,
            self.model_TAMS,
            self.model_BGB,
            self.model_Rmax_HSB,
        )
        self.tconv = conv_turnover_time(self.m_cenv, self.r_cenv, self.radius, self.lum)
        # gyration radius; k2g and k2r below are not needed but useful for testing
        self.rg2, self.k2g, self.k2r = gyration_radius(
            self.mass,
            self.m_core,
            self.radius,
            self.r_core,
            self.lum,
            self.log_Teff,
            self.age,
            self.Z_init,
        )
        # viscous tidal dissipation rate for convective envelopes
        self.k_over_T_conv = conv_dissipation_rate(
            self.m_cenv, self.r_cenv, self.mass, self.radius, self.lum, self.Z_init
        )

        # mass-loss rate
        dt = np.diff(self.age)
        dM = np.diff(self.mass)
        self.dM_dt = dM / dt

        # functions to interpolate M and log(R):
        self.M_func = interp1d(self.age, self.mass, fill_value="extrapolate")
        self.logR_func = interp1d(self.age, self.log_R, fill_value="extrapolate")
        self.rg2_func = interp1d(self.age, self.rg2, fill_value="extrapolate")
