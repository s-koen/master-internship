import numpy as np
import numpy as np
from numpy.typing import NDArray


class AccretionResult:
    def __init__(self, index, mixing_mass, final_mu) -> None:
        self.index = index
        self.mixing_mass = mixing_mass
        self.final_mu = final_mu
        self.mass_profile: None | NDArray[np.float64] = None
        self.mu_profile_mix: None | NDArray[np.float64] = None
        self.mu_profile_original: None | NDArray[np.float64] = None

    def add_integration_result(
        self,
        mass: NDArray[np.float64],
        mu_profile: NDArray[np.float64],
        mu_profile_original: NDArray[np.float64],
    ) -> None:
        self.mass_profile = mass
        self.mu_profile_mix = mu_profile
        self.mu_profile_original = mu_profile_original


class AccretorProfile:
    def __init__(self, profile: mr.MesaData) -> None:
        self.center_h1 = float(profile.center_h1)
        self.age = float(profile.star_age)
        self.profile = profile
        self.mass = np.array(profile.mass, dtype=np.float64)
        self.mu = np.array(profile.mu, dtype=np.float64)
        arg = np.argmin(self.mu)
        self.mu[:arg] = self.mu[arg]


class AccretorProfiles:
    def __init__(self) -> None:
        self.profiles = self.__get_profiles()

    def __get_profiles(self, fresh=False) -> dict[float, list[AccretorProfile]]:
        if fresh:
            print("In Accretor.__get_profiles:\n\tloading profiles")
            profiles_dict: dict[float, list[AccretorProfile]] = {}
            for i, _ in enumerate(range(37)):
                profiles: list[AccretorProfile] = []
                mass = np.round(0.8 + 0.1 * i, 1)
                print(f"\tloading mass {mass}")
                for j in range(1, 41):
                    profile = mr.MesaData(
                        f"/home/koen/master-internship/mesa-models/single-ms-stars/M{mass}/LOGS/MS/profile{j}.data"
                    )
                    profiles.append(AccretorProfile(profile))
                profiles_dict[mass] = profiles

            with open(
                f"/home/koen/master-internship/data/accretor-cache/profiles.pkl",
                "wb",
            ) as f:
                pickle.dump(profiles_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            return profiles_dict

        else:
            try:
                with open(
                    f"/home/koen/master-internship/data/accretor-cache/profiles.pkl",
                    "rb",
                ) as f:
                    return pickle.load(f)

            except FileNotFoundError:
                return self.__get_profiles(fresh=True)


class Accretor:
    def __init__(self, profiles: AccretorProfiles, age: float, mass: float) -> None:
        self.mass = mass
        self.age = age

        self.profiles = profiles.profiles
        self.masses = self.__determine_profile_masses()
        self.central_h1 = self.__get_center_h1()
        self.mu_curve, self.mass_curve = self.__get_mu_mass_curve()

    def __get_center_h1(self) -> float:
        interp_ages: list[float] = []
        for mass in self.masses:
            mass_profiles: list[AccretorProfile] = self.profiles[mass]
            h1s: list[float] = []
            ages: list[float] = []
            for profile in mass_profiles:
                h1s.append(profile.center_h1)
                ages.append(profile.age)

            interp_ages.append(np.interp(self.age, ages, h1s))

        return np.interp(self.mass, self.masses, interp_ages)

    def __determine_profile_masses(self) -> list[float]:
        masses: list[float] = []

        profile_masses: list[float] = list(self.profiles.keys())

        x: float = self.mass
        i: np.intp = np.searchsorted(profile_masses, x)

        lower: None | float
        upper: None | float

        if i == 0:
            lower = None
            upper = profile_masses[0]
        elif i == len(profile_masses):
            lower = profile_masses[-1]
            upper = None
        else:
            lower = profile_masses[i - 1]
            upper = profile_masses[i]

        if lower != None:
            masses.append(lower)
        if upper != None:
            masses.append(upper)

        return masses

    def __get_mu_mass_curve(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if len(self.masses) == 1:
            mu, mass = self.__get_mu_mass_curve_per_mass(self.masses[0])
            mu = mu[::-1]
            # arg = np.argmin(mu)
            # mu[:arg] = mu[arg]
            return mu, mass[::-1]

        mu_lower, mass_lower = self.__get_mu_mass_curve_per_mass(self.masses[0])
        mu_upper, mass_upper = self.__get_mu_mass_curve_per_mass(self.masses[1])

        mass = np.unique(np.concatenate([mass_lower, mass_upper]))
        mu_lower = np.interp(mass, mass_lower, mu_lower)
        mu_upper = np.interp(mass, mass_upper, mu_upper)

        # Interpolation fraction in central H
        f = (self.mass - self.masses[0]) / (self.masses[1] - self.masses[0])

        # Interpolate evolutionary state
        mu = (1 - f) * mu_lower + f * mu_upper
        mu = mu[::-1]
        # arg = np.argmin(mu)
        # mu[:arg] = mu[arg]
        return mu, mass[::-1]

    def __get_mu_mass_curve_per_mass(
        self, mass
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        surrounding_profiles: list[AccretorProfile] = []

        profiles = self.profiles[mass][::-1]
        all_center_h1s: list[float] = []
        for profile in profiles:
            all_center_h1s.append(profile.center_h1)

        x: float = self.central_h1
        i: np.intp = np.searchsorted(all_center_h1s, x)

        lower: None | AccretorProfile
        upper: None | AccretorProfile

        if i == 0:
            lower = None
            upper = profiles[0]
        elif i == len(all_center_h1s):
            lower = profiles[-1]
            upper = None
        else:
            lower = profiles[i - 1]
            upper = profiles[i]

        if lower != None:
            surrounding_profiles.append(lower)
        if upper != None:
            surrounding_profiles.append(upper)

        if len(surrounding_profiles) == 1:
            profile = surrounding_profiles[0]
            return np.array(profile.mu[::-1]), np.array(profile.mass[::-1])

        mass_lower = np.array(surrounding_profiles[0].mass)[::-1]
        mu_lower = np.array(surrounding_profiles[0].mu)[::-1]
        mass_upper = np.array(surrounding_profiles[1].mass)[::-1]
        mu_upper = np.array(surrounding_profiles[1].mu)[::-1]
        mass = np.unique(np.concatenate([mass_lower, mass_upper]))

        mu_lower = np.interp(mass, mass_lower, mu_lower)
        mu_upper = np.interp(mass, mass_upper, mu_upper)

        # Interpolation fraction in central H
        f = (self.central_h1 - surrounding_profiles[0].center_h1) / (
            surrounding_profiles[1].center_h1 - surrounding_profiles[0].center_h1
        )

        # Interpolate evolutionary state
        mu = (1 - f) * mu_lower + f * mu_upper

        return mu, mass

    def effective_mu_vs_depth(
        self, M_acc: float, mu_acc: float, save_profile: bool = False
    ):
        ms = self.mass_curve

        ms = -np.diff(self.mass_curve)
        mus = (self.mu_curve[1:] + self.mu_curve[:-1]) / 2
        mu_current = mu_acc
        m_current = M_acc

        mu_profile = [mu_acc]

        crossing_index: int = 0
        for i, (m, mu) in enumerate(zip(ms, mus)):
            mu_last = mu_current
            mu_current = (m + m_current) / ((m / mu) + (m_current / mu_current))
            m_current = m + m_current
            mu_profile.append(mu_current)
            if mu < mu_current:
                crossing_index = np.min([i + 3, len(mus)])
        mu_profile = np.array(mu_profile, dtype=np.float64)

        mu_start = np.array([mus[0], mus[0]], dtype=np.float64)
        mu_profile_original = np.concatenate([mu_start, mus])
        mu_profile = np.concatenate([[mu_acc], mu_profile])
        mass = np.concatenate([[self.mass_curve[0] + M_acc], self.mass_curve])
        result = AccretionResult(
            crossing_index,
            self.mass_curve[0] - mass[crossing_index],
            mu_profile[crossing_index],
        )

        if save_profile:
            result.add_integration_result(mass, mu_profile, mu_profile_original)

        return result
