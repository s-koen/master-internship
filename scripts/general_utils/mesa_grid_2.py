import json
from pathlib import Path
import mesa_reader as mr
import pickle
import numpy as np
import math
import re
import sys

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cache import get_star


def progressbar(current_value, total_value, bar_lengh, progress_char):
    percentage = int(
        (current_value / total_value) * 100
    )  # Percent Completed Calculation
    progress = int(
        (bar_lengh * current_value) / total_value
    )  # Progress Done Calculation
    loadbar = "Progress: [{:{len}}]{}%".format(
        progress * progress_char, percentage, len=bar_lengh
    )  # Progress Bar String
    print(loadbar, end="\r")


class MesaGrid:

    def __init__(self, grid_dir, fresh=False, loc="LOGS/"):

        self.grid_dir = Path(grid_dir)
        self.fresh = fresh
        print(self.fresh)
        self.loc = loc

        with open(f"{grid_dir }/grid_settings.json") as f:
            self.settings = json.load(f)

        self.axes = {
            "R": self.settings["Rs"],
            "q": self.settings["qs"],
            "beta": self.settings["mass_transfer"]["beta"],
            "delta": self.settings["mass_transfer"]["delta"],
        }

        self._load_models()

    def _finalize_grid(self):

        self.axes = {}

        for run in self.models:
            for key, value in run.params.items():

                if key not in self.axes:
                    self.axes[key] = set()

                self.axes[key].add(value)

        for key in self.axes:
            self.axes[key] = sorted(self.axes[key])

    def merge(self, other, overwrite=False):

        existing = {run.key: run for run in self.models}

        for run in other.models:

            if run.key in existing:
                if overwrite:
                    existing[run.key] = run
            else:
                existing[run.key] = run

        self.models = list(existing.values())
        self._finalize_grid()

    def _load_models(self):
        self.models = []
        self.failed_models = []

        l = len(list(self.grid_dir.iterdir()))
        for i, run_dir in enumerate(self.grid_dir.iterdir()):
            progressbar(i, l, 30, "■")

            if not run_dir.is_dir():
                continue

            try:
                self.models.append(MesaRun(run_dir, self.fresh, self.loc))
            except FileNotFoundError:
                self.failed_models.append(run_dir)

    def filter(self, rel_tol=1e-3, abs_tol=1e-12, **filters):

        for run in self.models:

            keep = True

            for key, values in filters.items():

                if not isinstance(values, (list, tuple, set)):
                    values = (values,)

                value = run.params[key]

                if isinstance(value, (int, float)):
                    if not any(
                        math.isclose(value, v, rel_tol=rel_tol, abs_tol=abs_tol)
                        for v in values
                    ):
                        keep = False
                        break
                else:
                    if value not in values:
                        keep = False
                        break

            if keep:
                yield run

    def array(self, value, x, y, **filters):

        xs = self.axes[x]
        ys = self.axes[y]

        arr = np.full((len(xs), len(ys)), np.nan)

        x_index = {np.round(v, 3): i for i, v in enumerate(xs)}
        y_index = {np.round(v, 3): i for i, v in enumerate(ys)}

        for run in self.filter(**filters):

            if callable(value):
                z = value(run)
            else:
                z = getattr(run, value)

            if x in ["beta", "delta"]:
                i = x_index[np.round(run.params[x] / (1 - run.params["eps"]), 3)]
            else:
                i = x_index[np.round(run.params[x], 3)]

            if y in ["beta", "delta"]:
                j = y_index[np.round(run.params[y] / (1 - run.params["eps"]), 3)]
            else:
                j = y_index[np.round(run.params[y], 3)]

            arr[i, j] = z

        return np.asarray(xs), np.asarray(ys), arr


class MesaRun:

    def __init__(self, run_dir, fresh, loc):
        self.run_dir = run_dir
        self.loc = loc

        with open(run_dir / "settings.json") as f:
            self.params = json.load(f)

        self.params["f_beta"] = self.params["beta"] / (1 - self.params["eps"])
        self.params["f_delta"] = self.params["delta"] / (1 - self.params["eps"])

        if "m" not in self.params:
            self.params["m"] = self.get_mass_from_model_path(self.params["model_name"])

        if "TP" not in self.params:
            self.params["TP"] = self.get_TP_from_model_path(self.params["model_name"])

        self.get_starting_model(fresh)
        self.get_history(fresh)
        self.get_profiles(fresh)

        self.env_mass = self.history.envelope_mass
        self.q = self.history.star_2_mass / self.history.star_1_mass
        star = get_star(m=self.params["m"])
        tpagb_age = star.age[star.ntpagb]
        self.age = self.starting_model.star_age + tpagb_age + self.star_age

    def __getattr__(self, name):
        return getattr(self.history, name)

    def get_mass_from_model_path(self, model_path):
        match = re.search(r"/M([0-9.]+)/models/", str(model_path))

        if match is None:
            raise ValueError(f"could not extract mass from {model_path}")

        return float(match.group(1))

    def get_TP_from_model_path(self, model_path):
        match = re.search(r"TP([0-9.]+).mod", str(model_path))

        if match is None:
            raise ValueError(f"could not extract TP from {model_path}")

        return float(match.group(1))

    def get_starting_model(self, fresh):
        if fresh:
            self.starting_model = mr.MesaData(self.params["model_name"])
            with open(f"{self.run_dir}/starting_model.pkl", "wb") as f:
                pickle.dump(self.starting_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        try:
            with open(f"{self.run_dir}/starting_model.pkl", "rb") as f:
                self.starting_model = pickle.load(f)
        except FileNotFoundError:
            self.starting_model = mr.MesaData(self.params["model_name"])
            with open(f"{self.run_dir}/starting_model.pkl", "wb") as f:
                pickle.dump(self.starting_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_history(self, fresh):
        if fresh:
            self.history = mr.MesaData(f"{self.run_dir}/{self.loc}history.data")
            with open(f"{self.run_dir}/history.pkl", "wb") as f:
                pickle.dump(self.history, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        try:
            with open(f"{self.run_dir}/history.pkl", "rb") as f:
                self.history = pickle.load(f)
        except FileNotFoundError:
            self.history = mr.MesaData(f"{self.run_dir}/{self.loc}/history.data")
            with open(f"{self.run_dir}/history.pkl", "wb") as f:
                pickle.dump(self.history, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_profiles(self, fresh):
        if fresh:
            profiles_dict = mr.MesaLogDir(f"{self.run_dir}/{self.loc}/")
            for i, profile in enumerate(profiles_dict.profile_numbers):
                _ = profiles_dict.profile_data(profile_number=profile)
            self.profiles = list(profiles_dict.profile_dict.values())
            with open(f"{self.run_dir}/profiles.pkl", "wb") as f:
                pickle.dump(self.profiles, f, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            with open(f"{self.run_dir}/profiles.pkl", "rb") as f:
                self.profiles = pickle.load(f)

        except FileNotFoundError:
            profiles_dict = mr.MesaLogDir(f"{self.run_dir}/{self.loc}/")
            for i, profile in enumerate(profiles_dict.profile_numbers):
                _ = profiles_dict.profile_data(profile_number=profile)
            self.profiles = list(profiles_dict.profile_dict.values())
            with open(f"{self.run_dir}/profiles.pkl", "wb") as f:
                pickle.dump(self.profiles, f, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def key(self):
        return tuple(
            sorted(
                (key, value) for key, value in self.params.items() if key != "grid_name"
            )
        )
