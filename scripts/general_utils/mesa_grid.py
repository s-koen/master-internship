import re
from pathlib import Path
import mesa_reader as mr
import numpy as np


class MesaModel:
    """
    container for a single mesa run
    """

    def __init__(self, binary_path=None, tpagb_path=None, initial_age=None):

        self.binary = mr.MesaData(str(binary_path)) if binary_path else None
        self.star = mr.MesaData(str(tpagb_path)) if tpagb_path else None
        self.initial_age = initial_age
        self.age = self.star.star_age if tpagb_path else None
        self.env_mass = (
            self.star.star_mass - self.star.he_core_mass if tpagb_path else None
        )

    def __str__(self):
        if self.star:
            return f"MESA model with R = {self.star.R[0]:.2f}, R_RL = {self.star.rl_1[0]:.2f} and q = {(self.star.star_2_mass[0] / self.star.star_1_mass[1]):.3f}"

        return f"Empty MESA model"

    def set_initial_age(self, age):
        self.initial_age = age
        self.age = self.star.star_age + self.initial_age


class MesaGrid:
    """
    load a grid of mesa runs stored as

        runs/Rxxx.xxqx.xxx/
            binary_history.data
            LOGS/TPAGB/history.data

    models are stored as
        self.models[(R1, q)]
    """

    def __init__(self, grid_dir, R1=None, q=None):
        self.grid_dir = Path(grid_dir)
        self.runs_dir = self.grid_dir / "runs"

        self.R1_filter = self._normalize_filter(R1)
        self.q_filter = self._normalize_filter(q)

        self.models = {}
        self.R1_vals = set()
        self.q_vals = set()

        self._get_tpagb_age()
        self._load_references()

        self._load_models()
        self._finalize_grid()

    def _load_references(self):

        dir = self.grid_dir / "reference-histories"

        self.ref_ms = mr.MesaData(str(dir / "ms.data"))
        self.ref_gb = mr.MesaData(str(dir / "gb.data"))
        self.ref_cheb = mr.MesaData(str(dir / "cheb.data"))
        self.ref_eagb = mr.MesaData(str(dir / "eagb.data"))
        self.ref_tpagb = mr.MesaData(str(dir / "tpagb.data"))

    def _get_tpagb_age(self):

        dir = self.grid_dir / "reference-histories"

        self.tpagb_age = 0
        for history in ["ms.data", "gb.data", "cheb.data", "eagb.data"]:
            history = mr.MesaData(str(dir / history))
            self.tpagb_age += history.star_age[-1]

    def _normalize_filter(self, val):
        """allow single value or iterable"""
        if val is None:
            return None
        if isinstance(val, (list, tuple, set)):
            return set(val)
        return {val}

    def _passes_filter(self, R1, q):

        if self.R1_filter is not None and R1 not in self.R1_filter:
            return False

        if self.q_filter is not None and q not in self.q_filter:
            return False

        return True

    def _parse_name(self, name):
        """
        parse Rxxx.xx_qx.xxx -> (R1, q)
        """
        m = re.match(r"R([0-9.]+)_q([0-9.]+)", name)
        if not m:
            return None

        R1 = float(m.group(1))
        q = float(m.group(2))
        return R1, q

    def _load_models(self):
        length = len(list(self.runs_dir.iterdir()))
        num_dig = len(str(length))
        for i, run_dir in enumerate(self.runs_dir.iterdir()):

            if not run_dir.is_dir():
                continue

            parsed = self._parse_name(run_dir.name)
            if parsed is None:
                continue

            R1, q = parsed

            if not self._passes_filter(R1, q):
                continue

            print(
                f"retrieving M = {R1:.2f}, q = {q:.3f}... ({i:0{num_dig}d}/{length+1})"
            )

            binary_path = run_dir / "binary_history.data"
            tpagb_path = run_dir / "LOGS" / "TPAGB" / "history.data"

            model = MesaModel(
                binary_path if binary_path.exists() else None,
                tpagb_path if tpagb_path.exists() else None,
            )

            age = self._find_initial_age(model)
            model.set_initial_age(age)

            self.models[(R1, q)] = model
            self.R1_vals.add(R1)
            self.q_vals.add(q)

    def _find_initial_age(self, model):
        initial_R = model.star.R[0]
        model_age = 0

        for history in [
            self.ref_ms,
            self.ref_gb,
            self.ref_cheb,
            self.ref_eagb,
            self.ref_tpagb,
        ]:
            try:
                arg = np.argwhere(history.log_R > np.log10(initial_R))[0][0]
                model_age += history.star_age[arg]
                print(f"found model age = {model_age}")
                break
            except IndexError:
                model_age += history.star_age[-1]
                print(f"excepted, new model age = {model_age}")

        return model_age - self.tpagb_age

    def _finalize_grid(self):
        """sort grid axes and keys"""
        self.R1_vals = sorted(self.R1_vals)
        self.q_vals = sorted(self.q_vals)

        self.sorted_keys = sorted(self.models.keys(), key=lambda x: (x[0], x[1]))

    def __getitem__(self, key):
        return self.models[key]

    def get(self, R1, q):
        return self.models.get((R1, q))

    def iter_models(self):
        """iterate in sorted order: R1 ascending, then q ascending"""
        for R1, q in self.sorted_keys:
            yield R1, q, self.models[(R1, q)]

    def get_R1_slice(self, R1):
        """
        return all models with given primary mass
        sorted by q
        """
        return [
            (q, self.models[(R1, q)]) for q in self.q_vals if (R1, q) in self.models
        ]

    def get_q_slice(self, q):
        """
        return all models with given mass ratio
        sorted by R1
        """
        return [
            (R1, self.models[(R1, q)]) for R1 in self.R1_vals if (R1, q) in self.models
        ]


# %%
