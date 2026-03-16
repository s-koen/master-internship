import re
from pathlib import Path
import mesa_reader as mr


class MesaGrid:
    """
    load a grid of mesa runs stored as

        runs/Rxxx.xxqx.xxx/
            binary_history.data
            LOGS/TPAGB/history.data

    models are stored as
        self.models[(m1, q)]
    """

    def __init__(self, runs_dir):
        self.runs_dir = Path(runs_dir)
        self.models = {}
        self.m1_vals = set()
        self.q_vals = set()
        self._load_models()

    def _parse_name(self, name):
        """
        parse Rxxx.xx_qx.xxx -> (m1, q)
        """
        m = re.match(r"R([0-9.]+)_q([0-9.]+)", name)
        if not m:
            return None

        m1 = float(m.group(1))
        q = float(m.group(2))
        return m1, q

    def _load_models(self):
        length = len(list(self.runs_dir.iterdir()))
        num_dig = len(str(length))
        for i, run_dir in enumerate(self.runs_dir.iterdir()):

            if not run_dir.is_dir():
                continue

            parsed = self._parse_name(run_dir.name)
            if parsed is None:
                continue

            m1, q = parsed
            print(f"retrieving M = {m1:.2f}, q = {q:.3f}... ({i:0{num_dig}d}/{length})")

            binary_path = run_dir / "binary_history.data"
            tpagb_path = run_dir / "LOGS" / "TPAGB" / "history.data"

            model = {}

            if binary_path.exists():
                model["binary_history"] = mr.MesaData(str(binary_path))

            if tpagb_path.exists():
                model["tpagb_history"] = mr.MesaData(str(tpagb_path))

            if model:
                self.models[(m1, q)] = model
                self.m1_vals.add(m1)
                self.q_vals.add(q)

        self.m1_vals = sorted(self.m1_vals)
        self.q_vals = sorted(self.q_vals)

    def __getitem__(self, key):
        return self.models[key]

    def get(self, m1, q):
        return self.models.get((m1, q))

    def iter_models(self):
        for (m1, q), model in self.models.items():
            yield m1, q, model
