import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import compute_m_DUP

plt.cplot = cplot

import time

# %%


with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)
# %%

print(intershell)

start = time.time()
Zs = np.unique(intershell["Z"])
for i in range(10):
    for Z in Zs:
        i = intershell.query(f"Z == {Z} and pmz == 2e-3 and last == 1").sort_values(
            "ntp"
        )

        print(np.unique(i["M1tp"]))
end = time.time()
print((end - start) * 16)
# %%

mass_Z_dict = {}
for Z in np.unique(intershell["Z"]):
    i = intershell.query(f"Z == {Z} and pmz == 2e-3 and last == 1").sort_values("ntp")
    mass_Z_dict[Z] = np.unique(i["M1tp"])

    print(np.unique(i["M1tp"]))
# %%


with open(
    f"/home/koen/master-internship/data/intershell-cache/MZ.pkl",
    "wb",
) as f:
    pickle.dump(mass_Z_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%

start = time.time()
for i in range(10):
    with open(
        f"/home/koen/master-internship/data/intershell-cache/MZ.pkl",
        "rb",
    ) as f:
        mass_Z_dict = pickle.load(f)
        print(mass_Z_dict)

end = time.time()
print((end - start) * 16)

# %%
