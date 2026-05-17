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
TPAGB1 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/rees2024-4M/LOGS/TPAGB/history.data"
)
# %%
TPAGB2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/standard-single/LOGS/TPAGB/history.data"
)

TPAGB2_logs = mr.MesaLogDir(
    "/home/koen/master-internship/mesa-models/standard-single/LOGS/TPAGB/"
)
# %%
fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

print(TPAGB2_logs.model_numbers)
plt.xlabel("")
plt.ylabel("")
axs[0].plot(TPAGB1.star_age, TPAGB1.lambda_DUP)
axs[0].plot(TPAGB2.star_age, TPAGB2.lambda_DUP)
axs[1].plot(TPAGB1.star_age, TPAGB1.R)
axs[1].plot(TPAGB2.star_age, TPAGB2.R)
axs[1].scatter(
    TPAGB2.star_age[TPAGB2_logs.model_numbers], TPAGB2.R[TPAGB2_logs.model_numbers]
)
plt.show()
plt.close()
# %%
