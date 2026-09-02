import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# %%

filepath = "/home/koen/master-internship/data/tp-info/"

directories = [
    filepath + "z0028models/",
    filepath + "z007models/",
    filepath + "z014models/",
]

# %%

pattern = re.compile(
    r"m(?P<initial_mass>\d+(?:\.\d+)?)"
    r"z(?P<z>\d+)"
    r"(?:y(?P<helium>\d+))?"
    r"(?:"
    r"(?:[._]nov=(?P<ov>\d+(?:\.\d+)?))"
    r"|"
    r"(?:_noovershoot(?P<noovershoot>))"
    r")?"
)

dfs = []

for z in directories:

    files = [files for _, _, files in os.walk(z)][0]

    for file in files:
        match = re.search(pattern, file)

        if match is None:
            print(f"could not parse filename: {file}")
            continue

        params = match.groupdict()

        if params["noovershoot"] is not None:
            params["ov"] = 0.0

        params.pop("noovershoot")

        # convert numerical parameters
        params["initial_mass"] = float(params["initial_mass"])
        params["z"] = float("0." + params["z"])

        if params["helium"] is not None:
            params["helium"] = float(params["helium"]) / 100
            if params["helium"] != 0.28:
                continue

        if params["ov"] is not None:
            params["ov"] = float(params["ov"])
        else:
            params["ov"] = 2

        if params["z"] == 0.014:
            match params["initial_mass"]:
                case 1.5:
                    if params["ov"] != 3:
                        continue
                case 1.75:
                    if params["ov"] != 2:
                        continue
                case _:
                    pass

        if params["z"] == 0.007:
            match params["initial_mass"]:
                case 1.5:
                    if params["ov"] != 1:
                        continue
                case 1.75:
                    if params["ov"] != 1:
                        continue
                case _:
                    pass

        if params["z"] == 0.0028:
            match params["initial_mass"]:
                case 1.15:
                    if params["ov"] != 1:
                        continue
                case 1.25:
                    if params["ov"] != 1:
                        continue
                case _:
                    pass

        # read the pulse table
        df = pd.read_csv(
            z + file,
            sep=r"\s+",
            comment="#",
            names=[
                "pulse",
                "Mcore",
                "M_csh",
                "t_csh",
                "Ddredge",
                "lambda",
                "DMh",
                "THeshell",
                "Tbce",
                "THshell",
                "interpulse",
                "Mtot",
                "MaxL",
                "MaxLHe",
                "maxR",
                "Mbol",
                "Teff",
            ],
            skiprows=1,
        )

        # add model parameters to every row
        for name, value in params.items():
            df[name] = value

        dfs.append(df)

# combine all files
df = pd.concat(dfs, ignore_index=True)

df = df.apply(pd.to_numeric)
# %%
M2z0014 = df[df["initial_mass"] == 1.75]
M2z0014 = M2z0014[M2z0014["z"] == 0.0028]
M2z0014["Ddredge"]
# %%
plt.plot(M2z0014["pulse"], M2z0014["Ddredge"])
plt.show()


# %%

with open(f"data/tp_info_pd_df.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
