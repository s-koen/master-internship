import pickle
import re
import pandas as pd


def _parse_model_header(line):
    """
    Parse a model header such as

        # Initial mass = 1.000, Z = 0.0028, Y = 0.250,
          M_mix = 0.00E+00, VW93

    Returns a dictionary of model properties.
    """
    header = line.lstrip("#").strip()

    data = {}

    for item in header.split(","):
        item = item.strip()

        if not item:
            continue

        if "=" in item:
            key, value = item.split("=", 1)
            if key.strip() == "Z":
                data["metallicity"] = value.strip()
            else:
                data[key.strip()] = value.strip()
        else:
            # flags such as VW93 and B95
            data[item] = True

    return data


def _parse_element_header(line):
    """
    Parse the elemental table header.

    Some column names contain spaces, e.g. `log e(X)` and `Net M(i)`.
    """
    header = line.lstrip("#").strip()

    # The known column names used by these Monash files.
    # Ordering matters because some names contain spaces.
    patterns = [
        r"log e\(X\)",
        r"\[X/H\]",
        r"\[X/Fe\]",
        r"Net M\(i\)",
        r"Mass\(i\)",
        r"X\(i\)",
        r"El",
        r"Z",
    ]

    pattern = "|".join(patterns)

    return re.findall(pattern, header)


def _is_summary_header(line):
    return line.lstrip("#").strip().startswith("[Rb/Zr]")


def read_monash_yields(filenames):
    """
    Read one or more Monash yield files into a single pandas DataFrame.

    Parameters
    ----------
    filenames : str or list[str]
        One filename or a list of filenames.

    Returns
    -------
    pandas.DataFrame
        One row per element per stellar model.

    Notes
    -----
    Model-level quantities such as `Initial mass`, `Z`, `Y`,
    `M_mix`, `Final mass`, etc. are repeated for every element.

    Arbitrary new model-level keys are automatically detected.

    Example
    -------
    df = read_monash_yields([
        "yields_z0028.dat",
        "yield_z007.dat",
    ])
    """

    if isinstance(filenames, (str, bytes)):
        filenames = [filenames]

    rows = []

    for filename in filenames:

        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        model = {}

        element_columns = None
        element_rows = []

        summary_columns = None
        summary_values = None

        def finish_model():
            """Add the current model to rows."""

            if not element_rows:
                return

            # Combine model metadata and summary quantities.
            model_data = {
                **model,
                **(summary_values or {}),
            }

            for row in element_rows:
                row = row.copy()
                row.update(model_data)

                # Explicitly use the model metallicity.
                # This overwrites the elemental atomic-number Z column,
                # which would otherwise conflict with the model Z.
                # if "Z" in model:
                #     row["metallicity"] = model["Z"]

                rows.append(row)

        for line in lines:

            # ---------------------------------------------------------
            # start of a new model
            # ---------------------------------------------------------
            if line.startswith("# Initial mass"):

                # Save previous model
                finish_model()

                model = _parse_model_header(line)

                element_columns = None
                element_rows = []

                summary_columns = None
                summary_values = None

                continue

            # ---------------------------------------------------------
            # model metadata
            # ---------------------------------------------------------
            if line.startswith("#"):

                header = line.lstrip("#").strip()

                # Element table header
                if header.startswith("El "):
                    element_columns = _parse_element_header(line)
                    continue

                # Summary header
                if _is_summary_header(line):
                    summary_columns = header.split()
                    continue

                # Summary values
                if summary_columns is not None:
                    values = header.split()

                    if len(values) == len(summary_columns):
                        summary_values = dict(zip(summary_columns, values))

                    summary_columns = None
                    continue

                # Other metadata headers
                if "=" in header:
                    for item in header.split(","):
                        item = item.strip()

                        if "=" in item:
                            key, value = item.split("=", 1)
                            model[key.strip()] = value.strip()
                        elif item:
                            model[item] = True

                continue

            # ---------------------------------------------------------
            # elemental data
            # ---------------------------------------------------------
            if element_columns is not None:

                values = line.split()

                if len(values) != len(element_columns):
                    if len(values) == 5:
                        continue
                    raise ValueError(
                        f"Could not parse line in {filename}:\n"
                        f"{line}\n\n"
                        f"Expected {len(element_columns)} values "
                        f"for columns {element_columns}, "
                        f"but found {len(values)}."
                    )

                element_rows.append(dict(zip(element_columns, values)))

        # Save final model
        finish_model()

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Convert numeric columns
    # -------------------------------------------------------------
    for column in df.columns:

        if column in {"El"}:
            continue

        # Flags such as VW93/B95 should remain boolean.
        if df[column].dtype == bool:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        # Only replace the column if conversion actually worked
        # for at least some values.
        if converted.notna().any():
            df[column] = converted

    return df


# %%
df = read_monash_yields(
    [
        "/home/koen/master-internship/data/yields/yields_z0028.dat",
        "/home/koen/master-internship/data/yields/yield_z007.dat",
        "/home/koen/master-internship/data/yields/yield_z014.dat",
    ]
)
# %%
filtered = df.query("B95 != 1.0 and N_ov != 0")
# %%
filtered
# %%

zs = np.unique(filtered["Z"])

for z in zs:
    filtered_z = filtered.query(f"Z == {z}")

    ms = np.unique(filtered_z["Initial mass"])
    for m in ms:
        filtered_mz = filtered_z.query(f"`Initial mass` == {m}")
        print(z, m, np.unique(filtered_mz["N_ov"]))

# %%

df = df.apply(pd.to_numeric)
# %%

with open(f"data/yields_pd_df.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
