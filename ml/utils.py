import numpy as np
import pandas as pd


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    q1_path = "data/mr-art_q1_incomplete/fsqc-results.csv"
    q2_path = "data/mr-art_q2_incomplete/fsqc-results.csv"
    q3_path = "data/mr-art_q3_incomplete/fsqc-results.csv"

    q1_df = pd.read_csv(q1_path)
    q2_df = pd.read_csv(q2_path)
    q3_df = pd.read_csv(q3_path)

    q1_df["QA"] = np.repeat(["EXC"], (q1_df.shape[0]))
    q2_df["QA"] = np.repeat(["MED"], (q2_df.shape[0]))
    q3_df["QA"] = np.repeat(["BAD"], (q3_df.shape[0]))

    df = pd.concat([q1_df, q2_df, q3_df])
    df["QA"] = df["QA"].astype("category").cat.codes

    X, y = df.drop(["subject", "QA"], axis=1), df["QA"]

    return X, y


def load_cleaned_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_dataset()

    X = X.drop(
        [
            "wm_snr_orig",
            "gm_snr_orig",
            "cc_size",
            "topo_lh",
            "topo_rh",
            "rot_tal_x",
            "rot_tal_y",
            "rot_tal_z",
        ],
        axis=1,
    )

    return X, y
