import numpy as np
import pandas as pd


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    q1_path = "data/mr-art_q1_incomplete/fsqc-results.csv"
    q2_path = "data/mr-art_q2_incomplete/fsqc-results.csv"
    q3_path = "data/mr-art_q3_incomplete/fsqc-results.csv"

    q1_df = pd.read_csv(q1_path)
    q2_df = pd.read_csv(q2_path)
    q3_df = pd.read_csv(q3_path)

    q1_df["QA"] = np.repeat([1], (q1_df.shape[0]))
    q2_df["QA"] = np.repeat([0], (q2_df.shape[0]))
    q3_df["QA"] = np.repeat([-1], (q3_df.shape[0]))

    df = pd.concat([q1_df, q2_df, q3_df])
    # df["QA"] = df["QA"].astype("category").cat.codes

    X, y = df.drop(["subject", "QA"], axis=1), df["QA"]

    return X, y


def load_cleaned_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_dataset()

    X = X[
        [
            "wm_snr_norm",
            "holes_lh",
            "holes_rh",
            "defects_lh",
            "defects_rh",
            "con_snr_lh",
            "con_snr_rh",
        ]
    ]

    X["defects"] = X["defects_lh"] + X["defects_rh"]
    X["con_snr"] = X["con_snr_lh"] * X["con_snr_rh"]
    X["holes"] = X["holes_lh"] + X["holes_rh"]

    X = X.drop(["defects_lh", "defects_rh", "holes_lh", "holes_rh"], axis=1)

    return X, y


def load_picked_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_dataset()

    X = X.drop(
        [
            "rot_tal_x",
            "rot_tal_y",
            "rot_tal_z",
            "cc_size",
        ],
        axis=1,
    )

    # X["Defects"] = X["defects_lh"] + X["defects_rh"]
    # X["CON_SNR"] = X["con_snr_lh"] + X["con_snr_rh"]

    # X = X.drop(["gm_snr_norm", "holes_lh", "holes_rh"], axis=1)

    # X = X[["Defects", "con_snr_lh", "con_snr_rh", "wm_snr_norm"]]

    return X, y
