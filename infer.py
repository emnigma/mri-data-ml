import argparse
import pathlib

import joblib
import pandas as pd

TEST_CSV_PATH = "data/mr-art_q3_incomplete/fsqc-results.csv"
MODEL_PATH = "catboost_model0.80_pipeline.pkl"


def main():
    parser = argparse.ArgumentParser(
        # prog="FreeSurfer/FastSurfer quality predictor",
        description="Predicts quality of FreeSurfer/FastSurfer reconstruction based on fsqc output.\
            1 - perfect quality,\
            0 - medium quality,\
            -1 - bad quality",
    )

    parser.add_argument(
        "--input-csv",
        type=pathlib.Path,
        help="Path to .csv file with fsqc output",
        dest="input_csv",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        help="Path to classifier",
        dest="model_path",
        required=True,
    )

    args = parser.parse_args()

    model = joblib.load(args.model_path)

    df = pd.read_csv(args.input_csv)
    df = df.drop(
        [
            "subject",
            "rot_tal_x",
            "rot_tal_y",
            "rot_tal_z",
            "cc_size",
        ],
        axis=1,
    )

    result_str = " ".join((str(*prediction) for prediction in model.predict(df)))
    print(result_str)


if __name__ == "__main__":
    main()
