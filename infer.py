import argparse
import pathlib

import joblib
import pandas as pd

TEST_CSV_PATH = "data/mr-art_q3_incomplete/fsqc-results.csv"
MODEL_PATH = "catboost_model0.80_pipeline.pkl"


def decode(prediction: list[int]):
    assert len(prediction) == 1
    quality = prediction[0]
    match quality:
        case 1:
            return 1
        case 0:
            return 2
        case -1:
            return 3
    raise RuntimeError(f"Unknown classification result: {quality}")


def main():
    parser = argparse.ArgumentParser(
        description="Predicts quality of FreeSurfer/FastSurfer reconstruction based on fsqc output.\
            1 - perfect quality,\
            2 - medium quality,\
            3 - bad quality",
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

    result_str = " ".join((str(decode(prediction)) for prediction in model.predict(df)))
    print(result_str)


if __name__ == "__main__":
    main()
