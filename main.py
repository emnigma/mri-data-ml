import logging
import typing as t

import pandas as pd
import tqdm
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from ml.estimators import ESTIMATORS, RS
from ml.utils import load_cleaned_dataset

logging.basicConfig(
    level=logging.DEBUG,
    filename="learn.log",
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

N_JOBS = 8


def make_ovo(
    est, params: dict[str, list[t.Any]]
) -> tuple[BaseEstimator, dict[str, list[t.Any]]]:
    params = {f"estimator__" + key: value for key, value in params.items()}

    return OneVsOneClassifier(estimator=est), params


ESTIMATORS = dict([make_ovo(k, v) for k, v in ESTIMATORS.items()])

X, y = load_cleaned_dataset()

X = StandardScaler().fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


run_results = []


for estimator in tqdm.tqdm(list(ESTIMATORS.keys())):
    X_train, y_train

    param_grid = ESTIMATORS[estimator]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    grid = GridSearchCV(estimator, param_grid, scoring="accuracy", cv=cv, n_jobs=N_JOBS)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    run_result = (
        estimator,
        grid.best_params_,
        grid.best_score_,
        accuracy_score(y_test, y_pred),
    )

    run_results.append(run_result)
    logging.debug(run_result)

    means = grid.cv_results_["mean_test_score"]
    params = grid.cv_results_["params"]

    for mean, param in zip(means, params):
        logging.debug(f"{estimator}:{mean:.3f} with: {param}")

pd.DataFrame(run_results).to_csv("run_results_ovo_cont_reduced_features.csv")
