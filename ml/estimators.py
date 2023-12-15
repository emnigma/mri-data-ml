from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

N_JOBS = 8
RS = 42

ESTIMATORS = {
    LogisticRegression(): {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
    },
    DecisionTreeClassifier(random_state=RS): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    SVC(): {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": [0.1, 1, 10, 100],
        "C": [0.1, 1, 10, 100, 1000],
    },
    AdaBoostClassifier(random_state=RS): {
        "estimator": [
            DecisionTreeClassifier(random_state=RS),
            RandomForestClassifier(random_state=RS),
        ],
        "n_estimators": [10, 25],
    },
    RandomForestClassifier(random_state=RS): {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15],
    },
    BaggingClassifier(random_state=RS): {
        "estimator": [
            DecisionTreeClassifier(random_state=RS),
            RandomForestClassifier(random_state=RS),
        ],
        "n_estimators": [10, 25],
    },
    GradientBoostingClassifier(random_state=RS): {
        "n_estimators": [10, 25],
        "max_depth": [1, 2, 5],
    },
    KNeighborsClassifier(): {"n_neighbors": [1, 3, 5, 10, 15, 20, 25]},
    # RadiusNeighborsClassifier(): {"radius": [10, 20]},
}
