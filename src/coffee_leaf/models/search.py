from joblib import parallel_config
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from coffee_leaf.config import load_config


CONFIG_PATH = "configs/default.yaml"


def get_search_configuration(configuration):
    if configuration is None:
        return load_config(CONFIG_PATH)
    return configuration


def build_parameter_grid(model_name, configuration=None):
    if model_name == "svm":
        estimator_name = "svc"
    elif model_name == "decision_tree":
        estimator_name = "decisiontreeclassifier"
    else:
        raise ValueError(f"Unknown model name: {model_name!r}.")
    configuration = get_search_configuration(configuration)
    parameter_grid = {}
    for parameter_name, parameter_values in configuration[model_name].items():
        grid_key = f"{estimator_name}__{parameter_name}"
        parameter_grid[grid_key] = list(parameter_values)
    return parameter_grid


def build_group_cross_validation(configuration):
    configuration = get_search_configuration(configuration)
    return StratifiedGroupKFold(
        n_splits=configuration["folds"], shuffle=True, random_state=configuration["seed"]
    )


def run_grid_search(pipeline, parameter_grid, features, labels, groups, configuration=None):
    cross_validation = build_group_cross_validation(configuration)
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    search = GridSearchCV(
        pipeline, parameter_grid, cv=cross_validation, scoring=scoring,
        refit="f1_macro", return_train_score=True, n_jobs=-1, error_score="raise", verbose=1,
    )
    with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        search.fit(features, labels, groups=groups)
    return search


def build_cv_predictions(pipeline, features, labels, groups, configuration):
    cross_validation = build_group_cross_validation(configuration)
    with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        return cross_val_predict(pipeline, features, labels, groups=groups, cv=cross_validation, n_jobs=-1)
