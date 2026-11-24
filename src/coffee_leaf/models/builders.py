from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from coffee_leaf.config import load_config


CONFIG_PATH = "configs/default.yaml"
CLASS_WEIGHT_OPTIONS = (None, "balanced")


def validate_class_weight(class_weight):
    if class_weight not in CLASS_WEIGHT_OPTIONS:
        raise ValueError("class_weight must be None or 'balanced'.")


def get_model_seed(configuration):
    if configuration is None:
        configuration = load_config(CONFIG_PATH)
    return configuration["seed"]


def build_svm_pipeline(class_weight, configuration=None):
    validate_class_weight(class_weight)
    seed = get_model_seed(configuration)
    classifier = SVC(kernel="rbf", random_state=seed, class_weight=class_weight)
    return make_pipeline(StandardScaler(), classifier)


def build_decision_tree_pipeline(class_weight, configuration=None):
    validate_class_weight(class_weight)
    seed = get_model_seed(configuration)
    classifier = DecisionTreeClassifier(random_state=seed, class_weight=class_weight)
    return make_pipeline(StandardScaler(), classifier)
