from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def create_model(name, **params):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "support_vector_machine":
        return SVC(**params)
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {name}")