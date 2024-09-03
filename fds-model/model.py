from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def create_model(name_model, **kwargs):
    models = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'isolation_forest': IsolationForest,
        'lof': LocalOutlierFactor
    }
    if name_model in models:
        return models[name_model]
    else:
        raise ValueError('Model not found')