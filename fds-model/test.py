import pandas as pd
from joblib import load
from sklearn.metrics import silhouette_score
import logging
import utils

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_predict(model_name, X):
    model = utils.load(model_name)
    return model.predict(X)

def evaluate_models(X, y):
    models = ['kmeans', 'dbscan', 'isolation_forest', 'lof']
    results = []
    for model_name in models:
        y_pred = load_and_predict(model_name, X)
        score = silhouette_score(X, y_pred)
        logging.info(f'Silhouette score for {model_name}: {score}')
        results.append({'model': model_name, 'silhouette_score': score})
    pd.DataFrame(results).to_csv('results_test.csv', index=False)

if __name__ == '__main__':
    dataset = pd.read_csv('data.csv')
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    scaler = load('scaler.pkl')
    X_scaled = scaler.transform(X)
    evaluate_models(X_scaled, y)