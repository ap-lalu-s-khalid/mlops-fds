import numpy as np
import pandas as pd
import os
import json
import logging
from sklearn.metrics import silhouette_score
import model
import utils  # Import the utils module
from utils import preprocess_data, save_model, load_config  # Assuming these functions exist in utils.py
from dataset import get_data_from_bigquery  # Importing the function from dataset.py
import model  # Assuming you have a model.py for your model's definition and training logic


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(X_train, X_test, y_train, y_test, model_config):
    results = []
    for name, params in model_config.items():
        m = model.create_model(name, **params)
        m.fit(X_train)
        y_pred = m.predict(X_test)
        score = silhouette_score(X_test, y_pred)
        logging.info(f'Silhouette score for {name}: {score}')
        results.append({'model': name, 'silhouette_score': score})
        utils.save_model(m, name)  # Use utils.save_model to save the model
    pd.DataFrame(results).to_csv('results.csv', index=False)

if __name__ == '__main__':
    # Load configuration (including BigQuery query and project ID)
    config = load_config("config.json")
    
    # Fetch dataset from BigQuery
    df = get_data_from_bigquery(config["bigquery"]["query"], config["bigquery"]["project_id"])
    
    # Preprocess the data (assuming this function expects a DataFrame and returns train/test splits)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    train_and_evaluate(X_train, X_test, y_train, y_test, config['models'])