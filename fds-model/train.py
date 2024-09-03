import numpy as np
import pandas as pd
import os
import json
import logging
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import utils  # Import the utils module
from utils import preprocess_data, save_model, load_config  # Assuming these functions exist in utils.py
from dataset import get_data_from_bigquery  # Importing the function from dataset.py
import model  # Assuming you have a model.py for your model's definition and training logic

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(X_train, X_test, y_train, y_test, model_config):
    results = []
    for name, params in model_config.items():
        logging.info(f'Training model: {name} with parameters: {params}')
        
        m = model.create_model(name, **params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')  # Use average='binary' for binary classification
        try:
            roc_auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])  # Assumes model has predict_proba method
        except AttributeError:
            roc_auc = None
            logging.warning(f'Model {name} does not support ROC AUC calculation')
        
        # Log metrics
        logging.info(f'Accuracy for {name}: {accuracy}')
        logging.info(f'F1 Score for {name}: {f1}')
        if roc_auc is not None:
            logging.info(f'ROC AUC Score for {name}: {roc_auc}')
        
        # Save the results
        results.append({
            'model': name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc_score': roc_auc
        })
        save_model(m, name)  # Use utils.save_model to save the model
    
    # Save the results to a CSV file
    pd.DataFrame(results).to_csv('results.csv', index=False)

if __name__ == '__main__':
    # Load configuration (including BigQuery query and project ID)
    config = load_config("config.json")
    
    # Fetch dataset from BigQuery
    df = get_data_from_bigquery(config["bigquery"]["query"], config["bigquery"]["project_id"], config["bigquery"]["credentials_path"])
    # Drop columns
    columns_drop = ['id', 'user_id', 'created_at', 'updated_at', 'blocked_at', 'expiry', 'payment_at', 'expiry_days']
    df = df.drop(columns=columns_drop, axis=1)
    target_column = 'fraud_status'
    df = df.fillna(0)

    # Preprocess the data (assuming this function expects a DataFrame and returns train/test splits)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column=target_column, normalize=True)

    # Train and evaluate models
    train_and_evaluate(X_train, X_test, y_train, y_test, config['models'])