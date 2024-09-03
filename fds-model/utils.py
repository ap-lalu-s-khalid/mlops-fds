import json
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    
    Parameters:
    - config_path: str, path to the configuration file.
    
    Returns:
    - dict, configuration data.
    """
    with open(config_path) as f:
        return json.load(f)

def load_data_from_bigquery(query, project_id, credentials_path=None):
    """
    Load data from BigQuery into a DataFrame.
    
    Parameters:
    - query: str, SQL query to fetch data.
    - project_id: str, Google Cloud project ID.
    - credentials_path: str, path to the service account key file (optional).
    
    Returns:
    - DataFrame containing the query results.
    """
    # Define credentials if provided
    credentials = None
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id, credentials=credentials)
    
    # Run the query and fetch the results
    df = client.query(query).to_dataframe()
    return df

def preprocess_data(data, target_column='target', normalize=False):
    """
    Preprocess data by scaling features and splitting into train and test sets.
    
    Parameters:
    - data: DataFrame, the dataset.
    - target_column: str, the name of the target column.
    
    Returns:
    - X_train, X_test, y_train, y_test: arrays, split and scaled feature and target data.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    if(normalize):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def save_model(model, name):
    """
    Save a model to disk using joblib.
    
    Parameters:
    - model: the model to be saved.
    - name: str, the filename under which to save the model.
    """
    # Create the directory if it doesn't exist
    directory = 'model_trained'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model inside the directory
    file_path = os.path.join(directory, f'{name}.pkl')
    dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(name):
    """
    Load a model from disk using joblib.
    
    Parameters:
    - name: str, the filename from which to load the model.
    
    Returns:
    - The loaded model.
    """
    return load(f'{name}.pkl')