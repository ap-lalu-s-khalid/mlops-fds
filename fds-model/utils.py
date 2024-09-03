import json
import pandas as pd
from google.cloud import bigquery, storage
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
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file {config_path}.")
        raise

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
    credentials = None
    if credentials_path:
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        except FileNotFoundError:
            print(f"Credentials file {credentials_path} not found.")
            raise
    
    client = bigquery.Client(project=project_id, credentials=credentials)
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f"An error occurred while querying BigQuery: {e}")
        raise
    
    return df

def preprocess_data(data, target_column='target', normalize=False):
    """
    Preprocess data by scaling features and splitting into train and test sets.
    
    Parameters:
    - data: DataFrame, the dataset.
    - target_column: str, the name of the target column.
    - normalize: bool, whether to normalize features.
    
    Returns:
    - X_train, X_test, y_train, y_test: arrays, split and scaled feature and target data.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def save_model(model, name, config):
    """
    Save a model to a Google Cloud Storage bucket.
    
    Parameters:
    - model: the model to be saved.
    - name: str, the filename under which to save the model.
    - config: dict, the configuration dictionary containing storage details.
    """
    try:
        storage_client = storage.Client.from_service_account_json(config['storage']['credentials_path'])
        bucket = storage_client.get_bucket(config['storage']['bucket_name'])
    except Exception as e:
        print(f"An error occurred while accessing GCS: {e}")
        raise
    
    local_file_path = f'{name}.pkl'
    
    try:
        dump(model, local_file_path)
        blob = bucket.blob(f'{name}.pkl')
        blob.upload_from_filename(local_file_path)
        os.remove(local_file_path)
        print(f"Model saved to GCS bucket {config['storage']['bucket_name']} as {name}.pkl")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        raise

def load_model_from_gcs(name, config):
    """
    Load a model from a Google Cloud Storage bucket.
    
    Parameters:
    - name: str, the filename of the model to load.
    - config: dict, the configuration dictionary containing storage details.
    
    Returns:
    - The loaded model.
    """
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client.from_service_account_json(config['storage']['credentials_path'])
        bucket = storage_client.get_bucket(config['storage']['bucket_name'])
        
        # Download the model file to a local file
        local_file_path = f'{name}.pkl'
        blob = bucket.blob(f'{name}.pkl')
        blob.download_to_filename(local_file_path)
        
        # Load the model from the local file
        model = load(local_file_path)

        print(f"Model loaded from GCS bucket {config['storage']['bucket_name']} as {name}.pkl")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model from GCS: {e}")
        raise