import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load

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

def preprocess_data(data_path):
    """
    Preprocess data by reading from a CSV, scaling features, and splitting into train and test sets.
    
    Parameters:
    - data_path: str, path to the dataset CSV file.
    
    Returns:
    - X_train, X_test, y_train, y_test: arrays, split and scaled feature and target data.
    """
    dataset = pd.read_csv(data_path)
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def save_model(model, name):
    """
    Save a model to disk using joblib.
    
    Parameters:
    - model: the model to be saved.
    - name: str, the filename under which to save the model.
    """
    dump(model, f'{name}.pkl')

def load_model(name):
    """
    Load a model from disk using joblib.
    
    Parameters:
    - name: str, the filename from which to load the model.
    
    Returns:
    - The loaded model.
    """
    return load(f'{name}.pkl')