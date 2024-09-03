import pandas as pd
from google.cloud import bigquery
import logging
from google.oauth2 import service_account

def get_data_from_bigquery(query, project_id, credentials_path=None):
    """
    Fetches data from Google BigQuery and returns it as a pandas DataFrame.

    Parameters:
    - query: str, the SQL query to execute.
    - project_id: str, the Google Cloud project ID.

    Returns:
    - DataFrame containing the query results.
    """
    try:

        # Define credentials if provided
        credentials = None
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        # Log the query execution
        logging.info("Executing query on BigQuery...")
        
        # Execute the query
        query_job = client.query(query)

        # Wait for the query to finish and return the results as a DataFrame
        results = query_job.result()

        # Log the number of rows fetched
        logging.info(f"Query executed successfully. {results.total_rows} rows fetched.")
        
        return results.to_dataframe()

    except Exception as e:
        logging.error(f"An error occurred while fetching data from BigQuery: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

if __name__ == "__main__":
    # Example usage
    project_id = "your-google-cloud-project-id"
    query = """
    SELECT *
    FROM `your-project-id.your-dataset.your-table`
    LIMIT 100
    """
    df = get_data_from_bigquery(query, project_id)
    print(df.head())