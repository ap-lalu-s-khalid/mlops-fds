from google.cloud import bigquery

def get_data_from_bigquery(query, project_id):
    """
    Fetches data from Google BigQuery and returns it as a pandas DataFrame.

    Parameters:
    - query: str, the SQL query to execute.
    - project_id: str, the Google Cloud project ID.

    Returns:
    - DataFrame containing the query results.
    """
    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    results = query_job.result()  # Waits for the query to finish

    return results.to_dataframe()

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