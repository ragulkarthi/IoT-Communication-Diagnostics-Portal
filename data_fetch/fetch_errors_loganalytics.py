from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import pandas as pd
from datetime import datetime, timezone
from config.config import CONFIG

def get_log_analytics_client():
    """Creates a LogsQueryClient using default Azure credentials."""
    creds = DefaultAzureCredential(exclude_environment_credential=True)
    return LogsQueryClient(creds)

def fetch_error_logs(device_id: str, days: int = 30) -> pd.DataFrame:
    """Fetches 'Error' level logs for a given device ID."""
    print(f"Fetching error logs for device: {device_id}")
    client = get_log_analytics_client()
    workspace_id = CONFIG["log_analytics"]["workspace_id"]

    # Kusto query for error logs
    kql = f"""
    AzureDiagnostics
    | where identity_g == "{device_id}"
    | where Level == "Error"
    | where TimeGenerated > ago({days}d)
    | project TimeGenerated, OperationName
    | sort by TimeGenerated asc
    """

    response = client.query_workspace(
        workspace_id=workspace_id,
        query=kql,
        timespan=None
    )

    if response.status != LogsQueryStatus.SUCCESS:
        raise Exception("Failed to query Log Analytics")

    table = response.tables[0]
    column_names = table.columns
    records = [dict(zip(column_names, row)) for row in table.rows]

    # Convert to DataFrame and ensure proper datetime handling
    df = pd.DataFrame(records)
    if not df.empty:
        df["TimeGenerated"] = pd.to_datetime(df["TimeGenerated"], utc=True)
        df = df.rename(columns={"TimeGenerated": "timestamp", "OperationName": "event"})
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df