from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import pandas as pd
import re
from datetime import datetime
from config.config import CONFIG

def get_log_analytics_client():
    """Creates a LogsQueryClient using default Azure credentials."""
    creds = DefaultAzureCredential(exclude_environment_credential=True)
    return LogsQueryClient(creds)

def extract_received_utc(message: str):
    """Extracts ReceivedUtc from the log message string."""
    try:
        match = re.search(r"ReceivedUtc\s*=\s*([\d/ :]+)", message)
        return match.group(1).strip() if match else None
    except Exception as e:
        print(f"Error parsing ReceivedUtc from message: {message}\n{e}")
        return None

def fetch_inbound_logs(device_id: str, days: int = 30) -> pd.DataFrame:
    """Fetches inbound telemetry logs and returns only ReceivedUtc values for a given device ID."""
    print(f"Fetching logs for device: {device_id}")
    client = get_log_analytics_client()
    workspace_id = CONFIG["log_analytics"]["workspace_id"]

    # Kusto query to fetch logs
    kql = f"""
    FunctionAppLogs
    | where AppName == "iot-stage-events"
    | where clientid_CF == "{device_id}"
    | where eventtype_CF == "Telemetry"
    | where TimeGenerated > ago({days}d)
    | sort by TimeGenerated asc
    | project Message
    """

    response = client.query_workspace(
        workspace_id=workspace_id,
        query=kql,
        timespan=None
    )

    if response.status != LogsQueryStatus.SUCCESS:
        raise Exception("Failed to query Log Analytics")

    table = response.tables[0]
    
    # Get column names directly from the table object
    column_names = table.columns
    records = [dict(zip(column_names, row)) for row in table.rows]

    # Extract only ReceivedUtc
    received_utc_list = [
        extract_received_utc(rec.get("Message", ""))
        for rec in records
        if extract_received_utc(rec.get("Message", "")) is not None
    ]
    # Convert to datetime objects and sort ascending
    received_utc_list_dt = [datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S") for dt_str in received_utc_list]
    received_utc_list_dt.sort()
    return pd.DataFrame(received_utc_list_dt, columns=["ReceivedUtc"])