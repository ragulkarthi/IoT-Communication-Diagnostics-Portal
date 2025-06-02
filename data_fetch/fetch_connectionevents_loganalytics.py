from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import pandas as pd
from datetime import datetime, timezone
from config.config import CONFIG
import json
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_log_analytics_client():
    """Creates a LogsQueryClient using default Azure credentials."""
    creds = DefaultAzureCredential(exclude_environment_credential=True)
    return LogsQueryClient(creds)

def fetch_connection_logs(device_id: str, days: int = 30) -> pd.DataFrame:
    """
    Fetches deviceConnect and deviceDisconnect logs for a given device ID.
    For deviceDisconnect events only, adds a status column (0=disconnect with statusCode, 1=normal disconnect)
    
    Args:
        device_id: The IoT device ID to search for
        days: Number of days to look back in logs
        
    Returns:
        DataFrame with columns: timestamp, event, status (status applies only to disconnect events)
    """
    logger.info(f"Fetching connection logs for device: {device_id}")
    client = get_log_analytics_client()
    workspace_id = CONFIG["log_analytics"]["workspace_id"]

    # Enhanced Kusto query to include properties_s for status code extraction
    kql = f"""
    AzureDiagnostics
    | where identity_g == "{device_id}"
    | where OperationName in ("deviceConnect", "deviceDisconnect")
    | where TimeGenerated > ago({days}d)
    | project TimeGenerated, OperationName, properties_s
    | sort by TimeGenerated asc
    """

    try:
        response = client.query_workspace(
            workspace_id=workspace_id,
            query=kql,
            timespan=None
        )

        if response.status != LogsQueryStatus.SUCCESS:
            logger.error("Failed to query Log Analytics")
            return pd.DataFrame(columns=["timestamp", "event", "status"])

        # Get the first table from the response
        table = response.tables[0]
        
        # Fix: Handle column names based on their type
        column_names = []
        if hasattr(table, 'columns') and table.columns:
            # Check if columns are objects with a name attribute or strings
            if isinstance(table.columns[0], str):
                column_names = table.columns
            else:
                column_names = [column.name for column in table.columns]
                
        records = []
        
        for row in table.rows:
            row_dict = dict(zip(column_names, row))
            event = row_dict.get("OperationName")
            timestamp = row_dict.get("TimeGenerated")
            
            # Create the base record with timestamp and event
            if event == "deviceConnect":
                record = {
                    "timestamp": timestamp,
                    "event": event
                }
            else:  # deviceDisconnect
                # Default status value is 1 (normal disconnect)
                status = 1
                
                # Check if statusCode and trackingId exist in properties
                if "properties_s" in row_dict:
                    try:
                        # Parse properties JSON
                        properties = json.loads(row_dict["properties_s"])
                        
                        # If both statusCode and trackingId exist, set status to 0
                        if "statusCode" in properties and "trackingId" in properties:
                            status = 0
                    except (json.JSONDecodeError, TypeError) as e:
                        # If JSON parsing fails, keep default status of 1
                        logger.warning(f"Failed to parse properties JSON: {e}")
                    record = {
                        "timestamp": timestamp,
                        "event": event,
                        "status": bool(status)  # Convert to boolean (0=False, 1=True)
                    }
                         
            records.append(record)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Successfully processed {len(df)} connection events")
        else:
            logger.info("No connection events found")

        return df
    
    except Exception as e:
        logger.error(f"Error fetching connection logs: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["timestamp", "event", "status"])