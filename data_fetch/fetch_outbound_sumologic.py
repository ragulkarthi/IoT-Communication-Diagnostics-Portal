import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
import time
import os
import json
from requests.auth import HTTPBasicAuth

# Initialize session with authentication
session = requests.Session()
session.auth = HTTPBasicAuth(
    os.getenv("SUMO_ACCESS_ID", "sucQuif3KVo0mk"),
    os.getenv("SUMO_ACCESS_KEY", "**************")
)
API_ENDPOINT = "https://api.sumologic.com/api/v1/search/jobs"


def create_search_job(query: str, start_time: datetime, end_time: datetime) -> str:
    """Create a search job and return the job ID"""
    job_data = {
        "query": query,
        "from": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "to": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timeZone": "UTC"
    }
    
    response = session.post(API_ENDPOINT, json=job_data)
    response.raise_for_status()
    return response.json()["id"]


def wait_for_job_completion(job_id: str, max_attempts: int = 30, delay: int = 2) -> None:
    """Wait for the search job to complete"""
    status_url = f"{API_ENDPOINT}/{job_id}"
    
    for _ in range(max_attempts):
        time.sleep(delay)
        response = session.get(status_url)
        response.raise_for_status()
        
        status = response.json()["state"]
        if status == "DONE GATHERING RESULTS":
            return
        elif status in ("CANCELLED", "FAILED"):
            raise Exception(f"Search job {status.lower()}")
    
    raise Exception("Search job timed out")


def get_job_results(job_id: str, limit: int = 1000) -> list:
    """Retrieve results from a completed job"""
    messages_url = f"{API_ENDPOINT}/{job_id}/messages?limit={limit}&offset=0"
    response = session.get(messages_url)
    response.raise_for_status()
    return response.json().get("messages", [])


def search_logs(query: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> list:
    """Execute a complete search operation"""
    job_id = create_search_job(query, start_time, end_time)
    wait_for_job_completion(job_id)
    records = get_job_results(job_id, limit)
    return [record.get("map", {}) for record in records]


def parse_log_record(raw_log: str) -> dict:
    """Parse a single log record and extract relevant fields"""
    try:
        log_data = json.loads(raw_log.strip())
        return {
            "timestamp": log_data.get("@t")
        }
    except json.JSONDecodeError:
        return None


def fetch_outbound_logs(device_id: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch outbound logs for a specific device from Sumo Logic.
    Returns a DataFrame with timestamp and status_code columns.
    
    Args:
        device_id: The device ID to search for
        days: Number of days to look back (default: 30)
        
    Returns:
        pd.DataFrame with columns ['timestamp', 'status_code']
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    # More specific query to reduce unnecessary data transfer
    query = f'_sourceCategory="IOT-NexGen/stg/api" AND "{device_id}/outbound_messages" | json field=_raw "StatusCode" | where %"StatusCode" = 200'
    
    records = search_logs(query, start_time, end_time)
    
    if not records:
        return pd.DataFrame(columns=["timestamp"])
    
    # Process records efficiently
    df = pd.DataFrame(records)
    if "_raw" not in df.columns:
        return pd.DataFrame(columns=["timestamp"])
    
    # Parse and filter records
    parsed_records = df["_raw"].apply(parse_log_record).dropna()
    if parsed_records.empty:
        return pd.DataFrame(columns=["timestamp"])
    
    df_out = pd.DataFrame(parsed_records.tolist())

    # Convert timestamp to datetime and sort
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True, errors="coerce")
    df_out = df_out.dropna(subset=["timestamp"])  # Drop rows with invalid timestamps
    df_out = df_out.sort_values("timestamp").reset_index(drop=True)

    return df_out