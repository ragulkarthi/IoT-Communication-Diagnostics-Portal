import argparse
from data_fetch.fetch_inbound_loganalytics import fetch_inbound_logs
from data_fetch.fetch_outbound_sumologic import fetch_outbound_logs
from data_fetch.fetch_connectionevents_loganalytics import fetch_connection_logs
from data_fetch.fetch_errors_loganalytics import fetch_error_logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_id", help="Device ID to analyze")
    args = parser.parse_args()

    print(f"Fetching logs for device: {args.device_id}")
    
    inbound = fetch_inbound_logs(args.device_id)
    #outbound = fetch_outbound_logs(args.device_id)
    connection = fetch_connection_logs(args.device_id)
    error = fetch_error_logs(args.device_id)

    inbound.to_csv(f"output/reports/{args.device_id}_inbound.csv", index=False)
    #outbound.to_csv(f"output/reports/{args.device_id}_outbound.csv", index=False)
    connection.to_csv(f"output/reports/{args.device_id}_connection.csv", index=False)
    error.to_csv(f"output/reports/{args.device_id}_error.csv", index=False)

    print("Log fetch completed. Data saved to output/reports/")

if __name__ == "__main__":
    main()
