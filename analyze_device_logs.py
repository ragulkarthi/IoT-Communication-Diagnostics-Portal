import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta, datetime
import os
import numpy as np

# Professional Configuration - Azure colors aligned
PROFESSIONAL_COLORS = {
    'connected': '#107C10',       # Azure green for connections (previously #0078D4 blue)
    'normal_disconnect': '#FFB900',  # Warning yellow for normal disconnects
    'error_disconnect': '#D13438',   # Critical red for error disconnects
    'inbound': '#0078D4',         # Azure blue for inbound
    'outbound': '#773ADC',        # Azure purple for outbound
    'both': '#D83B01',            # Azure orange for both
    'error': '#E81123',           # Azure error red
    'grid': '#F3F2F1',            # Azure gray for grid
    'background': '#FFFFFF',      # Pure white
    'text': '#323130',            # Azure text
    'highlight': '#767676',       # Medium gray
    'annotation': '#605E5C',      # Light gray
    'zoom_highlight': '#00BCF2'   # Azure bright blue for zoom highlights
}

FONT_FAMILY = "Segoe UI, Roboto, -apple-system, BlinkMacSystemFont, sans-serif"
TITLE_FONT = dict(family=FONT_FAMILY, size=14, color=PROFESSIONAL_COLORS['text'])
AXIS_FONT = dict(family=FONT_FAMILY, size=11, color='#605E5C')
PLOT_HEIGHT = 650  # Increased height for better visualization
ANNOTATION_FONT_SIZE = 11
MARKER_SIZE = 10
LINE_WIDTH = 4

def load_data_from_csv(device_id: str, reports_dir: str = "output/reports") -> dict:
    """
    Load device data from CSV files in the reports directory.
    
    Args:
        device_id: The Azure IoT device ID
        reports_dir: Directory containing the report CSV files
        
    Returns:
        Dictionary containing DataFrames for connection, inbound, outbound and error data
    """
    data = {}
    file_patterns = {
        'connection': f"{device_id}_connection.csv",
        'inbound': f"{device_id}_inbound.csv",
        'outbound': f"{device_id}_outbound.csv",
        'error': f"{device_id}_error.csv"
    }
    
    # Verify reports directory exists
    if not os.path.exists(reports_dir):
        print(f"Reports directory {reports_dir} does not exist. Creating it.")
        try:
            os.makedirs(reports_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating reports directory: {str(e)}")
    
    # Check if any of the expected files exist before proceeding
    files_exist = False
    for key, pattern in file_patterns.items():
        file_path = os.path.join(reports_dir, pattern)
        if os.path.exists(file_path):
            files_exist = True
            break
    
    if not files_exist:
        print(f"No data files found for device {device_id} in {reports_dir}")
        # Return empty dataframes
        data = {
            'connection': pd.DataFrame(columns=['timestamp', 'event', 'status']),
            'inbound': pd.DataFrame(columns=['timestamp']),
            'outbound': pd.DataFrame(columns=['timestamp']),
            'error': pd.DataFrame(columns=['timestamp', 'event'])
        }
        return data
    
    for key, pattern in file_patterns.items():
        file_path = os.path.join(reports_dir, pattern)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                if key == 'connection':
                    # Convert status column to boolean for device disconnect events
                    if 'status' in df.columns:
                        # Handle status conversion explicitly to avoid auto-conversion issues
                        df['status_parsed'] = None
                        
                        for idx, row in df.iterrows():
                            if row['event'] == 'deviceDisconnect':
                                # Handle empty or NaN values - default to True
                                if pd.isna(row['status']) or row['status'] == '':
                                    df.at[idx, 'status_parsed'] = True
                                # Convert various True/False formats to actual boolean
                                elif isinstance(row['status'], str):
                                    if row['status'].lower() == 'true':
                                        df.at[idx, 'status_parsed'] = True
                                    elif row['status'].lower() == 'false':
                                        df.at[idx, 'status_parsed'] = False
                                    # Handle numeric strings
                                    elif row['status'] == '0' or row['status'] == '0.0':
                                        df.at[idx, 'status_parsed'] = False
                                    elif row['status'] == '1' or row['status'] == '1.0':
                                        df.at[idx, 'status_parsed'] = True
                                elif isinstance(row['status'], (int, float)):
                                    # 1/1.0 -> True, 0/0.0 -> False
                                    df.at[idx, 'status_parsed'] = bool(row['status'])
                                else:
                                    # Default to True if can't parse
                                    df.at[idx, 'status_parsed'] = True
                                
                        # Replace original status column with parsed version
                        df['status'] = df['status_parsed']
                        df.drop('status_parsed', axis=1, inplace=True)
                
                # Convert timestamps to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    # Ensure timestamps have proper microsecond precision
                    if df['timestamp'].dt.microsecond.sum() == 0:
                        print(f"Warning: No microsecond precision in {key} timestamps")
                elif 'ReceivedUtc' in df.columns:  # For inbound messages
                    df['timestamp'] = pd.to_datetime(df['ReceivedUtc'], utc=True)
                    # Ensure timestamps have proper microsecond precision
                    if df['timestamp'].dt.microsecond.sum() == 0:
                        print(f"Warning: No microsecond precision in {key} timestamps")
                    
                data[key] = df
                print(f"Loaded {len(df)} records from {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                # Create empty DataFrame with appropriate columns for this file type
                if key == 'connection':
                    data[key] = pd.DataFrame(columns=['timestamp', 'event', 'status'])
                elif key == 'error':
                    data[key] = pd.DataFrame(columns=['timestamp', 'event'])
                else:  # inbound, outbound
                    data[key] = pd.DataFrame(columns=['timestamp'])
        else:
            # Create empty DataFrame with appropriate columns
            if key == 'connection':
                data[key] = pd.DataFrame(columns=['timestamp', 'event', 'status'])
            elif key == 'error':
                data[key] = pd.DataFrame(columns=['timestamp', 'event'])
            else:  # inbound, outbound
                data[key] = pd.DataFrame(columns=['timestamp'])
    
    return data

def add_connection_periods(fig, df):
    """Add connected time periods with professional styling"""
    connect_periods = []
    current_start = None
    
    # Sort by timestamp and remove duplicates to prevent multiple active connections
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'event'], keep='first')
    
    for _, row in df.iterrows():
        if row['event'] == 'deviceConnect':
            current_start = row['timestamp']
        elif row['event'] == 'deviceDisconnect' and current_start:
            connect_periods.append((current_start, row['timestamp']))
            current_start = None

    # Add only one trace for all connection periods
    if connect_periods:
        x_vals = []
        y_vals = []
        hover_texts = []
        
        for start, end in connect_periods:
            # Calculate precise duration including milliseconds
            duration_seconds = (end - start).total_seconds()
            hours = int(duration_seconds / 3600)
            minutes = int((duration_seconds % 3600) / 60)
            seconds = duration_seconds % 60
            
            # Format for hover display with high precision
            duration_str = f"{hours}h {minutes}m {seconds:.3f}s"
            
            x_vals.extend([start, end, None])  # None creates a gap between periods
            y_vals.extend([1, 1, None])
            hover_texts.extend([
                f"<b>Active Connection</b><br>"
                f"From: {start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}<br>"
                f"To: {end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}<br>"
                f"Duration: {duration_str}",
                f"<b>Active Connection</b><br>"
                f"From: {start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}<br>"
                f"To: {end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}<br>"
                f"Duration: {duration_str}",
                None
            ])
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color=PROFESSIONAL_COLORS['connected'], width=LINE_WIDTH),
            hoverinfo='text',
            hovertext=hover_texts,
            opacity=0.8,
            showlegend=False,
            name='Active Connection'
        ))

def add_professional_markers(fig, df, color, name, symbol, hover_template):
    """Add professional-grade event markers with proper vertical positioning"""
    if df is None or df.empty:
        return
        
    if 'y' not in df.columns:
        df['y'] = 1.0  # Default position
    
    # Format timestamps for hover with microsecond precision
    timestamp_texts = [ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for ts in df['timestamp']]
        
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['y'],
        mode='markers',
        marker=dict(
            color=color,
            size=MARKER_SIZE,
            symbol=symbol,
            line=dict(width=1.5, color='white'),
            opacity=0.9
        ),
        name=name,
        text=timestamp_texts,  # Use custom formatted timestamps
        hovertemplate=hover_template,
        hoverlabel=dict(
            bgcolor=color,
            font=dict(size=12, family=FONT_FAMILY, color='white'),
            bordercolor='rgba(0,0,0,0.2)'
        )
    ))

def handle_simultaneous_events(df_dict):
    """
    Process multiple dataframes and assign y-values to stack events occurring at the same timestamp.
    """
    # Step 1: Gather all timestamps and their sources
    all_timestamps = []
    for event_type, df in df_dict.items():
        if df is not None and not df.empty:
            for ts in df['timestamp']:
                all_timestamps.append((ts, event_type))
    
    # Step 2: Count events per timestamp and assign positions
    from collections import defaultdict
    timestamp_counts = defaultdict(int)
    timestamp_positions = defaultdict(dict)
    
    # Count events per timestamp
    for ts, event_type in all_timestamps:
        timestamp_counts[ts] += 1
    
    # For each timestamp, assign vertical positions based on event type priority
    priority_order = ['error', 'connection', 'inbound', 'outbound']
    
    for ts, count in timestamp_counts.items():
        if count > 1:  # Only process timestamps with multiple events
            position = 1.0  # Start position
            step = 0.1 / count  # Calculate step size based on number of events
            
            # First assign positions by priority
            for event_type in priority_order:
                for idx, (event_ts, event) in enumerate(all_timestamps):
                    if event_ts == ts and event == event_type:
                        timestamp_positions[ts][event_type] = position
                        position -= step
    
    # Step 3: Apply positions to each dataframe
    result = {}
    for event_type, df in df_dict.items():
        if df is None or df.empty:
            result[event_type] = df
            continue
            
        processed_df = df.copy()
        processed_df['y'] = 1.0  # Default position
        
        # Apply calculated positions for timestamps with multiple events
        for idx, row in processed_df.iterrows():
            ts = row['timestamp']
            if ts in timestamp_positions and event_type in timestamp_positions[ts]:
                processed_df.at[idx, 'y'] = timestamp_positions[ts][event_type]
        
        result[event_type] = processed_df
    
    return result

def add_millisecond_highlights(fig, df_conn, highlight_threshold_ms=10):
    """
    Add vertical highlights for events that occur very close together (within milliseconds)
    to help visually identify potential timing issues.
    
    Args:
        fig: The plotly figure to add highlights to
        df_conn: The connection events dataframe
        highlight_threshold_ms: The threshold in milliseconds to highlight
    """
    if df_conn is None or len(df_conn) < 2:
        return
    
    # Sort by timestamp
    df_sorted = df_conn.sort_values('timestamp')
    
    # Find events that occur within milliseconds of each other
    close_events = []
    for i in range(len(df_sorted) - 1):
        time_diff = (df_sorted.iloc[i+1]['timestamp'] - df_sorted.iloc[i]['timestamp']).total_seconds() * 1000
        if time_diff <= highlight_threshold_ms:
            close_events.append((
                df_sorted.iloc[i]['timestamp'],
                df_sorted.iloc[i+1]['timestamp'],
                time_diff
            ))
    
    # Add highlights for close events
    for start, end, diff_ms in close_events:
        # Center the highlight around the events
        mid_time = start + (end - start) / 2
        span_ms = max(5, diff_ms * 1.5)  # At least 5ms span, or 150% of the time difference
        
        # Convert span to timedelta
        span = timedelta(milliseconds=span_ms)
        
        fig.add_vrect(
            x0=mid_time - span/2,
            x1=mid_time + span/2,
            fillcolor=PROFESSIONAL_COLORS['zoom_highlight'],
            opacity=0.1,
            layer="above",
            line_width=0,
            annotation_text=f"{diff_ms:.3f}ms",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=PROFESSIONAL_COLORS['zoom_highlight']
        )

def format_device_telemetry_stats(df_conn, df_in, df_out, df_err, start_date, end_date):
    """
    Format Azure IoT Hub telemetry statistics for the device
    """
    stats = []
    
    # Calculate connection metrics
    connect_periods = []
    current_start = None
    for _, row in df_conn.sort_values('timestamp').iterrows():
        if row['event'] == 'deviceConnect':
            current_start = row['timestamp']
        elif row['event'] == 'deviceDisconnect' and current_start:
            connect_periods.append((current_start, row['timestamp']))
            current_start = None
    
    total_conn_time = sum((end-start).total_seconds() for start,end in connect_periods)/3600
    uptime_percentage = (total_conn_time / ((end_date - start_date).total_seconds()/3600)) * 100
    
    # Count disconnect types
    if 'status' in df_conn.columns:
        error_count = len(df_conn[(df_conn['event'] == 'deviceDisconnect') & (df_conn['status'] == False)])
        normal_count = len(df_conn[(df_conn['event'] == 'deviceDisconnect') & (df_conn['status'] == True)])
        total_disconnects = error_count + normal_count
        error_percentage = (error_count / total_disconnects * 100) if total_disconnects > 0 else 0
    else:
        error_count = 0
        normal_count = len(df_conn[df_conn['event'] == 'deviceDisconnect'])
        error_percentage = 0
    
    return "<br>".join(stats)

def create_device_timeline_from_csv(device_id: str, reports_dir: str = "output/reports", output_file: str = None):
    """
    Create and display a device timeline visualization from Azure IoT Hub CSV data files
    with enhanced millisecond precision and responsive layout.
    """
    # Load data from CSV files
    data = load_data_from_csv(device_id, reports_dir)
    
    df_conn = data.get('connection', pd.DataFrame(columns=['timestamp', 'event', 'status']))
    df_in = data.get('inbound', pd.DataFrame(columns=['timestamp']))
    df_out = data.get('outbound', pd.DataFrame(columns=['timestamp']))
    df_err = data.get('error', pd.DataFrame(columns=['timestamp', 'event']))
    
    # Get the first and last timestamp across all dataframes
    timestamps = []
    for df in [df_conn, df_in, df_out, df_err]:
        if df is not None and not df.empty and 'timestamp' in df.columns:
            timestamps.extend(df['timestamp'].tolist())
    
    if not timestamps:
        print(f"No data found for device {device_id}. Please check the device ID and ensure data files exist.")
        print(f"Expected files in directory: {reports_dir}")
        print(f"  - {device_id}_connection.csv")
        print(f"  - {device_id}_inbound.csv")
        print(f"  - {device_id}_outbound.csv")
        print(f"  - {device_id}_error.csv")
        return None
        
    start_date = min(timestamps) - timedelta(hours=1)
    end_date = max(timestamps) + timedelta(hours=1)
    
    # Create Figure with Professional Layout
    fig = go.Figure()
    
    # Add business hour background shading
    for day in pd.date_range(start_date, end_date, freq='D'):
        fig.add_vrect(
            x0=day.replace(hour=9, minute=0),
            x1=day.replace(hour=17, minute=0),
            fillcolor="rgba(0, 120, 212, 0.05)",  # Very light Azure blue
            layer="below",
            line_width=0,
        )
    
    # Add connection periods
    add_connection_periods(fig, df_conn)
    
    # Process all dataframes to handle simultaneous events
    dataframes_dict = {
        'inbound': df_in,
        'outbound': df_out,
        'connection': df_conn,
        'error': df_err
    }
    processed_dfs = handle_simultaneous_events(dataframes_dict)
    df_in = processed_dfs['inbound']
    df_out = processed_dfs['outbound']
    df_conn = processed_dfs['connection']
    df_err = processed_dfs['error']
    
    # Add Azure IoT Hub message markers
    if not df_in.empty:
        add_professional_markers(
            fig, df_in, 
            PROFESSIONAL_COLORS['inbound'], 
            'Inbound Message', 
            'circle',
            '<b>Inbound Message</b><br>Time: %{text}<extra></extra>'
        )
    
    if not df_out.empty:
        add_professional_markers(
            fig, df_out, 
            PROFESSIONAL_COLORS['outbound'], 
            'Outbound Message', 
            'square',
            '<b>Outbound Message</b><br>Time: %{text}<extra></extra>'
        )
    
    if not df_err.empty:
        add_professional_markers(
            fig, df_err, 
            PROFESSIONAL_COLORS['error'], 
            'Error Event', 
            'x',
            '<b>Error Event</b><br>Time: %{text}<extra></extra>'
        )
    
    # Add connection event markers
    connect_events = df_conn[df_conn['event'] == 'deviceConnect']
    add_professional_markers(
        fig, connect_events, 
        PROFESSIONAL_COLORS['connected'], 
        'Connected', 
        'triangle-up',
        '<b>Connected</b><br>Time: %{text}<extra></extra>'
    )
    
    # Add different types of disconnect markers
    if 'status' in df_conn.columns:
        normal_disconnects = df_conn[(df_conn['event'] == 'deviceDisconnect') & (df_conn['status'] == True)]
        error_disconnects = df_conn[(df_conn['event'] == 'deviceDisconnect') & (df_conn['status'] == False)]
        
        add_professional_markers(
            fig, normal_disconnects, 
            PROFESSIONAL_COLORS['normal_disconnect'], 
            'Normal Disconnect', 
            'triangle-down',
            '<b>Normal Disconnect</b><br>Time: %{text}<extra></extra>'
        )
        
        add_professional_markers(
            fig, error_disconnects, 
            PROFESSIONAL_COLORS['error_disconnect'], 
            'Error Disconnect', 
            'triangle-down',
            '<b>Error Disconnect</b><br>Time: %{text}<extra></extra>'
        )
    else:
        disconnect_events = df_conn[df_conn['event'] == 'deviceDisconnect']
        add_professional_markers(
            fig, disconnect_events, 
            PROFESSIONAL_COLORS['normal_disconnect'], 
            'Disconnected', 
            'triangle-down',
            '<b>Disconnected</b><br>Time: %{text}<extra></extra>'
        )
    
    # Add millisecond highlights for events occurring close together
    add_millisecond_highlights(fig, df_conn, highlight_threshold_ms=50)
    
    # Format telemetry statistics
    stats_text = format_device_telemetry_stats(df_conn, df_in, df_out, df_err, start_date, end_date)
    
    # Enhanced Layout Configuration with Millisecond Precision
    fig.update_layout(
        height=PLOT_HEIGHT,
        autosize=True,
        title=dict(
            text=f'<b>AZURE IOT HUB DEVICE TIMELINE</b><br>'
                f'<span style="font-size:0.8em">Device: {device_id}</span>',
            font=dict(size=18, family=FONT_FAMILY, color=PROFESSIONAL_COLORS['text']),
            x=0.05,
            y=0.98,  # Move title closer to top
            xanchor='left',
            yanchor='top',
            pad=dict(t=0, b=0)  # Reduce padding
        ),
        xaxis_title='<b>Timestamp (UTC)</b>',
        yaxis=dict(
            visible=False, 
            range=[0.85, 1.05],
            showgrid=False
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.0,  # Move legend up
            xanchor='right',
            x=1,
            font=dict(size=11, family=FONT_FAMILY),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        plot_bgcolor=PROFESSIONAL_COLORS['background'],
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=50),  # Reduced top and bottom margins
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family=FONT_FAMILY,
            bordercolor='rgba(0,0,0,0.1)'
        ),
        font=dict(
            family=FONT_FAMILY,
            size=12,
            color=PROFESSIONAL_COLORS['text']
        ),
        template="plotly_white"
    )

    # X-axis Configuration with Enhanced Millisecond Precision
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikecolor=PROFESSIONAL_COLORS['highlight'],
        spikesnap='cursor',
        spikethickness=0.5,
        spikedash='solid',
        gridcolor=PROFESSIONAL_COLORS['grid'],
        rangeslider=dict(
            visible=True,
            thickness=0.06,  # Make even thinner
            bgcolor='rgba(240,240,240,0.7)',
            bordercolor=PROFESSIONAL_COLORS['grid'],
            borderwidth=1
        ),
        tickformatstops=[
            dict(dtickrange=[None, 10], value="%H:%M:%S.%f"),
            dict(dtickrange=[10, 1000], value="%H:%M:%S.%L"),
            dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
            dict(dtickrange=[60000, 3600000], value="%H:%M"),
            dict(dtickrange=[3600000, 86400000], value="%H:%M"),
            dict(dtickrange=[86400000, 604800000], value="%e %b %Y"),
            dict(dtickrange=[604800000, "M1"], value="%e %b %Y"),
            dict(dtickrange=["M1", None], value="%b '%y")
        ],
        title_font=TITLE_FONT,
        tickfont=AXIS_FONT,
        showline=True,
        linecolor=PROFESSIONAL_COLORS['grid'],
        mirror=True,
        linewidth=1,
        range=[start_date, end_date],
        rangebreaks=[],
        title_text="<b>Timestamp (UTC)</b>",
        nticks=24,
        # Add this to reduce space below x-axis
        title_standoff=10,  # Reduce space between axis and title
        ticklen=5  # Make tick marks shorter
    )
    
    # Show the figure
    if output_file:
        # Save as HTML with fully responsive layout
        fig.write_html(
            output_file,
            config={
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',  # Reset zoom and autosize on double click
                'showTips': True,
                'responsive': True,  # Make chart responsive
                'displayModeBar': True,
                'modeBarButtonsToAdd': [
                    'zoom2d',
                    'pan2d',
                    'zoomIn2d',
                    'zoomOut2d',
                    'autoScale2d',
                    'resetScale2d'
                ],
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': f'{device_id}_iot_timeline',
                    'height': PLOT_HEIGHT,
                    'width': 1200,
                    'scale': 2  # Higher resolution
                },
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False
            },
            include_plotlyjs='cdn',
            auto_open=True,
            full_html=True,
            # Make fully responsive
            default_width='100%',
            default_height='100vh'  # Use viewport height for true responsiveness
        )
        print(f"Visualization saved to {output_file}")
    else:
        fig.show(config={
            'scrollZoom': True,
            'doubleClick': 'reset+autosize',
            'responsive': True
        })
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create an Azure IoT Hub device communication timeline with microsecond precision")
    parser.add_argument("device_id", help="IoT Hub device ID to visualize")
    parser.add_argument("--reports_dir", default="output/reports", help="Directory containing report CSV files")
    parser.add_argument("--output", help="Output HTML file path")
    parser.add_argument("--highlight_ms", type=int, default=50, help="Highlight events occurring within this many milliseconds")
    
    args = parser.parse_args()
    
    create_device_timeline_from_csv(args.device_id, args.reports_dir, args.output)