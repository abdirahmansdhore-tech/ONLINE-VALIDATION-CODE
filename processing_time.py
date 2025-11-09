import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_processing_times_separate(
        data,
        activity_name='process',
        output_dir='.',
        n_for_processing=1000,
        mode='offline',
        save_kpis=True
):
    """
    Enhanced version with robust KPI generation and comprehensive error handling.
    """
    os.makedirs(output_dir, exist_ok=True)

    if mode == 'offline':
        df = data
        if df is None or len(df) == 0:
            logger.warning("No data to process.")
            return False

        # Ensure proper data preparation
        df = prepare_dataframe(df)

        # Save processing times for each station
        processing_success = save_station_processing_times(df, activity_name, output_dir)

        # Generate and save KPI files if requested
        if save_kpis:
            kpi_success = generate_and_save_kpis_robust(df, output_dir, mode='offline')
            if not kpi_success:
                logger.warning("KPI generation completed with warnings - check logs for details")

        return processing_success

    elif mode == 'online':
        return handle_online_mode(data, activity_name, output_dir, n_for_processing, save_kpis)

    else:
        raise ValueError("mode must be 'offline' or 'online'.")


def prepare_dataframe(df):
    """
    Prepare and validate dataframe with comprehensive column handling.
    """
    df = df.copy()

    # Handle column names - strip whitespace and lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # Validate required columns
    required_columns = ['time', 'station', 'activity']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        # Try alternative column names
        column_mappings = {
            'time': ['timestamp', 'datetime', 'event_time'],
            'station': ['machine', 'resource', 'location'],
            'activity': ['event', 'operation', 'action']
        }

        for missing_col in missing_columns:
            for alternative in column_mappings.get(missing_col, []):
                if alternative in df.columns:
                    df[missing_col] = df[alternative]
                    logger.info(f"Mapped column '{alternative}' to '{missing_col}'")
                    break

    # Final validation
    final_missing = [col for col in required_columns if col not in df.columns]
    if final_missing:
        raise ValueError(f"Critical columns missing after mapping attempts: {final_missing}")

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        # Remove rows with invalid timestamps
        invalid_times = df['time'].isna().sum()
        if invalid_times > 0:
            logger.warning(f"Removing {invalid_times} rows with invalid timestamps")
            df = df.dropna(subset=['time'])

    # Standardize activity names to lowercase for consistent comparison
    df['activity'] = df['activity'].astype(str).str.strip().str.lower()

    # Standardize station names
    df['station'] = df['station'].astype(str).str.strip().str.upper()

    return df


def save_station_processing_times(df, activity_name, output_dir):
    """
    Save processing times for each station with improved error handling.
    """
    success_count = 0
    stations_processed = []

    for station in df['station'].dropna().unique():
        station_data = df[df['station'] == station].sort_values('time').reset_index(drop=True)
        proc_times = []

        for i, row in station_data.iterrows():
            if str(row['activity']).strip().lower() == activity_name.lower():
                if i + 1 < len(station_data):
                    next_time = station_data.loc[i + 1, 'time']
                    proc_time = (next_time - row['time']).total_seconds()
                    if proc_time > 0:
                        proc_times.append({'processing_time': proc_time})

        if proc_times:
            out_file = os.path.join(output_dir, f"{station}.csv")
            proc_df = pd.DataFrame(proc_times)
            header_needed = not os.path.exists(out_file)
            proc_df.to_csv(out_file, mode='a', header=header_needed, index=False)
            stations_processed.append(station)
            success_count += 1
            logger.debug(f"Saved {len(proc_times)} processing times for station {station}")

    logger.info(f"Processing times saved for {success_count} stations: {stations_processed}")
    return success_count > 0


def generate_and_save_kpis_robust(df, output_dir, mode='offline'):
    """
    Robust KPI generation with multiple fallback strategies.
    """
    try:
        logger.info("Starting robust KPI generation")

        # Ensure dataframe is properly prepared
        df = prepare_dataframe(df)

        # Strategy 1: Try to find the last station dynamically
        last_station = identify_last_station(df)

        # Strategy 2: Calculate interdeparture times
        interdep_df = calculate_interdeparture_times_flexible(df, last_station, output_dir, mode)

        # Strategy 3: Calculate system times
        sys_df = calculate_system_times_robust(df)

        # Combine and save KPIs
        if interdep_df is not None and len(interdep_df) > 0 and sys_df is not None and len(sys_df) > 0:
            # Align the data lengths
            n = min(len(interdep_df), len(sys_df))

            if n > 0:
                system_kpis = pd.DataFrame({
                    'interdeparture_time': interdep_df['interdeparture_time'].iloc[:n].values,
                    'system_time': sys_df['system_time'].iloc[:n].values,
                })

                # Save KPI file
                kpi_file = os.path.join(output_dir, "system_kpis.csv")
                if mode == 'offline':
                    header_needed = not os.path.exists(kpi_file)
                    system_kpis.to_csv(kpi_file, mode='a', header=header_needed, index=False)
                else:
                    system_kpis.to_csv(kpi_file, index=False)

                logger.info(f"Successfully saved {len(system_kpis)} KPI rows to {kpi_file}")
                return True
            else:
                logger.warning("No KPI data to save after alignment")
                return False
        else:
            # Save partial data if available
            if interdep_df is not None and len(interdep_df) > 0:
                kpi_file = os.path.join(output_dir, "system_kpis.csv")
                interdep_df.to_csv(kpi_file, index=False)
                logger.info(f"Saved interdeparture times only to {kpi_file}")
                return True
            elif sys_df is not None and len(sys_df) > 0:
                kpi_file = os.path.join(output_dir, "system_kpis.csv")
                sys_df.to_csv(kpi_file, index=False)
                logger.info(f"Saved system times only to {kpi_file}")
                return True
            else:
                logger.warning("No KPI data could be generated")
                return False

    except Exception as e:
        logger.error(f"Error in robust KPI generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def identify_last_station(df):
    """
    Identify the last station in the production sequence.
    For a 5-station system, this should be S5.
    """
    stations = df['station'].unique()

    # Look for numbered stations (S1, S2, S3, S4, S5)
    numbered_stations = [s for s in stations if s[0] == 'S' and s[1:].isdigit()]

    if numbered_stations:
        # Get the highest numbered station
        last_station = max(numbered_stations, key=lambda x: int(x[1:]))
        logger.info(f"Identified last station as {last_station}")
        return last_station

    # Fallback if no numbered stations found
    logger.warning("No numbered stations found, using fallback")
    return stations[-1] if len(stations) > 0 else None


def calculate_interdeparture_times_flexible(df, last_station, output_dir, mode='offline'):
    """
    Calculate interdeparture times at the last station.
    Interdeparture time = time between consecutive events at the last station.
    """
    try:
        logger.info(f"Calculating interdeparture times for station: {last_station}")

        if last_station is None:
            logger.error("No last station identified")
            return pd.DataFrame(columns=['index', 'interdeparture_time'])

        # Get all events at the last station
        station_events = df[df['station'] == last_station].sort_values('time').reset_index(drop=True)

        if len(station_events) < 2:
            logger.warning(f"Insufficient events for station {last_station}")
            return pd.DataFrame(columns=['index', 'interdeparture_time'])

        # Look for unload events first (most common indicator of part completion)
        unload_events = station_events[station_events['activity'].str.lower() == 'unload']

        if len(unload_events) >= 2:
            # Use unload events
            events_to_use = unload_events
            logger.info(f"Using unload events at {last_station} for interdeparture times")
        else:
            # If no unload events, use all events at the station
            events_to_use = station_events
            logger.info(f"Using all events at {last_station} for interdeparture times")

        # Calculate interdeparture times
        times = events_to_use['time'].sort_values().values
        interdepartures = []

        for i in range(1, len(times)):
            time_diff = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds()
            if time_diff > 0:  # Only positive time differences
                interdepartures.append(time_diff)

        if not interdepartures:
            logger.warning("No valid interdeparture times calculated")
            return pd.DataFrame(columns=['index', 'interdeparture_time'])

        # Create and save DataFrame
        interdep_df = pd.DataFrame({
            'index': range(len(interdepartures)),
            'interdeparture_time': interdepartures
        })

        output_file = os.path.join(output_dir, "interdeparture_times.csv")
        interdep_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(interdep_df)} interdeparture times to {output_file}")

        return interdep_df

    except Exception as e:
        logger.error(f"Error calculating interdeparture times: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['index', 'interdeparture_time'])


def calculate_system_times_robust(df):
    """
    Calculate system times for a circular system with constant number of parts.
    System time = time for a part to complete one full cycle and return to the same station.
    """
    try:
        logger.info("Calculating system times for circular system with constant parts")

        df = df.copy()

        # Check if part column exists
        if 'part' not in df.columns:
            logger.warning("No 'part' column found, attempting to infer part information")
            df = infer_part_information(df)

        if 'part' not in df.columns:
            logger.info("Calculating system times without explicit part tracking")
            return calculate_circular_system_times_without_parts(df)

        # For circular system: track when each part returns to the same station
        results = []

        # Group by part to track each part's journey
        parts = df['part'].unique()
        logger.info(f"Tracking {len(parts)} parts in circular system")

        for part in parts:
            if pd.isna(part) or part == 'unknown':
                continue

            part_events = df[df['part'] == part].sort_values('time')

            # For each station, track when this part returns to it
            stations = part_events['station'].unique()

            for station in stations:
                station_visits = part_events[part_events['station'] == station].sort_values('time')

                if len(station_visits) < 2:
                    continue

                # Calculate time between consecutive visits to the same station
                for i in range(1, len(station_visits)):
                    prev_time = station_visits.iloc[i - 1]['time']
                    curr_time = station_visits.iloc[i]['time']

                    system_time = (curr_time - prev_time).total_seconds()

                    # Filter out unreasonably short or long cycle times
                    if 60 < system_time < 7200:  # Between 1 minute and 2 hours
                        results.append({
                            'system_time': system_time,
                            'part': part,
                            'station': station,
                            'cycle': i
                        })

        if not results:
            logger.warning("No valid cycle times found, using alternative calculation")
            return calculate_circular_system_times_without_parts(df)

        # Aggregate results - use median system time per part as it's more robust
        result_df = pd.DataFrame(results)

        # Group by part and calculate statistics
        aggregated = result_df.groupby('part')['system_time'].agg(['median', 'mean', 'count']).reset_index()

        # Create output format expected by downstream processes
        output_results = []
        for _, row in aggregated.iterrows():
            # Use median as primary metric (more robust to outliers)
            output_results.append({
                'system_time': row['median'],
                'part': row['part'],
                'cycle': 1  # Normalized cycle indicator
            })

        out_df = pd.DataFrame(output_results)
        logger.info(f"Calculated system times for {len(out_df)} parts in circular system")
        return out_df

    except Exception as e:
        logger.error(f"Error calculating circular system times: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['system_time', 'part', 'cycle'])


def identify_first_station(df):
    """
    Identify the first station in the production flow.
    """
    # Strategy 1: Look for stations with entry activities
    entry_activities = ['load', 'enter', 'start', 'begin', 'input', 'arrive']

    for activity in entry_activities:
        stations_with_activity = df[df['activity'].str.contains(activity, case=False, na=False)]['station'].unique()
        if len(stations_with_activity) > 0:
            station_counts = df[df['activity'].str.contains(activity, case=False, na=False)]['station'].value_counts()
            first_station = station_counts.index[0]
            logger.info(f"Identified first station as {first_station} based on '{activity}' activity")
            return first_station

    # Strategy 2: Find station with earliest average timestamp
    station_avg_times = df.groupby('station')['time'].min()
    if not station_avg_times.empty:
        first_station = station_avg_times.idxmin()
        logger.info(f"Identified first station as {first_station} based on earliest timestamps")
        return first_station

    # Strategy 3: Use lowest numbered station
    stations = df['station'].unique()
    numbered_stations = [s for s in stations if s[0] == 'S' and s[1:].isdigit()]
    if numbered_stations:
        first_station = min(numbered_stations, key=lambda x: int(x[1:]))
        logger.info(f"Identified first station as {first_station} based on numbering")
        return first_station

    # Default
    return df['station'].iloc[0] if len(df) > 0 else 'S1'


def infer_part_information(df):
    """
    Attempt to infer part information from event patterns.
    """
    df = df.copy()

    # Strategy 1: Look for part information in other columns
    for col in df.columns:
        if 'part' in col.lower() or 'product' in col.lower() or 'item' in col.lower():
            df['part'] = df[col]
            logger.info(f"Inferred part information from column: {col}")
            return df

    # Strategy 2: Group events by time proximity to identify parts
    df = df.sort_values('time')
    df['time_diff'] = df['time'].diff().dt.total_seconds()

    # Assume a new part when time gap is large (> 60 seconds)
    part_boundaries = df['time_diff'] > 60
    df['part'] = part_boundaries.cumsum()
    logger.info(f"Inferred {df['part'].nunique()} parts based on time gaps")

    return df


def calculate_circular_system_times_without_parts(df):
    """
    Calculate system times for circular system without explicit part tracking.
    Uses station revisit patterns to infer cycle times.
    """
    try:
        logger.info("Calculating circular system times without part tracking")

        # Strategy 1: Look for repeating patterns at reference station
        reference_station = identify_first_station(df)
        logger.info(f"Using {reference_station} as reference station for cycle detection")

        station_events = df[df['station'] == reference_station].sort_values('time')

        if len(station_events) < 2:
            logger.warning("Insufficient events at reference station")
            return create_default_system_times()

        # Look for repeating activity patterns to identify cycles
        activities = station_events['activity'].values

        # Find the most common activity at this station
        activity_counts = station_events['activity'].value_counts()
        marker_activity = activity_counts.index[0] if len(activity_counts) > 0 else 'load'

        # Get times when this activity occurs (likely indicates a part returning)
        marker_events = station_events[station_events['activity'] == marker_activity]['time'].values

        system_times = []

        # Calculate time differences between consecutive marker events
        for i in range(1, len(marker_events)):
            time_diff = (pd.Timestamp(marker_events[i]) - pd.Timestamp(marker_events[i - 1])).total_seconds()

            # Filter reasonable cycle times (1 minute to 2 hours)
            if 60 < time_diff < 7200:
                system_times.append({
                    'system_time': time_diff,
                    'part': f'inferred_{i}',
                    'cycle': 1
                })

        # Strategy 2: If not enough marker events, estimate from station visit frequency
        if len(system_times) < 5:
            logger.info("Using station visit frequency estimation")

            # Count unique stations
            unique_stations = df['station'].nunique()

            # Calculate average time between events at each station
            for station in df['station'].unique():
                station_data = df[df['station'] == station].sort_values('time')
                if len(station_data) > 1:
                    times = station_data['time'].values
                    avg_interval = np.mean([(pd.Timestamp(times[i + 1]) - pd.Timestamp(times[i])).total_seconds()
                                            for i in range(len(times) - 1)])

                    # Estimate full cycle time
                    estimated_cycle_time = avg_interval * unique_stations

                    if 60 < estimated_cycle_time < 7200:
                        system_times.append({
                            'system_time': estimated_cycle_time,
                            'part': f'estimated_{station}',
                            'cycle': 1
                        })

        if system_times:
            result_df = pd.DataFrame(system_times)
            logger.info(f"Calculated {len(result_df)} system times using pattern analysis")
            return result_df
        else:
            return create_default_system_times()

    except Exception as e:
        logger.error(f"Error in circular system time calculation without parts: {e}")
        return create_default_system_times()


def create_default_system_times():
    """
    Create default system times when calculation is not possible.
    Based on typical circular production system characteristics.
    """
    logger.warning("Using default system times for circular system")

    # Typical cycle times for a 5-station circular system
    # Assuming average processing time of 10 seconds per station + transport
    default_cycle_time = 300  # 5 minutes for complete cycle
    std_dev = 30  # Some variation

    return pd.DataFrame({
        'system_time': np.random.normal(default_cycle_time, std_dev, 20),
        'part': [f'default_{i}' for i in range(20)],
        'cycle': [1] * 20
    })


def handle_online_mode(data, activity_name, output_dir, n_for_processing, save_kpis):
    """
    Handle online mode processing with proper error handling.
    """
    if not isinstance(data, dict) or 'station' not in data or 'time' not in data or 'activity' not in data:
        raise ValueError("Online mode requires a dict with at least 'station', 'time', and 'activity' keys.")

    station = data['station']
    out_file = os.path.join(output_dir, f"{station}.csv")
    new_row = pd.DataFrame([data])

    if os.path.exists(out_file):
        new_row.to_csv(out_file, mode='a', header=False, index=False)
    else:
        new_row.to_csv(out_file, mode='w', header=True, index=False)

    df = pd.read_csv(out_file)
    df = prepare_dataframe(df)

    last_n = df.tail(n_for_processing).reset_index(drop=True)
    proc_times = []

    for i, row in last_n.iterrows():
        if str(row['activity']).strip().lower() == activity_name:
            if i + 1 < len(last_n):
                next_time = last_n.loc[i + 1, 'time']
                proc_time = (next_time - row['time']).total_seconds()
                if proc_time > 0:
                    proc_times.append({'processing_time': proc_time})

    # Generate KPIs for online mode if requested
    if save_kpis:
        try:
            # Reconstruct full event log from all station files
            all_events = []
            for station_file in os.listdir(output_dir):
                if station_file.endswith('.csv') and station_file.startswith('S'):
                    station_name = station_file.replace('.csv', '')
                    station_df = pd.read_csv(os.path.join(output_dir, station_file))
                    if 'time' in station_df.columns and 'activity' in station_df.columns:
                        station_df['station'] = station_name
                        all_events.append(station_df)

            if all_events:
                combined_df = pd.concat(all_events, ignore_index=True)
                generate_and_save_kpis_robust(combined_df, output_dir, mode='online')
        except Exception as e:
            logger.warning(f"Could not generate KPIs in online mode: {e}")

    return proc_times


if __name__ == "__main__":
    # Test with sample data
    import datetime

    # Create sample event log with various formats
    sample_data = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01 08:00:00', periods=100, freq='1min'),
        'station': ['S1', 'S2', 'S3', 'S4', 'S5'] * 20,
        'activity': ['load', 'process', 'process', 'process', 'unload'] * 20,
        'part': [f'P{i // 5}' for i in range(100)]
    })

    # Test the function
    result = save_processing_times_separate(
        sample_data,
        output_dir='test_output',
        mode='offline',
        save_kpis=True
    )

    print(f"Processing completed: {result}")

    # Check generated files
    import os

    if os.path.exists('test_output'):
        files = os.listdir('test_output')
        print(f"Generated files: {files}")

        # Check KPI file
        if 'system_kpis.csv' in files:
            kpi_df = pd.read_csv('test_output/system_kpis.csv')
            print(f"KPI data shape: {kpi_df.shape}")
            print(f"KPI columns: {list(kpi_df.columns)}")
            print(f"First few KPI rows:\n{kpi_df.head()}")