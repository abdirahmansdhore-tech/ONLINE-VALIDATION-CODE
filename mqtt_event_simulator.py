'''
MQTT Event Simulator (Physical Station Mimic)

Function: Replays historical data from a CSV file as live MQTT events, exactly
          mimicking the real-time behavior of the physical stations.

- Reads a CSV file with historical event data.
- Calculates the time delay between each event based on timestamps.
- Publishes each event as an MQTT message after the calculated delay.
- Uses the same topic structure and payload format as the real LEGO factory.
'''

import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import pandas as pd

# --- Configuration ---
MQTT_BROKER = "localhost"  # Change to your MQTT broker IP if not running locally
MQTT_PORT = 1883
HISTORICAL_DATA_FILE = "event_log_250725_103848.csv" # The 7-hour data file

# MQTT topic structure from the LEGO factory system
TOPIC_TEMPLATE = "G1-5S-PL/station_event/{station_id}/all"

# --- Main Simulator Logic ---

def create_mqtt_client():
    """Creates and connects an MQTT client."""
    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print(f"Successfully connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f"Error: Could not connect to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}. {e}")
        print("Please ensure the broker is running and accessible.")
        return None

def replay_events(client, data_file):
    """Reads the historical data and replays it as MQTT events with real-time delays."""
    try:
        print(f"Loading historical data from {data_file}...")
        df = pd.read_csv(data_file, skiprows=1, header=None, names=['timestamp', 'station', 'pallet', 'event'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
        print(f"Found {len(df)} events to replay.")
    except FileNotFoundError:
        print(f"Error: Historical data file not found at '{data_file}'.")
        print("Please make sure the CSV file is in the same directory as the simulator.")
        return

    print("\n--- Starting MQTT Event Simulation ---")
    print("This will run for approximately 7 hours, mimicking the real system.")
    print("Press Ctrl+C to stop the simulation at any time.")

    # Get the timestamp of the first event to calculate relative delays
    last_event_time = df.iloc[0]['timestamp']
    total_events = len(df)
    
    for index, row in df.iterrows():
        try:
            current_event_time = row['timestamp']
            
            # Calculate the delay to wait before sending the next event
            delay = (current_event_time - last_event_time).total_seconds()

            # Ensure delay is not negative (for out-of-order events, though unlikely)
            if delay > 0:
                time.sleep(delay)

            # Prepare the MQTT message
            station_id = row['station']
            topic = TOPIC_TEMPLATE.format(station_id=station_id)
            payload = json.dumps({
                "event": row['event'],
                "pallet_id": row['pallet'],
                "station_id": station_id,
                "timestamp": row['timestamp'].isoformat() # Use ISO format for consistency
            })

            # Publish the event
            client.publish(topic, payload)
            
            # Log to console
            print(f"[{current_event_time.strftime('%H:%M:%S')}] Event {index + 1}/{total_events} | Topic: {topic} | Payload: {payload}")

            # Update the last event time
            last_event_time = current_event_time

        except KeyboardInterrupt:
            print("\n--- Simulation stopped by user. ---")
            break
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            break

    print("\n--- MQTT Event Simulation Finished ---")

if __name__ == "__main__":
    mqtt_client = create_mqtt_client()
    if mqtt_client:
        replay_events(mqtt_client, HISTORICAL_DATA_FILE)
        mqtt_client.disconnect()

