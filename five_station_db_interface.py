# -*- coding: utf-8 -*-
"""
Optimal Five Station Real-Time Database Interface
Combines the best features from both approaches
Integrates real five-station manufacturing system with InfluxDB
Supports both real-time and offline timestamp-driven processing
"""
import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
import time
import datetime
from time import sleep
import logging
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
import threading
import queue
from typing import Dict, List, Optional, Any

logger = logging.getLogger("FiveStationDBInterface")

class FiveStationDBInterface:
    """
    Optimal database interface for five-station manufacturing system
    Supports real-time MQTT data acquisition and InfluxDB storage
    Compatible with existing DTDC system structure
    """
    
    def __init__(self, db_ip, db_name, db_port=8086, mqtt_broker="localhost", mqtt_port=1883):
        # Initialize logger first
        self.logger = logging.getLogger("FiveStationDBInterface")
        # Database connections
        self.db_ip = db_ip
        self.db_name = db_name
        self.db_port = db_port
        self.client = InfluxDBClient(host=self.db_ip, port=self.db_port)
        self.client_df = DataFrameClient(host=self.db_ip, port=self.db_port)
        
        # MQTT configuration  
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # System configuration
        self.system_id = "G1-5S-PL"
        self.stations = ["S1", "S2", "S3", "S4", "S5"]
        self.station_capacities = {"S1": 12, "S2": 10, "S3": 8, "S4": 6, "S5": 4}
        
        # Data buffers and queues
        self.event_buffer = queue.Queue()
        self.real_time_data_buffer = queue.Queue()
        self.processing_time_buffer = {}
        self.station_states = {station: "IDLE" for station in self.stations}
        self.station_wips = {station: 0 for station in self.stations}
        self.last_events = {station: {'event': 'IDLE', 'pallet_id': 'None', 'timestamp': datetime.datetime.utcnow()} for station in self.stations}
        
        # Threading control
        self.data_processing_thread = None
        self.real_time_mode = False
        self.system_start_time = None
        self.last_timestamp = None
        
        # Create database if not exists
        self._initialize_database()
        
        logger.info("Five Station DB Interface initialized")
    
    def _initialize_database(self):
        """Initialize InfluxDB database and measurements"""
        try:
            databases = self.client.get_list_database()
            if not any(db['name'] == self.db_name for db in databases):
                self.client.create_database(self.db_name)
                logger.info(f"Created database: {self.db_name}")
            
            self.client.switch_database(self.db_name)
            logger.info(f"Connected to database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        logger.info(f"Connected to MQTT broker with code {rc}")
        
        # Subscribe to all station events
        topics = [
            f"{self.system_id}/station_event/+/all",
            f"{self.system_id}/station_wip/+/all", 
            f"{self.system_id}/station_status/+/all",
            f"{self.system_id}/system_time/master/all",
            f"{self.system_id}/system_status/master/all"
        ]
        
        for topic in topics:
            client.subscribe(topic, qos=2)
            logger.info(f"Subscribed to: {topic}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback - processes real-time station data"""
        try:
            topic_parts = msg.topic.split('/')
            if len(topic_parts) >= 4:
                system_id, context, source_id, target_id = topic_parts[:4]
                payload = msg.payload.decode("utf-8")
                timestamp = datetime.datetime.utcnow()
                
                if context == "station_event":
                    self._process_station_event(source_id, payload, timestamp)
                elif context == "station_wip":
                    self._process_station_wip(source_id, payload, timestamp)
                elif context == "station_status":
                    self._process_station_status(source_id, payload, timestamp)
                elif context == "system_time":
                    self._process_system_time(payload, timestamp)
                elif context == "system_status":
                    self._process_system_status(payload, timestamp)
                    
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _process_station_event(self, station_id, payload, timestamp):
        """Process station event data"""
        try:
            event_data = json.loads(payload)
            
            # Standardize event data format for DTDC compatibility
            processed_event = {
                'timestamp': timestamp,
                'station_id': station_id,
                'part_id': event_data.get('part_id'),
                'activity': event_data.get('activity', 'UNKNOWN'),
                'event_type': event_data.get('type', 'PROCESS'),
                'time_in_system': event_data.get('time', 0)
            }
            
            # Add to buffer for batch processing
            self.event_buffer.put(processed_event)
            
            # Calculate processing times if applicable
            if processed_event['activity'] in ['START', 'FINISH']:
                self._calculate_processing_time(processed_event)
            
            # Store immediately if in real-time mode
            if self.real_time_mode:
                self._store_event_to_db(processed_event)
                
        except Exception as e:
            logger.error(f"Error processing station event: {e}")
    
    def start_real_time_monitoring(self):
        """Start real-time data monitoring and processing"""
        if not self.real_time_mode:
            self.real_time_mode = True
            
            # Connect to MQTT broker
            try:
                self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
                self.mqtt_client.loop_start()
                logger.info("MQTT client started")
            except Exception as e:
                logger.error(f"Error connecting to MQTT broker: {e}")
                return False
            
            # Start data processing thread
            self.data_processing_thread = threading.Thread(target=self._data_processing_loop, daemon=True)
            self.data_processing_thread.start()
            logger.info("Real-time monitoring started")
            
        return True
    
    def stop_real_time_monitoring(self):
        """Stop real-time data monitoring"""
        if self.real_time_mode:
            self.real_time_mode = False
            
            # Stop MQTT client
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                logger.info("MQTT client stopped")
            except Exception as e:
                logger.error(f"Error stopping MQTT client: {e}")
            
            # Process remaining buffered data
            self._flush_buffers()
            logger.info("Real-time monitoring stopped")
    
    def get_timestamp_driven_data(self, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        """Get data for specific timestamp range - DTDC compatible"""
        try:
            # Query InfluxDB for events in timestamp range
            query = f"""
            SELECT * FROM eventlog_mqtt 
            WHERE time >= '{start_time.isoformat()}' 
            AND time <= '{end_time.isoformat()}'
            ORDER BY time ASC
            """
            
            result = self.client.query(query)
            
            if result:
                # Convert to DataFrame
                points = list(result.get_points())
                if points:
                    df = pd.DataFrame(points)
                    df['time'] = pd.to_datetime(df['time'])
                    return df
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting timestamp driven data: {e}")
            return pd.DataFrame()
    
    def query_processing_times(self, time_range: str, station_id: Optional[str] = None) -> pd.DataFrame:
        """Query processing times from database"""
        try:
            station_filter = f"AND station_id = '{station_id}'" if station_id else ""
            
            query = f"""
            SELECT * FROM real_perf 
            WHERE measures = 'processing_time_real' 
            AND time >= now() - {time_range}
            {station_filter}
            ORDER BY time ASC
            """
            
            result = self.client.query(query)
            
            if result:
                points = list(result.get_points())
                if points:
                    df = pd.DataFrame(points)
                    df['time'] = pd.to_datetime(df['time'])
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error querying processing times: {e}")
            return pd.DataFrame()
    
    def query_interdeparture_times(self, time_range: str, station_id: str = "S5") -> pd.DataFrame:
        """Query interdeparture times from database"""
        try:
            query = f"""
            SELECT * FROM real_perf 
            WHERE measures = 'interdeparture_time_real' 
            AND station_id = '{station_id}'
            AND time >= now() - {time_range}
            ORDER BY time ASC
            """
            
            result = self.client.query(query)
            
            if result:
                points = list(result.get_points())
                if points:
                    df = pd.DataFrame(points)
                    df['time'] = pd.to_datetime(df['time'])
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error querying interdeparture times: {e}")
            return pd.DataFrame()
    
    def get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state for DTDC dashboard"""
        try:
            state = {
                'mqtt_connected': self.real_time_mode,
                'stations_active': len([s for s in self.station_states.values() if s != 'IDLE']),
                'total_stations': len(self.stations),
                'station_states': self.station_states.copy(),
                'station_wips': self.station_wips.copy(),
                'system_start_time': self.system_start_time.isoformat() if self.system_start_time else None,
                'last_update': datetime.datetime.utcnow().isoformat()
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting system state: {e}")
            return {'error': str(e)}
    
    def _data_processing_loop(self):
        """Background thread for processing buffered data"""
        while self.real_time_mode:
            try:
                # Process events from buffer
                events_to_process = []
                while not self.event_buffer.empty() and len(events_to_process) < 100:
                    events_to_process.append(self.event_buffer.get_nowait())
                
                if events_to_process:
                    self._batch_store_events(events_to_process)
                
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
            except queue.Empty:
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                time.sleep(1)
    
    def _store_event_to_db(self, event):
        """Store single event to InfluxDB using DTDC-compatible format"""
        try:
            json_event = [{
                "measurement": "eventlog_mqtt",
                "time": event['timestamp'].isoformat(),
                "tags": {
                    "station_id": event['station_id'],
                    "activity": event['activity'],
                    "event_type": event['event_type']
                },
                "fields": {
                    "part_id": float(event['part_id']) if event['part_id'] else 0.0,
                    "time_in_system": float(event['time_in_system'])
                }
            }]
            
            self.client.write_points(json_event)
            
        except Exception as e:
            logger.error(f"Error storing event to DB: {e}")
    
    def _batch_store_events(self, events):
        """Store multiple events to InfluxDB in batch"""
        try:
            json_events = []
            for event in events:
                json_events.append({
                    "measurement": "eventlog_mqtt",
                    "time": event['timestamp'].isoformat(),
                    "tags": {
                        "station_id": event['station_id'],
                        "activity": event['activity'],
                        "event_type": event['event_type']
                    },
                    "fields": {
                        "part_id": float(event['part_id']) if event['part_id'] else 0.0,
                        "time_in_system": float(event['time_in_system'])
                    }
                })
            
            self.client.write_points(json_events)
            logger.debug(f"Stored {len(json_events)} events to database")
            
        except Exception as e:
            logger.error(f"Error batch storing events: {e}")
    
    def _calculate_processing_time(self, event):
        """Calculate processing time between START and FINISH events"""
        try:
            key = f"{event['station_id']}_{event['part_id']}"
            
            if event['activity'] == 'START':
                # Store start time
                if event['station_id'] not in self.processing_time_buffer:
                    self.processing_time_buffer[event['station_id']] = {}
                self.processing_time_buffer[event['station_id']][event['part_id']] = event['timestamp']
                
            elif event['activity'] == 'FINISH':
                # Calculate processing time
                if (event['station_id'] in self.processing_time_buffer and 
                    event['part_id'] in self.processing_time_buffer[event['station_id']]):
                    
                    start_time = self.processing_time_buffer[event['station_id']][event['part_id']]
                    processing_time = (event['timestamp'] - start_time).total_seconds()
                    
                    # Store processing time
                    self._store_processing_time_to_db({
                        'timestamp': event['timestamp'],
                        'station_id': event['station_id'],
                        'part_id': event['part_id'],
                        'processing_time': processing_time
                    })
                    
                    # Clean up buffer
                    del self.processing_time_buffer[event['station_id']][event['part_id']]
                    
        except Exception as e:
            logger.error(f"Error calculating processing time: {e}")
    
    def _store_processing_time_to_db(self, data):
        """Store processing time data to InfluxDB"""
        try:
            json_data = [{
                "measurement": "real_perf",
                "time": data['timestamp'].isoformat(),
                "tags": {
                    "measures": "processing_time_real",
                    "station_id": data['station_id'],
                    "part_id": str(data['part_id'])
                },
                "fields": {
                    "value": float(data['processing_time'])
                }
            }]
            
            self.client.write_points(json_data)
            
        except Exception as e:
            logger.error(f"Error storing processing time: {e}")
    
    def _flush_buffers(self):
        """Flush all remaining data in buffers to database"""
        try:
            # Process remaining events
            events_to_process = []
            while not self.event_buffer.empty():
                events_to_process.append(self.event_buffer.get_nowait())
            
            if events_to_process:
                self._batch_store_events(events_to_process)
                logger.info(f"Flushed {len(events_to_process)} events to database")
                
        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
    
    # Additional helper methods for DTDC compatibility
    def _process_station_wip(self, station_id, payload, timestamp):
        """Process station WIP data"""
        try:
            wip_count = int(payload)
            self.station_wips[station_id] = wip_count
            
            if self.real_time_mode:
                json_data = [{
                    "measurement": "station_wip_real",
                    "time": timestamp.isoformat(),
                    "tags": {"station_id": station_id},
                    "fields": {"wip_count": float(wip_count)}
                }]
                self.client.write_points(json_data)
                
        except Exception as e:
            logger.error(f"Error processing station WIP: {e}")
    
    def _process_station_status(self, station_id, payload, timestamp):
        """Process station status data"""
        try:
            status = payload.upper()
            self.station_states[station_id] = status
            
            if self.real_time_mode:
                json_data = [{
                    "measurement": "station_status_real",
                    "time": timestamp.isoformat(),
                    "tags": {"station_id": station_id},
                    "fields": {"status": status}
                }]
                self.client.write_points(json_data)
                
        except Exception as e:
            logger.error(f"Error processing station status: {e}")
    
    def _process_system_time(self, payload, timestamp):
        """Process system time synchronization"""
        try:
            system_time = datetime.datetime.fromisoformat(payload)
            self.system_start_time = system_time
            self.last_timestamp = timestamp
            logger.info(f"System time synchronized: {system_time}")
            
        except Exception as e:
            logger.error(f"Error processing system time: {e}")
    
    def _process_system_status(self, payload, timestamp):
        """Process system status changes"""
        try:
            status = payload.upper()
            if status == "START":
                self.start_real_time_monitoring()
            elif status == "STOP":
                self.stop_real_time_monitoring()
            
            logger.info(f"System status changed to: {status}")
            
        except Exception as e:
            logger.error(f"Error processing system status: {e}")


    def store_event(self, event_data: Dict[str, Any]) -> bool:
        """Store a single event in the database"""
        try:
            # Update station state
            station_id = event_data.get('station_id', 'Unknown')
            event_type = event_data.get('event', 'Unknown')
            pallet_id = event_data.get('pallet_id', 'Unknown')
            timestamp = event_data.get('timestamp', datetime.datetime.utcnow())
            
            # Update internal state
            if station_id in self.stations:
                self.station_states[station_id] = event_type
                self.last_events[station_id] = {
                    'event': event_type,
                    'pallet_id': pallet_id,
                    'timestamp': timestamp
                }
            
            # Store in database
            json_data = [{
                "measurement": "station_events",
                "time": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "tags": {
                    "station_id": station_id,
                    "pallet_id": str(pallet_id)
                },
                "fields": {
                    "event": event_type,
                    "value": 1.0
                }
            }]
            
            self.client.write_points(json_data)
            self.logger.debug(f"Stored event: {station_id} - {event_type} - Pallet {pallet_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing event: {e}")
            return False

    def get_last_event_per_station(self) -> Dict[str, Dict[str, Any]]:
        """Get the last event for each station."""
        last_events = {}
        try:
            for i in range(1, 6):
                station_id = f"S{i}"
                query = f"SELECT last(*) FROM \"station_events\" WHERE \"station_id\" = \'{station_id}\'"
                result = self.client.query(query)
                if result:
                    point = list(result.get_points())[0]
                    last_events[station_id] = {
                        "event": point["last_event"],
                        "pallet_id": point["last_pallet_id"],
                        "timestamp": point["time"]
                    }
        except Exception as e:
            self.logger.error(f"Error getting last event per station: {e}")
        return last_events

