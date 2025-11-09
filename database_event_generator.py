"""
Database Event Generator - Simulates Physical Five-Station System

This module reads historical data from the database and generates real-time events
to make the DTDC system behave as if the physical five-station system is running.

Key Features:
- Reads events from database (not files)
- Generates real-time event streams
- Simulates physical system behavior
- Creates live data for DTDC processing

Author: AI Assistant
Date: 2025-09-21
"""

import logging
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import queue

logger = logging.getLogger("DatabaseEventGenerator")

class DatabaseEventGenerator:
    """
    Generates real-time events from database to simulate physical system operation
    """
    
    def __init__(self, db_interface, historical_data_file=None):
        """Initialize the database event generator"""
        self.db_interface = db_interface
        self.historical_data_file = historical_data_file or "event_log_250725_103848.csv"
        
        # Event generation control
        self.is_running = False
        self.generation_thread = None
        self.event_queue = queue.Queue()
        
        # Timing control
        self.speed_multiplier = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        self.current_event_index = 0
        
        # Event data
        self.events_data = None
        self.total_events = 0
        
        # Statistics
        self.events_generated = 0
        self.start_time = None
        
        # Load historical data
        self.load_historical_data()
        
        logger.info(f"Database Event Generator initialized with {self.total_events} events")
    
    def load_historical_data(self):
        """Load historical event data from CSV file"""
        try:
            logger.info(f"Loading historical data from {self.historical_data_file}")
            
            # Read CSV file with proper headers
            self.events_data = pd.read_csv(
                self.historical_data_file,
                skiprows=1,  # Skip header row
                header=None,
                names=['timestamp', 'station', 'pallet', 'event']
            )
            
            # Convert timestamp to datetime
            self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
            
            # Sort by timestamp
            self.events_data = self.events_data.sort_values('timestamp').reset_index(drop=True)
            
            self.total_events = len(self.events_data)
            
            logger.info(f"Loaded {self.total_events} events from {self.events_data['timestamp'].min()} to {self.events_data['timestamp'].max()}")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            # Create dummy data if file loading fails
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy event data for testing"""
        logger.info("Creating dummy event data for testing")
        
        stations = ['S1', 'S2', 'S3', 'S4', 'S5']
        events = ['LOAD', 'PROCESS', 'UNLOAD', 'TRANSFER', 'BLOCK']
        
        dummy_events = []
        base_time = datetime.now()
        
        for i in range(100):  # Create 100 dummy events
            event_time = base_time + timedelta(seconds=i * 10)  # Event every 10 seconds
            station = stations[i % len(stations)]
            event = events[i % len(events)]
            pallet = f"P{(i % 12) + 1:02d}"  # 12 pallets cycling
            
            dummy_events.append({
                'timestamp': event_time,
                'station': station,
                'pallet': pallet,
                'event': event
            })
        
        self.events_data = pd.DataFrame(dummy_events)
        self.total_events = len(self.events_data)
    
    def start_event_generation(self, speed_multiplier=1.0):
        """Start generating events from database"""
        if self.is_running:
            logger.warning("Event generation already running")
            return False
        
        try:
            logger.info(f"Starting event generation at {speed_multiplier}x speed")
            
            self.speed_multiplier = speed_multiplier
            self.is_running = True
            self.events_generated = 0
            self.current_event_index = 0
            self.start_time = datetime.now()
            
            # Start generation thread
            self.generation_thread = threading.Thread(target=self._generate_events_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            
            logger.info("Event generation started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting event generation: {e}")
            self.is_running = False
            return False
    
    def stop_event_generation(self):
        """Stop generating events"""
        if not self.is_running:
            return True
        
        logger.info("Stopping event generation")
        self.is_running = False
        
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)
        
        logger.info(f"Event generation stopped. Generated {self.events_generated} events")
        return True
    
    def _generate_events_loop(self):
        """Main event generation loop"""
        logger.info("Event generation loop started")
        
        if self.events_data is None or len(self.events_data) == 0:
            logger.error("No event data available for generation")
            return
        
        # Calculate time differences between events
        time_diffs = []
        for i in range(1, len(self.events_data)):
            diff = (self.events_data.iloc[i]['timestamp'] - self.events_data.iloc[i-1]['timestamp']).total_seconds()
            time_diffs.append(max(diff, 0.1))  # Minimum 0.1 second between events
        
        # Generate events
        for i in range(len(self.events_data)):
            if not self.is_running:
                break
            
            try:
                # Get current event
                event_row = self.events_data.iloc[i]
                
                # Create event data
                event_data = {
                    'timestamp': datetime.now().isoformat(),
                    'original_timestamp': event_row['timestamp'].isoformat(),
                    'station': event_row['station'],
                    'pallet': event_row['pallet'],
                    'event': event_row['event'],
                    'event_index': i,
                    'total_events': self.total_events
                }
                
                # Store event in database
                self._store_event_in_database(event_data)
                
                # Add to queue for real-time processing
                self.event_queue.put(event_data)
                
                # Update statistics
                self.events_generated += 1
                self.current_event_index = i
                
                # Log progress
                if i % 100 == 0:
                    logger.info(f"Generated {i}/{self.total_events} events ({(i/self.total_events)*100:.1f}%)")
                
                # Wait for next event (adjusted by speed multiplier)
                if i < len(time_diffs):
                    wait_time = time_diffs[i] / self.speed_multiplier
                    time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error generating event {i}: {e}")
                continue
        
        logger.info(f"Event generation completed. Generated {self.events_generated} events")
        self.is_running = False
    
    def _store_event_in_database(self, event_data):
        """Store generated event in database"""
        try:
            # Store in InfluxDB
            self.db_interface.store_event(
                timestamp=event_data['timestamp'],
                station_id=event_data['station'],
                pallet_id=event_data['pallet'],
                event_type=event_data['event'],
                additional_data={
                    'original_timestamp': event_data['original_timestamp'],
                    'event_index': event_data['event_index'],
                    'generated': True
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing event in database: {e}")
    
    def get_next_event(self, timeout=1.0):
        """Get the next generated event from queue"""
        try:
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_generation_status(self):
        """Get current generation status"""
        return {
            'is_running': self.is_running,
            'events_generated': self.events_generated,
            'current_event_index': self.current_event_index,
            'total_events': self.total_events,
            'progress_percent': (self.current_event_index / self.total_events * 100) if self.total_events > 0 else 0,
            'speed_multiplier': self.speed_multiplier,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'queue_size': self.event_queue.qsize()
        }
    
    def get_current_station_states(self):
        """Get current state of all stations based on last events"""
        try:
            # Get last event for each station from database
            last_events = self.db_interface.get_last_event_per_station()
            
            # Format for display
            station_states = {}
            for station_id in ['S1', 'S2', 'S3', 'S4', 'S5']:
                if station_id in last_events:
                    event_info = last_events[station_id]
                    station_states[station_id] = {
                        'status': event_info.get('event', 'IDLE'),
                        'pallet': event_info.get('pallet', 'None'),
                        'timestamp': event_info.get('timestamp', 'N/A'),
                        'last_update': datetime.now().strftime('%H:%M:%S')
                    }
                else:
                    station_states[station_id] = {
                        'status': 'IDLE',
                        'pallet': 'None',
                        'timestamp': 'N/A',
                        'last_update': 'N/A'
                    }
            
            return station_states
            
        except Exception as e:
            logger.error(f"Error getting station states: {e}")
            return {}
    
    def reset_generation(self):
        """Reset generation to start from beginning"""
        self.stop_event_generation()
        self.current_event_index = 0
        self.events_generated = 0
        
        # Clear event queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Event generation reset")
    
    def set_speed_multiplier(self, multiplier):
        """Set the speed multiplier for event generation"""
        self.speed_multiplier = max(0.1, min(multiplier, 10.0))  # Limit between 0.1x and 10x
        logger.info(f"Speed multiplier set to {self.speed_multiplier}x")
    
    def get_events_in_time_range(self, start_time, end_time):
        """Get events from database in specified time range"""
        try:
            return self.db_interface.get_events_in_time_range(start_time, end_time)
        except Exception as e:
            logger.error(f"Error getting events in time range: {e}")
            return []
