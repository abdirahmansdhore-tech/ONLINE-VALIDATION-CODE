"""
Database Integrated Data Manager

This extends the original DataManager to work with database events while preserving
all original functionality for file-based workflows.

Key Features:
- Preserves ALL original DataManager functionality
- Adds database event integration for online mode
- Seamless switching between online (database) and offline (file) modes
- Real-time station status from database events

Author: AI Assistant
Date: 2025-09-21
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
from typing import Dict, List, Optional, Any

# Import original data manager
from logs.data_manager import DataManager
from five_station_db_interface import FiveStationDBInterface
from database_event_generator import DatabaseEventGenerator

logger = logging.getLogger("DatabaseIntegratedDataManager")

class DatabaseIntegratedDataManager(DataManager):
    """
    Extended DataManager that integrates database functionality while preserving original features
    """
    
    def __init__(self, config_file='config/system_config.json'):
        """Initialize with original functionality plus database integration"""
        
        # Initialize original DataManager
        super().__init__(config_file)
        
        # Database integration
        self.database_mode = False
        self.five_station_db = None
        self.event_generator = None
        
        # Station status tracking
        self.station_status = {
            'S1': {'status': 'IDLE', 'pallet': 'None', 'timestamp': 'N/A'},
            'S2': {'status': 'IDLE', 'pallet': 'None', 'timestamp': 'N/A'},
            'S3': {'status': 'IDLE', 'pallet': 'None', 'timestamp': 'N/A'},
            'S4': {'status': 'IDLE', 'pallet': 'None', 'timestamp': 'N/A'},
            'S5': {'status': 'IDLE', 'pallet': 'None', 'timestamp': 'N/A'}
        }
        
        # Event processing
        self.event_processing_active = False
        self.event_processing_thread = None
        self.processed_events_count = 0
        
        # Initialize database components
        self._initialize_database_components()
        
        logger.info("Database Integrated Data Manager initialized")
    
    def _initialize_database_components(self):
        """Initialize database components"""
        try:
            # Initialize database interface
            self.five_station_db = FiveStationDBInterface(
                db_ip='localhost',
                db_name='five_station_system',
                db_port=8086,
                mqtt_broker='localhost',
                mqtt_port=1883
            )
            
            # Initialize event generator
            self.event_generator = DatabaseEventGenerator(
                db_interface=self.five_station_db,
                historical_data_file="event_log_250725_103848.csv"
            )
            
            logger.info("Database components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database components: {e}")
            self.five_station_db = None
            self.event_generator = None
    
    # =============================================================================
    # ORIGINAL FUNCTIONALITY (Preserved - all methods from parent class work)
    # =============================================================================
    
    def get_system_status(self):
        """Enhanced system status including database mode"""
        try:
            # Get original status
            original_status = super().get_system_status()
            
            # Add database-specific status
            database_status = {
                'database_mode': self.database_mode,
                'database_connected': self.five_station_db is not None,
                'event_processing_active': self.event_processing_active,
                'processed_events_count': self.processed_events_count,
                'station_status': self.station_status.copy()
            }
            
            # Combine statuses
            original_status['database_integration'] = database_status
            
            return original_status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    # =============================================================================
    # DATABASE MODE FUNCTIONALITY
    # =============================================================================
    
    def switch_to_database_mode(self):
        """Switch to database mode (online simulation)"""
        try:
            logger.info("Switching to database mode")
            
            if self.five_station_db is None:
                logger.error("Database interface not available")
                return False
            
            self.database_mode = True
            self.data_source = 'online'
            
            # Start event generation and processing
            success = self.start_database_event_processing()
            
            if success:
                logger.info("Successfully switched to database mode")
                return True
            else:
                logger.error("Failed to start database event processing")
                self.database_mode = False
                self.data_source = 'offline'
                return False
                
        except Exception as e:
            logger.error(f"Error switching to database mode: {e}")
            self.database_mode = False
            return False
    
    def switch_to_file_mode(self):
        """Switch back to original file mode"""
        try:
            logger.info("Switching to file mode")
            
            # Stop database processing
            self.stop_database_event_processing()
            
            self.database_mode = False
            self.data_source = 'offline'
            
            logger.info("Successfully switched to file mode")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to file mode: {e}")
            return False
    
    def start_database_event_processing(self):
        """Start processing events from database"""
        try:
            if self.event_processing_active:
                logger.warning("Event processing already active")
                return True
            
            # Start event generation
            success = self.event_generator.start_event_generation(speed_multiplier=1.0)
            
            if success:
                # Start processing thread
                self.event_processing_active = True
                self.event_processing_thread = threading.Thread(target=self._process_database_events)
                self.event_processing_thread.daemon = True
                self.event_processing_thread.start()
                
                logger.info("Database event processing started")
                return True
            else:
                logger.error("Failed to start event generation")
                return False
                
        except Exception as e:
            logger.error(f"Error starting database event processing: {e}")
            return False
    
    def stop_database_event_processing(self):
        """Stop processing events from database"""
        try:
            logger.info("Stopping database event processing")
            
            # Stop event generation
            if self.event_generator:
                self.event_generator.stop_event_generation()
            
            # Stop processing thread
            self.event_processing_active = False
            
            if self.event_processing_thread and self.event_processing_thread.is_alive():
                self.event_processing_thread.join(timeout=5)
            
            logger.info("Database event processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping database event processing: {e}")
            return False
    
    def _process_database_events(self):
        """Process events from database in real-time"""
        logger.info("Database event processing thread started")
        
        while self.event_processing_active:
            try:
                # Get next event from generator
                event = self.event_generator.get_next_event(timeout=1.0)
                
                if event:
                    # Update station status
                    self._update_station_status(event)
                    
                    # Process event for KPI calculation
                    self._process_event_for_kpis(event)
                    
                    self.processed_events_count += 1
                    
                    # Log progress
                    if self.processed_events_count % 50 == 0:
                        logger.info(f"Processed {self.processed_events_count} database events")
                
            except Exception as e:
                logger.error(f"Error processing database event: {e}")
                time.sleep(1)
        
        logger.info("Database event processing thread stopped")
    
    def _update_station_status(self, event):
        """Update station status from event"""
        try:
            station_id = event['station']
            
            if station_id in self.station_status:
                self.station_status[station_id] = {
                    'status': event['event'],
                    'pallet': event['pallet'],
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
        except Exception as e:
            logger.error(f"Error updating station status: {e}")
    
    def _process_event_for_kpis(self, event):
        """Process event for KPI calculation (integrates with original DTDC processing)"""
        try:
            # Add event to buffer for batch processing
            with self._event_buffer_lock:
                self.event_buffer.append({
                    'timestamp': event['timestamp'],
                    'station': event['station'],
                    'pallet': event['pallet'],
                    'event': event['event']
                })
                
                # Trigger processing if buffer is full
                if len(self.event_buffer) >= self.batch_size:
                    self._process_event_buffer()
                    
        except Exception as e:
            logger.error(f"Error processing event for KPIs: {e}")
    
    def _process_event_buffer(self):
        """Process buffered events for KPI generation"""
        try:
            if not self.event_buffer:
                return
            
            # Convert events to DataFrame
            events_df = pd.DataFrame(self.event_buffer)
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            
            # Generate KPIs from events (this integrates with original DTDC processing)
            kpis = self._calculate_kpis_from_events(events_df)
            
            # Store KPIs in format compatible with original system
            self._store_kpis_for_validation(kpis)
            
            # Clear buffer
            self.event_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing event buffer: {e}")
    
    def _calculate_kpis_from_events(self, events_df):
        """Calculate KPIs from event data"""
        try:
            kpis = {}
            
            # Throughput calculation
            unload_events = events_df[events_df['event'] == 'UNLOAD']
            if len(unload_events) > 0:
                time_span = (events_df['timestamp'].max() - events_df['timestamp'].min()).total_seconds() / 3600
                if time_span > 0:
                    kpis['throughput'] = len(unload_events) / time_span
            
            # Processing times per station
            for station in ['S1', 'S2', 'S3', 'S4', 'S5']:
                station_events = events_df[events_df['station'] == station]
                
                # Calculate average processing time
                processing_times = []
                for pallet in station_events['pallet'].unique():
                    pallet_events = station_events[station_events['pallet'] == pallet].sort_values('timestamp')
                    
                    load_events = pallet_events[pallet_events['event'] == 'LOAD']
                    unload_events = pallet_events[pallet_events['event'] == 'UNLOAD']
                    
                    for _, load_event in load_events.iterrows():
                        subsequent_unloads = unload_events[unload_events['timestamp'] > load_event['timestamp']]
                        if len(subsequent_unloads) > 0:
                            unload_event = subsequent_unloads.iloc[0]
                            processing_time = (unload_event['timestamp'] - load_event['timestamp']).total_seconds()
                            processing_times.append(processing_time)
                
                if processing_times:
                    kpis[f'avg_processing_time_{station}'] = np.mean(processing_times)
                    kpis[f'std_processing_time_{station}'] = np.std(processing_times)
            
            # Interdeparture times
            interdeparture_times = []
            for station in ['S1', 'S2', 'S3', 'S4', 'S5']:
                station_unloads = events_df[
                    (events_df['station'] == station) & 
                    (events_df['event'] == 'UNLOAD')
                ].sort_values('timestamp')
                
                if len(station_unloads) > 1:
                    for i in range(1, len(station_unloads)):
                        interdep_time = (station_unloads.iloc[i]['timestamp'] - 
                                       station_unloads.iloc[i-1]['timestamp']).total_seconds()
                        interdeparture_times.append(interdep_time)
            
            if interdeparture_times:
                kpis['avg_interdeparture_time'] = np.mean(interdeparture_times)
                kpis['std_interdeparture_time'] = np.std(interdeparture_times)
            
            return kpis
            
        except Exception as e:
            logger.error(f"Error calculating KPIs from events: {e}")
            return {}
    
    def _store_kpis_for_validation(self, kpis):
        """Store KPIs in format compatible with original validation system"""
        try:
            # Create KPI data in format expected by original system
            kpi_data = {
                'timestamp': datetime.now().isoformat(),
                'kpis': kpis,
                'source': 'database_events'
            }
            
            # Store in _real_kpi_data for validation engine access
            if not hasattr(self, '_real_kpi_data') or self._real_kpi_data is None:
                self._real_kpi_data = []
            
            self._real_kpi_data.append(kpi_data)
            
            # Keep only recent data (last 100 entries)
            if len(self._real_kpi_data) > 100:
                self._real_kpi_data = self._real_kpi_data[-100:]
            
        except Exception as e:
            logger.error(f"Error storing KPIs for validation: {e}")
    
    # =============================================================================
    # ENHANCED METHODS FOR DATABASE INTEGRATION
    # =============================================================================
    
    def get_station_status(self):
        """Get current station status"""
        return self.station_status.copy()
    
    def get_real_kpi_data(self):
        """Get real KPI data (enhanced to include database-generated KPIs)"""
        if self.database_mode and hasattr(self, '_real_kpi_data') and self._real_kpi_data:
            # Return database-generated KPIs
            return self._real_kpi_data
        else:
            # Fall back to original file-based KPI data
            return super().get_real_kpi_data()
    
    def is_database_mode_active(self):
        """Check if database mode is active"""
        return self.database_mode and self.event_processing_active
    
    def start_campaign(self, campaign_config=None):
        """Start data preparation campaign (works in both database and file modes)"""
        try:
            logger.info("Starting data preparation campaign")
            
            if self.database_mode:
                # Database mode: prepare data from database events
                return self._start_database_campaign(campaign_config)
            else:
                # File mode: use original functionality
                return super().start_campaign(campaign_config)
                
        except Exception as e:
            logger.error(f"Error starting campaign: {e}")
            return False
    
    def _start_database_campaign(self, campaign_config=None):
        """Start campaign using database events"""
        try:
            logger.info("Starting database-driven data preparation campaign")
            
            # Ensure database event processing is active
            if not self.event_processing_active:
                success = self.start_database_event_processing()
                if not success:
                    logger.error("Failed to start database event processing")
                    return False
            
            # Set campaign parameters
            self.batch_size = 50  # Process events in batches
            self.event_buffer = []
            self._event_buffer_lock = threading.Lock()
            
            # Initialize KPI data storage
            self._real_kpi_data = []
            
            logger.info("Database campaign started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting database campaign: {e}")
            return False
    
    def prepare_data(self, data_source=None):
        """Prepare data (enhanced for database mode)"""
        try:
            if self.database_mode:
                # Database mode: data is prepared continuously from events
                logger.info("Data preparation active in database mode")
                return True
            else:
                # File mode: use original functionality
                return super().prepare_data(data_source)
                
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def save_processed_data_to_files(self):
        """Save processed database data to files for compatibility"""
        try:
            logger.info("Saving processed database data to files")
            
            # Create output directory if it doesn't exist
            import os
            output_dir = "database_outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save processed events to CSV
            if hasattr(self, 'event_buffer') and self.event_buffer:
                events_df = pd.DataFrame(self.event_buffer)
                events_file = os.path.join(output_dir, f"processed_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                events_df.to_csv(events_file, index=False)
                logger.info(f"Saved {len(events_df)} events to {events_file}")
            
            # Save KPI data to CSV
            if hasattr(self, '_real_kpi_data') and self._real_kpi_data:
                kpi_records = []
                for kpi_entry in self._real_kpi_data:
                    record = {'timestamp': kpi_entry['timestamp'], 'source': kpi_entry['source']}
                    record.update(kpi_entry['kpis'])
                    kpi_records.append(record)
                
                kpi_df = pd.DataFrame(kpi_records)
                kpi_file = os.path.join(output_dir, f"system_kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                kpi_df.to_csv(kpi_file, index=False)
                logger.info(f"Saved {len(kpi_df)} KPI records to {kpi_file}")
            
            # Save station status to JSON
            status_file = os.path.join(output_dir, f"station_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(status_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'station_status': self.station_status,
                    'processed_events_count': self.processed_events_count,
                    'database_mode': self.database_mode
                }, f, indent=2)
            logger.info(f"Saved station status to {status_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data to files: {e}")
            return False
    
    def _store_kpis_for_validation(self, kpis):
        """Store KPIs in format compatible with original validation system (enhanced with file saving)"""
        try:
            # Create KPI data in format expected by original system
            kpi_data = {
                'timestamp': datetime.now().isoformat(),
                'kpis': kpis,
                'source': 'database_events'
            }
            
            # Store in _real_kpi_data for validation engine access
            if not hasattr(self, '_real_kpi_data') or self._real_kpi_data is None:
                self._real_kpi_data = []
            
            self._real_kpi_data.append(kpi_data)
            
            # Keep only recent data (last 100 entries)
            if len(self._real_kpi_data) > 100:
                self._real_kpi_data = self._real_kpi_data[-100:]
            
            # Auto-save to files every 10 KPI entries
            if len(self._real_kpi_data) % 10 == 0:
                self.save_processed_data_to_files()
            
        except Exception as e:
            logger.error(f"Error storing KPIs for validation: {e}")
    
    def export_database_data_to_csv(self, time_range="1h"):
        """Export database data to CSV files"""
        try:
            logger.info(f"Exporting database data for last {time_range}")
            
            if not self.five_station_db:
                logger.error("Database interface not available")
                return False
            
            # Create export directory
            import os
            export_dir = "database_exports"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export events
            events_df = self.five_station_db.get_timestamp_driven_data(time_range)
            if not events_df.empty:
                events_file = os.path.join(export_dir, f"events_export_{timestamp}.csv")
                events_df.to_csv(events_file, index=False)
                logger.info(f"Exported {len(events_df)} events to {events_file}")
            
            # Export processing times
            processing_df = self.five_station_db.query_processing_times(time_range)
            if not processing_df.empty:
                processing_file = os.path.join(export_dir, f"processing_times_export_{timestamp}.csv")
                processing_df.to_csv(processing_file, index=False)
                logger.info(f"Exported {len(processing_df)} processing time records to {processing_file}")
            
            # Export interdeparture times
            interdep_df = self.five_station_db.query_interdeparture_times(time_range)
            if not interdep_df.empty:
                interdep_file = os.path.join(export_dir, f"interdeparture_times_export_{timestamp}.csv")
                interdep_df.to_csv(interdep_file, index=False)
                logger.info(f"Exported {len(interdep_df)} interdeparture time records to {interdep_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting database data: {e}")
            return False
