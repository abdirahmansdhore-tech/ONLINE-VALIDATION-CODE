import os
import time
import threading
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Any
import queue
from threading import Lock

logger = logging.getLogger('SystemController')


class SystemController:
    def __init__(self, data_manager, validation_engine, digital_model_interface, calibration_engine,
                 issue_identification, config: Dict[str, Any]):
        self.data_manager = data_manager
        self.validation_engine = validation_engine
        self.digital_model_interface = digital_model_interface
        self.calibration_engine = calibration_engine
        self.issue_identification = issue_identification
        self.config = config

        # Thread-safe state management
        self._state_lock = Lock()

        # SEPARATED WORKFLOWS STATE MANAGEMENT
        self.data_preparation_active = False
        self.validation_campaign_active = False
        self.arena_integration_enabled = False

        # Workflow threads
        self.data_preparation_thread = None
        self.validation_campaign_thread = None

        # Configuration and intervals
        self.update_interval = config.get('update_interval', 5.0)
        self.validation_count = 0
        self.last_validation_results = {}

        # Error handling and recovery
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_backoff_time = 10  # seconds

        # Statistics tracking (separated by workflow)
        self.data_preparation_statistics = {
            'start_time': None,
            'files_processed': 0,
            'kpi_files_generated': 0,  # NEW: Track KPI file generation
            'last_error': None,
            'status': 'inactive'
        }

        self.validation_statistics = {
            'start_time': None,
            'windows_processed': 0,
            'validation_failures': 0,
            'calibrations_performed': 0,
            'last_error': None,
            'status': 'inactive'
        }

        # Configuration - load once and cache
        self.validation_config = config.get('validation_config', {
            'batch_size': 50,
            'validation_mode': 'kpi',  # 'kpi', 'event', or 'both'
            'window_stride': 50,
            'enable_arena_integration': False,  # Now manual control only
            'enable_kpi_generation': True  # NEW: Control KPI generation
        })

        # Cache for system configuration
        self._system_config = None
        self._config_last_loaded = None

        logger.info("SystemController initialized with separated workflow management and enhanced file format support")

    def connect_components(self):
        """Initialize connections to core components (NO ARENA AUTO-CONNECTION)"""
        try:
            # REMOVED: Arena auto-connection - now manual only
            logger.info("Components connected to SystemController (Arena connection is manual)")
            return {'success': True, 'message': 'Core components connected (Arena manual)'}

        except Exception as e:
            logger.error(f"Error connecting components: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self):
        """Get comprehensive thread-safe system status"""
        with self._state_lock:
            status = {
                # Separated workflow states
                'data_preparation_active': self.data_preparation_active,
                'validation_campaign_active': self.validation_campaign_active,
                'arena_integration_enabled': self.arena_integration_enabled,

                # Legacy compatibility
                'campaign_active': self.data_preparation_active or self.validation_campaign_active,

                # Validation info
                'validation_count': self.validation_count,
                'last_validation': self.last_validation_results.copy(),
                'update_interval': self.update_interval,

                # Separated statistics
                'data_preparation_statistics': self.data_preparation_statistics.copy(),
                'validation_statistics': self.validation_statistics.copy(),

                # Error tracking
                'consecutive_errors': self.consecutive_errors,

                # Arena status (if digital model interface available)
                'arena_integration': self.arena_integration_enabled
            }

            # Add Arena status if integration enabled and interface available
            if self.arena_integration_enabled and self.digital_model_interface:
                try:
                    status['arena_status'] = self.digital_model_interface.get_status()
                except Exception as e:
                    if "CoInitialize" in str(e):
                        logger.debug(f"COM handled by digital_model_interface: {e}")
                        status['arena_status'] = {'error': 'COM initialization handled internally'}
                    else:
                        logger.warning(f"Arena status error: {e}")
                        status['arena_status'] = {'error': str(e)}

            return status

    # =============================================================================
    # DATA PREPARATION WORKFLOW (Start/Stop buttons)
    # =============================================================================

    def start_data_preparation(self):
        """Start data preparation workflow - initializes components and prepares input data with KPI generation"""
        with self._state_lock:
            if self.data_preparation_active:
                return {'success': False, 'message': 'Data preparation already active'}

            # Reset statistics
            self.data_preparation_statistics = {
                'start_time': datetime.now(),
                'files_processed': 0,
                'kpi_files_generated': 0,  # NEW: Track KPI generation
                'last_error': None,
                'status': 'active'
            }
            self.consecutive_errors = 0

            self.data_preparation_active = True
            self.data_preparation_thread = threading.Thread(target=self._data_preparation_loop, daemon=True)
            self.data_preparation_thread.start()

        logger.info("Data preparation workflow started with KPI generation enabled")
        return {'success': True, 'message': 'Data preparation started with KPI generation'}

    def stop_data_preparation(self):
        """Stop data preparation workflow"""
        with self._state_lock:
            if not self.data_preparation_active:
                return {'success': True, 'message': 'Data preparation not active'}

            self.data_preparation_active = False
            if self.data_preparation_statistics:
                self.data_preparation_statistics['status'] = 'stopped'

        # Wait for thread to finish
        if self.data_preparation_thread and self.data_preparation_thread.is_alive():
            self.data_preparation_thread.join(timeout=10)
            if self.data_preparation_thread.is_alive():
                logger.warning("Data preparation thread did not stop gracefully")

        logger.info("Data preparation workflow stopped")
        return {'success': True, 'message': 'Data preparation stopped'}

    def _data_preparation_loop(self):
        """Data preparation workflow loop with integrated KPI generation"""
        logger.info("Data preparation loop started with KPI generation")

        try:
            # Load configuration
            config = self._get_system_config()
            if not config:
                logger.error("Cannot start data preparation - no valid configuration")
                return

            station_config = config.get("station_config", {})
            stations = station_config.get("stations", [])
            dist_configs = config.get("dist_configs", {})

            # Start data manager campaign
            self.data_manager.start_campaign()
            logger.info("Data manager campaign started")

            window_size = self.validation_config.get('batch_size', 50)
            current_window_start = 0

            while self._is_data_preparation_active():
                try:
                    window_start_time = time.time()
                    logger.info(f"Processing data window starting at position {current_window_start}")

                    # === [1] Extract raw events for this window ===
                    window_events = self.data_manager.get_raw_events(
                        window_size=window_size,
                        start_position=current_window_start
                    )

                    if window_events is None or len(window_events) == 0:
                        logger.info(f"No raw events for window starting at {current_window_start}")
                        if not self._has_more_data(current_window_start, window_size):
                            logger.info("No more data available, restarting from beginning")
                            current_window_start = 0
                            continue
                        else:
                            break

                    # === [2] Process window data with KPI generation ===
                    processing_success = self._process_window_data_with_kpis(window_events, stations, dist_configs,
                                                                             window_size)
                    if processing_success:
                        with self._state_lock:
                            self.data_preparation_statistics['files_processed'] += 1
                            self.data_preparation_statistics['kpi_files_generated'] += 1  # NEW: Track KPI generation
                    else:
                        logger.warning(f"Failed to process data for window {current_window_start}")

                    # === Move to next window ===
                    current_window_start += self.validation_config.get('window_stride', window_size)

                    # Reset error counter on successful iteration
                    self.consecutive_errors = 0

                    # === Adaptive sleep based on processing time ===
                    processing_time = time.time() - window_start_time
                    sleep_time = max(0, self.update_interval - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    self._handle_data_preparation_error(e, current_window_start)
                    current_window_start += window_size  # Skip problematic window

        except Exception as e:
            logger.error(f"Fatal error in data preparation loop: {e}")
            with self._state_lock:
                self.data_preparation_statistics['last_error'] = str(e)
                self.data_preparation_statistics['status'] = 'error'

        finally:
            logger.info("Data preparation loop ended")

    def _is_data_preparation_active(self):
        """Thread-safe check for data preparation status"""
        with self._state_lock:
            return self.data_preparation_active

    def _handle_data_preparation_error(self, error, window_start):
        """Handle errors in data preparation loop"""
        self.consecutive_errors += 1
        error_msg = f"Error in data preparation window {window_start}: {error}"
        logger.error(error_msg)

        with self._state_lock:
            self.data_preparation_statistics['last_error'] = error_msg

        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping data preparation")
            with self._state_lock:
                self.data_preparation_active = False
                self.data_preparation_statistics['status'] = 'error'
            return

        # Exponential backoff
        backoff_time = min(self.error_backoff_time * (2 ** (self.consecutive_errors - 1)), 60)
        logger.info(f"Backing off for {backoff_time} seconds after {self.consecutive_errors} consecutive errors")
        time.sleep(backoff_time)

    # =============================================================================
    # VALIDATION CAMPAIGN WORKFLOW (Start Validation button)
    # =============================================================================

    def start_validation_campaign(self):
        """Start validation campaign workflow - runs validation and saves results"""
        with self._state_lock:
            if self.validation_campaign_active:
                return {'success': False, 'message': 'Validation campaign already active'}

            # Reset statistics
            self.validation_statistics = {
                'start_time': datetime.now(),
                'windows_processed': 0,
                'validation_failures': 0,
                'calibrations_performed': 0,
                'last_error': None,
                'status': 'active'
            }

            self.validation_campaign_active = True
            self.validation_campaign_thread = threading.Thread(target=self._validation_campaign_loop, daemon=True)
            self.validation_campaign_thread.start()

        logger.info("Validation campaign started")
        return {'success': True, 'message': 'Validation campaign started'}

    def stop_validation_campaign(self):
        """Stop validation campaign workflow"""
        with self._state_lock:
            if not self.validation_campaign_active:
                return {'success': True, 'message': 'Validation campaign not active'}

            self.validation_campaign_active = False
            if self.validation_statistics:
                self.validation_statistics['status'] = 'stopped'

        # Wait for thread to finish
        if self.validation_campaign_thread and self.validation_campaign_thread.is_alive():
            self.validation_campaign_thread.join(timeout=10)
            if self.validation_campaign_thread.is_alive():
                logger.warning("Validation campaign thread did not stop gracefully")

        logger.info("Validation campaign stopped")
        return {'success': True, 'message': 'Validation campaign stopped'}

    def _validation_campaign_loop(self):
        """Validation campaign workflow loop"""
        logger.info("Validation campaign loop started")

        try:
            # Load configuration
            config = self._get_system_config()
            if not config:
                logger.error("Cannot start validation campaign - no valid configuration")
                return

            window_size = self.validation_config.get('batch_size', 50)
            current_window_start = 0

            while self._is_validation_campaign_active():
                try:
                    window_start_time = time.time()
                    logger.info(f"Validation window starting at position {current_window_start}")

                    # === Run validation for this window ===
                    validation_results = self._run_window_validation(current_window_start, window_size)

                    # === Check validation results and handle failures ===
                    validation_failed = self._check_validation_failure(validation_results, current_window_start)

                    if validation_failed:
                        calibration_success = self._handle_validation_failure(validation_results, current_window_start)
                        if calibration_success:
                            with self._state_lock:
                                self.validation_statistics['calibrations_performed'] += 1

                    # === Update statistics ===
                    with self._state_lock:
                        self.validation_statistics['windows_processed'] += 1
                        if validation_failed:
                            self.validation_statistics['validation_failures'] += 1
                        self.last_validation_results.update(validation_results)

                    # === Save validation results to file ===
                    self._save_validation_results_to_file(validation_results, current_window_start)

                    # === Move to next window ===
                    current_window_start += self.validation_config.get('window_stride', window_size)

                    # === Adaptive sleep ===
                    processing_time = time.time() - window_start_time
                    sleep_time = max(0, self.update_interval - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    self._handle_validation_campaign_error(e, current_window_start)
                    current_window_start += window_size  # Skip problematic window

        except Exception as e:
            logger.error(f"Fatal error in validation campaign loop: {e}")
            with self._state_lock:
                self.validation_statistics['last_error'] = str(e)
                self.validation_statistics['status'] = 'error'

        finally:
            logger.info("Validation campaign loop ended")

    def _is_validation_campaign_active(self):
        """Thread-safe check for validation campaign status"""
        with self._state_lock:
            return self.validation_campaign_active

    def _handle_validation_campaign_error(self, error, window_start):
        """Handle errors in validation campaign loop"""
        error_msg = f"Error in validation campaign window {window_start}: {error}"
        logger.error(error_msg)

        with self._state_lock:
            self.validation_statistics['last_error'] = error_msg

        # Simple backoff for validation campaign
        time.sleep(self.error_backoff_time)

    def _save_validation_results_to_file(self, validation_results, window_start):
        """Save validation results to CSV file"""
        try:
            import pandas as pd

            # Prepare results data
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'mlcss': validation_results.get('mlcss', 0.0),
                'dtw': validation_results.get('dtw', 0.0),
                'lcss': validation_results.get('lcss', 0.0)
            }

            # Save to file
            results_file = 'validation_results.csv'
            file_exists = os.path.exists(results_file)

            df = pd.DataFrame([results_data])
            df.to_csv(results_file, mode='a', header=not file_exists, index=False)

            logger.debug(f"Validation results saved to {results_file}")

        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

    # =============================================================================
    # STOP ALL OPERATIONS (Stop button)
    # =============================================================================

    def stop_all_operations(self):
        """Stop all operations - data preparation, validation, and optionally Arena"""
        results = []

        # Stop data preparation
        if self.data_preparation_active:
            result = self.stop_data_preparation()
            results.append(f"Data preparation: {result['message']}")

        # Stop validation campaign
        if self.validation_campaign_active:
            result = self.stop_validation_campaign()
            results.append(f"Validation: {result['message']}")

        # Optionally stop Arena if integration is enabled
        if self.arena_integration_enabled and self.digital_model_interface:
            try:
                arena_result = self.digital_model_interface.stop_simulation()
                if arena_result.get('success'):
                    results.append("Arena simulation: Stopped")
                else:
                    results.append(f"Arena simulation: {arena_result.get('error', 'Stop failed')}")
            except Exception as e:
                results.append(f"Arena simulation: Error - {str(e)}")

        # Stop data manager
        try:
            self.data_manager.stop()
            results.append("Data manager: Stopped")
        except Exception as e:
            logger.warning(f"Error stopping data manager: {e}")
            results.append(f"Data manager: Warning - {str(e)}")

        logger.info("All operations stopped")
        return {
            'success': True,
            'message': 'All operations stopped',
            'details': results
        }

    # =============================================================================
    # ARENA INTEGRATION MANAGEMENT
    # =============================================================================

    def enable_arena_integration(self):
        """Enable Arena integration (called when Arena is manually connected)"""
        with self._state_lock:
            self.arena_integration_enabled = True
        logger.info("Arena integration enabled")

    def disable_arena_integration(self):
        """Disable Arena integration"""
        with self._state_lock:
            self.arena_integration_enabled = False
        logger.info("Arena integration disabled")

    def get_arena_real_time_data(self):
        """Get real-time data from Arena simulation (if integration enabled)"""
        if not self.arena_integration_enabled or not self.digital_model_interface:
            return None

        try:
            status = self.digital_model_interface.get_comprehensive_status()
            return status
        except Exception as e:
            if "CoInitialize" in str(e):
                logger.debug(f"COM initialization handled by digital_model_interface: {e}")
                return None
            else:
                logger.error(f"Error getting Arena real-time data: {e}")
                return None

    # =============================================================================
    # SHARED HELPER METHODS - UPDATED WITH KPI GENERATION
    # =============================================================================

    def _get_system_config(self):
        """Load and cache system configuration"""
        try:
            config_path = self.config.get('system_config_path',
                                          os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json'))

            # Check if we need to reload config
            if (self._system_config is None or
                    self._config_last_loaded is None or
                    (datetime.now() - self._config_last_loaded).seconds > 300):  # Reload every 5 minutes

                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self._system_config = json.load(f)
                    self._config_last_loaded = datetime.now()
                    logger.debug("System configuration reloaded")
                else:
                    logger.error(f"Configuration file not found: {config_path}")
                    return None

            return self._system_config

        except Exception as e:
            logger.error(f"Error loading system configuration: {e}")
            return None

    def _process_window_data_with_kpis(self, window_events, stations, dist_configs, window_size):
        """Process window data with proper error handling and KPI generation"""
        try:
            # Import functions dynamically to avoid startup dependencies
            import importlib.util

            # Process processing times AND generate KPIs
            try:
                processing_time_module = importlib.import_module('processing_time')
                # UPDATED: Add save_kpis=True to generate KPI files during window processing
                kpi_generation_enabled = self.validation_config.get('enable_kpi_generation', True)

                processing_time_module.save_processing_times_separate(
                    window_events,
                    output_dir='.',
                    mode='offline',
                    save_kpis=kpi_generation_enabled  # NEW: Generate KPI files during data preparation
                )

                if kpi_generation_enabled:
                    logger.info(f"Processing times and KPIs saved for window of {len(window_events)} events")
                else:
                    logger.info(
                        f"Processing times saved for window of {len(window_events)} events (KPI generation disabled)")

            except ImportError as e:
                logger.warning(f"Could not import processing_time module: {e}")
                return False
            except Exception as e:
                logger.error(f"Error in processing_time module: {e}")
                return False

            # Generate digital traces
            try:
                digital_generator_module = importlib.import_module('digital_input_online_generator')
                digital_generator_module.generate_traces_for_all_stations(
                    stations=stations,
                    processing_times_dir='.',
                    output_dir='correlated_traces',
                    dist_configs=dist_configs,
                    window_size=window_size
                )
                logger.debug(f"Digital traces generated for {len(stations)} stations")
            except ImportError as e:
                logger.warning(f"Could not import digital_input_online_generator module: {e}")
                return False
            except Exception as e:
                logger.error(f"Error in digital_input_online_generator module: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error processing window data with KPIs: {e}")
            return False

    # LEGACY METHOD for backward compatibility
    def _process_window_data(self, window_events, stations, dist_configs, window_size):
        """Legacy method - calls the new KPI-enabled version"""
        return self._process_window_data_with_kpis(window_events, stations, dist_configs, window_size)

    def _run_window_validation(self, current_window_start, window_size):
        """Run validation for current window with enhanced file format error handling"""
        validation_mode = self.validation_config.get('validation_mode', 'kpi')
        window_validation_results = {}

        try:
            if validation_mode in ["kpi", "both"]:
                try:
                    real_kpi = self.data_manager.get_real_system_kpi_data(
                        window_size=window_size,
                        start_position=current_window_start
                    )
                    digital_kpi = self.data_manager.get_digital_model_kpi_data(
                        window_size=window_size,
                        start_position=current_window_start
                    )

                    if real_kpi is not None and digital_kpi is not None:
                        # ENHANCED: Check data compatibility before validation
                        if len(real_kpi) == 0:
                            logger.warning(f"Window {current_window_start}: Real KPI data is empty")
                        elif len(digital_kpi) == 0:
                            logger.warning(f"Window {current_window_start}: Digital KPI data is empty")
                        else:
                            kpi_results = self.validation_engine.run_validation(real_kpi, digital_kpi)
                            window_validation_results.update(kpi_results)
                            logger.debug(f"Window {current_window_start}: KPI validation completed")
                    else:
                        logger.warning(
                            f"Window {current_window_start}: KPI data unavailable - Real: {real_kpi is not None}, Digital: {digital_kpi is not None}")

                except Exception as e:
                    logger.error(f"Window {current_window_start}: KPI validation error - {e}")
                    # Continue with event validation if available

            if validation_mode in ["event", "both"]:
                try:
                    real_output = self.data_manager.get_real_system_output_data(
                        window_size=window_size,
                        start_position=current_window_start
                    )
                    digital_output = self.data_manager.get_digital_model_output_data(
                        window_size=window_size,
                        start_position=current_window_start
                    )

                    if real_output is not None and digital_output is not None:
                        if len(real_output) == 0:
                            logger.warning(f"Window {current_window_start}: Real output data is empty")
                        elif len(digital_output) == 0:
                            logger.warning(f"Window {current_window_start}: Digital output data is empty")
                        else:
                            event_results = self.validation_engine.run_validation(real_output, digital_output)
                            window_validation_results.update(event_results)
                            logger.debug(f"Window {current_window_start}: Event validation completed")
                    else:
                        logger.warning(
                            f"Window {current_window_start}: Event data unavailable - Real: {real_output is not None}, Digital: {digital_output is not None}")

                except Exception as e:
                    logger.error(f"Window {current_window_start}: Event validation error - {e}")

            # ENHANCED: Ensure we always return some results, even if validation fails
            if not window_validation_results:
                logger.warning(f"Window {current_window_start}: No validation results obtained, returning defaults")
                window_validation_results = {'mlcss': 0.0, 'dtw': 0.0, 'lcss': 0.0}

        except Exception as e:
            logger.error(f"Error in window validation: {e}")
            # Return default results to prevent system failure
            window_validation_results = {'mlcss': 0.0, 'dtw': 0.0, 'lcss': 0.0}

        return window_validation_results

    def _check_validation_failure(self, validation_results, window_start):
        """Check if validation failed for current window"""
        if not validation_results:
            return False

        try:
            selected_algorithm = self.validation_engine.current_config.get('kpi_algorithm', 'mlcss')
            validation_level = self.validation_engine.current_config.get('validation_level', 'kpi')
            algorithm_thresholds = self.validation_engine.thresholds

            if validation_level == 'event':
                algorithm_score = validation_results.get('lcss', 0.0)
                threshold = algorithm_thresholds.get('lcss', 0.85)
                if algorithm_score < threshold:
                    logger.info(f"Window {window_start}: Validation failed (LCSS: {algorithm_score:.3f} < {threshold})")
                    return True
            else:
                if selected_algorithm == 'mlcss':
                    algorithm_score = validation_results.get('mlcss', 0.0)
                    threshold = algorithm_thresholds.get('mlcss', 0.90)
                    if algorithm_score < threshold:
                        logger.info(
                            f"Window {window_start}: Validation failed (mLCSS: {algorithm_score:.3f} < {threshold})")
                        return True
                elif selected_algorithm == 'dtw':
                    algorithm_score = validation_results.get('dtw', 0.0)
                    threshold = algorithm_thresholds.get('dtw', 0.95)
                    if algorithm_score < threshold:
                        logger.info(
                            f"Window {window_start}: Validation failed (DTW: {algorithm_score:.3f} < {threshold})")
                        return True
                elif selected_algorithm == 'both':
                    mlcss_score = validation_results.get('mlcss', 0.0)
                    dtw_score = validation_results.get('dtw', 0.0)
                    threshold_m = algorithm_thresholds.get('mlcss', 0.90)
                    threshold_d = algorithm_thresholds.get('dtw', 0.95)
                    if mlcss_score < threshold_m or dtw_score < threshold_d:
                        logger.info(
                            f"Window {window_start}: Validation failed (mLCSS: {mlcss_score:.3f} < {threshold_m}, DTW: {dtw_score:.3f} < {threshold_d})")
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking validation failure: {e}")
            return False

    def _handle_validation_failure(self, validation_results, window_start):
        """Handle validation failure with issue identification and calibration"""
        if not self.issue_identification:
            logger.warning("No issue identification engine available")
            return False

        try:
            logger.info(f"Window {window_start}: Triggering issue identification")

            selected_algorithm = self.validation_engine.current_config.get('kpi_algorithm', 'mlcss')
            analysis_results, problematic_stations = self.issue_identification.run_station_level_analysis(
                selected_algorithm=selected_algorithm
            )

            if problematic_stations:
                logger.info(f"Window {window_start}: Problematic stations identified: {problematic_stations}")

                if self.calibration_engine:
                    calib_requests = {}
                    for station in problematic_stations:
                        real_times = self.issue_identification._load_real_processing_times(station)
                        if real_times:
                            calib_requests[station] = real_times

                    if calib_requests:
                        # Run calibration in background thread to avoid blocking validation
                        def background_calibration():
                            try:
                                logger.info(f"Window {window_start}: Starting background calibration for {len(calib_requests)} stations")
                                calib_results = self.calibration_engine.calibrate_multiple_stations(calib_requests)
                                logger.info(f"Window {window_start}: Background calibration completed for {len(calib_results)} stations")

                                # Update parameters and regenerate correlated times
                                self._update_parameters_and_regenerate_correlated_times(calib_results)
                                logger.info(f"Window {window_start}: Parameters updated and correlated times regenerated")
                            except Exception as e:
                                logger.error(f"Window {window_start}: Error in background calibration: {e}")

                        # Start calibration in background thread
                        calibration_thread = threading.Thread(target=background_calibration, daemon=True)
                        calibration_thread.start()
                        logger.info(f"Window {window_start}: Background calibration thread started")
                        return True
            else:
                logger.info(f"Window {window_start}: No problematic stations identified")

        except Exception as e:
            logger.error(f"Window {window_start}: Error in issue identification: {e}")

        return False

    def _has_more_data(self, start_position, window_size):
        """Check if more data is available"""
        try:
            real_kpi = self.data_manager.get_real_system_kpi_data(
                window_size=window_size,
                start_position=start_position
            )
            return real_kpi is not None and len(real_kpi) >= window_size
        except Exception:
            return False

    def _update_parameters_and_regenerate_correlated_times(self, calibration_results):
        """Update parameters and regenerate correlated times with better error handling"""
        try:
            logger.info("Updating distribution parameters and regenerating correlated times")

            config = self._get_system_config()
            if not config:
                logger.error("Cannot update parameters - no valid configuration")
                return

            config_path = self.config.get('system_config_path',
                                          os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json'))

            updated_stations = []
            for station_id, calib_data in calibration_results.items():
                if 'params' in calib_data and calib_data['params']:
                    if 'dist_configs' in config and station_id in config['dist_configs']:
                        old_params = config['dist_configs'][station_id]['params']
                        config['dist_configs'][station_id]['params'] = calib_data['params']
                        updated_stations.append(station_id)
                        logger.info(f"Updated {station_id} parameters: {old_params} -> {calib_data['params']}")

            if updated_stations:
                # Write updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Configuration updated for stations: {updated_stations}")

                # Invalidate cached config
                self._system_config = None

                # Regenerate correlated processing times
                self._regenerate_correlated_times(updated_stations, config['dist_configs'])

        except Exception as e:
            logger.error(f"Error updating parameters and regenerating correlated times: {e}")

    def _regenerate_correlated_times(self, station_list, dist_configs):
        """Regenerate correlated times with improved error handling"""
        try:
            import importlib.util

            # Use current directory for processing time files
            pt_dir = os.getcwd()
            output_dir = os.path.join(pt_dir, 'correlated_traces')

            logger.info(f"Regenerating correlated times for stations: {station_list}")

            # Dynamic import to avoid startup dependencies
            digital_generator_module = importlib.import_module('digital_input_online_generator')
            digital_generator_module.generate_traces_for_all_stations(
                stations=station_list,
                processing_times_dir=pt_dir,
                output_dir=output_dir,
                dist_configs=dist_configs
            )

            logger.info(f"Successfully regenerated correlated times for {len(station_list)} stations")

        except Exception as e:
            logger.error(f"Error regenerating correlated times: {e}")

    # =============================================================================
    # KPI MANAGEMENT METHODS
    # =============================================================================

    def enable_kpi_generation(self):
        """Enable KPI generation during data preparation"""
        with self._state_lock:
            self.validation_config['enable_kpi_generation'] = True
        logger.info("KPI generation enabled")
        return {'success': True, 'message': 'KPI generation enabled'}

    def disable_kpi_generation(self):
        """Disable KPI generation during data preparation"""
        with self._state_lock:
            self.validation_config['enable_kpi_generation'] = False
        logger.info("KPI generation disabled")
        return {'success': True, 'message': 'KPI generation disabled'}

    def get_kpi_generation_status(self):
        """Get current KPI generation status"""
        with self._state_lock:
            return {
                'enabled': self.validation_config.get('enable_kpi_generation', True),
                'files_generated': self.data_preparation_statistics.get('kpi_files_generated', 0),
                'last_generation': self.data_preparation_statistics.get('start_time')
            }

    def force_kpi_generation(self, window_size=None):
        """Force KPI generation for current data (manual trigger)"""
        try:
            window_size = window_size or self.validation_config.get('batch_size', 50)

            # Get latest window of raw events
            raw_events = self.data_manager.get_raw_events(window_size=window_size, start_position=0)

            if raw_events is None or len(raw_events) == 0:
                return {'success': False, 'message': 'No raw events available for KPI generation'}

            # Load configuration
            config = self._get_system_config()
            if not config:
                return {'success': False, 'message': 'No valid configuration available'}

            station_config = config.get("station_config", {})
            stations = station_config.get("stations", [])
            dist_configs = config.get("dist_configs", {})

            # Process with KPI generation
            success = self._process_window_data_with_kpis(raw_events, stations, dist_configs, window_size)

            if success:
                with self._state_lock:
                    self.data_preparation_statistics['kpi_files_generated'] += 1

                logger.info(f"Manual KPI generation completed for {len(raw_events)} events")
                return {
                    'success': True,
                    'message': f'KPI generation completed for {len(raw_events)} events',
                    'events_processed': len(raw_events)
                }
            else:
                return {'success': False, 'message': 'KPI generation failed'}

        except Exception as e:
            logger.error(f"Error in manual KPI generation: {e}")
            return {'success': False, 'error': str(e)}

    # =============================================================================
    # ENHANCED STATUS METHODS
    # =============================================================================

    def get_comprehensive_status(self):
        """Get comprehensive system status including KPI information"""
        base_status = self.get_status()

        # Add KPI-specific information
        kpi_status = self.get_kpi_generation_status()
        base_status['kpi_generation'] = kpi_status

        # Add file existence checks with format detection
        base_status['file_status'] = {
            'system_kpis_exists': os.path.exists('system_kpis.csv'),
            'interdeparture_times_exists': os.path.exists('interdeparture_times.csv'),
            'validation_results_exists': os.path.exists('validation_results.csv'),
            'real_kpi_format': self.data_manager.get_status().get('real_kpi_format', 'Unknown'),
            'digital_kpi_format': self.data_manager.get_status().get('digital_kpi_format', 'Unknown')
        }

        # Add workflow-specific timestamps
        if self.data_preparation_statistics.get('start_time'):
            base_status['data_preparation_statistics']['duration'] = (
                    datetime.now() - self.data_preparation_statistics['start_time']
            ).total_seconds()

        if self.validation_statistics.get('start_time'):
            base_status['validation_statistics']['duration'] = (
                    datetime.now() - self.validation_statistics['start_time']
            ).total_seconds()

        return base_status

    # =============================================================================
    # LEGACY METHODS (for backward compatibility)
    # =============================================================================

    def start_campaign(self):
        """Legacy method - now starts data preparation workflow"""
        return self.start_data_preparation()

    def stop_campaign(self):
        """Legacy method - now stops all operations"""
        return self.stop_all_operations()

    def run_offline_kpi_validation(self, window_size=50, stride=50):
        """Run offline KPI validation with enhanced error handling and file format support"""
        try:
            real_kpi = self.data_manager.get_real_system_kpi_data()
            digital_kpi = self.data_manager.get_digital_model_kpi_data()

            if real_kpi is None or digital_kpi is None:
                logger.error("Could not load KPI data for offline validation")
                logger.error(f"Real KPI data: {'Available' if real_kpi is not None else 'Not available'}")
                logger.error(f"Digital KPI data: {'Available' if digital_kpi is not None else 'Not available'}")
                return []

            # ENHANCED: Check data compatibility
            if len(real_kpi) == 0:
                logger.error("Real KPI data is empty")
                return []
            if len(digital_kpi) == 0:
                logger.error("Digital KPI data is empty")
                return []

            logger.info(f"Starting offline KPI validation: Real={len(real_kpi)} rows, Digital={len(digital_kpi)} rows")

            results = self.validation_engine.run_offline_validation(
                real_kpi, digital_kpi, window_size=window_size, stride=stride
            )

            logger.info(f"Offline KPI validation completed: {len(results)} windows processed")
            return results

        except Exception as e:
            logger.error(f"Error in offline KPI validation: {e}")
            return []

    def run_offline_event_validation(self, window_size=50, stride=50):
        """Run offline event validation with enhanced error handling"""
        try:
            real_events = self.data_manager.get_real_system_output_data()
            digital_events = self.data_manager.get_digital_model_output_data()

            if real_events is None or digital_events is None:
                logger.error("Could not load event data for offline validation")
                logger.error(f"Real event data: {'Available' if real_events is not None else 'Not available'}")
                logger.error(f"Digital event data: {'Available' if digital_events is not None else 'Not available'}")
                return []

            # ENHANCED: Check data compatibility
            if len(real_events) == 0:
                logger.error("Real event data is empty")
                return []
            if len(digital_events) == 0:
                logger.error("Digital event data is empty")
                return []

            logger.info(
                f"Starting offline event validation: Real={len(real_events)} rows, Digital={len(digital_events)} rows")

            results = self.validation_engine.run_offline_validation(
                real_events, digital_events, window_size=window_size, stride=stride
            )

            logger.info(f"Offline event validation completed: {len(results)} windows processed")
            return results

        except Exception as e:
            logger.error(f"Error in offline event validation: {e}")
            return []