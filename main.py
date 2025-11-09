import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import threading
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'validation_algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from database_integrated_data_manager import DatabaseIntegratedDataManager
from validation_engine import ValidationEngine
from digital_model_interface_linux import DigitalModelInterface
from system_controller import SystemController
from calibration_engine import CalibrationEngine
from issue_identification import IssueIdentificationModule
from file_monitor_improved import FileMonitor

CURRENT_UTC = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = os.environ.get("USER", "unknown")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DigitalTwinValidationSystem')

validation_results_history = []
validation_results_file = 'validation_results.csv'


def save_validation_results(results):
    """Save validation results to CSV file for further analysis"""
    try:
        mlcss = results.get('mlcss', 0.0)
        dtw = results.get('dtw', 0.0)
        lcss = results.get('lcss', 0.0)

        if mlcss > 0 or dtw > 0 or lcss > 0:
            results_with_timestamp = {
                'timestamp': datetime.now().isoformat(),
                'mlcss': mlcss,
                'dtw': dtw,
                'lcss': lcss
            }
            validation_results_history.append(results_with_timestamp)
            df = pd.DataFrame(validation_results_history)
            df.to_csv(validation_results_file, index=False)
            logger.info(f"Validation results saved to {validation_results_file}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")


class DigitalTwinValidationSystem:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        logger.info(f"System initialization started by {CURRENT_USER} at {CURRENT_UTC}")

        # Load configuration first
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'system_config.json')
        self.config = self.load_config(config_path)

        # Add system-level configuration
        self.config.update({
            'system_config_path': config_path,
            'update_interval': self.config.get('validation_config', {}).get('update_interval', 120) / 10,
            'stop_arena_on_campaign_end': False,  # Don't auto-stop Arena
            'arena_sync_mode': 'time_based'
        })

        # Initialize core components
        self.data_manager = DatabaseIntegratedDataManager(config_file=config_path)
        self.validation_engine = ValidationEngine(self.config)
        self.digital_model_interface = DigitalModelInterface(self.config)
        self.calibration_engine = CalibrationEngine(self.config)
        self.issue_identification = IssueIdentificationModule(
            tic_threshold=0.10,
            validation_threshold=0.8
        )

        # Initialize SystemController with separated workflows
        self.system_controller = SystemController(
            data_manager=self.data_manager,
            validation_engine=self.validation_engine,
            digital_model_interface=self.digital_model_interface,
            calibration_engine=self.calibration_engine,
            issue_identification=self.issue_identification,
            config=self.config
        )

        # File monitoring - will be managed manually
        self.file_monitor = None

        # Ensure validation config has required fields
        if 'validation_config' not in self.config:
            self.config['validation_config'] = {}
        if 'selected_algorithm' not in self.config['validation_config']:
            self.config['validation_config']['selected_algorithm'] = 'mlcss'
        if 'validation_level' not in self.config['validation_config']:
            self.config['validation_config']['validation_level'] = 'kpi'

        # Connect core components (NO ARENA AUTO-CONNECTION)
        self._initialize_system_components()

        self.setup_routes()
        logger.info(f"Digital Twin Validation System initialized by {CURRENT_USER}")

    def load_config(self, config_path):
        """Load system configuration with enhanced defaults"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Ensure all required config sections exist
            default_config = {
                "system_id": "G1-5S-PL",
                "arena_config": {
                    "model_path": "",
                    "output_file": "log.txt",
                    "historical_file": "log.csv",
                    "polling_interval": 1.0
                },
                "validation_config": {
                    "update_interval": 120,
                    "batch_size": 50,
                    "window_stride": 50,
                    "validation_mode": "kpi",
                    "enable_arena_integration": False,  # Now manual only
                    "thresholds": {
                        "lcss": 0.85,
                        "mlcss": 0.90,
                        "dtw": 0.95
                    },
                    "selected_algorithm": "mlcss",
                    "validation_level": "kpi"
                },
                "operational_mode": "offline"
            }

            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue

            return config

        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            return default_config

    def _initialize_system_components(self):
        """Initialize and connect core components (NO ARENA AUTO-CONNECTION)"""
        try:
            # Connect core components only (Arena is manual)
            connection_result = self.system_controller.connect_components()
            if not connection_result.get('success', False):
                logger.warning(f"Component connection issues: {connection_result}")

            logger.info("Core system components initialized (Arena connection is manual)")

        except Exception as e:
            logger.error(f"Error initializing system components: {e}")

    def convert_to_json_serializable(self, obj):
        """Enhanced JSON serialization for complex objects"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
            return obj.isoformat()
        else:
            return obj

    def setup_routes(self):
        """Setup Flask routes for the separated workflow system"""

        @self.app.route('/')
        def dashboard():
            return send_file('dashboard_with_trending.html')

        @self.app.route('/api/system/status')
        def get_system_status():
            """Get comprehensive system status including all separated workflows"""
            try:
                # Get status from SystemController with separated workflows
                system_status = self.system_controller.get_status()
                validation_status = self.validation_engine.get_status()
                operational_mode = self.config.get('operational_mode', 'offline')

                # Get station states (real vs digital)
                stations = {
                    'real': self.data_manager.get_real_station_states() if operational_mode == 'online' else None,
                    'digital': self._get_integrated_digital_states()
                }

                # Get validation results
                validation_results = self._get_integrated_validation_results()

                # Get Arena real-time data if integration enabled
                arena_real_time = None
                if system_status.get('arena_integration_enabled', False):
                    arena_real_time = self.system_controller.get_arena_real_time_data()

                status_data = {
                    'station_states': stations,
                    'validation_results': validation_results,
                    'system_status': system_status,
                    'connection_status': self.data_manager.get_status(),
                    'digital_model': system_status.get('arena_status', {}),
                    'arena_real_time': arena_real_time,
                    'validation_count': validation_status.get('validation_count', 0),
                    'current_user': CURRENT_USER,
                    'last_update': datetime.now().isoformat(),
                    'operational_mode': operational_mode,
                    'kpi_algorithm': self.config['validation_config'].get('selected_algorithm', 'mlcss'),
                    'validation_level': self.config['validation_config'].get('validation_level', 'kpi'),

                    # Separated workflow states
                    'data_preparation_active': system_status.get('data_preparation_active', False),
                    'validation_campaign_active': system_status.get('validation_campaign_active', False),
                    'arena_integration_enabled': system_status.get('arena_integration_enabled', False),
                    'data_preparation_statistics': system_status.get('data_preparation_statistics', {}),
                    'validation_statistics': system_status.get('validation_statistics', {})
                }

                return jsonify(self.convert_to_json_serializable(status_data))

            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/config/update', methods=['POST'])
        def update_config():
            """Update system configuration and propagate to all components"""
            try:
                data = request.get_json()

                # Update main config
                config_updated = False
                if 'kpi_algorithm' in data:
                    self.config['validation_config']['selected_algorithm'] = data['kpi_algorithm']
                    config_updated = True
                if 'validation_level' in data:
                    self.config['validation_config']['validation_level'] = data['validation_level']
                    config_updated = True

                # Update validation engine
                self.validation_engine.update_configuration(data)

                # Update SystemController if needed
                if config_updated:
                    self.system_controller.validation_config.update(self.config['validation_config'])

                logger.info(
                    f"Config updated: Algorithm={data.get('kpi_algorithm')}, Level={data.get('validation_level')}")
                return jsonify({'success': True, 'message': 'Configuration updated successfully'})

            except Exception as e:
                logger.error(f"Error updating config: {e}")
                return jsonify({'success': False, 'error': str(e)})

        # =============================================================================
        # DATA PREPARATION WORKFLOW ROUTES (Start/Stop buttons)
        # =============================================================================

        @self.app.route('/api/data-preparation/start', methods=['POST'])
        def start_data_preparation():
            """Start data preparation workflow"""
            try:
                # Start file monitoring if needed
                if self.file_monitor is None:
                    sim_file = self.config['arena_config']['output_file']
                    hist_file = self.config['arena_config']['historical_file']
                    self.file_monitor = FileMonitor(self.data_manager, sim_file, hist_file)
                    self.file_monitor.start_monitoring()
                    logger.info("File monitoring started")

                # Start data preparation workflow
                result = self.system_controller.start_data_preparation()

                if result.get('success'):
                    logger.info("Data preparation workflow started")

                return jsonify(self.convert_to_json_serializable(result))

            except Exception as e:
                logger.error(f"Error starting data preparation: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/data-preparation/stop', methods=['POST'])
        def stop_data_preparation():
            """Stop data preparation workflow"""
            try:
                result = self.system_controller.stop_data_preparation()
                logger.info("Data preparation workflow stopped")
                return jsonify(self.convert_to_json_serializable(result))

            except Exception as e:
                logger.error(f"Error stopping data preparation: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/system/stop-all', methods=['POST'])
        def stop_all_operations():
            """Stop all operations - data preparation, validation, and Arena"""
            try:
                # Stop file monitoring
                if self.file_monitor:
                    self.file_monitor.stop_monitoring()
                    self.file_monitor = None
                    logger.info("File monitoring stopped")

                # Stop all operations via SystemController
                result = self.system_controller.stop_all_operations()

                logger.info("All system operations stopped")
                return jsonify(self.convert_to_json_serializable(result))

            except Exception as e:
                logger.error(f"Error stopping all operations: {e}")
                return jsonify({'success': False, 'error': str(e)})

        # =============================================================================
        # VALIDATION CAMPAIGN WORKFLOW ROUTES (Start Validation button)
        # =============================================================================

        @self.app.route('/api/validation/start', methods=['POST'])
        def start_validation_campaign():
            """Start validation campaign workflow"""
            try:
                result = self.system_controller.start_validation_campaign()

                if result.get('success'):
                    logger.info("Validation campaign started")

                return jsonify(self.convert_to_json_serializable(result))

            except Exception as e:
                logger.error(f"Error starting validation campaign: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/validation/stop', methods=['POST'])
        def stop_validation_campaign():
            """Stop validation campaign workflow"""
            try:
                result = self.system_controller.stop_validation_campaign()
                logger.info("Validation campaign stopped")
                return jsonify(self.convert_to_json_serializable(result))

            except Exception as e:
                logger.error(f"Error stopping validation campaign: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/validation/results')
        def get_validation_results():
            """Get latest validation results"""
            try:
                results = self._get_integrated_validation_results()

                # Filter based on current algorithm and level settings
                selected_algo = self.config['validation_config'].get('selected_algorithm', 'mlcss')
                validation_level = self.config['validation_config'].get('validation_level', 'kpi')

                filtered = {'timestamp': results.get('timestamp', datetime.now().isoformat())}

                if validation_level == 'event':
                    filtered['lcss'] = results.get('lcss', 0.0)
                else:
                    if selected_algo == 'mlcss':
                        filtered['mlcss'] = results.get('mlcss', 0.0)
                    elif selected_algo == 'dtw':
                        filtered['dtw'] = results.get('dtw', 0.0)
                    elif selected_algo == 'both':
                        filtered['mlcss'] = results.get('mlcss', 0.0)
                        filtered['dtw'] = results.get('dtw', 0.0)

                # Save to history
                save_validation_results(filtered)

                return jsonify({'success': True, 'data': self.convert_to_json_serializable(filtered)})

            except Exception as e:
                logger.error(f"Error getting validation results: {e}")
                return jsonify({'success': False, 'error': str(e)})

        # =============================================================================
        # ARENA CONTROL ROUTES (Independent Manual Control)
        # =============================================================================

        @self.app.route('/api/arena/connect', methods=['POST'])
        def connect_arena():
            """Connect to Arena manually"""
            try:
                result = self.digital_model_interface.connect_to_arena()

                if result.get('success'):
                    # Enable Arena integration in SystemController
                    self.system_controller.enable_arena_integration()

                    arena_version = result.get('arena_version', 'Unknown')
                    logger.info(f"✓ Arena connected manually: Version {arena_version}")
                    return jsonify({
                        'success': True,
                        'message': f'✓ Arena connected: Version {arena_version}',
                        'details': {
                            'arena_version': arena_version,
                            'connection_method': 'Manual COM interface'
                        }
                    })
                else:
                    error_msg = result.get('error', 'Unknown connection error')
                    logger.error(f"✗ Arena connection failed: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': f'✗ Arena connection failed: {error_msg}',
                        'action_required': result.get('action_required', 'Check Arena installation')
                    })

            except Exception as e:
                logger.error(f"Error in Arena connect route: {e}")
                return jsonify({'success': False, 'error': f'Connection error: {str(e)}'})

        @self.app.route('/api/arena/load', methods=['POST'])
        def load_arena_model():
            """Load Arena model manually"""
            try:
                data = request.get_json() or {}

                # Get model path from request OR config
                model_path = data.get('model_path', '')
                if not model_path:
                    model_path = self.config.get('arena_config', {}).get('model_path', '')

                if not model_path:
                    return jsonify({
                        'success': False,
                        'error': '✗ Model load failed: No model path specified',
                        'action_required': 'Set model path in configuration'
                    })

                # Update config with model path
                self.config['arena_config']['model_path'] = model_path
                self.digital_model_interface.model_path = model_path

                # Check if file exists
                if not os.path.exists(model_path):
                    return jsonify({
                        'success': False,
                        'error': f'✗ Model load failed: File not found at {model_path}',
                        'action_required': 'Verify the model file path exists'
                    })

                # Load the model
                result = self.digital_model_interface.load_arena_model()

                if result.get('success'):
                    discovery = result.get('discovery', {})
                    discovered_resources = discovery.get('discovered', [])
                    logger.info(f"✓ Arena model loaded: Found {len(discovered_resources)} resources")

                    return jsonify({
                        'success': True,
                        'message': f'✓ Arena model loaded: Found {len(discovered_resources)} resources',
                        'details': {
                            'model_path': model_path,
                            'resources_found': len(discovered_resources),
                            'discovery': discovery
                        }
                    })
                else:
                    error_msg = result.get('error', 'Unknown load error')
                    logger.error(f"✗ Model load failed: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': f'✗ Model load failed: {error_msg}',
                        'action_required': result.get('action_required', 'Check model file and Arena connection')
                    })

            except Exception as e:
                logger.error(f"Error in Arena load route: {e}")
                return jsonify({'success': False, 'error': f'Model load error: {str(e)}'})

        @self.app.route('/api/arena/start', methods=['POST'])
        def start_simulation():
            """Start Arena simulation manually"""
            try:
                # Check if model is loaded first
                status = self.digital_model_interface.get_status()
                if not status.get('model_loaded', False):
                    return jsonify({
                        'success': False,
                        'error': '✗ Simulation start failed: No model loaded',
                        'action_required': 'Load Arena model first'
                    })

                result = self.digital_model_interface.start_simulation()

                if result.get('success'):
                    sim_state = result.get('simulation_state', 'Unknown')
                    logger.info(f"✓ Arena simulation started: State: {sim_state}")

                    return jsonify({
                        'success': True,
                        'message': f'✓ Simulation started: State: {sim_state}',
                        'details': {
                            'simulation_state': sim_state,
                            'command_sent': 'arena_model.Go()'
                        }
                    })
                else:
                    error_msg = result.get('error', 'Unknown start error')
                    logger.error(f"✗ Simulation start failed: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': f'✗ Simulation start failed: {error_msg}',
                        'action_required': result.get('action_required', 'Check Arena model')
                    })

            except Exception as e:
                logger.error(f"Error in Arena start route: {e}")
                return jsonify({'success': False, 'error': f'Simulation start error: {str(e)}'})

        @self.app.route('/api/arena/stop', methods=['POST'])
        def stop_simulation():
            """Stop Arena simulation manually"""
            try:
                result = self.digital_model_interface.stop_simulation()

                if result.get('success'):
                    sim_state = result.get('simulation_state', 'Stopped')
                    logger.info(f"✓ Arena simulation stopped: State: {sim_state}")

                    return jsonify({
                        'success': True,
                        'message': f'✓ Simulation stopped: State: {sim_state}',
                        'details': {
                            'simulation_state': sim_state,
                            'command_sent': 'arena_model.End()'
                        }
                    })
                else:
                    error_msg = result.get('error', 'Unknown stop error')
                    logger.error(f"✗ Simulation stop failed: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': f'✗ Simulation stop failed: {error_msg}',
                        'action_required': result.get('action_required', 'Try stopping manually in Arena')
                    })

            except Exception as e:
                logger.error(f"Error in Arena stop route: {e}")
                return jsonify({'success': False, 'error': f'Simulation stop error: {str(e)}'})

        @self.app.route('/api/arena/status')
        def get_arena_status():
            """Get detailed Arena status and real-time data"""
            try:
                # Check if Arena integration is enabled
                if not self.system_controller.arena_integration_enabled:
                    return jsonify({
                        "success": True,
                        "data": {
                            "connected": False,
                            "message": "Arena integration not enabled"
                        }
                    })

                # Get Arena data via SystemController
                arena_data = self.system_controller.get_arena_real_time_data()

                if arena_data:
                    return jsonify({"success": True, "data": self.convert_to_json_serializable(arena_data)})
                else:
                    # Fallback to basic status
                    basic_status = self.digital_model_interface.get_status()
                    return jsonify({"success": True, "data": self.convert_to_json_serializable(basic_status)})

            except Exception as e:
                logger.error(f"Error getting Arena status: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/stations/model')
        def get_arena_station_states():
            """Get Arena station states"""
            try:
                states = self._get_integrated_digital_states()

                if states:
                    return jsonify({"success": True, "data": self.convert_to_json_serializable(states)})
                else:
                    return jsonify({"success": True, "data": {}})

            except Exception as e:
                logger.error(f"Error getting Arena station states: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        # =============================================================================
        # OFFLINE VALIDATION ROUTES
        # =============================================================================

        @self.app.route('/api/validation/offline/kpi', methods=['POST'])
        def run_offline_kpi_validation():
            """Run offline KPI validation"""
            try:
                data = request.get_json()
                window_size = data.get('window_size', 50)
                stride = data.get('stride', 50)

                results = self.system_controller.run_offline_kpi_validation(window_size, stride)
                return jsonify({
                    'success': True,
                    'data': self.convert_to_json_serializable(results),
                    'windows_processed': len(results)
                })

            except Exception as e:
                logger.error(f"Error in offline KPI validation: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/validation/offline/event', methods=['POST'])
        def run_offline_event_validation():
            """Run offline event validation"""
            try:
                data = request.get_json()
                window_size = data.get('window_size', 50)
                stride = data.get('stride', 50)

                results = self.system_controller.run_offline_event_validation(window_size, stride)
                return jsonify({
                    'success': True,
                    'data': self.convert_to_json_serializable(results),
                    'windows_processed': len(results)
                })

            except Exception as e:
                logger.error(f"Error in offline event validation: {e}")
                return jsonify({'success': False, 'error': str(e)})

        # =============================================================================
        # DATABASE MODE CONTROL ROUTES (Five-Station Integration)
        # =============================================================================

        @self.app.route('/api/database/switch_online', methods=['POST'])
        def switch_to_online_mode():
            """Switch to online mode (database-driven simulation)"""
            try:
                success = self.data_manager.switch_to_database_mode()
                
                if success:
                    logger.info("Successfully switched to online mode")
                    return jsonify({
                        'success': True,
                        'message': 'Switched to online mode - processing database events',
                        'mode': 'online'
                    })
                else:
                    logger.error("Failed to switch to online mode")
                    return jsonify({
                        'success': False,
                        'error': 'Failed to switch to online mode',
                        'action_required': 'Check database connection and MQTT simulator'
                    })
                    
            except Exception as e:
                logger.error(f"Error switching to online mode: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/database/switch_offline', methods=['POST'])
        def switch_to_offline_mode():
            """Switch to offline mode (file-based processing)"""
            try:
                success = self.data_manager.switch_to_file_mode()
                
                if success:
                    logger.info("Successfully switched to offline mode")
                    return jsonify({
                        'success': True,
                        'message': 'Switched to offline mode - using file-based processing',
                        'mode': 'offline'
                    })
                else:
                    logger.error("Failed to switch to offline mode")
                    return jsonify({
                        'success': False,
                        'error': 'Failed to switch to offline mode'
                    })
                    
            except Exception as e:
                logger.error(f"Error switching to offline mode: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/database/station_status', methods=['GET'])
        def get_station_status():
            """Get current station status from database mode"""
            try:
                if hasattr(self.data_manager, 'get_station_status'):
                    status = self.data_manager.get_station_status()
                    return jsonify({'success': True, 'station_status': status})
                else:
                    return jsonify({'success': False, 'error': 'Station status not available'})
            except Exception as e:
                logger.error(f"Error getting station status: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/database/save_files', methods=['POST'])
        def save_processed_files():
            """Save processed database data to files"""
            try:
                if hasattr(self.data_manager, 'save_processed_data_to_files'):
                    success = self.data_manager.save_processed_data_to_files()
                    if success:
                        return jsonify({'success': True, 'message': 'Files saved successfully'})
                    else:
                        return jsonify({'success': False, 'error': 'Failed to save files'})
                else:
                    return jsonify({'success': False, 'error': 'File saving not available'})
            except Exception as e:
                logger.error(f"Error saving files: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/database/export_csv', methods=['POST'])
        def export_database_csv():
            """Export database data to CSV files"""
            try:
                data = request.get_json() or {}
                time_range = data.get('time_range', '1h')
                
                if hasattr(self.data_manager, 'export_database_data_to_csv'):
                    success = self.data_manager.export_database_data_to_csv(time_range)
                    if success:
                        return jsonify({'success': True, 'message': f'Database data exported for last {time_range}'})
                    else:
                        return jsonify({'success': False, 'error': 'Failed to export data'})
                else:
                    return jsonify({'success': False, 'error': 'Export functionality not available'})
            except Exception as e:
                logger.error(f"Error exporting database data: {e}")
                return jsonify({'success': False, 'error': str(e)})        # =============================================================================
        # LEGACY COMPATIBILITY ROUTES
        # =============================================================================

        @self.app.route('/api/campaign/start', methods=['POST'])
        def legacy_start_campaign():
            """Legacy route - redirects to data preparation start"""
            return start_data_preparation()

        @self.app.route('/api/campaign/stop', methods=['POST'])
        def legacy_stop_campaign():
            """Legacy route - redirects to stop all operations"""
            return stop_all_operations()

    def _get_integrated_validation_results(self):
        """Get validation results from the integrated workflow"""
        try:
            # Primary source: SystemController validation campaign
            if hasattr(self.system_controller,
                       'last_validation_results') and self.system_controller.last_validation_results:
                results = self.system_controller.last_validation_results.copy()
                results['timestamp'] = datetime.now().isoformat()
                return results

            # Fallback: ValidationEngine
            results = self.validation_engine.get_latest_results()
            if not results.get('timestamp'):
                results['timestamp'] = datetime.now().isoformat()
            return results

        except Exception as e:
            logger.error(f"Error getting integrated validation results: {e}")
            return {'timestamp': datetime.now().isoformat()}

    def _get_integrated_digital_states(self):
        """Get digital station states from integrated interface"""
        try:
            # Only get Arena states if integration is enabled
            if not self.system_controller.arena_integration_enabled:
                return {}

            # Try via SystemController (Arena integration)
            arena_data = self.system_controller.get_arena_real_time_data()
            if arena_data and 'machines' in arena_data:
                return arena_data['machines']

            # Fallback to direct interface
            if hasattr(self.digital_model_interface, "get_all_machine_states"):
                return self.digital_model_interface.get_all_machine_states()

            return {}

        except Exception as e:
            logger.error(f"Error getting integrated digital states: {e}")
            return {}



    def run(self):
        """Run the separated workflow digital twin validation system"""
        logger.info(f"Starting Digital Twin Validation System with Separated Workflows")
        logger.info(f"User: {CURRENT_USER}, Time: {CURRENT_UTC}")
        logger.info(f"Arena Integration: Manual Control Only")
        logger.info("Dashboard available at: http://localhost:5000")

        try:
            self.app.run(host='0.0.0.0', port=5000, debug=False)
        except Exception as e:
            logger.error(f"Error running system: {e}")
        finally:
            # Cleanup on shutdown
            try:
                if self.file_monitor:
                    self.file_monitor.stop_monitoring()
                if hasattr(self.system_controller, 'stop_all_operations'):
                    self.system_controller.stop_all_operations()
                logger.info("System shutdown completed")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


if __name__ == '__main__':
    try:
        # Enhanced startup with error checking
        logger.info("=" * 60)
        logger.info("Digital Twin Validation System - Starting (Fixed Version)")
        logger.info("=" * 60)

        # Check if configuration file exists
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'system_config.json')
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Please ensure config/system_config.json exists")
            sys.exit(1)

        # Initialize system
        logger.info("Initializing system components...")
        system = DigitalTwinValidationSystem()

        # Log startup information
        logger.info("System initialized successfully")
        logger.info("Web server starting on http://localhost:5000")
        logger.info("Dashboard available at: http://localhost:5000/dashboard_with_trending.html")
        logger.info("Arena control fixes applied - Enhanced error handling active")
        logger.info("=" * 60)

        # Run the system
        system.run()

    except KeyboardInterrupt:
        logger.info("System shutdown requested by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Critical error during system startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Digital Twin Validation System shutdown complete")