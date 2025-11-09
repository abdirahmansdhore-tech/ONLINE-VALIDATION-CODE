"""
Enhanced Arena Resource Discovery & Interface
Supports configurable resource names and comprehensive discovery strategies
"""
import logging
import pythoncom
import win32com.client
from datetime import datetime
import time
import os
import threading
import json

logger = logging.getLogger('DigitalModelInterface')

class DigitalModelInterface:
    def __init__(self, config):
        self.config = config
        self.arena_config = config.get('arena_config', {})
        self.model_path = self.arena_config.get('model_path', '')

        # Resource configuration - now configurable through config file
        self.resource_config = self.arena_config.get('resource_names', {})
        self.custom_resource_patterns = self.resource_config.get('patterns', [])
        self.expected_resource_count = self.resource_config.get('expected_count', 5)
        self.resource_prefix = self.resource_config.get('prefix', '')

        # Arena objects
        self.arena_app = None
        self.arena_model = None
        self.siman = None
        self.connected = False
        self.model_loaded = False

        # Connection management
        self._connection_lock = threading.Lock()
        self._last_connection_attempt = 0
        self._connection_cooldown = 3.0

        # COM thread safety
        self._thread_local = threading.local()

        # State mappings - configurable through config
        self.state_mappings = self.arena_config.get('state_mappings', {
            1: "Idle",
            2: "Busy",
            3: "Inactive",
            4: "Failed",
            5: "Blocked",
            6: "Starved"
        })

        # Discovery results
        self.verified_resources = []
        self.resource_indices = {}
        self.discovered_resources = []
        self.discovery_complete = False
        self.discovery_metadata = {}

        logger.info("Enhanced DigitalModelInterface initialized with configurable resource discovery")

    def _ensure_com_initialized(self):
        """Ensure COM is initialized for current thread"""
        try:
            if not hasattr(self._thread_local, 'com_initialized'):
                pythoncom.CoInitialize()
                self._thread_local.com_initialized = True
                logger.debug(f"COM initialized for thread {threading.current_thread().ident}")
        except Exception as e:
            logger.debug(f"COM initialization note: {e}")

    def connect_to_arena(self):
        """Connect to Arena with improved error handling and retry logic"""
        with self._connection_lock:
            if self.connected and self.arena_app:
                try:
                    version = self.arena_app.Version
                    self.arena_app.Visible = True
                    logger.info(f"Using existing Arena connection - Version: {version}")
                    return {'success': True, 'message': f'Already connected to Arena {version}'}
                except Exception as e:
                    logger.warning(f"Existing connection failed: {e}")
                    self.connected = False
                    self.arena_app = None

            if not self._should_attempt_connection():
                wait_time = self._connection_cooldown - (time.time() - self._last_connection_attempt)
                return {'success': False, 'error': f'Connection cooldown active ({wait_time:.1f}s remaining)'}

            self._last_connection_attempt = time.time()

            try:
                self._ensure_com_initialized()

                # Try to get existing Arena instance first
                try:
                    logger.info("Attempting to connect to existing Arena instance...")
                    self.arena_app = win32com.client.GetObject("", "Arena.Application")
                    logger.info("Connected to existing Arena instance")
                except Exception:
                    logger.info("No existing Arena found, creating new instance...")
                    try:
                        self.arena_app = win32com.client.Dispatch("Arena.Application")
                        logger.info("Created new Arena instance")
                    except Exception as e:
                        logger.error(f"Failed to create Arena instance: {e}")
                        return {
                            'success': False,
                            'error': f'Cannot create Arena instance: {str(e)}',
                            'action_required': 'Ensure Arena is installed and COM registration is valid'
                        }

                # Make Arena visible and ensure it stays active
                self.arena_app.Visible = True
                time.sleep(1.0)

                # Verify connection
                version = self.arena_app.Version
                logger.info(f"Arena connection verified - Version: {version}")
                self.connected = True

                return {
                    'success': True,
                    'message': f'Connected to Arena version {version}',
                    'arena_version': version
                }

            except Exception as e:
                logger.error(f"Arena connection failed: {e}")
                self.arena_app = None
                self.connected = False
                return {
                    'success': False,
                    'error': f'Arena connection error: {str(e)}',
                    'action_required': 'Check Arena installation and Windows COM settings'
                }

    def load_arena_model(self):
        """Load Arena model with comprehensive resource discovery"""
        if not self.connected or not self.arena_app:
            connect_result = self.connect_to_arena()
            if not connect_result.get('success'):
                return connect_result

        if not self.model_path or not os.path.exists(self.model_path):
            return {
                'success': False,
                'error': f'Model file not found: {self.model_path}',
                'action_required': 'Verify model file path in configuration'
            }

        logger.info(f"Loading Arena model: {self.model_path}")

        try:
            self._ensure_com_initialized()

            # Load the model
            try:
                self.arena_model = self.arena_app.Models.Open(self.model_path)
                logger.info("Model file opened successfully")
            except Exception as e:
                logger.error(f"Failed to open model: {e}")
                return {
                    'success': False,
                    'error': f'Cannot open model file: {str(e)}',
                    'action_required': 'Ensure Arena can open this model manually'
                }

            if not self.arena_model:
                return {
                    'success': False,
                    'error': 'Model loading returned None',
                    'action_required': 'Model file may be corrupted or incompatible'
                }

            # Initialize SIMAN for resource access
            try:
                logger.info("Initializing SIMAN interface...")
                self.arena_model.Check()
                time.sleep(1.0)
                self.siman = self.arena_model.SIMAN
                logger.info("SIMAN interface acquired")
            except Exception as e:
                logger.warning(f"SIMAN initialization warning: {e}")
                # Continue without SIMAN - some models don't support it

            self.model_loaded = True

            # Perform comprehensive resource discovery
            discovery_result = self._comprehensive_resource_discovery()

            return {
                'success': True,
                'message': f'Model loaded successfully with {len(self.verified_resources)} resources discovered',
                'model_path': self.model_path,
                'discovery': discovery_result,
                'verified_resources': self.verified_resources,
                'resource_mapping': self.resource_indices
            }

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return {
                'success': False,
                'error': f'Model loading failed: {str(e)}',
                'action_required': 'Check model compatibility and Arena version'
            }

    def _comprehensive_resource_discovery(self):
        """Comprehensive resource discovery using multiple strategies"""
        discovered = []
        discovery_strategies = []

        logger.info("Starting comprehensive resource discovery...")

        # Strategy 1: Use configured custom resource names
        if self.resource_config.get('explicit_names'):
            strategy_result = self._discover_explicit_resources()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'Explicit Configuration',
                'found': len(strategy_result['resources']),
                'resources': strategy_result['resources']
            })

        # Strategy 2: Use configured patterns
        if self.custom_resource_patterns:
            strategy_result = self._discover_by_patterns()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'Pattern Matching',
                'found': len(strategy_result['resources']),
                'patterns': self.custom_resource_patterns
            })

        # Strategy 3: Systematic exploration with prefix
        if self.resource_prefix:
            strategy_result = self._discover_with_prefix()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'Prefix Exploration',
                'found': len(strategy_result['resources']),
                'prefix': self.resource_prefix
            })

        # Strategy 4: Common naming conventions
        if len(discovered) < self.expected_resource_count:
            strategy_result = self._discover_common_names()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'Common Conventions',
                'found': len(strategy_result['resources'])
            })

        # Strategy 5: Index-based discovery
        if len(discovered) < self.expected_resource_count:
            strategy_result = self._discover_by_index()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'Index Exploration',
                'found': len(strategy_result['resources']),
                'range_checked': strategy_result.get('range_checked', 0)
            })

        # Strategy 6: SIMAN symbol table exploration
        if self.siman and len(discovered) < self.expected_resource_count:
            strategy_result = self._explore_siman_symbols()
            discovered.extend(strategy_result['resources'])
            discovery_strategies.append({
                'strategy': 'SIMAN Symbol Table',
                'found': len(strategy_result['resources'])
            })

        # Remove duplicates while preserving order
        unique_resources = []
        seen_names = set()
        for resource in discovered:
            if resource['name'] not in seen_names:
                unique_resources.append(resource)
                seen_names.add(resource['name'])

        # Update internal tracking
        if unique_resources:
            self.verified_resources = [r['name'] for r in unique_resources]
            self.resource_indices = {r['name']: r['index'] for r in unique_resources}
            self.discovered_resources = unique_resources
            self.discovery_complete = True

            logger.info(f"Discovery complete: Found {len(unique_resources)} unique resources")
            for resource in unique_resources:
                logger.info(f"  - {resource['name']} (index: {resource['index']}, type: {resource.get('type', 'unknown')})")
        else:
            logger.warning("No resources discovered - model may not contain standard resources")

        return {
            'success': len(unique_resources) > 0,
            'discovered': unique_resources,
            'strategies_used': discovery_strategies,
            'total_found': len(unique_resources),
            'expected': self.expected_resource_count
        }

    def _discover_explicit_resources(self):
        """Discover explicitly configured resource names"""
        resources = []
        explicit_names = self.resource_config.get('explicit_names', [])

        if not explicit_names or not self.siman:
            return {'resources': []}

        logger.info(f"Checking {len(explicit_names)} explicitly configured resource names")

        for name in explicit_names:
            try:
                index = self.siman.SymbolNumber(name)
                if index > 0:
                    resources.append({
                        'name': name,
                        'index': index,
                        'type': 'configured',
                        'discovery_method': 'explicit'
                    })
                    logger.debug(f"Found configured resource: {name} (index: {index})")
            except Exception as e:
                logger.debug(f"Resource '{name}' not found: {e}")

        return {'resources': resources}

    def _discover_by_patterns(self):
        """Discover resources matching configured patterns"""
        resources = []

        if not self.custom_resource_patterns or not self.siman:
            return {'resources': []}

        logger.info(f"Testing resource patterns: {self.custom_resource_patterns}")

        for pattern in self.custom_resource_patterns:
            # Try pattern with numbers 1-20
            for i in range(1, 21):
                name = pattern.format(i=i, num=i, index=i)
                try:
                    index = self.siman.SymbolNumber(name)
                    if index > 0:
                        resources.append({
                            'name': name,
                            'index': index,
                            'type': 'pattern_match',
                            'pattern': pattern
                        })
                        logger.debug(f"Pattern '{pattern}' matched: {name} (index: {index})")
                except:
                    continue

        return {'resources': resources}

    def _discover_with_prefix(self):
        """Discover resources with configured prefix"""
        resources = []

        if not self.resource_prefix or not self.siman:
            return {'resources': []}

        logger.info(f"Exploring resources with prefix: {self.resource_prefix}")

        # Try various suffixes with the prefix
        suffixes = ['1', '2', '3', '4', '5', '01', '02', '03', '04', '05',
                   '_1', '_2', '_3', '_4', '_5', ' 1', ' 2', ' 3', ' 4', ' 5']

        for suffix in suffixes:
            name = f"{self.resource_prefix}{suffix}"
            try:
                index = self.siman.SymbolNumber(name)
                if index > 0:
                    resources.append({
                        'name': name,
                        'index': index,
                        'type': 'prefix_match',
                        'prefix': self.resource_prefix
                    })
                    logger.debug(f"Prefix match found: {name} (index: {index})")
            except:
                continue

        return {'resources': resources}

    def _discover_common_names(self):
        """Discover resources using common naming conventions"""
        resources = []

        if not self.siman:
            return {'resources': []}

        # Extended list of common resource naming patterns
        common_patterns = [
            # Standard Arena names
            "Resource {}", "Resource{}", "Resource_{}",
            # Machine names
            "Machine {}", "Machine{}", "M{}", "Machine_{}",
            # Station names
            "Station {}", "Station{}", "S{}", "Station_{}",
            # Process names
            "Process {}", "Process{}", "P{}", "Process_{}",
            # Server names
            "Server {}", "Server{}", "Server_{}",
            # Workstation names
            "Workstation {}", "Workstation{}", "WS{}",
            # Operator names
            "Operator {}", "Operator{}", "Op{}",
            # Equipment names
            "Equipment {}", "Equipment{}", "Eq{}"
        ]

        logger.info("Testing common resource naming conventions")

        for pattern in common_patterns:
            for i in range(1, 11):  # Check first 10 of each pattern
                name = pattern.format(i)
                try:
                    index = self.siman.SymbolNumber(name)
                    if index > 0:
                        resources.append({
                            'name': name,
                            'index': index,
                            'type': 'common_convention',
                            'pattern': pattern
                        })
                        logger.debug(f"Common convention found: {name} (index: {index})")
                except:
                    continue

        return {'resources': resources}

    def _discover_by_index(self):
        """Discover resources by systematic index exploration"""
        resources = []

        if not self.siman:
            return {'resources': [], 'range_checked': 0}

        logger.info("Exploring resources by index range")

        max_index = 100  # Reasonable upper limit for resource indices
        consecutive_failures = 0
        max_consecutive_failures = 10

        for i in range(1, max_index + 1):
            try:
                # Try to get resource state by index
                state = self.siman.ResourceState(i)

                # If successful, we found a resource
                resources.append({
                    'name': f'Resource_{i}',  # Generate name based on index
                    'index': i,
                    'type': 'index_discovery',
                    'current_state': state,
                    'state_name': self.state_mappings.get(state, f"State_{state}")
                })
                logger.debug(f"Resource found at index {i}, state: {state}")
                consecutive_failures = 0

            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.debug(f"Stopping index exploration after {max_consecutive_failures} consecutive failures")
                    break

        return {'resources': resources, 'range_checked': i}

    def _explore_siman_symbols(self):
        """Explore SIMAN symbol table for resource identifiers"""
        resources = []

        if not self.siman:
            return {'resources': []}

        logger.info("Exploring SIMAN symbol table")

        # Try to access various SIMAN collections that might contain resources
        try:
            # Common SIMAN symbol prefixes for resources
            prefixes = ['R_', 'RES_', 'RESOURCE_', 'M_', 'MACH_', 'S_', 'STAT_']

            for prefix in prefixes:
                for i in range(1, 21):
                    name = f"{prefix}{i}"
                    try:
                        index = self.siman.SymbolNumber(name)
                        if index > 0:
                            resources.append({
                                'name': name,
                                'index': index,
                                'type': 'siman_symbol',
                                'prefix': prefix
                            })
                            logger.debug(f"SIMAN symbol found: {name} (index: {index})")
                    except:
                        continue
        except Exception as e:
            logger.debug(f"SIMAN symbol exploration note: {e}")

        return {'resources': resources}

    def _should_attempt_connection(self):
        """Check if enough time has passed since last connection attempt"""
        current_time = time.time()
        return (current_time - self._last_connection_attempt) >= self._connection_cooldown

    def get_machine_state(self, machine_name):
        """Get state of a specific machine/resource"""
        if not self.model_loaded:
            return {
                'state': 'ERROR',
                'error': 'No model loaded',
                'action_required': 'Load Arena model first'
            }

        if not self.discovery_complete or not self.verified_resources:
            return {
                'state': 'ERROR',
                'error': 'No resources discovered',
                'action_required': 'Complete resource discovery first'
            }

        # Handle various naming formats
        machine_name = str(machine_name).strip()

        # Check if it's a verified resource
        if machine_name not in self.verified_resources:
            # Try to find a matching resource
            matched_resource = self._find_matching_resource(machine_name)
            if matched_resource:
                machine_name = matched_resource
            else:
                return {
                    'state': 'UNKNOWN',
                    'error': f'Resource {machine_name} not found',
                    'available_resources': self.verified_resources,
                    'action_required': 'Use one of the discovered resource names'
                }

        try:
            self._ensure_com_initialized()

            resource_index = self.resource_indices.get(machine_name)
            if not resource_index:
                return {
                    'state': 'ERROR',
                    'error': f'No index mapping for {machine_name}',
                    'action_required': 'Re-run resource discovery'
                }

            state_num = self.siman.ResourceState(resource_index)
            state_name = self.state_mappings.get(state_num, f"State_{state_num}")

            return {
                'state': state_name,
                'state_number': state_num,
                'resource_index': resource_index,
                'resource_name': machine_name
            }

        except Exception as e:
            logger.error(f"Error reading {machine_name} state: {e}")
            return {
                'state': 'ERROR',
                'error': str(e),
                'action_required': 'Check SIMAN interface connection'
            }

    def _find_matching_resource(self, requested_name):
        """Find a matching resource name from discovered resources"""
        requested_lower = requested_name.lower()

        # Direct match
        for resource in self.verified_resources:
            if resource.lower() == requested_lower:
                return resource

        # Partial match
        for resource in self.verified_resources:
            if requested_lower in resource.lower() or resource.lower() in requested_lower:
                return resource

        # Number-based match (e.g., "S1" might match "Station 1")
        import re
        requested_numbers = re.findall(r'\d+', requested_name)
        if requested_numbers:
            for resource in self.verified_resources:
                resource_numbers = re.findall(r'\d+', resource)
                if resource_numbers and requested_numbers[0] == resource_numbers[0]:
                    return resource

        return None

    def get_all_machine_states(self):
        """Get states of all discovered resources"""
        if not self.model_loaded:
            return {
                'error': 'No model loaded',
                'action_required': 'Load Arena model first'
            }

        if not self.discovery_complete:
            return {
                'error': 'Resource discovery not complete',
                'action_required': 'Complete model loading first'
            }

        states = {}
        for machine_name in self.verified_resources:
            state_result = self.get_machine_state(machine_name)

            if 'error' in state_result and state_result['state'] == 'ERROR':
                states[machine_name] = {
                    "status": "ERROR",
                    "error": state_result['error'],
                    "last_updated": datetime.now().isoformat()
                }
            else:
                states[machine_name] = {
                    "status": state_result.get('state', 'UNKNOWN'),
                    "state_number": state_result.get('state_number'),
                    "resource_index": state_result.get('resource_index'),
                    "last_updated": datetime.now().isoformat()
                }

        return states

    def get_simulation_time(self):
        """Get current simulation time"""
        if not self.model_loaded or not self.siman:
            return None

        try:
            self._ensure_com_initialized()

            # Try multiple methods to get simulation time
            try:
                sim_time = self.siman.RunCurrentTime
                return sim_time
            except:
                try:
                    sim_time = self.siman.SimTime
                    return sim_time
                except:
                    # Last resort - use TNOW
                    sim_time = self.siman.SymbolValue("TNOW")
                    return sim_time
        except Exception as e:
            logger.debug(f"Cannot read simulation time: {e}")
            return None

    def get_simulation_state(self):
        """Determine current simulation state"""
        if not self.model_loaded:
            return "NoModelLoaded"

        try:
            # Check if simulation is running by comparing time samples
            time1 = self.get_simulation_time()
            if time1 is None:
                return "Unknown"

            time.sleep(0.1)
            time2 = self.get_simulation_time()

            if time2 is None:
                return "Unknown"
            elif time2 > time1:
                return "Running"
            else:
                return "Stopped"

        except Exception as e:
            logger.debug(f"Cannot determine simulation state: {e}")
            return "Unknown"

    def start_simulation(self):
        """Start Arena simulation"""
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'No model loaded',
                'action_required': 'Load Arena model first'
            }

        try:
            self._ensure_com_initialized()

            # Start simulation
            self.arena_model.Go()
            logger.info("Simulation start command sent")

            time.sleep(0.5)
            new_state = self.get_simulation_state()

            return {
                'success': True,
                'message': f'Simulation started - State: {new_state}',
                'simulation_state': new_state
            }

        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            return {
                'success': False,
                'error': f'Start simulation failed: {str(e)}',
                'action_required': 'Check Arena model configuration'
            }

    def stop_simulation(self):
        """Stop Arena simulation"""
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'No model loaded',
                'action_required': 'Load Arena model first'
            }

        try:
            self._ensure_com_initialized()

            self.arena_model.End()
            logger.info("Simulation stop command sent")

            time.sleep(0.5)
            new_state = self.get_simulation_state()

            return {
                'success': True,
                'message': 'Simulation stopped',
                'simulation_state': new_state
            }

        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            return {
                'success': False,
                'error': f'Stop simulation failed: {str(e)}',
                'action_required': 'Try stopping manually in Arena'
            }

    def get_comprehensive_status(self):
        """Get comprehensive status including discovery details"""
        status = {
            'connection': {
                'connected': self.connected,
                'model_loaded': self.model_loaded,
                'model_path': self.model_path,
                'model_path_exists': os.path.exists(self.model_path) if self.model_path else False,
                'verified_resources': self.verified_resources,
                'discovery_complete': self.discovery_complete,
                'resource_count': len(self.verified_resources),
                'expected_resources': self.expected_resource_count,
                'resource_configuration': self.resource_config
            },
            'timestamp': datetime.now().isoformat()
        }

        if self.connected:
            try:
                status['connection']['arena_version'] = self.arena_app.Version
            except:
                status['connection']['arena_version'] = 'Unknown'

        if self.model_loaded:
            sim_time = self.get_simulation_time()
            sim_state = self.get_simulation_state()

            status['connection'].update({
                'simulation_state': sim_state,
                'simulation_time': sim_time
            })

            # Get machine states
            status['machines'] = self.get_all_machine_states()

            # Add discovery metadata
            status['discovery_metadata'] = {
                'total_resources': len(self.discovered_resources),
                'resource_types': list(set(r.get('type', 'unknown') for r in self.discovered_resources)),
                'discovery_methods': list(set(r.get('discovery_method', 'unknown') for r in self.discovered_resources if 'discovery_method' in r))
            }
        else:
            status['error'] = 'No Arena model loaded'
            status['action_required'] = 'Load Arena model to discover resources'

        return status

    def get_status(self):
        """Get basic status information"""
        return {
            'connected': self.connected,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'simulation_state': self.get_simulation_state() if self.model_loaded else 'NoModelLoaded',
            'simulation_time': self.get_simulation_time() if self.model_loaded else None,
            'verified_resources': self.verified_resources,
            'resource_count': len(self.verified_resources),
            'discovery_complete': self.discovery_complete,
            'last_checked': datetime.now().isoformat()
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            with self._connection_lock:
                if hasattr(self._thread_local, 'com_initialized'):
                    try:
                        pythoncom.CoUninitialize()
                    except:
                        pass
            logger.info("COM resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")