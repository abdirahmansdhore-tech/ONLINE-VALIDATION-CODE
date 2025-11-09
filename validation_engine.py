import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import threading

# Add validation_algorithms to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation_algorithms'))

from mLCSS_TIC import mLCSS_TIC_Validator
from dtw_tic_validator import DTW_TIC_Validator
from LCSS import LCSS_Validator

logger = logging.getLogger('ValidationEngine')


class ValidationEngine:
    def __init__(self, config):
        self.config = config
        self.validation_config = config.get('validation_config', {})

        # Validation thresholds for different algorithms
        self.thresholds = self.validation_config.get('thresholds', {
            'lcss': 0.85,
            'mlcss': 0.90,
            'dtw': 0.95
        })

        # LCSS epsilon factor for event-level validation
        self.lcss_epsilon_factor = self.validation_config.get('lcss_epsilon_factor', 0.2)

        # Initialize validation algorithms
        self.mlcss_validator = mLCSS_TIC_Validator()
        self.dtw_validator = DTW_TIC_Validator()
        self.lcss_validator = LCSS_Validator()

        # Latest validation results storage
        self.latest_results = {
            'lcss': 0.0,
            'mlcss': 0.0,
            'dtw': 0.0,
            'timestamp': datetime.now()
        }

        self.validation_count = 0

        # Current validation configuration
        self.current_config = {
            'validation_level': 'kpi',
            'kpi_algorithm': 'mlcss',
            'kpi_metric': 'interdeparture_time',
            'target_station': 'S5'
        }

        # Load initial configuration
        self._load_initial_config()

        logger.info("ValidationEngine initialized - FIXED implementation")

    def _load_initial_config(self):
        """Load initial configuration from config file"""
        val_config = self.validation_config

        if 'kpi_metric' in val_config:
            self.current_config['kpi_metric'] = val_config['kpi_metric']

        if 'selected_algorithm' in val_config:
            self.current_config['kpi_algorithm'] = val_config['selected_algorithm']

        if 'validation_level' in val_config:
            self.current_config['validation_level'] = val_config['validation_level']

        logger.info(f"Config loaded: Level={self.current_config['validation_level']}, "
                    f"Algorithm={self.current_config['kpi_algorithm']}, "
                    f"Metric={self.current_config['kpi_metric']}")

    def update_configuration(self, config_data):
        """Update validation configuration"""
        try:
            if 'kpi_metric' in config_data:
                valid_metrics = ['interdeparture_time', 'system_time']
                if config_data['kpi_metric'] not in valid_metrics:
                    logger.error(f"Invalid KPI metric: {config_data['kpi_metric']}")
                    return False

                old_metric = self.current_config['kpi_metric']
                self.current_config['kpi_metric'] = config_data['kpi_metric']
                logger.info(f"KPI metric changed: {old_metric} -> {config_data['kpi_metric']}")

            for key in ['validation_level', 'kpi_algorithm', 'target_station']:
                if key in config_data:
                    old_value = self.current_config.get(key)
                    self.current_config[key] = config_data[key]
                    logger.info(f"Config {key} changed: {old_value} -> {config_data[key]}")

            return True

        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

    def run_validation(self, historical_data, simulation_data):
        """
        MAIN VALIDATION METHOD - COMPLETELY FIXED
        """
        try:
            # FIXED: Simple data preparation - NO DATA CORRUPTION
            hist_df = self._prepare_data_safely(historical_data)
            sim_df = self._prepare_data_safely(simulation_data)

            if hist_df is None or sim_df is None:
                logger.error("Invalid data provided for validation")
                return {'lcss': 0.0, 'mlcss': 0.0, 'dtw': 0.0}

            logger.info(f"Validation input: hist={len(hist_df)} rows, sim={len(sim_df)} rows")
            logger.info(f"Historical columns: {list(hist_df.columns)}")
            logger.info(f"Simulation columns: {list(sim_df.columns)}")

            results = {}
            validation_level = self.current_config.get('validation_level', 'kpi')
            kpi_algorithm = self.current_config.get('kpi_algorithm', 'mlcss')
            kpi_metric = self.current_config.get('kpi_metric', 'interdeparture_time')

            # KPI-level validation
            if validation_level == 'kpi':
                logger.info(f"KPI validation: {kpi_algorithm} on {kpi_metric}")

                # FIXED: Use only selected algorithm - no fallbacks
                if kpi_algorithm == 'mlcss':
                    mlcss_result = self.run_mlcss_validation(hist_df, sim_df, kpi_metric)
                    results['mlcss'] = mlcss_result
                    results['dtw'] = 0.0
                elif kpi_algorithm == 'dtw':
                    dtw_result = self.run_dtw_validation(hist_df, sim_df, kpi_metric)
                    results['dtw'] = dtw_result
                    results['mlcss'] = 0.0

                results['lcss'] = 0.0

            # Event-level validation
            elif validation_level == 'event':
                logger.info("Event-level LCSS validation")
                lcss_result = self.run_lcss_validation(hist_df, sim_df)
                results['lcss'] = lcss_result
                results['mlcss'] = 0.0
                results['dtw'] = 0.0

            # Ensure all keys exist
            for key in ['mlcss', 'dtw', 'lcss']:
                results.setdefault(key, 0.0)

            # Update results
            self.latest_results.update(results)
            self.latest_results['timestamp'] = datetime.now()
            self.validation_count += 1

            # Log final results
            active_metric = kpi_metric if validation_level == 'kpi' else 'events'
            active_result = results.get(kpi_algorithm, results.get('lcss', 0.0))
            logger.info(f"Validation #{self.validation_count}: {active_metric} -> {active_result:.3f}")

            return results

        except Exception as e:
            logger.error(f"Critical error in run_validation: {e}")
            import traceback
            traceback.print_exc()
            return {'lcss': 0.0, 'mlcss': 0.0, 'dtw': 0.0}

    def _prepare_data_safely(self, data):
        """FIXED data preparation - minimal processing to avoid corruption"""
        if data is None:
            return None

        if hasattr(data, 'empty') and data.empty:
            return None

        # Simple copy
        df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)

        # ONLY strip column whitespace - NO OTHER MODIFICATIONS
        df.columns = [str(col).strip() for col in df.columns]

        return df

    def run_mlcss_validation(self, historical_data, simulation_data, kpi_metric):
        """COMPLETELY FIXED mLCSS validation"""
        try:
            logger.info(f"=== mLCSS VALIDATION: {kpi_metric} ===")

            # STRICT: Verify exact column exists - NO FALLBACKS
            if kpi_metric not in historical_data.columns:
                logger.error(f"Historical data MISSING column '{kpi_metric}'")
                logger.error(f"Available: {list(historical_data.columns)}")
                return 0.0

            if kpi_metric not in simulation_data.columns:
                logger.error(f"Simulation data MISSING column '{kpi_metric}'")
                logger.error(f"Available: {list(simulation_data.columns)}")
                return 0.0

            # FIXED: Extract values and align properly
            hist_raw = historical_data[kpi_metric]
            sim_raw = simulation_data[kpi_metric]

            logger.info(f"Raw data: hist={len(hist_raw)} rows, sim={len(sim_raw)} rows")

            # FIXED: Convert to numeric
            hist_numeric = pd.to_numeric(hist_raw, errors='coerce')
            sim_numeric = pd.to_numeric(sim_raw, errors='coerce')

            # CRITICAL FIX: Ensure same length BEFORE filtering
            min_length = min(len(hist_numeric), len(sim_numeric))
            hist_trimmed = hist_numeric.iloc[:min_length]
            sim_trimmed = sim_numeric.iloc[:min_length]

            # CRITICAL FIX: Remove NaN PAIRS together - maintain alignment
            valid_pairs_mask = hist_trimmed.notna() & sim_trimmed.notna()
            hist_clean = hist_trimmed[valid_pairs_mask].values
            sim_clean = sim_trimmed[valid_pairs_mask].values

            logger.info(f"After cleaning: {len(hist_clean)} valid aligned pairs")

            if len(hist_clean) < 2:
                logger.error(f"Insufficient clean data: {len(hist_clean)} pairs")
                return 0.0

            # FIXED: Proper statistical analysis
            hist_mean = np.mean(hist_clean)
            hist_std = np.std(hist_clean)
            sim_mean = np.mean(sim_clean)

            logger.info(f"Stats - Hist: mean={hist_mean:.2f}, std={hist_std:.2f}")
            logger.info(f"Stats - Sim: mean={sim_mean:.2f}")

            # FIXED: Reasonable delta calculation
            # Use standard deviation as primary criterion, with minimum of 10% mean
            delta = max(hist_std * 0.8, hist_mean * 0.1)

            logger.info(f"mLCSS delta: {delta:.3f}")

            # Test correlation first
            if len(hist_clean) > 1:
                correlation = np.corrcoef(hist_clean, sim_clean)[0, 1]
                logger.info(f"Data correlation: {correlation:.3f}")

            # Run mLCSS validation
            similarity = float(self.mlcss_validator.validate(hist_clean, sim_clean, delta=delta))

            logger.info(f"mLCSS FINAL result: {similarity:.3f}")
            logger.info(f"========================================")

            return similarity

        except Exception as e:
            logger.error(f"mLCSS validation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def run_dtw_validation(self, historical_data, simulation_data, kpi_metric):
        """COMPLETELY FIXED DTW validation"""
        try:
            logger.info(f"=== DTW VALIDATION: {kpi_metric} ===")

            # STRICT: No fallbacks
            if kpi_metric not in historical_data.columns:
                logger.error(f"Historical missing '{kpi_metric}' for DTW")
                return 0.0

            if kpi_metric not in simulation_data.columns:
                logger.error(f"Simulation missing '{kpi_metric}' for DTW")
                return 0.0

            # Extract and align data
            hist_raw = historical_data[kpi_metric]
            sim_raw = simulation_data[kpi_metric]

            hist_numeric = pd.to_numeric(hist_raw, errors='coerce')
            sim_numeric = pd.to_numeric(sim_raw, errors='coerce')

            # Ensure same length before filtering
            min_length = min(len(hist_numeric), len(sim_numeric))
            hist_trimmed = hist_numeric.iloc[:min_length]
            sim_trimmed = sim_numeric.iloc[:min_length]

            # Remove invalid pairs together
            valid_pairs_mask = hist_trimmed.notna() & sim_trimmed.notna()
            hist_clean = hist_trimmed[valid_pairs_mask].values
            sim_clean = sim_trimmed[valid_pairs_mask].values

            logger.info(f"DTW aligned data: {len(hist_clean)} pairs")

            if len(hist_clean) < 2:
                logger.error(f"Insufficient DTW data: {len(hist_clean)} pairs")
                return 0.0

            # Run DTW validation
            similarity = float(self.dtw_validator.validate(hist_clean, sim_clean))

            logger.info(f"DTW FINAL result: {similarity:.3f}")
            logger.info(f"====================================")

            return similarity

        except Exception as e:
            logger.error(f"DTW validation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def run_lcss_validation(self, historical_data, simulation_data):
        """FIXED LCSS validation for event sequences"""
        try:
            logger.info(f"=== LCSS EVENT VALIDATION ===")

            # Extract event sequences
            hist_seq = self.extract_event_sequence_with_time(historical_data)
            sim_seq = self.extract_event_sequence_with_time(simulation_data)

            logger.info(f"Event sequences: hist={len(hist_seq)}, sim={len(sim_seq)}")

            if len(hist_seq) < 2 or len(sim_seq) < 2:
                logger.error(f"Insufficient events for LCSS: hist={len(hist_seq)}, sim={len(sim_seq)}")
                return 0.0

            # Calculate epsilon based on historical timing
            epsilon = self._calculate_epsilon_from_events(hist_seq)
            logger.info(f"LCSS epsilon: {epsilon:.2f} seconds")

            # Run LCSS validation
            similarity = float(self.lcss_validator.validate(hist_seq, sim_seq, time_epsilon=epsilon))

            logger.info(f"LCSS FINAL result: {similarity:.3f}")
            logger.info(f"==============================")

            return similarity

        except Exception as e:
            logger.error(f"LCSS validation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def extract_event_sequence_with_time(self, data):
        """FIXED event sequence extraction"""
        try:
            if data is None or len(data) == 0:
                return []

            df = data.copy()

            # Check required columns
            if 'activity' not in df.columns or 'time' not in df.columns:
                logger.warning(f"Missing required columns. Available: {list(df.columns)}")
                return []

            # FIXED: Safe time conversion
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

            # Remove only rows with invalid timestamps
            valid_time_mask = df['time'].notna()
            df = df[valid_time_mask]

            if len(df) == 0:
                logger.warning("No valid timestamps found")
                return []

            # Sort by time and create sequence
            df = df.sort_values('time')

            sequence = []
            for _, row in df.iterrows():
                activity = str(row['activity']).strip()
                timestamp = row['time']
                sequence.append((activity, timestamp))

            return sequence

        except Exception as e:
            logger.error(f"Error extracting event sequence: {e}")
            return []

    def _calculate_epsilon_from_events(self, event_sequence):
        """Calculate appropriate epsilon for LCSS"""
        try:
            if len(event_sequence) < 2:
                return 5.0

            # Extract timestamps and calculate intervals
            timestamps = [t for _, t in event_sequence]

            time_seconds = []
            for t in timestamps:
                if isinstance(t, datetime):
                    time_seconds.append(t.timestamp())
                else:
                    time_seconds.append(float(pd.to_datetime(t).timestamp()))

            if len(time_seconds) > 1:
                intervals = np.diff(sorted(time_seconds))
                mean_interval = np.mean(intervals) if len(intervals) > 0 else 1.0
                epsilon = mean_interval * self.lcss_epsilon_factor
                return max(epsilon, 1.0)  # At least 1 second
            else:
                return 5.0

        except Exception as e:
            logger.error(f"Error calculating epsilon: {e}")
            return 5.0

    def run_offline_validation(self, historical_data, simulation_data, window_size=50, stride=50):
        """Run validation on moving windows for offline analysis"""
        results_list = []

        try:
            n = min(len(historical_data), len(simulation_data))
            logger.info(f"Offline validation: {n} rows, window={window_size}, stride={stride}")

            windows_processed = 0
            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                h_window = historical_data.iloc[start:end].copy()
                s_window = simulation_data.iloc[start:end].copy()

                logger.debug(f"Processing window {start}:{end}")

                window_results = self.run_validation(h_window, s_window)

                results_list.append({
                    'start': start,
                    'end': end,
                    'results': window_results,
                    'window_id': windows_processed,
                    'kpi_metric': self.current_config['kpi_metric'] if self.current_config[
                                                                           'validation_level'] == 'kpi' else 'N/A'
                })

                windows_processed += 1

            logger.info(f"Offline validation complete: {windows_processed} windows processed")
            return results_list

        except Exception as e:
            logger.error(f"Offline validation error: {e}")
            return []

    def get_latest_results(self):
        """Get the latest validation results"""
        return self.latest_results.copy()

    def get_status(self):
        """Get comprehensive validation engine status"""
        return {
            'validation_count': self.validation_count,
            'algorithms_available': {
                'mlcss': self.mlcss_validator is not None,
                'dtw': self.dtw_validator is not None,
                'lcss': self.lcss_validator is not None
            },
            'thresholds': self.thresholds,
            'current_config': self.current_config.copy(),
            'selected_kpi_metric': self.current_config['kpi_metric'],
            'last_validation': self.latest_results['timestamp'].isoformat() if self.latest_results[
                'timestamp'] else None
        }

    def get_validation_threshold(self):
        """Get threshold for currently selected algorithm"""
        level = self.current_config['validation_level']

        if level == 'kpi':
            algorithm = self.current_config['kpi_algorithm']
            return self.thresholds.get(algorithm, 0.90)
        else:
            return self.thresholds.get('lcss', 0.85)

    def is_validation_successful(self, results):
        """Determine if validation passed"""
        level = self.current_config['validation_level']

        if level == 'kpi':
            algorithm = self.current_config['kpi_algorithm']
            score = results.get(algorithm, 0.0)
            threshold = self.thresholds.get(algorithm, 0.90)
            return score >= threshold
        else:
            score = results.get('lcss', 0.0)
            threshold = self.thresholds.get('lcss', 0.85)
            return score >= threshold

    # =================== DEBUGGING METHODS ===================

    def debug_data_before_validation(self, historical_data, simulation_data, kpi_metric):
        """Debug data before validation to identify issues"""
        try:
            logger.info(f"=== PRE-VALIDATION DEBUG ===")
            logger.info(f"Target KPI metric: {kpi_metric}")

            # Check basic data structure
            logger.info(
                f"Historical data: {historical_data.shape if hasattr(historical_data, 'shape') else 'No shape'}")
            logger.info(
                f"Simulation data: {simulation_data.shape if hasattr(simulation_data, 'shape') else 'No shape'}")

            # Check column existence
            hist_columns = list(historical_data.columns) if hasattr(historical_data, 'columns') else []
            sim_columns = list(simulation_data.columns) if hasattr(simulation_data, 'columns') else []

            logger.info(f"Historical columns: {hist_columns}")
            logger.info(f"Simulation columns: {sim_columns}")

            has_hist_metric = kpi_metric in hist_columns
            has_sim_metric = kpi_metric in sim_columns

            logger.info(f"Historical has {kpi_metric}: {has_hist_metric}")
            logger.info(f"Simulation has {kpi_metric}: {has_sim_metric}")

            if has_hist_metric and has_sim_metric:
                # Analyze the data values
                hist_data = pd.to_numeric(historical_data[kpi_metric], errors='coerce')
                sim_data = pd.to_numeric(simulation_data[kpi_metric], errors='coerce')

                hist_valid = hist_data.notna().sum()
                sim_valid = sim_data.notna().sum()

                logger.info(f"Valid values: hist={hist_valid}/{len(hist_data)}, sim={sim_valid}/{len(sim_data)}")

                if hist_valid > 0:
                    logger.info(
                        f"Historical stats: min={hist_data.min():.2f}, max={hist_data.max():.2f}, mean={hist_data.mean():.2f}")

                if sim_valid > 0:
                    logger.info(
                        f"Simulation stats: min={sim_data.min():.2f}, max={sim_data.max():.2f}, mean={sim_data.mean():.2f}")

                # Show sample values
                logger.info(f"Historical sample: {hist_data.dropna().head(5).tolist()}")
                logger.info(f"Simulation sample: {sim_data.dropna().head(5).tolist()}")

            logger.info(f"===============================")

        except Exception as e:
            logger.error(f"Debug analysis error: {e}")

    def test_algorithms_with_known_data(self):
        """Test validation algorithms with known data to verify correctness"""
        try:
            logger.info("=== ALGORITHM CORRECTNESS TEST ===")

            # Test data - identical sequences should give 1.0 similarity
            test_data = [10.5, 12.3, 11.8, 13.1, 10.9, 12.7, 11.4, 10.2]

            # Test mLCSS with identical data
            delta = np.std(test_data)
            mlcss_identical = float(self.mlcss_validator.validate(test_data, test_data, delta=delta))
            logger.info(f"mLCSS identical test: {mlcss_identical:.3f} (expect ~1.0)")

            # Test DTW with identical data
            dtw_identical = float(self.dtw_validator.validate(test_data, test_data))
            logger.info(f"DTW identical test: {dtw_identical:.3f} (expect ~1.0)")

            # Test with slightly different data
            test_data2 = [x + np.random.normal(0, 0.1) for x in test_data]

            mlcss_similar = float(self.mlcss_validator.validate(test_data, test_data2, delta=delta))
            dtw_similar = float(self.dtw_validator.validate(test_data, test_data2))

            logger.info(f"mLCSS similar test: {mlcss_similar:.3f} (expect >0.8)")
            logger.info(f"DTW similar test: {dtw_similar:.3f} (expect >0.8)")

            # Verify algorithm correctness
            if mlcss_identical < 0.95:
                logger.error("mLCSS algorithm implementation is BROKEN!")
            if dtw_identical < 0.95:
                logger.error("DTW algorithm implementation is BROKEN!")

            logger.info(f"===================================")

            return {
                'mlcss_identical': mlcss_identical,
                'dtw_identical': dtw_identical,
                'mlcss_similar': mlcss_similar,
                'dtw_similar': dtw_similar
            }

        except Exception as e:
            logger.error(f"Algorithm test error: {e}")
            return None

    # ================ LEGACY SUPPORT ================

    def safe_read_csv(self, path):
        """Legacy method - kept for compatibility"""
        if not os.path.exists(path):
            logger.info(f"File not found: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            df.columns = [col.strip() for col in df.columns]
            return df
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return pd.DataFrame()

    def online_validation_loop(self, hist_path, sim_path, kpi_metric='interdeparture_time', interval=30):
        """Legacy online validation loop"""

        def loop():
            hist_df = self.safe_read_csv(hist_path)
            sim_df = self.safe_read_csv(sim_path)
            if not hist_df.empty and not sim_df.empty:
                self.run_validation(hist_df, sim_df)
            else:
                logger.info("Waiting for data files...")

            # Schedule next validation
            self._online_loop_thread = threading.Timer(interval, loop)
            self._online_loop_thread.daemon = True
            self._online_loop_thread.start()

        loop()