import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Add validation_algorithms to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation_algorithms'))

from mLCSS_TIC import mLCSS_TIC_Validator
from dtw_tic_validator import DTW_TIC_Validator
from LCSS import LCSS_Validator

logger = logging.getLogger('IssueIdentificationModule')


def theils_inequality_coefficient(real_data, predicted_data):
    """
    Calculate Theil's Inequality Coefficient (TIC) between real and predicted data.
    
    TIC = 0: Perfect prediction
    TIC = 1: Worst possible prediction (no better than naive forecast)
    TIC > 1: Worse than naive forecast
    
    Args:
        real_data (array-like): Real observed values
        predicted_data (array-like): Predicted values
        
    Returns:
        float: TIC value
    """
    real_data = np.array(real_data)
    predicted_data = np.array(predicted_data)
    
    # Ensure same length
    min_len = min(len(real_data), len(predicted_data))
    real_data = real_data[:min_len]
    predicted_data = predicted_data[:min_len]
    
    if len(real_data) < 2:
        return 1.0  # Return worst case for insufficient data
    
    # Calculate mean squared error
    mse = np.mean((real_data - predicted_data) ** 2)
    
    # Calculate denominator (sum of squared real values and predicted values)
    denominator = np.mean(real_data ** 2) + np.mean(predicted_data ** 2)
    
    if denominator == 0:
        return 0.0  # Perfect match when both are zero
    
    tic = np.sqrt(mse) / np.sqrt(denominator)
    return tic


class IssueIdentificationModule:
    def __init__(self, tic_threshold=0.1, validation_threshold=0.8):
        """
        Initialize Issue Identification Module
        
        Args:
            tic_threshold (float): TIC threshold for identifying problematic stations
            validation_threshold (float): Validation threshold (kept for compatibility)
        """
        self.tic_threshold = tic_threshold
        self.validation_threshold = validation_threshold
        
        # Initialize validation algorithms (kept for compatibility but not used for calibration decisions)
        self.mlcss_validator = mLCSS_TIC_Validator()
        self.dtw_validator = DTW_TIC_Validator()
        self.lcss_validator = LCSS_Validator()
        
        logger.info(f"IssueIdentificationModule initialized with TIC threshold: {tic_threshold}, Validation threshold: {validation_threshold}")

        # Internal state
        self.latest_validation_results = {}
        self.problematic_stations = []
        self.calibration_required = False

    def _get_available_stations(self):
        """Get list of available stations based on existing processing time files"""
        stations = []
        for station in ['S1', 'S2', 'S3', 'S4', 'S5']:
            real_file = f"{station}.csv"
            if os.path.exists(real_file):
                stations.append(station)
        return stations

    def _load_real_processing_times(self, station):
        """
        Load real processing times from station CSV file
        
        Args:
            station (str): Station name (e.g., 'S1', 'S2')
            
        Returns:
            list: Real processing times
        """
        try:
            file_path = f"{station}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'processing_time' in df.columns:
                    times = df['processing_time'].dropna().tolist()
                    logger.debug(f"Loaded {len(times)} real processing times for {station}")
                    return times
                else:
                    logger.warning(f"No 'processing_time' column found in {file_path}")
                    return []
            else:
                logger.warning(f"Real processing times file not found: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading real processing times for {station}: {e}")
            return []

    def _load_correlated_processing_times(self, station):
        """
        Load correlated processing times from correlated_traces/correlated_processing_time_{station}.txt
        
        Args:
            station (str): Station name (e.g., 'S1', 'S2')
            
        Returns:
            list: Correlated processing times
        """
        try:
            file_path = f"correlated_traces/correlated_processing_time_{station}.txt"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    times = [float(line.strip()) for line in f if line.strip()]
                logger.debug(f"Loaded {len(times)} correlated processing times for {station}")
                return times
            else:
                logger.warning(f"Correlated processing times file not found: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading correlated processing times for {station}: {e}")
            return []

    def _calculate_station_validation_indicators(self, real_times, correlated_times, selected_algorithm='mlcss'):
        """
        Calculate TIC (Theil's Inequality Coefficient) ONLY - no validation algorithms
        This method now focuses on TIC analysis only, removing station-level validation
        
        Args:
            real_times (list): Real processing times
            correlated_times (list): Correlated processing times
            selected_algorithm (str): Not used, kept for compatibility
            
        Returns:
            dict: TIC analysis result only
        """
        try:
            # Ensure we have data
            if len(real_times) == 0 or len(correlated_times) == 0:
                return {
                    'algorithm': 'tic_only',
                    'tic_value': None,
                    'error': 'No data available'
                }
            
            # Limit data size to prevent memory issues (use last 1000 points)
            max_points = 1000
            if len(real_times) > max_points:
                real_times = real_times[-max_points:]
            if len(correlated_times) > max_points:
                correlated_times = correlated_times[-max_points:]
            
            # Equalize lengths for comparison
            min_len = min(len(real_times), len(correlated_times))
            if min_len < 2:
                return {
                    'algorithm': 'tic_only',
                    'tic_value': None,
                    'error': 'Insufficient data points'
                }
            
            real_arr = np.array(real_times[:min_len])
            corr_arr = np.array(correlated_times[:min_len])
            
            # Calculate ONLY TIC - no validation algorithms
            tic_value = theils_inequality_coefficient(real_arr, corr_arr)
            
            return {
                'algorithm': 'tic_only',
                'tic_value': round(tic_value, 4),
                'data_points': min_len
            }
            
        except Exception as e:
            logger.error(f"Error calculating TIC: {e}")
            return {
                'algorithm': 'tic_only',
                'tic_value': None,
                'error': str(e)
            }

    def analyze_station_performance(self, station, selected_algorithm='mlcss', validation_threshold=None):
        """
        Analyze performance of a single station using TIC analysis only
        
        Args:
            station (str): Station name (e.g., 'S1', 'S2')
            selected_algorithm (str): Not used, kept for compatibility
            validation_threshold (float): Not used, kept for compatibility
            
        Returns:
            dict: Station analysis results based on TIC only
        """
        # Load processing times
        real_times = self._load_real_processing_times(station)
        correlated_times = self._load_correlated_processing_times(station)
        
        # Calculate TIC only (no validation algorithms)
        tic_results = self._calculate_station_validation_indicators(real_times, correlated_times, selected_algorithm)
        tic_value = tic_results.get('tic_value')
        
        # Determine if station is problematic based on TIC threshold ONLY
        # System-level validation failure triggers TIC analysis, but calibration
        # should only happen when TIC shows actual dissimilarity (TIC > threshold)
        is_problematic = False
        reasons = []
        
        # Only check TIC threshold for calibration decision
        if tic_value is not None and tic_value > self.tic_threshold:
            is_problematic = True
            reasons.append(f"TIC value ({tic_value:.3f}) above threshold ({self.tic_threshold})")
        else:
            if tic_value is not None:
                reasons.append(f"TIC value ({tic_value:.3f}) acceptable (≤ {self.tic_threshold})")
            else:
                reasons.append("TIC calculation failed - insufficient data")
        
        return {
            'station': station,
            'real_processing_times_count': len(real_times),
            'correlated_processing_times_count': len(correlated_times),
            'tic_result': tic_results,
            'selected_algorithm': 'tic_only',
            'tic_threshold': self.tic_threshold,
            'tic_value': tic_value,
            'is_problematic': is_problematic,
            'reasons': reasons,
            'needs_calibration': is_problematic,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def run_station_level_analysis(self, stations=None, selected_algorithm='mlcss'):
        """
        Run TIC-only analysis for all or specified stations
        
        Args:
            stations (list, optional): List of station names to analyze. If None, analyzes all available stations.
            selected_algorithm (str): Not used, kept for compatibility
            
        Returns:
            tuple: (analysis_results, problematic_stations)
        """
        if stations is None:
            stations = self._get_available_stations()
        
        if not stations:
            logger.warning("No stations found for analysis")
            return {}, []
        
        analysis_results = {}
        problematic_stations = []
        
        logger.info(f"Starting TIC-only analysis for stations: {stations}")
        
        for station in stations:
            try:
                result = self.analyze_station_performance(station, selected_algorithm)
                analysis_results[station] = result
                
                if result['is_problematic']:
                    problematic_stations.append(station)
                    logger.info(f"Station {station} identified as problematic: {', '.join(result['reasons'])}")
                else:
                    logger.info(f"Station {station} performance is acceptable")
                    
            except Exception as e:
                logger.error(f"Error analyzing station {station}: {e}")
                analysis_results[station] = {
                    'station': station,
                    'error': str(e),
                    'is_problematic': False,
                    'needs_calibration': False,
                    'selected_algorithm': 'tic_only',
                    'analysis_timestamp': datetime.now().isoformat()
                }
        
        # Update internal state
        self.latest_validation_results = analysis_results
        self.problematic_stations = problematic_stations
        self.calibration_required = len(problematic_stations) > 0
        
        logger.info(f"TIC analysis complete: {len(analysis_results)} stations analyzed, {len(problematic_stations)} problematic")
        
        return analysis_results, problematic_stations

    # Legacy methods for backward compatibility with existing system
    def run_tic_analysis(self, digital_input_file=None, real_data_source=None, mode="offline", tic_threshold=None):
        """
        Legacy method for backward compatibility - now uses station-level analysis
        """
        logger.info("Running TIC analysis (legacy method - using station-level analysis)")
        
        if tic_threshold is not None:
            self.tic_threshold = tic_threshold
        
        analysis_results, problematic_stations = self.run_station_level_analysis()
        
        return {
            'analysis_results': analysis_results,
            'problematic_stations': problematic_stations,
            'calibration_required': len(problematic_stations) > 0,
            'tic_threshold': self.tic_threshold
        }

    def get_latest_results(self):
        """Get latest analysis results"""
        return {
            'validation_results': self.latest_validation_results,
            'problematic_stations': self.problematic_stations,
            'calibration_required': self.calibration_required,
            'timestamp': datetime.now().isoformat()
        }

    def reset_state(self):
        """Reset internal state"""
        self.latest_validation_results = {}
        self.problematic_stations = []
        self.calibration_required = False
        logger.info("Issue identification state reset")

