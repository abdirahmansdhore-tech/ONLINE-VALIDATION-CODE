
#!/usr/bin/env python3
"""
Calibration Engine - Bootstrap Particle Filter for parameter calibration
======================================================================
"""

import os
import sys
import numpy as np
from datetime import datetime
import logging
import pandas as pd

# ---- Import required functions ----
from utils.quasi_trace_generator import input_trace
from utils.dtw_tic_validator import theils_inequality_coefficient

logger = logging.getLogger("CalibrationEngine")

class CalibrationEngine:
    def __init__(self, config):
        self.config = config
        self.calibration_config = config.get("calibration_config", {})
        self.n_particles = self.calibration_config.get("n_particles", 20)
        self.n_iterations = self.calibration_config.get("n_iterations", 5)
        self.window_size = self.calibration_config.get("window_size", 50)
        self.distribution_config = config.get("dist_configs", {})
        self.calibrated_parameters = {}
        self.calibration_results = {}
        self.calibration_history = []

        logger.info("[OK] CalibrationEngine initialized")

    def calibrate_multiple_stations(self, calibration_requests):
        """
        Calibrate parameters for each flagged station using their real processing times.
        calibration_requests: dict {station: real_processing_times}
        """
        calibration_results = {}
        for station, real_proc_times in calibration_requests.items():
            result = self.calibrate_station(station, real_proc_times)
            if result:
                calibration_results[station] = result
        return calibration_results

    def calibrate_station(self, station, real_processing_times):
        """
        Calibrate parameters for a single station using a particle filter.
        """
        if len(real_processing_times) < 5:
            logger.warning(f"[Calibration] Not enough data for station {station}")
            return None

        # Current config for this station
        cfg = self.distribution_config.get(station, {"dist_code": 3, "params": [10.0, 2.0]})
        dist_code = cfg["dist_code"]
        current_params = cfg["params"]

        logger.info(f"[Calibration] Starting for station {station} (dist_code={dist_code}, params={current_params})")

        # Divide real data into non-overlapping windows
        N = len(real_processing_times)
        window_size = self.window_size
        windows = [real_processing_times[i:i+window_size] for i in range(0, N, window_size) if len(real_processing_times[i:i+window_size]) >= 5]

        window_results = []
        best_all_params = []
        for widx, window in enumerate(windows):
            # 1. Initialize particles around data statistics
            particles = self._init_particles(dist_code, window, self.n_particles)
            best_params = None
            best_tic = float("inf")
            for _ in range(self.n_iterations):
                # 2. For each particle, generate synthetic and evaluate TIC
                weights = []
                for i in range(self.n_particles):
                    synth = input_trace(window, dist_code, particles[i].tolist())
                    tic = theils_inequality_coefficient(window, synth)
                    weight = np.exp(-tic * 10) if tic is not None else 0.0
                    weights.append(weight)
                    if tic is not None and tic < best_tic:
                        best_tic = tic
                        best_params = particles[i].tolist()
                # 3. Resample with noise
                particles = self._resample_particles(particles, weights)
            if best_params is not None:
                window_results.append({
                    "window_index": widx,
                    "calibrated_params": best_params,
                    "tic": best_tic,
                    "window_size": len(window)
                })
                best_all_params.append(best_params)
            logger.info(f"[Calibration] Station {station} window {widx}: Best TIC={best_tic:.4f}, Params={best_params}")

        # Weighted mean of params
        if window_results:
            param_array = np.array([wr['calibrated_params'] for wr in window_results])
            sizes = np.array([wr['window_size'] for wr in window_results])
            weighted_params = np.average(param_array, axis=0, weights=sizes)
            self.calibrated_parameters[station] = {
                "dist_code": dist_code,
                "params": weighted_params.tolist()
            }
            result = {
                "station": station,
                "dist_code": dist_code,
                "params": weighted_params.tolist(),
                "window_results": window_results,
                "calibration_time": datetime.now().isoformat()
            }
            self.calibration_results[station] = result
            self.calibration_history.append({
                "station": station,
                "params": weighted_params.tolist(),
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"[Calibration] Station {station} finished: {weighted_params.tolist()}")
            return result
        else:
            logger.warning(f"[Calibration] No calibration windows succeeded for {station}")
            return None

    def _init_particles(self, dist_code, data, n_particles):
        """
        Initialize particle positions for the chosen distribution.
        """
        mean = np.mean(data)
        std = np.std(data)
        particles = []
        if dist_code == 3:  # Normal: [mu, sigma]
            for _ in range(n_particles):
                mu = np.random.normal(mean, std*0.2)
                sigma = abs(np.random.normal(std, std*0.2)) or 1e-3
                particles.append([mu, sigma])
        elif dist_code == 1:  # Uniform: [a, b]
            dmin, dmax = np.min(data), np.max(data)
            for _ in range(n_particles):
                a = np.random.uniform(dmin*0.9, dmin*1.1)
                b = np.random.uniform(dmax*0.9, dmax*1.1)
                if b < a: a, b = b, a
                particles.append([a, b])
        elif dist_code == 2:  # Triangular: [a, b, c]
            dmin, dmax = np.min(data), np.max(data)
            for _ in range(n_particles):
                a = np.random.uniform(dmin*0.8, dmin*1.2)
                b = np.random.uniform(dmax*0.8, dmax*1.2)
                c = np.random.uniform(a, b)
                particles.append([a, b, c])
        # Add more elifs for other distributions as needed
        else:  # Default: normal
            for _ in range(n_particles):
                mu = np.random.normal(mean, std*0.2)
                sigma = abs(np.random.normal(std, std*0.2)) or 1e-3
                particles.append([mu, sigma])
        return np.array(particles)

    def _resample_particles(self, particles, weights):
        """
        Resample particles according to weights with added noise.
        """
        weights = np.array(weights)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= np.sum(weights)
        idxs = np.random.choice(len(particles), size=len(particles), p=weights)
        new_particles = particles[idxs]
        # Add small noise
        noise = np.random.normal(0, np.std(new_particles, axis=0)*0.05 + 1e-4, new_particles.shape)
        return new_particles + noise

    def get_calibrated_parameters(self, station=None):
        """Return calibrated parameters for a station or all."""
        if station:
            return self.calibrated_parameters.get(station, None)
        return self.calibrated_parameters

    def save_calibration_results(self, filepath=None):
        """Save calibration results/history as JSON."""
        import json
        if filepath is None:
            filepath = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump({
                "calibration_results": self.calibration_results,
                "calibrated_parameters": self.calibrated_parameters,
                "calibration_history": self.calibration_history,
            }, f, indent=2)
        logger.info(f"[OK] Calibration results saved to {filepath}")
        return filepath
