#!/usr/bin/env python3
"""
Signal Processor Module
=======================

This module handles signal preprocessing and filtering for heartbeat detection.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from typing import Dict, List, Optional, Tuple
import collections

class SignalProcessor:
    """Class for processing and filtering sensor signals."""
    
    def __init__(self, sampling_rate=100, filter_low=1.0, filter_high=100.0, 
                 filter_order=4, smoothing_window=5):
        """Initialize the signal processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            filter_low: Low frequency cutoff for bandpass filter
            filter_high: High frequency cutoff for bandpass filter
            filter_order: Order of the Butterworth filter
            smoothing_window: Window size for smoothing filter
        """
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        self.smoothing_window = smoothing_window
        
        # Data buffers for processing
        self.accel_buffer = collections.deque(maxlen=sampling_rate * 10)  # 10 seconds
        self.gyro_buffer = collections.deque(maxlen=sampling_rate * 10)
        self.timestamps = collections.deque(maxlen=sampling_rate * 10)
        
        # Filter coefficients
        self.b, self.a = self._design_bandpass_filter()
        
        # Processing state
        self.initialized = False
        self.sample_count = 0
        
        print(f"Signal processor initialized: {sampling_rate}Hz, {filter_low}-{filter_high}Hz bandpass")
    
    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design a bandpass Butterworth filter.
        
        Returns:
            Tuple of (b, a) filter coefficients
        """
        # Normalize frequencies by Nyquist frequency
        nyquist = self.sampling_rate / 2.0
        low_norm = self.filter_low / nyquist
        high_norm = self.filter_high / nyquist
        
        # Ensure frequencies are within valid range
        low_norm = max(0.001, min(low_norm, 0.99))
        high_norm = max(0.001, min(high_norm, 0.99))
        
        # Design filter
        b, a = butter(self.filter_order, [low_norm, high_norm], btype='band')
        return b, a
    
    def process(self, sensor_data: Dict) -> Optional[Dict]:
        """Process sensor data for heartbeat detection.
        
        Args:
            sensor_data: Dictionary containing sensor data
            
        Returns:
            Dictionary containing processed data
        """
        if not sensor_data:
            return None
        
        try:
            # Extract data
            timestamp = sensor_data.get('timestamp', 0)
            accel_z = sensor_data.get('accel_z', 0.0)
            gyro_z = sensor_data.get('gyro_z', 0.0)
            
            # Add to buffers
            self.accel_buffer.append(accel_z)
            self.gyro_buffer.append(gyro_z)
            self.timestamps.append(timestamp)
            self.sample_count += 1
            
            # Wait for enough data to process
            if len(self.accel_buffer) < self.sampling_rate:
                return None
            
            # Convert buffers to numpy arrays
            accel_array = np.array(self.accel_buffer)
            gyro_array = np.array(self.gyro_buffer)
            time_array = np.array(self.timestamps)
            
            # Apply bandpass filter
            filtered_accel = self._apply_bandpass_filter(accel_array)
            filtered_gyro = self._apply_bandpass_filter(gyro_array)
            
            # Apply smoothing
            smoothed_accel = self._apply_smoothing(filtered_accel)
            smoothed_gyro = self._apply_smoothing(filtered_gyro)
            
            # Calculate signal statistics
            accel_stats = self._calculate_statistics(smoothed_accel)
            gyro_stats = self._calculate_statistics(smoothed_gyro)
            
            # Return processed data
            return {
                'timestamp': timestamp,
                'raw_accel_z': accel_z,
                'raw_gyro_z': gyro_z,
                'filtered_accel_z': filtered_accel[-1] if len(filtered_accel) > 0 else 0.0,
                'filtered_gyro_z': filtered_gyro[-1] if len(filtered_gyro) > 0 else 0.0,
                'smoothed_accel_z': smoothed_accel[-1] if len(smoothed_accel) > 0 else 0.0,
                'smoothed_gyro_z': smoothed_gyro[-1] if len(smoothed_gyro) > 0 else 0.0,
                'accel_stats': accel_stats,
                'gyro_stats': gyro_stats,
                'sample_count': self.sample_count
            }
            
        except Exception as e:
            print(f"Error processing signal: {e}")
            return None
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to data.
        
        Args:
            data: Input signal data
            
        Returns:
            Filtered signal data
        """
        try:
            # Apply zero-phase filtering to avoid phase distortion
            filtered = filtfilt(self.b, self.a, data)
            return filtered
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            return data
    
    def _apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing filter to data.
        
        Args:
            data: Input signal data
            
        Returns:
            Smoothed signal data
        """
        try:
            if len(data) < self.smoothing_window:
                return data
            
            # Use Savitzky-Golay filter for smoothing
            window_length = min(self.smoothing_window, len(data))
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            
            if window_length < 3:
                return data
            
            smoothed = savgol_filter(data, window_length, 2)
            return smoothed
            
        except Exception as e:
            print(f"Error applying smoothing: {e}")
            return data
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict:
        """Calculate statistics for the signal.
        
        Args:
            data: Input signal data
            
        Returns:
            Dictionary containing signal statistics
        """
        try:
            if len(data) == 0:
                return {'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'peak': 0.0}
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            rms_val = np.sqrt(np.mean(data**2))
            peak_val = np.max(np.abs(data))
            
            return {
                'mean': float(mean_val),
                'std': float(std_val),
                'rms': float(rms_val),
                'peak': float(peak_val)
            }
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'peak': 0.0}
    
    def get_buffer_data(self) -> Dict:
        """Get current buffer data for analysis.
        
        Returns:
            Dictionary containing buffer data
        """
        return {
            'accel_buffer': list(self.accel_buffer),
            'gyro_buffer': list(self.gyro_buffer),
            'timestamps': list(self.timestamps),
            'sample_count': self.sample_count
        }
    
    def reset_buffers(self):
        """Reset all data buffers."""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.timestamps.clear()
        self.sample_count = 0
        print("Signal processor buffers reset")

class AdvancedSignalProcessor(SignalProcessor):
    """Advanced signal processor with additional features."""
    
    def __init__(self, sampling_rate=100, filter_low=1.0, filter_high=100.0, 
                 filter_order=4, smoothing_window=5, notch_freq=50.0):
        """Initialize advanced signal processor.
        
        Args:
            notch_freq: Frequency for notch filter (e.g., 50Hz for power line interference)
        """
        super().__init__(sampling_rate, filter_low, filter_high, filter_order, smoothing_window)
        self.notch_freq = notch_freq
        self.notch_b, self.notch_a = self._design_notch_filter()
    
    def _design_notch_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design a notch filter to remove power line interference."""
        try:
            # Normalize frequency
            nyquist = self.sampling_rate / 2.0
            notch_norm = self.notch_freq / nyquist
            
            # Design notch filter
            b, a = signal.iirnotch(self.notch_freq, 30, self.sampling_rate)
            return b, a
        except Exception as e:
            print(f"Error designing notch filter: {e}")
            # Return simple pass-through filter
            return np.array([1.0]), np.array([1.0])
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line interference."""
        try:
            filtered = filtfilt(self.notch_b, self.notch_a, data)
            return filtered
        except Exception as e:
            print(f"Error applying notch filter: {e}")
            return data
    
    def process(self, sensor_data: Dict) -> Optional[Dict]:
        """Process sensor data with advanced filtering."""
        if not sensor_data:
            return None
        
        try:
            # Extract data
            timestamp = sensor_data.get('timestamp', 0)
            accel_z = sensor_data.get('accel_z', 0.0)
            gyro_z = sensor_data.get('gyro_z', 0.0)
            
            # Add to buffers
            self.accel_buffer.append(accel_z)
            self.gyro_buffer.append(gyro_z)
            self.timestamps.append(timestamp)
            self.sample_count += 1
            
            # Wait for enough data
            if len(self.accel_buffer) < self.sampling_rate:
                return None
            
            # Convert to arrays
            accel_array = np.array(self.accel_buffer)
            gyro_array = np.array(self.gyro_buffer)
            
            # Apply notch filter first
            notch_accel = self._apply_notch_filter(accel_array)
            notch_gyro = self._apply_notch_filter(gyro_array)
            
            # Apply bandpass filter
            filtered_accel = self._apply_bandpass_filter(notch_accel)
            filtered_gyro = self._apply_bandpass_filter(notch_gyro)
            
            # Apply smoothing
            smoothed_accel = self._apply_smoothing(filtered_accel)
            smoothed_gyro = self._apply_smoothing(filtered_gyro)
            
            # Calculate statistics
            accel_stats = self._calculate_statistics(smoothed_accel)
            gyro_stats = self._calculate_statistics(smoothed_gyro)
            
            return {
                'timestamp': timestamp,
                'raw_accel_z': accel_z,
                'raw_gyro_z': gyro_z,
                'notch_accel_z': notch_accel[-1] if len(notch_accel) > 0 else 0.0,
                'notch_gyro_z': notch_gyro[-1] if len(notch_gyro) > 0 else 0.0,
                'filtered_accel_z': filtered_accel[-1] if len(filtered_accel) > 0 else 0.0,
                'filtered_gyro_z': filtered_gyro[-1] if len(filtered_gyro) > 0 else 0.0,
                'smoothed_accel_z': smoothed_accel[-1] if len(smoothed_accel) > 0 else 0.0,
                'smoothed_gyro_z': smoothed_gyro[-1] if len(smoothed_gyro) > 0 else 0.0,
                'accel_stats': accel_stats,
                'gyro_stats': gyro_stats,
                'sample_count': self.sample_count
            }
            
        except Exception as e:
            print(f"Error in advanced signal processing: {e}")
            return None 