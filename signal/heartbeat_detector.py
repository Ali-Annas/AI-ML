#!/usr/bin/env python3
"""
Heartbeat Detector Module
=========================

This module implements algorithms for detecting heartbeats from processed sensor data.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple
import collections
import time

class HeartbeatDetector:
    """Class for detecting heartbeats from processed sensor data."""
    
    def __init__(self, threshold=0.1, min_distance=0.5, window_size=100):
        """Initialize the heartbeat detector.
        
        Args:
            threshold: Peak detection threshold
            min_distance: Minimum distance between peaks (seconds)
            window_size: Size of the analysis window
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.window_size = window_size
        
        # Data buffers
        self.signal_buffer = collections.deque(maxlen=window_size)
        self.timestamp_buffer = collections.deque(maxlen=window_size)
        self.peak_times = collections.deque(maxlen=50)  # Store last 50 peak times
        
        # Detection state
        self.last_heartbeat_time = 0
        self.heartbeat_count = 0
        self.detection_enabled = True
        
        print(f"Heartbeat detector initialized: threshold={threshold}, min_distance={min_distance}s")
    
    def detect(self, processed_data: Dict) -> Optional[Dict]:
        """Detect heartbeats from processed sensor data.
        
        Args:
            processed_data: Dictionary containing processed sensor data
            
        Returns:
            Dictionary containing heartbeat detection results
        """
        if not processed_data or not self.detection_enabled:
            return None
        
        try:
            # Extract data
            timestamp = processed_data.get('timestamp', 0)
            signal_value = processed_data.get('smoothed_accel_z', 0.0)
            
            # Add to buffers
            self.signal_buffer.append(signal_value)
            self.timestamp_buffer.append(timestamp)
            
            # Wait for enough data
            if len(self.signal_buffer) < self.window_size // 2:
                return None
            
            # Detect peaks
            peaks = self._detect_peaks()
            
            # Calculate heart rate
            heart_rate = self._calculate_heart_rate()
            
            # Determine if this is a new heartbeat
            is_new_heartbeat = self._is_new_heartbeat(timestamp, peaks)
            
            if is_new_heartbeat:
                self.heartbeat_count += 1
                self.last_heartbeat_time = timestamp
                
                return {
                    'timestamp': timestamp,
                    'heartbeat_detected': True,
                    'heart_rate': heart_rate,
                    'peak_amplitude': signal_value,
                    'heartbeat_count': self.heartbeat_count,
                    'confidence': self._calculate_confidence(peaks)
                }
            else:
                return {
                    'timestamp': timestamp,
                    'heartbeat_detected': False,
                    'heart_rate': heart_rate,
                    'peak_amplitude': signal_value,
                    'heartbeat_count': self.heartbeat_count,
                    'confidence': 0.0
                }
                
        except Exception as e:
            print(f"Error in heartbeat detection: {e}")
            return None
    
    def _detect_peaks(self) -> List[int]:
        """Detect peaks in the signal buffer.
        
        Returns:
            List of peak indices
        """
        try:
            if len(self.signal_buffer) < 10:
                return []
            
            # Convert to numpy array
            signal_array = np.array(self.signal_buffer)
            
            # Calculate adaptive threshold
            threshold = max(self.threshold, np.std(signal_array) * 2)
            
            # Find peaks
            min_distance_samples = int(self.min_distance * 100)  # Assuming 100Hz sampling
            peaks, _ = find_peaks(signal_array, height=threshold, distance=min_distance_samples)
            
            return peaks.tolist()
            
        except Exception as e:
            print(f"Error detecting peaks: {e}")
            return []
    
    def _calculate_heart_rate(self) -> float:
        """Calculate heart rate from peak intervals.
        
        Returns:
            Heart rate in beats per minute
        """
        try:
            if len(self.peak_times) < 2:
                return 0.0
            
            # Calculate intervals between peaks
            intervals = []
            peak_times_list = list(self.peak_times)
            
            for i in range(1, len(peak_times_list)):
                interval = peak_times_list[i] - peak_times_list[i-1]
                if interval > 0.3 and interval < 3.0:  # Valid heartbeat interval (0.3-3.0 seconds)
                    intervals.append(interval)
            
            if not intervals:
                return 0.0
            
            # Calculate average interval
            avg_interval = np.mean(intervals)
            
            # Convert to BPM
            heart_rate = 60.0 / avg_interval
            
            # Limit to reasonable range (40-200 BPM)
            heart_rate = max(40.0, min(200.0, heart_rate))
            
            return heart_rate
            
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
            return 0.0
    
    def _is_new_heartbeat(self, timestamp: float, peaks: List[int]) -> bool:
        """Determine if a new heartbeat was detected.
        
        Args:
            timestamp: Current timestamp
            peaks: List of detected peak indices
            
        Returns:
            True if a new heartbeat was detected
        """
        try:
            if not peaks:
                return False
            
            # Check if the most recent peak is new
            latest_peak_idx = peaks[-1]
            latest_peak_time = self.timestamp_buffer[latest_peak_idx]
            
            # Check if this peak is recent enough
            if timestamp - latest_peak_time > 0.1:  # Peak should be within 100ms
                return False
            
            # Check if this is a new peak (not already recorded)
            if latest_peak_time not in self.peak_times:
                self.peak_times.append(latest_peak_time)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for new heartbeat: {e}")
            return False
    
    def _calculate_confidence(self, peaks: List[int]) -> float:
        """Calculate confidence in the heartbeat detection.
        
        Args:
            peaks: List of detected peak indices
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            if not peaks:
                return 0.0
            
            # Calculate signal-to-noise ratio
            signal_array = np.array(self.signal_buffer)
            peak_values = signal_array[peaks]
            
            if len(peak_values) == 0:
                return 0.0
            
            # Calculate SNR
            peak_mean = np.mean(peak_values)
            signal_std = np.std(signal_array)
            
            if signal_std == 0:
                return 0.0
            
            snr = peak_mean / signal_std
            
            # Convert SNR to confidence (0-1)
            confidence = min(1.0, snr / 5.0)  # Normalize to 0-1 range
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.0
    
    def get_statistics(self) -> Dict:
        """Get heartbeat detection statistics.
        
        Returns:
            Dictionary containing detection statistics
        """
        try:
            heart_rate = self._calculate_heart_rate()
            
            return {
                'total_heartbeats': self.heartbeat_count,
                'current_heart_rate': heart_rate,
                'last_heartbeat_time': self.last_heartbeat_time,
                'buffer_size': len(self.signal_buffer),
                'peak_count': len(self.peak_times)
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def reset(self):
        """Reset the heartbeat detector."""
        self.signal_buffer.clear()
        self.timestamp_buffer.clear()
        self.peak_times.clear()
        self.last_heartbeat_time = 0
        self.heartbeat_count = 0
        print("Heartbeat detector reset")

class AdvancedHeartbeatDetector(HeartbeatDetector):
    """Advanced heartbeat detector with additional features."""
    
    def __init__(self, threshold=0.1, min_distance=0.5, window_size=100, 
                 adaptive_threshold=True, noise_reduction=True):
        """Initialize advanced heartbeat detector.
        
        Args:
            adaptive_threshold: Use adaptive thresholding
            noise_reduction: Apply noise reduction techniques
        """
        super().__init__(threshold, min_distance, window_size)
        self.adaptive_threshold = adaptive_threshold
        self.noise_reduction = noise_reduction
        
        # Additional buffers for advanced processing
        self.filtered_buffer = collections.deque(maxlen=window_size)
        self.threshold_history = collections.deque(maxlen=50)
    
    def _detect_peaks(self) -> List[int]:
        """Advanced peak detection with adaptive thresholding."""
        try:
            if len(self.signal_buffer) < 10:
                return []
            
            signal_array = np.array(self.signal_buffer)
            
            # Apply noise reduction if enabled
            if self.noise_reduction:
                signal_array = self._apply_noise_reduction(signal_array)
                self.filtered_buffer.extend(signal_array)
            
            # Calculate adaptive threshold
            if self.adaptive_threshold:
                threshold = self._calculate_adaptive_threshold(signal_array)
            else:
                threshold = self.threshold
            
            self.threshold_history.append(threshold)
            
            # Find peaks with more sophisticated criteria
            min_distance_samples = int(self.min_distance * 100)
            
            # Use prominence for better peak detection
            peaks, properties = find_peaks(
                signal_array, 
                height=threshold, 
                distance=min_distance_samples,
                prominence=threshold * 0.5  # Minimum prominence
            )
            
            return peaks.tolist()
            
        except Exception as e:
            print(f"Error in advanced peak detection: {e}")
            return []
    
    def _apply_noise_reduction(self, signal_array: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the signal."""
        try:
            # Simple moving average filter
            window_size = 3
            if len(signal_array) >= window_size:
                filtered = np.convolve(signal_array, np.ones(window_size)/window_size, mode='same')
                return filtered
            else:
                return signal_array
                
        except Exception as e:
            print(f"Error applying noise reduction: {e}")
            return signal_array
    
    def _calculate_adaptive_threshold(self, signal_array: np.ndarray) -> float:
        """Calculate adaptive threshold based on signal characteristics."""
        try:
            # Calculate baseline threshold
            baseline = np.mean(signal_array)
            std_dev = np.std(signal_array)
            
            # Adaptive threshold based on signal variability
            adaptive_threshold = baseline + (std_dev * 2.0)
            
            # Ensure minimum threshold
            min_threshold = self.threshold
            adaptive_threshold = max(min_threshold, adaptive_threshold)
            
            return adaptive_threshold
            
        except Exception as e:
            print(f"Error calculating adaptive threshold: {e}")
            return self.threshold
    
    def _calculate_confidence(self, peaks: List[int]) -> float:
        """Enhanced confidence calculation."""
        try:
            if not peaks:
                return 0.0
            
            # Get filtered signal if available
            if self.noise_reduction and len(self.filtered_buffer) > 0:
                signal_array = np.array(list(self.filtered_buffer)[-len(self.signal_buffer):])
            else:
                signal_array = np.array(self.signal_buffer)
            
            peak_values = signal_array[peaks]
            
            if len(peak_values) == 0:
                return 0.0
            
            # Calculate multiple confidence factors
            snr = np.mean(peak_values) / np.std(signal_array)
            peak_consistency = 1.0 - np.std(peak_values) / np.mean(peak_values)
            temporal_consistency = self._calculate_temporal_consistency()
            
            # Combine confidence factors
            confidence = (snr * 0.4 + peak_consistency * 0.3 + temporal_consistency * 0.3)
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating enhanced confidence: {e}")
            return 0.0
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of detected heartbeats."""
        try:
            if len(self.peak_times) < 3:
                return 0.0
            
            # Calculate intervals
            intervals = []
            peak_times_list = list(self.peak_times)
            
            for i in range(1, len(peak_times_list)):
                interval = peak_times_list[i] - peak_times_list[i-1]
                if 0.3 < interval < 3.0:  # Valid heartbeat interval
                    intervals.append(interval)
            
            if len(intervals) < 2:
                return 0.0
            
            # Calculate coefficient of variation (lower is better)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval == 0:
                return 0.0
            
            cv = std_interval / mean_interval
            consistency = max(0.0, 1.0 - cv)
            
            return consistency
            
        except Exception as e:
            print(f"Error calculating temporal consistency: {e}")
            return 0.0 