#!/usr/bin/env python3
"""
Heartbeat Detection System using BNO086 IMU Sensor
==================================================

This script implements a complete pipeline for detecting subtle body movements
(like heartbeat vibrations) using the SparkFun BNO086 IMU connected to a 
Raspberry Pi Zero 2 W via I²C.

Author: [Your Name]
Date: [Current Date]
"""

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os

# Import our custom modules
from sensor_interface import BNO086Interface
from data_logger import DataLogger
from signal_processor import SignalProcessor
from heartbeat_detector import HeartbeatDetector
from visualizer import RealTimeVisualizer

class HeartbeatDetectionSystem:
    """Main class for the heartbeat detection system."""
    
    def __init__(self, config=None):
        """Initialize the heartbeat detection system."""
        self.config = config or self._get_default_config()
        self.sensor = None
        self.logger = None
        self.processor = None
        self.detector = None
        self.visualizer = None
        self.running = False
        
    def _get_default_config(self):
        """Get default configuration parameters."""
        return {
            'sampling_rate': 100,  # Hz
            'filter_low': 1.0,     # Hz
            'filter_high': 100.0,  # Hz
            'filter_order': 4,
            'smoothing_window': 5,
            'peak_threshold': 0.1,
            'min_peak_distance': 0.5,  # seconds
            'live_plot': False,
            'output_file': f"heartbeat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    
    def initialize(self):
        """Initialize all system components."""
        print("Initializing Heartbeat Detection System...")
        
        try:
            # Initialize sensor interface
            self.sensor = BNO086Interface()
            self.sensor.initialize()
            print("✓ Sensor interface initialized")
            
            # Initialize data logger
            self.logger = DataLogger(self.config['output_file'])
            print("✓ Data logger initialized")
            
            # Initialize signal processor
            self.processor = SignalProcessor(
                sampling_rate=self.config['sampling_rate'],
                filter_low=self.config['filter_low'],
                filter_high=self.config['filter_high'],
                filter_order=self.config['filter_order'],
                smoothing_window=self.config['smoothing_window']
            )
            print("✓ Signal processor initialized")
            
            # Initialize heartbeat detector
            self.detector = HeartbeatDetector(
                threshold=self.config['peak_threshold'],
                min_distance=self.config['min_peak_distance']
            )
            print("✓ Heartbeat detector initialized")
            
            # Initialize visualizer if requested
            if self.config['live_plot']:
                self.visualizer = RealTimeVisualizer()
                print("✓ Real-time visualizer initialized")
            
            print("System initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def run(self, duration=None):
        """Run the heartbeat detection system."""
        if not self.initialize():
            return
        
        print(f"Starting heartbeat detection...")
        print(f"Sampling rate: {self.config['sampling_rate']} Hz")
        print(f"Output file: {self.config['output_file']}")
        if duration:
            print(f"Duration: {duration} seconds")
        print("Press Ctrl+C to stop")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Read sensor data
                sensor_data = self.sensor.read_data()
                if sensor_data is None:
                    continue
                
                # Log raw data
                self.logger.log_data(sensor_data)
                
                # Process signal
                processed_data = self.processor.process(sensor_data)
                
                # Detect heartbeat
                if processed_data is not None:
                    heartbeat_info = self.detector.detect(processed_data)
                    if heartbeat_info:
                        print(f"Heartbeat detected! Rate: {heartbeat_info['rate']:.1f} BPM")
                
                # Update visualization
                if self.visualizer and processed_data is not None:
                    self.visualizer.update(processed_data, heartbeat_info)
                
                # Maintain sampling rate
                time.sleep(1.0 / self.config['sampling_rate'])
                
        except KeyboardInterrupt:
            print("\nStopping heartbeat detection...")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources."""
        self.running = False
        
        if self.sensor:
            self.sensor.cleanup()
        
        if self.logger:
            self.logger.close()
        
        if self.visualizer:
            self.visualizer.close()
        
        print("System cleanup complete!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Heartbeat Detection System')
    parser.add_argument('--duration', type=float, help='Recording duration in seconds')
    parser.add_argument('--rate', type=int, default=100, help='Sampling rate in Hz')
    parser.add_argument('--output', type=str, help='Output CSV file name')
    parser.add_argument('--live-plot', action='store_true', help='Enable live plotting')
    parser.add_argument('--filter-low', type=float, default=1.0, help='Low frequency cutoff')
    parser.add_argument('--filter-high', type=float, default=100.0, help='High frequency cutoff')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'sampling_rate': args.rate,
        'filter_low': args.filter_low,
        'filter_high': args.filter_high,
        'live_plot': args.live_plot,
        'output_file': args.output or f"heartbeat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    }
    
    # Create and run system
    system = HeartbeatDetectionSystem(config)
    system.run(duration=args.duration)

if __name__ == "__main__":
    main()
