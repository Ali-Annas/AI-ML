#!/usr/bin/env python3
"""
Data Logger Module
==================

This module handles logging sensor data to CSV files with timestamps.

Author: [Your Name]
Date: [Current Date]
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional
import time

class DataLogger:
    """Class for logging sensor data to CSV files."""
    
    def __init__(self, filename: str, auto_create=True):
        """Initialize the data logger.
        
        Args:
            filename: Output CSV file name
            auto_create: Whether to create the file if it doesn't exist
        """
        self.filename = filename
        self.file = None
        self.writer = None
        self.initialized = False
        
        if auto_create:
            self.initialize()
    
    def initialize(self):
        """Initialize the CSV file and writer."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(self.filename)
            
            # Open file in append mode
            self.file = open(self.filename, 'a', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            
            # Write headers if file is new
            if not file_exists:
                self._write_headers()
            
            self.initialized = True
            print(f"Data logger initialized: {self.filename}")
            
        except Exception as e:
            print(f"Failed to initialize data logger: {e}")
            raise
    
    def _write_headers(self):
        """Write CSV headers."""
        headers = [
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'temperature'
        ]
        self.writer.writerow(headers)
        self.file.flush()
    
    def log_data(self, data: Dict):
        """Log sensor data to CSV file.
        
        Args:
            data: Dictionary containing sensor data
        """
        if not self.initialized or not data:
            return
        
        try:
            # Extract data values
            row = [
                data.get('timestamp', time.time()),
                data.get('accel_x', 0.0),
                data.get('accel_y', 0.0),
                data.get('accel_z', 0.0),
                data.get('gyro_x', 0.0),
                data.get('gyro_y', 0.0),
                data.get('gyro_z', 0.0),
                data.get('mag_x', 0.0),
                data.get('mag_y', 0.0),
                data.get('mag_z', 0.0),
                data.get('temperature', 0.0)
            ]
            
            # Write to CSV
            self.writer.writerow(row)
            self.file.flush()  # Ensure data is written immediately
            
        except Exception as e:
            print(f"Error logging data: {e}")
    
    def log_custom_data(self, data_dict: Dict):
        """Log custom data with flexible structure.
        
        Args:
            data_dict: Dictionary with custom data to log
        """
        if not self.initialized:
            return
        
        try:
            # Convert dictionary to list of values
            row = []
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    row.append(value)
                else:
                    row.append(str(value))
            
            self.writer.writerow(row)
            self.file.flush()
            
        except Exception as e:
            print(f"Error logging custom data: {e}")
    
    def get_file_info(self) -> Dict:
        """Get information about the log file.
        
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(self.filename):
            return {'exists': False}
        
        try:
            stat = os.stat(self.filename)
            return {
                'exists': True,
                'size_bytes': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': os.path.abspath(self.filename)
            }
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    def get_line_count(self) -> int:
        """Get the number of lines in the log file.
        
        Returns:
            Number of lines (including header)
        """
        if not os.path.exists(self.filename):
            return 0
        
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def close(self):
        """Close the log file."""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self.initialized = False
            print(f"Data logger closed: {self.filename}")

class TimestampedDataLogger(DataLogger):
    """Enhanced data logger with automatic timestamping."""
    
    def __init__(self, filename: str, include_datetime=True):
        """Initialize timestamped data logger.
        
        Args:
            filename: Output CSV file name
            include_datetime: Whether to include human-readable datetime
        """
        self.include_datetime = include_datetime
        super().__init__(filename)
    
    def _write_headers(self):
        """Write CSV headers with optional datetime column."""
        headers = ['timestamp']
        if self.include_datetime:
            headers.append('datetime')
        
        headers.extend([
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z',
            'temperature'
        ])
        
        self.writer.writerow(headers)
        self.file.flush()
    
    def log_data(self, data: Dict):
        """Log sensor data with automatic timestamping."""
        if not self.initialized or not data:
            return
        
        try:
            timestamp = data.get('timestamp', time.time())
            
            # Build row with timestamp
            row = [timestamp]
            
            if self.include_datetime:
                row.append(datetime.fromtimestamp(timestamp).isoformat())
            
            # Add sensor data
            row.extend([
                data.get('accel_x', 0.0),
                data.get('accel_y', 0.0),
                data.get('accel_z', 0.0),
                data.get('gyro_x', 0.0),
                data.get('gyro_y', 0.0),
                data.get('gyro_z', 0.0),
                data.get('mag_x', 0.0),
                data.get('mag_y', 0.0),
                data.get('mag_z', 0.0),
                data.get('temperature', 0.0)
            ])
            
            self.writer.writerow(row)
            self.file.flush()
            
        except Exception as e:
            print(f"Error logging timestamped data: {e}")

class BufferedDataLogger(DataLogger):
    """Data logger with buffering for better performance."""
    
    def __init__(self, filename: str, buffer_size=100):
        """Initialize buffered data logger.
        
        Args:
            filename: Output CSV file name
            buffer_size: Number of rows to buffer before writing
        """
        self.buffer_size = buffer_size
        self.buffer = []
        super().__init__(filename)
    
    def log_data(self, data: Dict):
        """Log data with buffering."""
        if not self.initialized or not data:
            return
        
        try:
            # Add to buffer
            row = [
                data.get('timestamp', time.time()),
                data.get('accel_x', 0.0),
                data.get('accel_y', 0.0),
                data.get('accel_z', 0.0),
                data.get('gyro_x', 0.0),
                data.get('gyro_y', 0.0),
                data.get('gyro_z', 0.0),
                data.get('mag_x', 0.0),
                data.get('mag_y', 0.0),
                data.get('mag_z', 0.0),
                data.get('temperature', 0.0)
            ]
            
            self.buffer.append(row)
            
            # Flush buffer if full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
                
        except Exception as e:
            print(f"Error buffering data: {e}")
    
    def _flush_buffer(self):
        """Flush the buffer to disk."""
        if not self.buffer:
            return
        
        try:
            for row in self.buffer:
                self.writer.writerow(row)
            self.file.flush()
            self.buffer.clear()
            
        except Exception as e:
            print(f"Error flushing buffer: {e}")
    
    def close(self):
        """Close the logger and flush any remaining data."""
        self._flush_buffer()
        super().close() 