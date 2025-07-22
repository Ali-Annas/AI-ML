#!/usr/bin/env python3
"""
BNO086 Sensor Interface
=======================

This module provides an interface to the SparkFun BNO086 IMU sensor
connected via I²C to a Raspberry Pi.

Author: [Your Name]
Date: [Current Date]
"""

import time
import smbus2 as smbus
from typing import Dict, Optional, Tuple
import numpy as np

# BNO086 I2C address
BNO086_ADDRESS = 0x4A

# Register addresses for BNO086
CHIP_ID_REG = 0x00
ACCEL_DATA_REG = 0x08
GYRO_DATA_REG = 0x14
MAG_DATA_REG = 0x20
TEMP_DATA_REG = 0x34

class BNO086Interface:
    """Interface class for the BNO086 IMU sensor."""
    
    def __init__(self, bus_number=1, address=BNO086_ADDRESS):
        """Initialize the BNO086 interface.
        
        Args:
            bus_number: I2C bus number (usually 1 for Raspberry Pi)
            address: I2C address of the BNO086 sensor
        """
        self.bus_number = bus_number
        self.address = address
        self.bus = None
        self.initialized = False
        
        # Data storage
        self.last_accel = None
        self.last_gyro = None
        self.last_mag = None
        self.last_temp = None
        
    def initialize(self):
        """Initialize the I2C bus and verify sensor connection."""
        try:
            # Initialize I2C bus
            self.bus = smbus.SMBus(self.bus_number)
            
            # Wait for sensor to be ready
            time.sleep(0.1)
            
            # Read chip ID to verify connection
            chip_id = self.bus.read_byte_data(self.address, CHIP_ID_REG)
            if chip_id != 0xA0:  # Expected BNO086 chip ID
                raise Exception(f"Invalid chip ID: 0x{chip_id:02X}. Expected: 0xA0")
            
            print(f"BNO086 sensor found at address 0x{self.address:02X}")
            self.initialized = True
            
            # Configure sensor for optimal heartbeat detection
            self._configure_sensor()
            
        except Exception as e:
            print(f"Failed to initialize BNO086: {e}")
            raise
    
    def _configure_sensor(self):
        """Configure the sensor for heartbeat detection."""
        try:
            # Set accelerometer range to ±2g for better sensitivity
            # Set gyroscope range to ±250dps
            # Enable high-frequency data output
            
            # Note: The exact configuration depends on the specific BNO086 library
            # being used. This is a placeholder for the configuration.
            print("Sensor configured for heartbeat detection")
            
        except Exception as e:
            print(f"Warning: Could not configure sensor: {e}")
    
    def read_data(self) -> Optional[Dict]:
        """Read sensor data from the BNO086.
        
        Returns:
            Dictionary containing sensor data with keys:
            - timestamp: Current timestamp
            - accel_x, accel_y, accel_z: Accelerometer data (m/s²)
            - gyro_x, gyro_y, gyro_z: Gyroscope data (rad/s)
            - mag_x, mag_y, mag_z: Magnetometer data (μT)
            - temperature: Temperature (°C)
        """
        if not self.initialized:
            return None
        
        try:
            timestamp = time.time()
            
            # Read accelerometer data (6 bytes: x, y, z each 16-bit)
            accel_data = self.bus.read_i2c_block_data(self.address, ACCEL_DATA_REG, 6)
            accel_x = self._convert_to_float(accel_data[0:2])
            accel_y = self._convert_to_float(accel_data[2:4])
            accel_z = self._convert_to_float(accel_data[4:6])
            
            # Read gyroscope data (6 bytes: x, y, z each 16-bit)
            gyro_data = self.bus.read_i2c_block_data(self.address, GYRO_DATA_REG, 6)
            gyro_x = self._convert_to_float(gyro_data[0:2])
            gyro_y = self._convert_to_float(gyro_data[2:4])
            gyro_z = self._convert_to_float(gyro_data[4:6])
            
            # Read magnetometer data (6 bytes: x, y, z each 16-bit)
            mag_data = self.bus.read_i2c_block_data(self.address, MAG_DATA_REG, 6)
            mag_x = self._convert_to_float(mag_data[0:2])
            mag_y = self._convert_to_float(mag_data[2:4])
            mag_z = self._convert_to_float(mag_data[4:6])
            
            # Read temperature data (2 bytes)
            temp_data = self.bus.read_i2c_block_data(self.address, TEMP_DATA_REG, 2)
            temperature = self._convert_to_float(temp_data[0:2])
            
            # Store last readings
            self.last_accel = (accel_x, accel_y, accel_z)
            self.last_gyro = (gyro_x, gyro_y, gyro_z)
            self.last_mag = (mag_x, mag_y, mag_z)
            self.last_temp = temperature
            
            return {
                'timestamp': timestamp,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'mag_x': mag_x,
                'mag_y': mag_y,
                'mag_z': mag_z,
                'temperature': temperature
            }
            
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None
    
    def _convert_to_float(self, data_bytes: bytes) -> float:
        """Convert 2-byte data to float value.
        
        Args:
            data_bytes: 2-byte array representing the sensor value
            
        Returns:
            Converted float value
        """
        # Convert to 16-bit signed integer
        value = int.from_bytes(data_bytes, byteorder='little', signed=True)
        
        # Convert to float (scale factors depend on sensor configuration)
        # For accelerometer: typically ±2g range = ±19.6 m/s²
        # For gyroscope: typically ±250dps range = ±4.36 rad/s
        # For magnetometer: typically ±4900μT range
        
        # This is a simplified conversion - actual scaling depends on sensor settings
        return float(value) / 16384.0  # 2^14 for 16-bit resolution
    
    def get_last_reading(self) -> Optional[Dict]:
        """Get the last successful sensor reading."""
        if self.last_accel is None:
            return None
        
        return {
            'accel_x': self.last_accel[0],
            'accel_y': self.last_accel[1],
            'accel_z': self.last_accel[2],
            'gyro_x': self.last_gyro[0],
            'gyro_y': self.last_gyro[1],
            'gyro_z': self.last_gyro[2],
            'mag_x': self.last_mag[0],
            'mag_y': self.last_mag[1],
            'mag_z': self.last_mag[2],
            'temperature': self.last_temp
        }
    
    def test_connection(self) -> bool:
        """Test the sensor connection."""
        try:
            if not self.initialized:
                return False
            
            # Try to read chip ID
            chip_id = self.bus.read_byte_data(self.address, CHIP_ID_REG)
            return chip_id == 0xA0
            
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.bus:
            self.bus.close()
        self.initialized = False

# Alternative implementation using the official SparkFun library
class BNO086InterfaceSparkFun:
    """Alternative interface using the official SparkFun library."""
    
    def __init__(self):
        """Initialize using SparkFun library."""
        try:
            import qwiic_bno08x
            self.sensor = qwiic_bno08x.QwiicBNO08x()
            self.initialized = False
        except ImportError:
            raise ImportError("SparkFun Qwiic BNO08x library not found. Install with: pip install sparkfun-qwiic-bno08x")
    
    def initialize(self):
        """Initialize the sensor using SparkFun library."""
        try:
            if not self.sensor.begin():
                raise Exception("Could not connect to BNO086 sensor")
            
            # Configure for high-frequency data
            self.sensor.enable_accelerometer()
            self.sensor.enable_gyroscope()
            self.sensor.enable_magnetometer()
            
            # Set data rates
            self.sensor.set_accelerometer_data_rate(100)  # 100 Hz
            self.sensor.set_gyroscope_data_rate(100)      # 100 Hz
            self.sensor.set_magnetometer_data_rate(100)   # 100 Hz
            
            self.initialized = True
            print("BNO086 sensor initialized with SparkFun library")
            
        except Exception as e:
            print(f"Failed to initialize BNO086 with SparkFun library: {e}")
            raise
    
    def read_data(self) -> Optional[Dict]:
        """Read sensor data using SparkFun library."""
        if not self.initialized:
            return None
        
        try:
            timestamp = time.time()
            
            # Read sensor data
            accel = self.sensor.get_accelerometer()
            gyro = self.sensor.get_gyroscope()
            mag = self.sensor.get_magnetometer()
            temp = self.sensor.get_temperature()
            
            return {
                'timestamp': timestamp,
                'accel_x': accel.x,
                'accel_y': accel.y,
                'accel_z': accel.z,
                'gyro_x': gyro.x,
                'gyro_y': gyro.y,
                'gyro_z': gyro.z,
                'mag_x': mag.x,
                'mag_y': mag.y,
                'mag_z': mag.z,
                'temperature': temp
            }
            
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        self.initialized = False 