#!/usr/bin/env python3
"""
Visualizer Module
=================

This module provides real-time visualization of sensor data and heartbeat detection.

Author: [Your Name]
Date: [Current Date]
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Optional
import collections
import tkinter as tk
from tkinter import ttk
import threading
import time

class RealTimeVisualizer:
    """Class for real-time visualization of sensor data and heartbeat detection."""
    
    def __init__(self, window_size=1000, update_interval=100):
        """Initialize the real-time visualizer.
        
        Args:
            window_size: Number of data points to display
            update_interval: Update interval in milliseconds
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Data buffers
        self.timestamps = collections.deque(maxlen=window_size)
        self.raw_accel = collections.deque(maxlen=window_size)
        self.filtered_accel = collections.deque(maxlen=window_size)
        self.smoothed_accel = collections.deque(maxlen=window_size)
        self.heartbeat_times = collections.deque(maxlen=50)
        self.heartbeat_rates = collections.deque(maxlen=50)
        
        # Plotting setup
        self.fig = None
        self.ax1 = None  # Raw and filtered signals
        self.ax2 = None  # Heart rate
        self.canvas = None
        self.root = None
        
        # Animation
        self.ani = None
        self.running = False
        
        print("Real-time visualizer initialized")
    
    def initialize(self):
        """Initialize the matplotlib figure and GUI."""
        try:
            # Create Tkinter root window
            self.root = tk.Tk()
            self.root.title("Heartbeat Detection - Real-time Monitor")
            self.root.geometry("1200x800")
            
            # Create matplotlib figure
            self.fig = Figure(figsize=(12, 8), dpi=100)
            
            # Create subplots
            self.ax1 = self.fig.add_subplot(2, 1, 1)
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            
            # Setup plots
            self._setup_plots()
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            # Create control panel
            self._create_control_panel()
            
            # Start animation
            self.running = True
            self.ani = animation.FuncAnimation(
                self.fig, self._update_plot, interval=self.update_interval, 
                blit=False, cache_frame_data=False
            )
            
            print("Real-time visualizer GUI initialized")
            
        except Exception as e:
            print(f"Error initializing visualizer: {e}")
            raise
    
    def _setup_plots(self):
        """Setup the initial plot appearance."""
        # Signal plot
        self.ax1.set_title("Sensor Signals", fontsize=14, fontweight='bold')
        self.ax1.set_ylabel("Amplitude", fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Heart rate plot
        self.ax2.set_title("Heart Rate", fontsize=14, fontweight='bold')
        self.ax2.set_xlabel("Time (s)", fontsize=12)
        self.ax2.set_ylabel("BPM", fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(40, 200)  # Reasonable heart rate range
    
    def _create_control_panel(self):
        """Create control panel with buttons and status."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Running")
        self.status_label.pack(side=tk.LEFT)
        
        # Control buttons
        ttk.Button(control_frame, text="Pause", command=self.pause).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Resume", command=self.resume).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Clear", command=self.clear).pack(side=tk.RIGHT, padx=5)
        
        # Statistics labels
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(side=tk.LEFT, padx=20)
        
        self.heart_rate_label = ttk.Label(stats_frame, text="Heart Rate: -- BPM")
        self.heart_rate_label.pack(side=tk.LEFT, padx=10)
        
        self.beat_count_label = ttk.Label(stats_frame, text="Beats: 0")
        self.beat_count_label.pack(side=tk.LEFT, padx=10)
    
    def update(self, processed_data: Dict, heartbeat_info: Optional[Dict] = None):
        """Update the visualizer with new data.
        
        Args:
            processed_data: Processed sensor data
            heartbeat_info: Heartbeat detection results
        """
        if not self.running:
            return
        
        try:
            # Extract data
            timestamp = processed_data.get('timestamp', time.time())
            raw_accel = processed_data.get('raw_accel_z', 0.0)
            filtered_accel = processed_data.get('filtered_accel_z', 0.0)
            smoothed_accel = processed_data.get('smoothed_accel_z', 0.0)
            
            # Add to buffers
            self.timestamps.append(timestamp)
            self.raw_accel.append(raw_accel)
            self.filtered_accel.append(filtered_accel)
            self.smoothed_accel.append(smoothed_accel)
            
            # Update heartbeat data
            if heartbeat_info and heartbeat_info.get('heartbeat_detected', False):
                self.heartbeat_times.append(timestamp)
                heart_rate = heartbeat_info.get('heart_rate', 0.0)
                self.heartbeat_rates.append(heart_rate)
                
                # Update status labels
                self._update_status_labels(heartbeat_info)
            
        except Exception as e:
            print(f"Error updating visualizer: {e}")
    
    def _update_status_labels(self, heartbeat_info: Dict):
        """Update status labels with heartbeat information."""
        try:
            heart_rate = heartbeat_info.get('heart_rate', 0.0)
            beat_count = heartbeat_info.get('heartbeat_count', 0)
            
            self.heart_rate_label.config(text=f"Heart Rate: {heart_rate:.1f} BPM")
            self.beat_count_label.config(text=f"Beats: {beat_count}")
            
        except Exception as e:
            print(f"Error updating status labels: {e}")
    
    def _update_plot(self, frame):
        """Update the plot with current data."""
        try:
            if not self.running or len(self.timestamps) < 2:
                return
            
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Convert to arrays
            times = np.array(self.timestamps)
            raw_data = np.array(self.raw_accel)
            filtered_data = np.array(self.filtered_accel)
            smoothed_data = np.array(self.smoothed_accel)
            
            # Normalize time to start from 0
            times_normalized = times - times[0]
            
            # Plot signals
            self.ax1.plot(times_normalized, raw_data, 'gray', alpha=0.5, label='Raw Signal', linewidth=1)
            self.ax1.plot(times_normalized, filtered_data, 'blue', alpha=0.7, label='Filtered Signal', linewidth=1.5)
            self.ax1.plot(times_normalized, smoothed_data, 'red', label='Smoothed Signal', linewidth=2)
            
            # Mark heartbeat peaks
            if len(self.heartbeat_times) > 0:
                heartbeat_times_norm = np.array(self.heartbeat_times) - times[0]
                heartbeat_amplitudes = []
                
                for hb_time in self.heartbeat_times:
                    # Find closest timestamp index
                    idx = np.argmin(np.abs(times - hb_time))
                    if idx < len(smoothed_data):
                        heartbeat_amplitudes.append(smoothed_data[idx])
                
                if heartbeat_amplitudes:
                    self.ax1.scatter(heartbeat_times_norm, heartbeat_amplitudes, 
                                   color='green', s=50, zorder=5, label='Heartbeat Peaks')
            
            # Setup signal plot
            self.ax1.set_title("Sensor Signals", fontsize=14, fontweight='bold')
            self.ax1.set_ylabel("Amplitude", fontsize=12)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            
            # Plot heart rate
            if len(self.heartbeat_times) > 1:
                hb_times_norm = np.array(self.heartbeat_times) - times[0]
                hb_rates = np.array(self.heartbeat_rates)
                
                self.ax2.plot(hb_times_norm, hb_rates, 'green', linewidth=2, marker='o')
                self.ax2.set_ylim(40, 200)
            
            # Setup heart rate plot
            self.ax2.set_title("Heart Rate", fontsize=14, fontweight='bold')
            self.ax2.set_xlabel("Time (s)", fontsize=12)
            self.ax2.set_ylabel("BPM", fontsize=12)
            self.ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            self.fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def pause(self):
        """Pause the visualization."""
        self.running = False
        self.status_label.config(text="Status: Paused")
    
    def resume(self):
        """Resume the visualization."""
        self.running = True
        self.status_label.config(text="Status: Running")
    
    def clear(self):
        """Clear all data buffers."""
        self.timestamps.clear()
        self.raw_accel.clear()
        self.filtered_accel.clear()
        self.smoothed_accel.clear()
        self.heartbeat_times.clear()
        self.heartbeat_rates.clear()
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self._setup_plots()
        self.canvas.draw()
        
        # Reset labels
        self.heart_rate_label.config(text="Heart Rate: -- BPM")
        self.beat_count_label.config(text="Beats: 0")
    
    def start(self):
        """Start the visualization GUI."""
        if self.root:
            self.root.mainloop()
    
    def close(self):
        """Close the visualization."""
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
        if self.root:
            self.root.quit()
            self.root.destroy()

class SimpleVisualizer:
    """Simple visualizer without GUI for basic plotting."""
    
    def __init__(self, window_size=1000):
        """Initialize simple visualizer.
        
        Args:
            window_size: Number of data points to store
        """
        self.window_size = window_size
        
        # Data buffers
        self.timestamps = collections.deque(maxlen=window_size)
        self.raw_accel = collections.deque(maxlen=window_size)
        self.filtered_accel = collections.deque(maxlen=window_size)
        self.heartbeat_times = collections.deque(maxlen=50)
        self.heartbeat_rates = collections.deque(maxlen=50)
        
        print("Simple visualizer initialized")
    
    def update(self, processed_data: Dict, heartbeat_info: Optional[Dict] = None):
        """Update with new data."""
        try:
            timestamp = processed_data.get('timestamp', time.time())
            raw_accel = processed_data.get('raw_accel_z', 0.0)
            filtered_accel = processed_data.get('filtered_accel_z', 0.0)
            
            self.timestamps.append(timestamp)
            self.raw_accel.append(raw_accel)
            self.filtered_accel.append(filtered_accel)
            
            if heartbeat_info and heartbeat_info.get('heartbeat_detected', False):
                self.heartbeat_times.append(timestamp)
                self.heartbeat_rates.append(heartbeat_info.get('heart_rate', 0.0))
                
        except Exception as e:
            print(f"Error updating simple visualizer: {e}")
    
    def plot_data(self, save_path: Optional[str] = None):
        """Plot the collected data."""
        try:
            if len(self.timestamps) < 2:
                print("Not enough data to plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Convert to arrays
            times = np.array(self.timestamps)
            raw_data = np.array(self.raw_accel)
            filtered_data = np.array(self.filtered_accel)
            
            # Normalize time
            times_norm = times - times[0]
            
            # Plot signals
            ax1.plot(times_norm, raw_data, 'gray', alpha=0.5, label='Raw Signal')
            ax1.plot(times_norm, filtered_data, 'blue', label='Filtered Signal')
            
            # Mark heartbeats
            if len(self.heartbeat_times) > 0:
                hb_times_norm = np.array(self.heartbeat_times) - times[0]
                hb_amplitudes = []
                
                for hb_time in self.heartbeat_times:
                    idx = np.argmin(np.abs(times - hb_time))
                    if idx < len(filtered_data):
                        hb_amplitudes.append(filtered_data[idx])
                
                if hb_amplitudes:
                    ax1.scatter(hb_times_norm, hb_amplitudes, 
                              color='red', s=50, zorder=5, label='Heartbeats')
            
            ax1.set_title("Sensor Signals")
            ax1.set_ylabel("Amplitude")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot heart rate
            if len(self.heartbeat_times) > 1:
                hb_times_norm = np.array(self.heartbeat_times) - times[0]
                hb_rates = np.array(self.heartbeat_rates)
                
                ax2.plot(hb_times_norm, hb_rates, 'green', marker='o')
                ax2.set_ylim(40, 200)
            
            ax2.set_title("Heart Rate")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("BPM")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error plotting data: {e}")
    
    def clear(self):
        """Clear all data."""
        self.timestamps.clear()
        self.raw_accel.clear()
        self.filtered_accel.clear()
        self.heartbeat_times.clear()
        self.heartbeat_rates.clear() 