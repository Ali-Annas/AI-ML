#!/usr/bin/env python3
"""
Detect Heartbeats from Existing CSV Data
========================================

This script uses our heartbeat detection algorithms to analyze
the existing phone experiment data for heartbeat detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, List, Tuple

class HeartbeatDetectorFromData:
    """Detect heartbeats from existing CSV data."""
    
    def __init__(self, sampling_rate=400):
        """Initialize the detector.
        
        Args:
            sampling_rate: Sampling rate of the data (Hz)
        """
        self.sampling_rate = sampling_rate
        self.filter_low = 1.0  # Hz
        self.filter_high = 100.0  # Hz
        self.filter_order = 4
        
        # Design bandpass filter
        nyquist = sampling_rate / 2.0
        low_norm = self.filter_low / nyquist
        high_norm = self.filter_high / nyquist
        self.b, self.a = butter(self.filter_order, [low_norm, high_norm], btype='band')
        
        print(f"Heartbeat detector initialized: {sampling_rate}Hz, {self.filter_low}-{self.filter_high}Hz bandpass")
    
    def load_data(self, filepath: str) -> Dict:
        """Load data from CSV file."""
        print(f"Loading data from: {filepath}")
        
        try:
            data = pd.read_csv(filepath)
            
            if 'Time (s)' in data.columns and 'Acceleration z (m/s^2)' in data.columns:
                # Extract time and acceleration data
                time_data = data['Time (s)'].values
                accel_z = data['Acceleration z (m/s^2)'].values
                
                # Convert scientific notation
                time_data = [float(str(t).replace('E', 'e')) for t in time_data]
                accel_z = [float(str(a).replace('E', 'e')) for a in accel_z]
                
                # Remove gravity (approximately 9.81 m/s²)
                accel_z = np.array(accel_z) - 9.81
                
                return {
                    'time': np.array(time_data),
                    'accel_z': np.array(accel_z),
                    'placement': 'Chest' if 'Chest' in filepath else 'Abdominal',
                    'filepath': filepath
                }
            else:
                print(f"Required columns not found in {filepath}")
                return None
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to the signal."""
        try:
            # Apply zero-phase filtering
            filtered = filtfilt(self.b, self.a, signal)
            return filtered
        except Exception as e:
            print(f"Error filtering signal: {e}")
            return signal
    
    def detect_peaks(self, signal: np.ndarray, threshold: float = None, min_distance: int = None) -> Tuple[np.ndarray, Dict]:
        """Detect peaks in the filtered signal."""
        try:
            if threshold is None:
                # Adaptive threshold
                threshold = np.std(signal) * 2.0
            
            if min_distance is None:
                # Minimum distance between peaks (0.5 seconds)
                min_distance = int(0.5 * self.sampling_rate)
            
            # Find peaks
            peaks, properties = find_peaks(signal, height=threshold, distance=min_distance)
            
            return peaks, properties
            
        except Exception as e:
            print(f"Error detecting peaks: {e}")
            return np.array([]), {}
    
    def calculate_heart_rate(self, peaks: np.ndarray, time: np.ndarray) -> float:
        """Calculate heart rate from peak intervals."""
        try:
            if len(peaks) < 2:
                return 0.0
            
            # Calculate intervals between peaks
            peak_times = time[peaks]
            intervals = np.diff(peak_times)
            
            # Filter valid intervals (0.3 to 3.0 seconds)
            valid_intervals = intervals[(intervals > 0.3) & (intervals < 3.0)]
            
            if len(valid_intervals) == 0:
                return 0.0
            
            # Calculate average interval and convert to BPM
            avg_interval = np.mean(valid_intervals)
            heart_rate = 60.0 / avg_interval
            
            # Limit to reasonable range (40-200 BPM)
            heart_rate = max(40.0, min(200.0, heart_rate))
            
            return heart_rate
            
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
            return 0.0
    
    def analyze_file(self, filepath: str) -> Dict:
        """Complete analysis of a single file."""
        print(f"\nAnalyzing: {filepath}")
        
        # Load data
        data = self.load_data(filepath)
        if not data:
            return None
        
        # Filter signal
        filtered_signal = self.filter_signal(data['accel_z'])
        
        # Detect peaks
        peaks, properties = self.detect_peaks(filtered_signal)
        
        # Calculate heart rate
        heart_rate = self.calculate_heart_rate(peaks, data['time'])
        
        # Calculate statistics
        if len(peaks) > 0:
            peak_amplitudes = filtered_signal[peaks]
            peak_times = data['time'][peaks]
            
            # Calculate intervals
            intervals = np.diff(peak_times)
            valid_intervals = intervals[(intervals > 0.3) & (intervals < 3.0)]
            
            if len(valid_intervals) > 0:
                interval_std = np.std(valid_intervals)
                interval_cv = interval_std / np.mean(valid_intervals)  # Coefficient of variation
            else:
                interval_std = interval_cv = 0
        else:
            peak_amplitudes = []
            interval_std = interval_cv = 0
        
        # Results
        results = {
            'filepath': filepath,
            'placement': data['placement'],
            'duration': data['time'][-1] - data['time'][0],
            'total_samples': len(data['accel_z']),
            'heartbeats_detected': len(peaks),
            'heart_rate': heart_rate,
            'peak_count': len(peaks),
            'interval_std': interval_std,
            'interval_cv': interval_cv,
            'signal_mean': np.mean(data['accel_z']),
            'signal_std': np.std(data['accel_z']),
            'filtered_mean': np.mean(filtered_signal),
            'filtered_std': np.std(filtered_signal),
            'peak_times': data['time'][peaks] if len(peaks) > 0 else [],
            'peak_amplitudes': peak_amplitudes,
            'raw_signal': data['accel_z'],
            'filtered_signal': filtered_signal,
            'time': data['time']
        }
        
        print(f"  Heartbeats detected: {len(peaks)}")
        print(f"  Heart rate: {heart_rate:.1f} BPM")
        print(f"  Duration: {results['duration']:.1f} seconds")
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot the analysis results."""
        if not results:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Raw signal
        ax1 = axes[0]
        ax1.plot(results['time'], results['raw_signal'], 'gray', alpha=0.7, label='Raw Signal')
        ax1.set_title(f'Raw Accelerometer Signal - {results["placement"]} Placement')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Filtered signal with peaks
        ax2 = axes[1]
        ax2.plot(results['time'], results['filtered_signal'], 'blue', label='Filtered Signal')
        
        if len(results['peak_times']) > 0:
            ax2.scatter(results['peak_times'], results['peak_amplitudes'], 
                       color='red', s=50, zorder=5, label='Detected Peaks')
        
        ax2.set_title(f'Filtered Signal with Heartbeat Peaks - {results["placement"]} Placement')
        ax2.set_ylabel('Filtered Acceleration (m/s²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Heart rate over time
        ax3 = axes[2]
        if len(results['peak_times']) > 1:
            # Calculate heart rate for each interval
            intervals = np.diff(results['peak_times'])
            valid_mask = (intervals > 0.3) & (intervals < 3.0)
            
            if np.any(valid_mask):
                valid_intervals = intervals[valid_mask]
                heart_rates = 60.0 / valid_intervals
                interval_times = results['peak_times'][1:][valid_mask]
                
                ax3.plot(interval_times, heart_rates, 'green', marker='o', linewidth=2)
                ax3.axhline(y=results['heart_rate'], color='red', linestyle='--', 
                           label=f'Average: {results["heart_rate"]:.1f} BPM')
                ax3.set_title(f'Heart Rate Over Time - {results["placement"]} Placement')
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Heart Rate (BPM)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(40, 200)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {save_path}")
        
        plt.show()
    
    def compare_placements(self, results_list: List[Dict]):
        """Compare results between different placements."""
        if len(results_list) < 2:
            print("Need at least 2 results for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        placements = [r['placement'] for r in results_list]
        heart_rates = [r['heart_rate'] for r in results_list]
        heartbeat_counts = [r['heartbeats_detected'] for r in results_list]
        signal_stds = [r['signal_std'] for r in results_list]
        filtered_stds = [r['filtered_std'] for r in results_list]
        
        # Plot 1: Heart rate comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(placements, heart_rates, color=['blue', 'green'])
        ax1.set_title('Heart Rate by Placement')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_ylim(0, max(heart_rates) * 1.2 if heart_rates else 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, heart_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}', ha='center', va='bottom')
        
        # Plot 2: Heartbeat count comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(placements, heartbeat_counts, color=['orange', 'red'])
        ax2.set_title('Heartbeats Detected by Placement')
        ax2.set_ylabel('Number of Heartbeats')
        
        # Add value labels on bars
        for bar, count in zip(bars2, heartbeat_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Plot 3: Signal variability comparison
        ax3 = axes[1, 0]
        x = np.arange(len(placements))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, signal_stds, width, label='Raw Signal', color='lightblue')
        bars3b = ax3.bar(x + width/2, filtered_stds, width, label='Filtered Signal', color='lightgreen')
        
        ax3.set_title('Signal Variability by Placement')
        ax3.set_ylabel('Standard Deviation (m/s²)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(placements)
        ax3.legend()
        
        # Plot 4: Duration and sampling info
        ax4 = axes[1, 1]
        durations = [r['duration'] for r in results_list]
        samples = [r['total_samples'] for r in results_list]
        
        bars4a = ax4.bar(x - width/2, durations, width, label='Duration (s)', color='lightcoral')
        ax4_twin = ax4.twinx()
        bars4b = ax4_twin.bar(x + width/2, samples, width, label='Samples', color='lightyellow', alpha=0.7)
        
        ax4.set_title('Recording Information by Placement')
        ax4.set_ylabel('Duration (seconds)')
        ax4_twin.set_ylabel('Number of Samples')
        ax4.set_xticks(x)
        ax4.set_xticklabels(placements)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filename = f"placement_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as: {comparison_filename}")
        
        plt.show()

def main():
    """Main function to analyze all CSV files."""
    print("Heartbeat Detection from Existing Data")
    print("=" * 50)
    
    # Initialize detector
    detector = HeartbeatDetectorFromData(sampling_rate=400)
    
    # Find accelerometer CSV files
    csv_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv') and 'Accelerometer' in file:
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} accelerometer files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Analyze each file
    results = []
    for csv_file in csv_files:
        result = detector.analyze_file(csv_file)
        if result:
            results.append(result)
            
            # Plot individual results
            plot_filename = f"heartbeat_analysis_{result['placement']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            detector.plot_results(result, save_path=plot_filename)
    
    # Compare placements
    if len(results) >= 2:
        detector.compare_placements(results)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("HEARTBEAT DETECTION SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['placement']} Placement:")
        print(f"  File: {os.path.basename(result['filepath'])}")
        print(f"  Duration: {result['duration']:.1f} seconds")
        print(f"  Heartbeats detected: {result['heartbeats_detected']}")
        print(f"  Heart rate: {result['heart_rate']:.1f} BPM")
        print(f"  Signal variability: {result['signal_std']:.3f} m/s²")
        print(f"  Filtered variability: {result['filtered_std']:.3f} m/s²")
        
        if result['interval_cv'] > 0:
            print(f"  Heart rate consistency: {1 - result['interval_cv']:.2f}")
    
    # Save detailed report
    report_filename = f"heartbeat_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("HEARTBEAT DETECTION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files analyzed: {len(results)}\n\n")
        
        for result in results:
            f.write(f"{result['placement']} Placement:\n")
            f.write(f"  File: {result['filepath']}\n")
            f.write(f"  Duration: {result['duration']:.1f} seconds\n")
            f.write(f"  Total samples: {result['total_samples']}\n")
            f.write(f"  Heartbeats detected: {result['heartbeats_detected']}\n")
            f.write(f"  Heart rate: {result['heart_rate']:.1f} BPM\n")
            f.write(f"  Signal mean: {result['signal_mean']:.3f} m/s²\n")
            f.write(f"  Signal std: {result['signal_std']:.3f} m/s²\n")
            f.write(f"  Filtered mean: {result['filtered_mean']:.3f} m/s²\n")
            f.write(f"  Filtered std: {result['filtered_std']:.3f} m/s²\n")
            f.write(f"  Interval std: {result['interval_std']:.3f} s\n")
            f.write(f"  Interval CV: {result['interval_cv']:.3f}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\nDetailed report saved as: {report_filename}")

if __name__ == "__main__":
    main() 