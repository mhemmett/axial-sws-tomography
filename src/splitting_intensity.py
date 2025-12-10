"""
Splitting Intensity Calculation Module

This module implements splitting intensity (SI) calculation from seismic waveforms
following the Chevrot (2000) method. Splitting intensity provides a linear observable
complementary to traditional splitting parameters (φ, δt) for anisotropy inversion.

Splitting intensity measures the energy of the transverse component relative to 
the radial component in a time window around the S-wave arrival, providing
information about anisotropy strength that is independent of fast axis orientation.

References:
- Chevrot, S. (2000). Multichannel analysis of shear wave splitting. 
  Journal of Geophysical Research, 105(B9), 21579-21590.
  
Author: Axial Seamount Research Team
Date: December 2024
"""

import numpy as np
import obspy
from obspy import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.filter import bandpass
from obspy.signal.util import next_pow_2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings


class SplittingIntensityCalculator:
    """
    Calculate splitting intensity from three-component seismic waveforms.
    
    This class implements the Chevrot (2000) method for computing splitting
    intensity, which provides a linear measure of anisotropy that complements
    traditional splitting parameter analysis.
    """
    
    def __init__(self, freq_min: float = 2.0, freq_max: float = 8.0,
                 window_length: float = 2.0, taper_fraction: float = 0.1):
        """
        Initialize the splitting intensity calculator.
        
        Parameters:
        -----------
        freq_min : float
            Minimum frequency for bandpass filter (Hz)
        freq_max : float  
            Maximum frequency for bandpass filter (Hz)
        window_length : float
            Length of analysis window around S-arrival (seconds)
        taper_fraction : float
            Fraction of window to taper at edges
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window_length = window_length
        self.taper_fraction = taper_fraction
        
        # Quality control thresholds
        self.min_snr = 2.0
        self.max_noise_level = 0.5
        
    def calculate_si_from_stream(self, stream: Stream, s_arrival: UTCDateTime,
                                back_azimuth: float, incidence_angle: Optional[float] = None,
                                quality_control: bool = True) -> Dict:
        """
        Calculate splitting intensity from a three-component stream.
        
        Parameters:
        -----------
        stream : obspy.Stream
            Three-component seismic data (Z, N, E or Z, R, T)
        s_arrival : obspy.UTCDateTime
            S-wave arrival time
        back_azimuth : float
            Back-azimuth from source to receiver (degrees)
        incidence_angle : float, optional
            Incidence angle at receiver (degrees)
        quality_control : bool
            Whether to apply quality control checks
            
        Returns:
        --------
        result : dict
            Dictionary containing SI value, uncertainty, and quality metrics
        """
        try:
            # Copy stream to avoid modifying original
            st = stream.copy()
            
            # Basic validation
            if len(st) != 3:
                raise ValueError(f"Expected 3 components, got {len(st)}")
            
            # Sort traces by component (Z, N, E order)
            st = self._sort_components(st)
            
            # Check sampling rates
            if not self._check_sampling_rates(st):
                raise ValueError("Inconsistent sampling rates")
            
            # Apply preprocessing
            st = self._preprocess_stream(st, s_arrival)
            
            # Rotate to radial-transverse if needed
            if not self._is_radial_transverse(st):
                st = self._rotate_to_radial_transverse(st, back_azimuth)
            
            # Extract analysis window
            z_trace, r_trace, t_trace = st[0], st[1], st[2]
            
            # Calculate time window around S-arrival
            dt = r_trace.stats.delta
            s_sample = int((s_arrival - r_trace.stats.starttime) / dt)
            
            # Define signal window (S-wave pulse): tS-0.1s to tS+1.2s
            signal_pre_samples = int(0.1 / dt)  # 0.1s before S
            signal_post_samples = int(1.2 / dt)  # 1.2s after S
            signal_start = s_sample - signal_pre_samples
            signal_end = s_sample + signal_post_samples
            
            # Define noise window (pre-S): tS-3.0s to tS-0.5s  
            noise_pre_samples = int(3.0 / dt)  # 3.0s before S
            noise_buffer_samples = int(0.5 / dt)  # 0.5s buffer
            noise_start = s_sample - noise_pre_samples
            noise_end = s_sample - noise_buffer_samples
            
            # Check window bounds
            if (signal_start < 0 or signal_end >= len(r_trace.data) or 
                noise_start < 0 or noise_end >= len(r_trace.data)):
                raise ValueError("Analysis windows extend beyond trace bounds")
            
            # Extract signal windows (r_sig, t_sig)
            r_sig = r_trace.data[signal_start:signal_end].copy()
            t_sig = t_trace.data[signal_start:signal_end].copy()
            
            # Extract noise window (t_noise)
            t_noise = t_trace.data[noise_start:noise_end].copy()
            
            # Apply taper to signal windows only
            r_sig = self._apply_taper(r_sig)
            t_sig = self._apply_taper(t_sig)
            
            # Quality control checks
            qc_metrics = {}
            if quality_control:
                qc_metrics = self._quality_control_checks(r_sig, t_sig, t_noise, 
                                                        r_trace, s_sample)
                if not qc_metrics['passed']:
                    return {
                        'SI': np.nan,
                        'SI_uncertainty': np.nan,
                        'quality_metrics': qc_metrics,
                        'passed_qc': False
                    }
            
            # Calculate splitting intensity using correct Chevrot method
            si_value = self._compute_splitting_intensity(r_sig, t_sig, r_trace.stats.sampling_rate)
            
            # Estimate uncertainty using Chevrot noise projection
            si_uncertainty = self._estimate_si_uncertainty(r_sig, t_sig, t_noise, r_trace.stats.sampling_rate)
            
            # Additional metrics
            energy_ratio = np.sum(t_sig**2) / np.sum(r_sig**2) if np.sum(r_sig**2) > 0 else 0
            total_energy = np.sum(r_sig**2) + np.sum(t_sig**2)
            
            result = {
                'SI': si_value,
                'SI_uncertainty': si_uncertainty,
                'energy_ratio': energy_ratio,
                'total_energy': total_energy,
                'signal_window_start': r_trace.stats.starttime + signal_start * dt,
                'signal_window_length': (signal_end - signal_start) * dt,
                'noise_window_start': r_trace.stats.starttime + noise_start * dt,
                'noise_window_length': (noise_end - noise_start) * dt,
                'back_azimuth': back_azimuth,
                'quality_metrics': qc_metrics,
                'passed_qc': qc_metrics.get('passed', True)
            }
            
            return result
            
        except Exception as e:
            warnings.warn(f"SI calculation failed: {str(e)}")
            return {
                'SI': np.nan,
                'SI_uncertainty': np.nan,
                'quality_metrics': {'error': str(e)},
                'passed_qc': False
            }
    
    def _sort_components(self, stream: Stream) -> Stream:
        """Sort traces by component in Z, N, E order."""
        component_order = ['Z', 'N', 'E', '1', '2', 'R', 'T']
        
        # Try to sort by last character of channel code
        try:
            sorted_traces = []
            for comp in component_order:
                matching = [tr for tr in stream if tr.stats.channel[-1].upper() == comp]
                if matching:
                    sorted_traces.extend(matching)
            
            if len(sorted_traces) == 3:
                stream.traces = sorted_traces
            else:
                # Fallback: sort by channel name alphabetically
                stream.sort(['channel'])
                
        except:
            # If sorting fails, use original order
            pass
        
        return stream
    
    def _check_sampling_rates(self, stream: Stream) -> bool:
        """Check if all traces have the same sampling rate."""
        sampling_rates = [tr.stats.sampling_rate for tr in stream]
        return len(set(sampling_rates)) == 1
    
    def _preprocess_stream(self, stream: Stream, s_arrival: UTCDateTime) -> Stream:
        """Apply preprocessing filters and corrections."""
        for tr in stream:
            # Remove trend and mean
            tr.detrend('linear')
            tr.detrend('demean')
            
            # Apply bandpass filter
            try:
                tr.filter('bandpass', freqmin=self.freq_min, freqmax=self.freq_max, 
                         corners=3, zerophase=True)
            except:
                # Fallback for older ObsPy versions
                tr.data = bandpass(tr.data, self.freq_min, self.freq_max, 
                                 tr.stats.sampling_rate, corners=3, zerophase=True)
            
            # Apply taper
            tr.taper(max_percentage=0.05, type='hann')
        
        return stream
    
    def _is_radial_transverse(self, stream: Stream) -> bool:
        """Check if stream is already in radial-transverse coordinates."""
        channels = [tr.stats.channel[-1].upper() for tr in stream]
        return 'R' in channels and 'T' in channels
    
    def _rotate_to_radial_transverse(self, stream: Stream, back_azimuth: float) -> Stream:
        """
        Rotate horizontal components to radial-transverse system.
        
        Parameters:
        -----------
        stream : obspy.Stream
            Stream with Z, N, E components
        back_azimuth : float
            Back-azimuth in degrees from source to receiver
            
        Returns:
        --------
        rotated_stream : obspy.Stream  
            Stream with Z, R, T components
        """
        # Find horizontal components
        z_trace = None
        n_trace = None  
        e_trace = None
        
        for tr in stream:
            comp = tr.stats.channel[-1].upper()
            if comp in ['Z', '3']:
                z_trace = tr
            elif comp in ['N', '1', 'Y']:
                n_trace = tr
            elif comp in ['E', '2', 'X']:
                e_trace = tr
        
        if n_trace is None or e_trace is None:
            raise ValueError("Could not identify N and E components")
        
        # Convert back-azimuth to radians
        baz_rad = np.deg2rad(back_azimuth)
        
        # Rotation matrix (clockwise rotation by back-azimuth)
        # R = N * cos(baz) + E * sin(baz)
        # T = -N * sin(baz) + E * cos(baz)
        r_data = n_trace.data * np.cos(baz_rad) + e_trace.data * np.sin(baz_rad)
        t_data = -n_trace.data * np.sin(baz_rad) + e_trace.data * np.cos(baz_rad)
        
        # Create new traces
        r_trace = n_trace.copy()
        r_trace.data = r_data
        r_trace.stats.channel = r_trace.stats.channel[:-1] + 'R'
        
        t_trace = e_trace.copy()
        t_trace.data = t_data
        t_trace.stats.channel = t_trace.stats.channel[:-1] + 'T'
        
        # Create new stream
        rotated_stream = Stream()
        if z_trace:
            rotated_stream += z_trace
        rotated_stream += r_trace
        rotated_stream += t_trace
        
        return rotated_stream
    
    def _apply_taper(self, data: np.ndarray) -> np.ndarray:
        """Apply Hanning taper to window edges."""
        n = len(data)
        taper_samples = int(n * self.taper_fraction)
        
        if taper_samples > 0:
            # Hanning taper
            taper = np.ones(n)
            
            # Left edge
            for i in range(taper_samples):
                w = 0.5 * (1 - np.cos(np.pi * i / taper_samples))
                taper[i] = w
            
            # Right edge  
            for i in range(taper_samples):
                w = 0.5 * (1 - np.cos(np.pi * i / taper_samples))
                taper[n - 1 - i] = w
            
            data = data * taper
        
        return data
    
    def _quality_control_checks(self, r_sig: np.ndarray, t_sig: np.ndarray,
                               t_noise: np.ndarray, r_trace: Trace, s_sample: int) -> Dict:
        """
        Perform quality control checks on windowed data.
        
        Returns:
        --------
        qc_metrics : dict
            Quality control results and metrics
        """
        qc_metrics = {'passed': True}
        
        # 1. Signal-to-noise ratio check using provided noise window
        try:
            signal_power = np.mean(r_sig**2)
            noise_power = np.mean(t_noise**2)  # Use the extracted noise window
            
            if noise_power > 0:
                snr = signal_power / noise_power
                qc_metrics['snr'] = snr
                
                if snr < self.min_snr:
                    qc_metrics['passed'] = False
                    qc_metrics['fail_reason'] = f'Low SNR: {snr:.2f} < {self.min_snr}'
        except:
            pass  # SNR check failed, but don't fail overall
        
        # 2. Excessive transverse energy check
        r_rms = np.sqrt(np.mean(r_sig**2))
        t_rms = np.sqrt(np.mean(t_sig**2))
        
        if r_rms > 0:
            noise_ratio = t_rms / (r_rms + t_rms)
            qc_metrics['noise_ratio'] = noise_ratio
            
            if noise_ratio > self.max_noise_level:
                qc_metrics['passed'] = False
                qc_metrics['fail_reason'] = f'High noise: {noise_ratio:.2f} > {self.max_noise_level}'
        
        # 3. Check for data gaps or spikes in signal windows
        if np.any(np.isnan(r_sig)) or np.any(np.isnan(t_sig)) or np.any(np.isnan(t_noise)):
            qc_metrics['passed'] = False
            qc_metrics['fail_reason'] = 'Data contains NaN values'
        
        # Check for extreme values (potential spikes) in signal windows
        r_std = np.std(r_sig)
        t_std = np.std(t_sig)
        
        if r_std > 0 and np.any(np.abs(r_sig) > 10 * r_std):
            qc_metrics['passed'] = False
            qc_metrics['fail_reason'] = 'Radial signal contains extreme values'
        
        if t_std > 0 and np.any(np.abs(t_sig) > 10 * t_std):
            qc_metrics['passed'] = False
            qc_metrics['fail_reason'] = 'Transverse signal contains extreme values'
        
        return qc_metrics
    
    def _compute_splitting_intensity(self, r_sig: np.ndarray, t_sig: np.ndarray, sampling_rate: float) -> float:
        """
        Compute splitting intensity using the Chevrot (2000) method.
        
        SI = ∫ t_sig(t) * r_sig'(t) dt / ∫ r_sig'²(t) dt
        
        where:
        - r_sig is the radial component S-wave signal window
        - t_sig is the transverse component S-wave signal window  
        - r_sig'(t) is the time derivative of the radial component
        """
        # Compute time derivative of radial component (not transverse!)
        dt = 1.0 / sampling_rate  # Correct time step
        r_derivative = np.gradient(r_sig, dt)
        
        # Numerator: integral of t_sig * d(r_sig)/dt
        numerator = np.trapz(t_sig * r_derivative, dx=dt)

        # Denominator: radial derivative energy
        denominator = np.trapz(r_derivative**2, dx=dt)

        if denominator <= 0:
            return 0.0

        # Splitting intensity
        si = numerator / denominator
        return si
    
    def _estimate_si_uncertainty(self, r_sig: np.ndarray, t_sig: np.ndarray, 
                                t_noise: np.ndarray, sampling_rate: float) -> float:
        """
        Estimate uncertainty in splitting intensity following Chevrot-style
        noise projection onto the SI functional.

        SI = ∫ t_sig * d(r_sig)/dt dt / ∫ (d(r_sig)/dt)^2 dt

        σ(SI)² ≈ ∫ (n_T * d(r_sig)/dt)² dt / (∫ (d(r_sig)/dt)² dt)²
        
        where n_T is the transverse noise from pre-S window.
        """
        dt = 1.0 / sampling_rate

        # Derivative of radial signal in S-wave window
        r_derivative = np.gradient(r_sig, dt)

        # Denominator: radial derivative energy (squared for uncertainty)
        denom = np.trapz(r_derivative**2, dx=dt)
        if denom <= 0:
            return 1.0  # Cannot compute, return large uncertainty

        # Match noise length to signal length
        if len(t_noise) >= len(r_derivative):
            # Truncate noise to match signal length
            n_T = t_noise[:len(r_derivative)]
        else:
            # Pad noise with zeros if too short
            n_T = np.pad(t_noise, (0, len(r_derivative) - len(t_noise)), 'constant')

        # Numerator: noise projection onto radial derivative operator
        numerator = np.trapz((n_T * r_derivative)**2, dx=dt)

        # SI uncertainty
        sigma_si = np.sqrt(numerator) / denom
        return sigma_si



def batch_calculate_si(organized_waveforms: Dict, stations_df,
                      si_calculator: Optional[SplittingIntensityCalculator] = None,
                      quality_control: bool = True) -> Dict:
    """
    Calculate splitting intensity for a batch of organized waveforms.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary of waveform data organized by event ID
    stations_df : pandas.DataFrame
        Station information including coordinates
    si_calculator : SplittingIntensityCalculator, optional
        SI calculator instance (created if None)
    quality_control : bool
        Whether to apply quality control
        
    Returns:
    --------
    si_results : dict
        Dictionary of SI results by event ID
    """
    if si_calculator is None:
        si_calculator = SplittingIntensityCalculator()
    
    si_results = {}
    
    for event_id, event_data in organized_waveforms.items():
        try:
            # Get required data
            traces = event_data.get('traces')
            s_arrival_str = event_data.get('origin_time') + event_data.get('s_arrival_time')
            back_azimuth = event_data.get('back_azimuth')
            
            if traces is None or s_arrival_str is None:
                continue
            
            # Convert S-arrival to UTCDateTime
            if isinstance(s_arrival_str, str):
                s_arrival = UTCDateTime(s_arrival_str)
            else:
                s_arrival = UTCDateTime(s_arrival_str)
            
            # Calculate SI
            si_result = si_calculator.calculate_si_from_stream(
                traces, s_arrival, back_azimuth, quality_control=quality_control
            )
            
            # Add metadata
            si_result['event_id'] = event_id
            si_result['station'] = event_data.get('station')
            
            si_results[event_id] = si_result
            
        except Exception as e:
            print(f"SI calculation failed for event {event_id}: {e}")
            si_results[event_id] = {
                'SI': np.nan,
                'SI_uncertainty': np.nan, 
                'passed_qc': False,
                'error': str(e)
            }
    
    return si_results


