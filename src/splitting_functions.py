"""
Master Shear-Wave Splitting Analysis Functions

This module contains all the functions used in the master shear-wave splitting workflow,
extracted from the notebook for better code organization and reusability.

Author: Michael Hemmett with Claude 4.0
Date: 11-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime
import obspy
from tqdm import tqdm
import swspy

def create_extended_catalog(catalog_df, pre_event_time=4.0, post_event_time=15.0):
    """
    Create extended start and end times for waveform retrieval.
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Input catalog with datetime column
    pre_event_time : float
        Time before event origin (seconds) for noise/P-wave analysis
    post_event_time : float
        Time after event origin (seconds) for S-wave analysis
    
    Returns:
    --------
    pandas.DataFrame
        Catalog with extended starttime and endtime columns
    """
    extended_catalog = catalog_df.copy()
    
    # Convert datetime to UTCDateTime objects
    origin_times = [UTCDateTime(dt) for dt in extended_catalog['datetime']]
    
    # Create extended time windows
    extended_catalog['starttime'] = [ot - pre_event_time for ot in origin_times]
    extended_catalog['endtime'] = [ot + post_event_time for ot in origin_times]
    extended_catalog['origin_utc'] = origin_times
    
    # Add timing parameters for reference
    extended_catalog['pre_event_sec'] = pre_event_time
    extended_catalog['post_event_sec'] = post_event_time
    extended_catalog['total_duration'] = pre_event_time + post_event_time
    
    return extended_catalog


def organize_waveform_data(traces_dict, extended_catalog):
    """
    Organize waveform data by event ID and ensure proper timing alignment.
    
    Parameters:
    -----------
    traces_dict : dict
        Dictionary of trace data keyed by event ID
    extended_catalog : pandas.DataFrame
        Catalog with extended timing information
    
    Returns:
    --------
    dict
        Organized waveform data with timing metadata
    """
    organized_data = {}
    
    for idx, row in extended_catalog.iterrows():
        event_id = row['id']
        
        if event_id in traces_dict:
            # Get the traces for this event
            event_traces = traces_dict[event_id]
            
            # Add all metadata needed for QC and analysis
            organized_data[event_id] = {
                'traces': event_traces,
                'origin_time': row['origin_utc'],
                'datetime': row['datetime'],  # For SNR calculations
                'starttime': row['starttime'], 
                'endtime': row['endtime'],
                'pre_event_sec': row['pre_event_sec'],
                'post_event_sec': row['post_event_sec'],
                'magnitude': row['mag'],
                'depth': row['dep'],
                'latitude': row['lat'],
                'longitude': row['lon'],
                'station': row['station'],  # For back-azimuth and station lookups
                'p_arrival_time': row.get('p_arrival_time', np.nan),  # For SNR P-window
                's_arrival_time': row.get('s_arrival_time', np.nan)   # For SNR S-window
            }
    
    return organized_data


def snr(trace, noise_start_offset=0.5, noise_duration=2.0, 
                  signal_start_offset=0.5, signal_duration=3.0):
    """
    Calculate signal-to-noise ratio for a seismic trace.
    
    Parameters:
    -----------
    trace : obspy.Trace
        Seismic trace data
    noise_start_offset : float
        Seconds from trace start to begin noise window
    noise_duration : float
        Duration of noise window in seconds
    signal_start_offset : float
        Seconds from P-arrival to begin signal window
    signal_duration : float
        Duration of signal window in seconds
    
    Returns:
    --------
    float
        Signal-to-noise ratio
    """
    try:
        sampling_rate = trace.stats.sampling_rate
        
        # Calculate sample indices for noise window
        noise_start_idx = int(noise_start_offset * sampling_rate)
        noise_end_idx = int((noise_start_offset + noise_duration) * sampling_rate)
        
        # Assume P-arrival is at pre_event_time from start
        p_arrival_idx = int(4.0 * sampling_rate)  # 4 seconds pre-event time
        signal_start_idx = p_arrival_idx + int(signal_start_offset * sampling_rate)
        signal_end_idx = signal_start_idx + int(signal_duration * sampling_rate)
        
        # Ensure indices are within trace bounds
        if (noise_end_idx >= len(trace.data) or 
            signal_end_idx >= len(trace.data) or
            noise_start_idx < 0 or signal_start_idx < 0):
            return np.nan
        
        # Calculate RMS amplitudes
        noise_rms = np.sqrt(np.mean(trace.data[noise_start_idx:noise_end_idx]**2))
        signal_rms = np.sqrt(np.mean(trace.data[signal_start_idx:signal_end_idx]**2))
        
        # Avoid division by zero
        if noise_rms == 0:
            return np.inf if signal_rms > 0 else np.nan
        
        return signal_rms / noise_rms
    
    except Exception as e:
        print(f"SNR calculation error: {e}")
        return np.nan


def analyze_p_wave_rectilinearity(trace_z, trace_n, trace_e, p_arrival_offset=4.0, 
                                 analysis_window=1.0):
    """
    Analyze P-wave rectilinearity using covariance matrix analysis.
    
    Parameters:
    -----------
    trace_z, trace_n, trace_e : obspy.Trace
        Vertical, North, and East component traces
    p_arrival_offset : float
        Time of P-arrival from trace start (seconds)
    analysis_window : float
        Window duration for P-wave analysis (seconds)
    
    Returns:
    --------
    dict
        Rectilinearity metrics including eigenvalue ratios and polarization
    """
    try:
        # Ensure all traces have same sampling rate
        sampling_rate = trace_z.stats.sampling_rate
        if not (trace_n.stats.sampling_rate == sampling_rate and 
                trace_e.stats.sampling_rate == sampling_rate):
            return {'rectilinearity': np.nan, 'azimuth': np.nan, 'incidence': np.nan}
        
        # Calculate sample indices for P-wave window
        p_arrival_idx = int(p_arrival_offset * sampling_rate)
        window_samples = int(analysis_window * sampling_rate)
        start_idx = p_arrival_idx
        end_idx = p_arrival_idx + window_samples
        
        # Ensure window is within trace bounds
        min_length = min(len(trace_z.data), len(trace_n.data), len(trace_e.data))
        if end_idx >= min_length or start_idx < 0:
            return {'rectilinearity': np.nan, 'incidence': np.nan}
        
        # Extract P-wave data
        z_data = trace_z.data[start_idx:end_idx]
        n_data = trace_n.data[start_idx:end_idx]
        e_data = trace_e.data[start_idx:end_idx]
        
        # Create data matrix
        data_matrix = np.column_stack([z_data, n_data, e_data])
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(data_matrix.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Calculate rectilinearity (linearity measure)
        rectilinearity = 1 - (eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0])
        
        # Principal polarization vector (largest eigenvalue)
        principal_vector = eigenvectors[:, 0]
        
        # Calculate azimuth and incidence angle
        incidence = np.degrees(np.arccos(abs(principal_vector[0])))  # Z component
        
        return {
            'rectilinearity': rectilinearity,
            'incidence': incidence,
            'eigenvalues': eigenvalues,
            'principal_vector': principal_vector
        }
    
    except Exception as e:
        print(f"P-wave rectilinearity analysis error: {e}")
        return {'rectilinearity': np.nan, 'incidence': np.nan}

def calculate_p_wave_metrics_for_organized_waveforms(organized_waveforms,
                                                      p_arrival_offset=4.0,
                                                      analysis_window=1.0):
    """
    Calculate P-wave rectilinearity and incidence angle for all events in organized_waveforms.
    
    This function runs the P-wave covariance matrix analysis on all events,
    extracting QC metrics (rectilinearity and incidence angle) needed for 
    quality control filtering.
    
    All required metadata is already in organized_waveforms, so no external 
    catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_offset : float, optional
        Time of P-arrival from trace start (seconds), default=4.0
    analysis_window : float, optional
        Window duration for P-wave analysis (seconds), default=1.0
        
    Returns:
    --------
    dict
        Updated organized_waveforms with P-wave metrics added to each event:
        - 'rectilinearity': P-wave linearity measure (0-1, higher is better)
        - 'incidence': Angle from vertical (degrees, lower is better)
        - 'p_wave_azimuth': P-wave polarization direction (optional QC)
    """
    
    print(f"Calculating P-wave metrics for {len(organized_waveforms)} events...")
    print(f"P-arrival offset: {p_arrival_offset}s, Analysis window: {analysis_window}s")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['rectilinearity'] = np.nan
            event_data['incidence'] = np.nan
            event_data['p_wave_azimuth'] = np.nan
            continue
        
        # Convert to stream if it's a list
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        print(f"  Found {len(event_stream)} traces")
        
        # Find Z, N, E components
        trace_z = None
        trace_n = None
        trace_e = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component == 'Z':
                trace_z = tr
            elif component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
        
        # Check if we have all three components
        if trace_z is None or trace_n is None or trace_e is None:
            print(f"  Missing components: Z={trace_z is not None}, "
                  f"N={trace_n is not None}, E={trace_e is not None}")
            event_data['rectilinearity'] = np.nan
            event_data['incidence'] = np.nan
            event_data['p_wave_azimuth'] = np.nan
            continue
        
        # Perform P-wave rectilinearity analysis
        try:
            p_wave_metrics = analyze_p_wave_rectilinearity(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_offset,
                analysis_window=analysis_window
            )
            
            # Add results to event_data
            event_data['rectilinearity'] = p_wave_metrics['rectilinearity']
            event_data['incidence'] = p_wave_metrics['incidence']
            
            print(f"  Results:")
            print(f"    Rectilinearity: {p_wave_metrics['rectilinearity']:.3f}")
            print(f"    Incidence angle: {p_wave_metrics['incidence']:.1f}°")
            
            if not np.isnan(p_wave_metrics['rectilinearity']):
                success_count += 1
                
        except Exception as e:
            print(f"  Error calculating P-wave metrics: {e}")
            event_data['rectilinearity'] = np.nan
            event_data['incidence'] = np.nan
            event_data['p_wave_azimuth'] = np.nan
    
    print(f"\n{'='*60}")
    print("P-Wave Metrics Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid P-wave metrics: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    rect_values = [data.get('rectilinearity', np.nan) for data in organized_waveforms.values()]
    incidence_values = [data.get('incidence', np.nan) for data in organized_waveforms.values()]
    
    valid_rect = [v for v in rect_values if not np.isnan(v)]
    valid_incidence = [v for v in incidence_values if not np.isnan(v)]
    
    if valid_rect:
        print(f"\nRectilinearity Statistics:")
        print(f"  Range: {min(valid_rect):.3f} to {max(valid_rect):.3f}")
        print(f"  Mean: {np.mean(valid_rect):.3f}")
        print(f"  Median: {np.median(valid_rect):.3f}")
        
        # Show how many pass typical QC threshold
        passing_rect = sum(1 for v in valid_rect if v >= 0.7)
        print(f"  Passing QC (≥0.7): {passing_rect}/{len(valid_rect)} ({100*passing_rect/len(valid_rect):.1f}%)")
    
    if valid_incidence:
        print(f"\nIncidence Angle Statistics:")
        print(f"  Range: {min(valid_incidence):.1f}° to {max(valid_incidence):.1f}°")
        print(f"  Mean: {np.mean(valid_incidence):.1f}°")
        print(f"  Median: {np.median(valid_incidence):.1f}°")
        
        # Show how many pass typical QC threshold
        passing_incidence = sum(1 for v in valid_incidence if v <= 30.0)
        print(f"  Passing QC (≤30°): {passing_incidence}/{len(valid_incidence)} ({100*passing_incidence/len(valid_incidence):.1f}%)")
    
    return organized_waveforms


def perform_splitting_analysis(event_data, use_dynamic_params=True, plot_results=True):
    """
    Perform shear-wave splitting analysis using data from organized_waveforms.
    
    This function now uses dynamic parameter calculation to optimize the analysis
    window and filtering based on the event's spectral characteristics. All required
    data (traces, timing, metadata) comes from event_data.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with N/E/Z components
        - station: Station name
        - back_azimuth: Geographic back-azimuth
        - incidence: P-wave incidence angle
        - s_arrival_time: S-wave arrival (seconds from origin)
        - datetime: Event origin time
        - magnitude: Event magnitude
        - All QC metrics (snr_horizontal, rectilinearity, etc.)
    use_dynamic_params : bool, optional
        Whether to use dynamic parameter estimation (default=True)
        If True, calculates optimal window and filter parameters from data
        If False, uses fixed default parameters
    plot_results : bool, optional
        Whether to generate diagnostic plots (default=True)
    
    Returns:
    --------
    tuple
        (result_dict, splitting_obj) containing:
        - result_dict: Dict with splitting parameters and metadata
        - splitting_obj: SWSPy splitting object with sws_result_df and plots
    """
    
    station_name = event_data.get('station', 'UNKNOWN')
    
    try:
        print(f"  Creating SWSPy splitting object with dynamic parameters...")
        
        # Create the splitting analysis object using our helper function
        # This handles all the setup: filtering, windowing, parameter optimization
        splitting_obj = create_splitting_analysis(event_data, use_dynamic_params=use_dynamic_params)
        
        # Get dynamic parameters if they were calculated
        dynamic_params = getattr(splitting_obj, 'dynamic_params', None)
        
        if dynamic_params:
            t_dom = dynamic_params['t_dom']
            window_duration = dynamic_params['dynamic_window_length']
            freq_min = dynamic_params['optimal_freq_min']
            freq_max = dynamic_params['optimal_freq_max']
        else:
            # Use default values
            t_dom = 0.1
            window_duration = 1.0
            freq_min = 5.0
            freq_max = 40.0
        
        # Perform the actual splitting analysis
        print(f"  Running SWSPy splitting measurement...")
        splitting_obj.perform_sws_analysis(coord_system="ZNE", sws_method="EV_and_XC")
        
        # Extract results from sws_result_df DataFrame
        print(f"  Extracting results from sws_result_df...")
        
        if not hasattr(splitting_obj, 'sws_result_df'):
            raise ValueError("splitting_obj does not have 'sws_result_df' attribute after perform_sws_analysis()")
        
        results_df = splitting_obj.sws_result_df
        
        if results_df is None or len(results_df) == 0:
            raise ValueError("sws_result_df is empty or None")
        
        # Extract splitting parameters from the first row (should be only one station)
        result_row = results_df.iloc[0]
        
        # Get splitting parameters (adjust column names based on actual DataFrame)
        phi_best = result_row.get('fast', result_row.get('phi_from_N', np.nan))
        dt_best = result_row.get('lag', result_row.get('dt', np.nan))
        
        # Get error estimates if available
        phi_error = result_row.get('fast_error', result_row.get('phi_err', np.nan))
        dt_error = result_row.get('lag_error', result_row.get('dt_err', np.nan))
        
        # Get other metadata from event_data
        avg_snr = event_data.get('snr_horizontal', np.nan)
        magnitude = event_data.get('magnitude', np.nan)
        s_arrival_time = event_data.get('s_arrival_time', np.nan)
        
        result_dict = {
            'station': station_name,
            'phi': float(phi_best) if not pd.isna(phi_best) else np.nan,
            'dt': float(dt_best) if not pd.isna(dt_best) else np.nan,
            'phi_error': float(phi_error) if not pd.isna(phi_error) else np.nan,
            'dt_error': float(dt_error) if not pd.isna(dt_error) else np.nan,
            'snr_avg': avg_snr,
            's_arrival_time': s_arrival_time,
            'magnitude': magnitude,
            'filter_freq_min': freq_min,
            'filter_freq_max': freq_max,
            'window_duration': window_duration,
            'success': True if not np.isnan(phi_best) and not np.isnan(dt_best) else False,
            'sws_result_df': results_df  # Include full DataFrame for reference
        }
        
        # Add dynamic parameters to result if used
        if use_dynamic_params and dynamic_params is not None:
            result_dict['dynamic_params'] = dynamic_params
            result_dict['dominant_period'] = t_dom
        
        # Generate diagnostic plots
        if plot_results:
            print(f"  Generating diagnostic plots...")
            try:
                splitting_obj.plot()
                plt.show()
            except Exception as plot_error:
                print(f"  Warning: Could not generate plots: {plot_error}")
        
        print(f"  ✓ Splitting analysis complete!")
        print(f"    Fast axis (φ): {result_dict['phi']:.1f}°" if not np.isnan(result_dict['phi']) else "    Fast axis (φ): N/A")
        print(f"    Delay time (δt): {result_dict['dt']:.3f}s" if not np.isnan(result_dict['dt']) else "    Delay time (δt): N/A")
        
        # Return both the result dict and the splitting object
        return result_dict, splitting_obj
    
    except Exception as e:
        import traceback
        error_dict = {
            'station': station_name,
            'phi': np.nan,
            'dt': np.nan,
            'phi_error': np.nan,
            'dt_error': np.nan,
            'quality': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }
        print(f"  ✗ Error: {e}")
        return error_dict, None


def apply_quality_control(organized_waveforms, qc_thresholds):
    """
    Apply quality control filters to organized_waveforms and return only events that pass all thresholds.
    
    This function assumes that all QC metrics have already been calculated and stored in organized_waveforms:
    - snr_horizontal: S-wave signal-to-noise ratio
    - rectilinearity: P-wave linearity measure
    - incidence: P-wave angle from vertical
    - magnitude: Event magnitude (optional)
    - back_azimuth: Geographic back-azimuth (for reference)
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing traces and all QC metrics
    qc_thresholds : dict
        Quality control threshold values with keys:
        - 'min_snr': Minimum horizontal SNR (e.g., 2.0)
        - 'min_rectilinearity': Minimum P-wave rectilinearity (e.g., 0.7)
        - 'max_incidence': Maximum incidence angle in degrees (e.g., 30.0)
        - 'min_magnitude': Minimum magnitude (optional, e.g., 0.0)
        
    Returns:
    --------
    dict
        Filtered organized_waveforms containing only events that pass ALL QC thresholds
    """
    
    print(f"\n{'='*60}")
    print("Applying Quality Control Filters")
    print(f"{'='*60}")
    print(f"Initial events: {len(organized_waveforms)}")
    print(f"\nQC Thresholds:")
    print(f"  Minimum SNR: {qc_thresholds.get('min_snr', 2.0)}")
    print(f"  Minimum Rectilinearity: {qc_thresholds.get('min_rectilinearity', 0.7)}")
    print(f"  Maximum Incidence: {qc_thresholds.get('max_incidence', 30.0)}°")
    print(f"  Minimum Magnitude: {qc_thresholds.get('min_magnitude', 0.0)}")
    
    # Extract thresholds with defaults
    min_snr = qc_thresholds.get('min_snr', 2.0)
    min_rectilinearity = qc_thresholds.get('min_rectilinearity', 0.7)
    max_incidence = qc_thresholds.get('max_incidence', 30.0)
    min_magnitude = qc_thresholds.get('min_magnitude', 0.0)
    
    # Track QC statistics
    qc_stats = {
        'total': len(organized_waveforms),
        'failed_snr': 0,
        'failed_rectilinearity': 0,
        'failed_incidence': 0,
        'failed_magnitude': 0,
        'failed_missing_data': 0,
        'passed': 0
    }
    
    filtered_waveforms = {}
    
    print(f"\n{'='*60}")
    print("Checking Individual Events")
    print(f"{'='*60}")
    
    for event_id, event_data in organized_waveforms.items():
        # Track failure reasons for this event
        failure_reasons = []
        
        # Check for missing QC metrics
        snr_horizontal = event_data.get('snr_horizontal', np.nan)
        rectilinearity = event_data.get('rectilinearity', np.nan)
        incidence = event_data.get('incidence', np.nan)
        magnitude = event_data.get('magnitude', 0.0)
        
        # Check if any required metrics are missing
        if np.isnan(snr_horizontal):
            failure_reasons.append('Missing SNR')
            qc_stats['failed_missing_data'] += 1
        
        if np.isnan(rectilinearity):
            failure_reasons.append('Missing rectilinearity')
            qc_stats['failed_missing_data'] += 1
        
        if np.isnan(incidence):
            failure_reasons.append('Missing incidence')
            qc_stats['failed_missing_data'] += 1
        
        # Apply QC thresholds
        if not np.isnan(snr_horizontal) and snr_horizontal < min_snr:
            failure_reasons.append(f'SNR too low ({snr_horizontal:.2f} < {min_snr})')
            qc_stats['failed_snr'] += 1
        
        if not np.isnan(rectilinearity) and rectilinearity < min_rectilinearity:
            failure_reasons.append(f'Rectilinearity too low ({rectilinearity:.3f} < {min_rectilinearity})')
            qc_stats['failed_rectilinearity'] += 1
        
        if not np.isnan(incidence) and incidence > max_incidence:
            failure_reasons.append(f'Incidence too high ({incidence:.1f}° > {max_incidence}°)')
            qc_stats['failed_incidence'] += 1
        
        if magnitude < min_magnitude:
            failure_reasons.append(f'Magnitude too low ({magnitude:.1f} < {min_magnitude})')
            qc_stats['failed_magnitude'] += 1
        
        # Event passes if no failure reasons
        if len(failure_reasons) == 0:
            filtered_waveforms[event_id] = event_data
            qc_stats['passed'] += 1
            print(f"✓ Event {event_id}: PASS")
            print(f"    SNR={snr_horizontal:.2f}, Rect={rectilinearity:.3f}, Inc={incidence:.1f}°, Mag={magnitude:.1f}")
        else:
            print(f"✗ Event {event_id}: FAIL - {', '.join(failure_reasons)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Quality Control Summary")
    print(f"{'='*60}")
    print(f"Total events processed: {qc_stats['total']}")
    print(f"Events passed: {qc_stats['passed']} ({100*qc_stats['passed']/qc_stats['total']:.1f}%)")
    print(f"Events failed: {qc_stats['total'] - qc_stats['passed']}")
    print(f"\nFailure Breakdown:")
    print(f"  Failed SNR threshold: {qc_stats['failed_snr']}")
    print(f"  Failed rectilinearity threshold: {qc_stats['failed_rectilinearity']}")
    print(f"  Failed incidence threshold: {qc_stats['failed_incidence']}")
    print(f"  Failed magnitude threshold: {qc_stats['failed_magnitude']}")
    print(f"  Missing QC data: {qc_stats['failed_missing_data']}")
    print(f"\nReturning {len(filtered_waveforms)} events for splitting analysis")
    
    return filtered_waveforms


def perform_splitting_on_organized_waveforms(organized_waveforms):
    """
    Perform shear-wave splitting analysis on all events in organized_waveforms.
    
    This function assumes organized_waveforms has been filtered by apply_quality_control()
    and contains only events that pass QC thresholds. It extracts horizontal components
    and performs splitting analysis using the back_azimuth for coordinate rotation.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing QC-filtered traces and metadata
        Must have: traces, back_azimuth, station, and all other event metadata
        
    Returns:
    --------
    dict
        Dictionary with event IDs as keys, each containing:
        {
            event_id: {
                'result': result_dict,  # Splitting parameters and metadata
                'splitting_obj': splitting_obj  # SWSPy splitting object
            },
            ...
        }
    """
    
    print(f"\n{'='*60}")
    print("Performing Shear-Wave Splitting Analysis")
    print(f"{'='*60}")
    print(f"Processing {len(organized_waveforms)} QC-filtered events...")
    
    event_results = {}
    
    # Track statistics
    stats = {
        'total_events': len(organized_waveforms),
        'missing_components': 0,
        'missing_back_azimuth': 0,
        'splitting_errors': 0,
        'successful_splits': 0
    }
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\n{'─'*60}")
        print(f"Event {event_id}")
        print(f"{'─'*60}")
        
        # Get traces
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  ✗ No traces found")
            stats['missing_components'] += 1
            continue
        
        # Convert to stream if needed
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        # Find N and E components
        trace_n = None
        trace_e = None
        trace_z = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
            elif component == 'Z':
                trace_z = tr
        
        # Check if we have horizontal components
        if trace_n is None or trace_e is None:
            print(f"  ✗ Missing horizontal components (N: {trace_n is not None}, E: {trace_e is not None})")
            stats['missing_components'] += 1
            continue
        
        print(f"  ✓ Found horizontal components")
        
        # Check for back-azimuth
        back_azimuth = event_data.get('back_azimuth')
        if back_azimuth is None or np.isnan(back_azimuth):
            print(f"  ✗ Missing back-azimuth")
            stats['missing_back_azimuth'] += 1
            continue
        
        print(f"  ✓ Back-azimuth: {back_azimuth:.2f}°")
        
        # Get station and other metadata
        station_name = event_data.get('station', 'UNKNOWN')
        magnitude = event_data.get('magnitude', np.nan)
        snr_horizontal = event_data.get('snr_horizontal', np.nan)
        rectilinearity = event_data.get('rectilinearity', np.nan)
        incidence = event_data.get('incidence', np.nan)
        
        print(f"  Station: {station_name}")
        print(f"  Magnitude: {magnitude:.1f}")
        print(f"  SNR: {snr_horizontal:.2f}")
        print(f"  Rectilinearity: {rectilinearity:.3f}")
        print(f"  Incidence: {incidence:.1f}°")
        
        # Perform splitting analysis using dynamic parameters
        print(f"\n  → Running splitting analysis with dynamic parameters...")
        
        try:
            # perform_splitting_analysis now returns (result_dict, splitting_obj)
            splitting_result, splitting_obj = perform_splitting_analysis(
                event_data, use_dynamic_params=True, plot_results=True
            )
            
            if splitting_result.get('success', False):
                # Add additional metadata to result
                splitting_result['event_id'] = event_id
                splitting_result['back_azimuth'] = back_azimuth
                splitting_result['snr_horizontal'] = snr_horizontal
                splitting_result['rectilinearity'] = rectilinearity
                splitting_result['incidence'] = incidence
                splitting_result['event_lat'] = event_data.get('latitude')
                splitting_result['event_lon'] = event_data.get('longitude')
                splitting_result['event_depth'] = event_data.get('depth')
                splitting_result['event_datetime'] = event_data.get('datetime')
                
                # Store both result dict and splitting object for later use
                event_results[event_id] = {
                    'result': splitting_result,
                    'splitting_obj': splitting_obj
                }
                stats['successful_splits'] += 1
                
                print(f"  ✓ SUCCESS!")
                if not np.isnan(splitting_result['phi']):
                    print(f"    Fast axis (φ): {splitting_result['phi']:.1f}°")
                else:
                    print(f"    Fast axis (φ): N/A")
                    
                if not np.isnan(splitting_result['dt']):
                    print(f"    Delay time (δt): {splitting_result['dt']:.3f}s")
                else:
                    print(f"    Delay time (δt): N/A")
                    
                if not np.isnan(splitting_result.get('phi_error', np.nan)):
                    print(f"    φ error: ±{splitting_result['phi_error']:.1f}°")
                if not np.isnan(splitting_result.get('dt_error', np.nan)):
                    print(f"    δt error: ±{splitting_result['dt_error']:.3f}s")
                if 'dominant_period' in splitting_result:
                    print(f"    Dominant period: {splitting_result['dominant_period']:.3f}s")
            else:
                error_msg = splitting_result.get('error', 'Unknown error')
                print(f"  ✗ FAILED: {error_msg}")
                if 'traceback' in splitting_result:
                    print(f"  Traceback:\n{splitting_result['traceback']}")
                stats['splitting_errors'] += 1
                
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR during splitting: {e}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            stats['splitting_errors'] += 1
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("Splitting Analysis Summary")
    print(f"{'='*60}")
    print(f"Total events processed: {stats['total_events']}")
    print(f"Successful splits: {stats['successful_splits']} ({100*stats['successful_splits']/stats['total_events']:.1f}%)")
    print(f"\nFailure breakdown:")
    print(f"  Missing components: {stats['missing_components']}")
    print(f"  Missing back-azimuth: {stats['missing_back_azimuth']}")
    print(f"  Splitting errors: {stats['splitting_errors']}")
    
    # Calculate splitting parameter statistics if we have results
    if event_results:
        # Extract phi and dt values from nested structure
        phi_values = [r['result']['phi'] for r in event_results.values() 
                      if not np.isnan(r['result']['phi'])]
        dt_values = [r['result']['dt'] for r in event_results.values() 
                     if not np.isnan(r['result']['dt'])]
        
        if phi_values and dt_values:
            print(f"\n{'─'*60}")
            print("Splitting Parameter Statistics")
            print(f"{'─'*60}")
            print(f"Fast axis direction (φ):")
            print(f"  Mean: {np.mean(phi_values):.1f}° ± {np.std(phi_values):.1f}°")
            print(f"  Range: {np.min(phi_values):.1f}° to {np.max(phi_values):.1f}°")
            print(f"  Median: {np.median(phi_values):.1f}°")
            
            print(f"\nDelay time (δt):")
            print(f"  Mean: {np.mean(dt_values):.3f} ± {np.std(dt_values):.3f}s")
            print(f"  Range: {np.min(dt_values):.3f}s to {np.max(dt_values):.3f}s")
            print(f"  Median: {np.median(dt_values):.3f}s")
            
            # Add to stats
            stats['phi_mean'] = float(np.mean(phi_values))
            stats['phi_std'] = float(np.std(phi_values))
            stats['dt_mean'] = float(np.mean(dt_values))
            stats['dt_std'] = float(np.std(dt_values))
        else:
            print(f"\n  Note: No valid splitting parameters for statistics")
    
    # Return the event_results dictionary directly
    # Each entry contains both 'result' and 'splitting_obj'
    return event_results


def process_all_events(waveform_data, extended_catalog, qc_thresholds):
    """
    Process all events through the complete splitting analysis workflow.
    
    Parameters:
    -----------
    waveform_data : dict
        Organized waveform data by event ID
    extended_catalog : pandas.DataFrame
        Extended catalog with timing information
    qc_thresholds : dict
        Quality control threshold values
    
    Returns:
    --------
    dict
        Complete results including QC and splitting analysis
    """
    
    all_results = {
        'qc_results': {},
        'splitting_results': {},
        'summary_stats': {}
    }
    
    processed_count = 0
    qc_passed_count = 0
    splitting_success_count = 0
    
    print(f"Processing {len(waveform_data)} events...")
    
    for event_id in tqdm(waveform_data.keys(), desc="Processing events"):
        event_info = waveform_data[event_id]
        event_traces = event_info['traces']
        
        if not event_traces:
            continue
            
        processed_count += 1
        
        # Apply quality control
        qc_result = apply_quality_control(event_traces, event_info, qc_thresholds)
        all_results['qc_results'][event_id] = qc_result
        
        if qc_result['overall_pass']:
            qc_passed_count += 1
            
            # Organize traces by station and component for splitting analysis
            traces_by_station = {}
            for trace in event_traces:
                station = trace.stats.station
                component = trace.stats.channel[-1]
                
                if station not in traces_by_station:
                    traces_by_station[station] = {}
                traces_by_station[station][component] = trace
            
            # Perform splitting analysis on stations with horizontal components
            event_splitting_results = {}
            
            for station, components in traces_by_station.items():
                # Check if we have horizontal components
                trace_n = components.get('N') or components.get('1')
                trace_e = components.get('E') or components.get('2')
                
                if trace_n and trace_e:
                    # Check if this station passed QC
                    if (station in qc_result['station_results'] and 
                        qc_result['station_results'][station]['snr_pass']):
                        
                        splitting_result = perform_splitting_analysis(
                            trace_n, trace_e, event_info, station
                        )
                        
                        if splitting_result['success']:
                            event_splitting_results[station] = splitting_result
                            splitting_success_count += 1
            
            all_results['splitting_results'][event_id] = event_splitting_results
    
    # Calculate summary statistics
    all_results['summary_stats'] = {
        'total_events': len(waveform_data),
        'processed_events': processed_count,
        'qc_passed_events': qc_passed_count,
        'splitting_success_count': splitting_success_count,
        'qc_pass_rate': qc_passed_count / processed_count if processed_count > 0 else 0,
        'splitting_success_rate': splitting_success_count / qc_passed_count if qc_passed_count > 0 else 0
    }
    
    return all_results

    
def compile_results_dataframe(splitting_results, extended_catalog):
    """
    Compile splitting results into a structured DataFrame for analysis.
    
    Parameters:
    -----------
    splitting_results : dict
        Splitting analysis results by event containing 'result' and 'splitting_obj'
    extended_catalog : pandas.DataFrame
        Extended catalog with event metadata
    
    Returns:
    --------
    pandas.DataFrame
        Compiled results with splitting parameters and metadata
    """
    
    results_list = []
    
    for event_id, event_result in splitting_results.items():
        # Extract the result dictionary (not 'splitting_obj')
        split_data = event_result.get('result')
        
        if split_data is None or not split_data.get('success', False):
            continue
        
        # Get event metadata from catalog
        event_row = extended_catalog[extended_catalog['id'] == event_id]
        if event_row.empty:
            # Try using the metadata stored in split_data
            event_lat = split_data.get('event_lat', np.nan)
            event_lon = split_data.get('event_lon', np.nan)
            event_depth = split_data.get('event_depth', np.nan)
        else:
            event_meta = event_row.iloc[0]
            event_lat = event_meta.get('lat', np.nan)
            event_lon = event_meta.get('lon', np.nan)
            event_depth = event_meta.get('dep', np.nan)
        
        # Compile result entry
        result_entry = {
            'event_id': event_id,
            'station': split_data.get('station', 'UNKNOWN'),
            'phi': split_data.get('phi', np.nan),
            'dt': split_data.get('dt', np.nan),
            'phi_error': split_data.get('phi_error', np.nan),
            'dt_error': split_data.get('dt_error', np.nan),
            'avg_snr': split_data.get('snr_avg', split_data.get('snr_horizontal', np.nan)),
            'magnitude': split_data.get('magnitude', np.nan),
            'event_lat': event_lat,
            'event_lon': event_lon,
            'event_depth': event_depth,
            'window_duration': split_data.get('window_duration', np.nan),
            'quality': split_data.get('quality', 'unknown'),
            'back_azimuth': split_data.get('back_azimuth', np.nan),
            'rectilinearity': split_data.get('rectilinearity', np.nan),
            'incidence': split_data.get('incidence', np.nan)
        }
        
        results_list.append(result_entry)
    
    if results_list:
        return pd.DataFrame(results_list)
    else:
        print("Warning: No successful splitting results found to compile")
        return pd.DataFrame()


def create_splitting_plots(results_df):
    """
    Create comprehensive plots of splitting analysis results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Compiled splitting results
    """
    
    if results_df.empty:
        print("No results available for plotting")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Fast axis directions (phi) histogram
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(results_df['phi'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Fast Axis Direction φ (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fast Axis Directions')
    plt.axvline(results_df['phi'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["phi"].mean():.1f}°')
    plt.legend()
    
    # Plot 2: Delay times (dt) histogram  
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(results_df['dt'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Delay Time δt (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delay Times')
    plt.axvline(results_df['dt'].mean(), color='red', linestyle='--',
                label=f'Mean: {results_df["dt"].mean():.3f}s')
    plt.legend()
    
    # Plot 3: Phi vs Magnitude
    ax3 = plt.subplot(2, 3, 3)
    scatter = plt.scatter(results_df['magnitude'], results_df['phi'], 
                         c=results_df['dt'], cmap='viridis', alpha=0.6)
    plt.xlabel('Magnitude')
    plt.ylabel('Fast Axis Direction φ (degrees)')
    plt.title('φ vs Magnitude (colored by δt)')
    plt.colorbar(scatter, label='δt (seconds)')
    
    # Plot 4: Delay time vs Magnitude
    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(results_df['magnitude'], results_df['dt'], alpha=0.6, color='orange')
    plt.xlabel('Magnitude')
    plt.ylabel('Delay Time δt (seconds)')
    plt.title('Delay Time vs Magnitude')
    
    # Plot 5: Station-wise splitting parameters
    ax5 = plt.subplot(2, 3, 5)
    station_means = results_df.groupby('station').agg({
        'phi': 'mean',
        'dt': 'mean'
    }).reset_index()
    
    plt.scatter(station_means['phi'], station_means['dt'], s=100, alpha=0.7)
    for idx, row in station_means.iterrows():
        plt.annotate(row['station'], (row['phi'], row['dt']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Mean Fast Axis Direction φ (degrees)')
    plt.ylabel('Mean Delay Time δt (seconds)')
    plt.title('Station-Average Splitting Parameters')
    
    # Plot 6: Quality metrics
    ax6 = plt.subplot(2, 3, 6)
    plt.scatter(results_df['avg_snr'], results_df['dt'], alpha=0.6, color='green')
    plt.xlabel('Average SNR')
    plt.ylabel('Delay Time δt (seconds)')
    plt.title('Delay Time vs Signal Quality')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics table
    print(f"\n{'='*60}")
    print("SPLITTING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total measurements: {len(results_df)}")
    print(f"Unique stations: {results_df['station'].nunique()}")
    print(f"Unique events: {results_df['event_id'].nunique()}")
    print(f"\nFast Axis Direction (φ):")
    print(f"  Mean: {results_df['phi'].mean():.1f}° ± {results_df['phi'].std():.1f}°")
    print(f"  Range: {results_df['phi'].min():.1f}° to {results_df['phi'].max():.1f}°")
    print(f"\nDelay Time (δt):")
    print(f"  Mean: {results_df['dt'].mean():.3f} ± {results_df['dt'].std():.3f} seconds")
    print(f"  Range: {results_df['dt'].min():.3f} to {results_df['dt'].max():.3f} seconds")
    print(f"\nQuality Metrics:")
    print(f"  Average SNR: {results_df['avg_snr'].mean():.1f}")
    print(f"  Magnitude range: {results_df['magnitude'].min():.1f} to {results_df['magnitude'].max():.1f}")


def create_analysis_documentation(results, qc_thresholds, results_df):
    """
    Create comprehensive documentation of the analysis workflow and parameters.
    
    Parameters:
    -----------
    results : dict
        Complete analysis results
    qc_thresholds : dict
        Quality control parameters used
    results_df : pandas.DataFrame
        Compiled splitting results
    
    Returns:
    --------
    dict
        Documentation dictionary
    """
    
    doc = {
        'analysis_info': {
            'workflow': 'Master Shear-Wave Splitting Analysis',
            'date_created': datetime.now().isoformat(),
            'software_versions': {
                'python': '3.x',
                'obspy': 'latest',
                'swspy': 'latest'
            }
        },
        'parameters': {
            'quality_control': qc_thresholds,
            'time_windows': {
                'pre_event_time': 4.0,
                'post_event_time': 15.0,
                'p_wave_analysis_window': 1.0,
                's_wave_base_window': 2.0
            },
            'splitting_analysis': {
                'phi_range': '(-90, 95) degrees in 5° steps',
                'dt_range': '(0.0, 0.25) seconds in 0.01s steps',
                'window_scaling': 'Dynamic based on magnitude'
            }
        },
        'statistics': results.get('summary_stats', {}),
        'file_locations': {
            'input_catalog': '../data/final_catalog.csv',
            'trace_data': '../scripts/all_earthquakes_trace_data.pkl',
            'output_results': '../results/master_splitting_results.csv'
        }
    }
    
    if not results_df.empty:
        doc['result_statistics'] = {
            'phi_stats': {
                'mean': float(results_df['phi'].mean()),
                'std': float(results_df['phi'].std()),
                'min': float(results_df['phi'].min()),
                'max': float(results_df['phi'].max())
            },
            'dt_stats': {
                'mean': float(results_df['dt'].mean()),
                'std': float(results_df['dt'].std()),
                'min': float(results_df['dt'].min()),
                'max': float(results_df['dt'].max())
            },
            'measurement_counts': {
                'total_measurements': len(results_df),
                'unique_stations': int(results_df['station'].nunique()),
                'unique_events': int(results_df['event_id'].nunique())
            }
        }
    
    return doc


def export_results_multiple_formats(results, results_df, documentation):
    """
    Export results in multiple formats for different use cases.
    """
    
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. CSV export (main results)
    if not results_df.empty:
        csv_path = output_dir / 'master_splitting_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Results exported to CSV: {csv_path}")
    
    # 2. JSON export (complete results with metadata)
    json_path = output_dir / f'complete_analysis_results_{timestamp}.json'
    
    # Convert numpy types to Python native types for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == 'splitting_results':
            # Convert splitting results
            json_results[key] = {}
            for event_id, stations in value.items():
                json_results[key][event_id] = {}
                for station, data in stations.items():
                    # Convert numpy values
                    clean_data = {}
                    for k, v in data.items():
                        if isinstance(v, np.ndarray):
                            clean_data[k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            clean_data[k] = v.item()
                        else:
                            clean_data[k] = v
                    json_results[key][event_id][station] = clean_data
        else:
            json_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump({'results': json_results, 'documentation': documentation}, f, indent=2)
    print(f"Complete results exported to JSON: {json_path}")
    
    # 3. Documentation export
    doc_path = output_dir / f'analysis_documentation_{timestamp}.json'
    with open(doc_path, 'w') as f:
        json.dump(documentation, f, indent=2)
    print(f"Analysis documentation exported: {doc_path}")
    
    # 4. Summary report
    report_path = output_dir / f'analysis_summary_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("AXIAL SEAMOUNT SHEAR-WAVE SPLITTING ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis Date: {documentation['analysis_info']['date_created']}\n")
        f.write(f"Workflow: {documentation['analysis_info']['workflow']}\n\n")
        
        stats = documentation['statistics']
        f.write("PROCESSING STATISTICS:\n")
        f.write(f"Total events processed: {stats.get('processed_events', 'N/A')}\n")
        f.write(f"Events passing QC: {stats.get('qc_passed_events', 'N/A')}\n")
        f.write(f"Successful splitting analyses: {stats.get('splitting_success_count', 'N/A')}\n")
        f.write(f"QC pass rate: {stats.get('qc_pass_rate', 0):.1%}\n")
        f.write(f"Splitting success rate: {stats.get('splitting_success_rate', 0):.1%}\n\n")
        
        if 'result_statistics' in documentation:
            result_stats = documentation['result_statistics']
            f.write("SPLITTING PARAMETERS:\n")
            f.write(f"Fast axis direction (φ): {result_stats['phi_stats']['mean']:.1f}° ± {result_stats['phi_stats']['std']:.1f}°\n")
            f.write(f"Delay time (δt): {result_stats['dt_stats']['mean']:.3f} ± {result_stats['dt_stats']['std']:.3f} seconds\n")
            f.write(f"Total measurements: {result_stats['measurement_counts']['total_measurements']}\n")
            f.write(f"Stations analyzed: {result_stats['measurement_counts']['unique_stations']}\n")
    
    print(f"Summary report exported: {report_path}")


def parse_phase_file(filename):
    """
    Parse the phase file format used at Axial Seamount:
    Event lines: # YYYY MM DD HH MM SS.ss LAT LON DEPTH MAG ... EVENT_ID E
    Phase lines: STATION ARRIVAL_TIME WEIGHT PHASE_TYPE QUALITY
    
    Returns:
    --------
    events_df : pandas.DataFrame
        DataFrame with earthquake event information
    phases_df : pandas.DataFrame  
        DataFrame with phase picks for each event
    """
    
    events = []
    phases = []
    current_event_id = None
    current_event_info = None
    
    print(f"Parsing phase file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines...")
    
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        parts = stripped.split()
        
        # Check if this is an event line (starts with #)
        if stripped.startswith('#') and len(parts) >= 10:
            try:
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = float(parts[6])
                lat = float(parts[7])
                lon = float(parts[8])
                depth = float(parts[9])
                
                # Find the event ID (second to last element, before 'E')
                event_id = parts[-2] if len(parts) >= 2 else str(len(events))
                
                # Create UTC datetime string
                datetime_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}Z"
                
                event_info = {
                    'event_id': event_id,
                    'year': year,
                    'month': month,
                    'day': day, 
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'datetime_str': datetime_str,
                    'lat': lat,
                    'lon': lon,
                    'depth': depth
                }
                
                # Add magnitude if available (typically at index 10)
                if len(parts) > 10:
                    try:
                        event_info['magnitude'] = float(parts[10])
                    except ValueError:
                        event_info['magnitude'] = np.nan
                else:
                    event_info['magnitude'] = np.nan
                
                events.append(event_info)
                current_event_info = event_info
                current_event_id = event_id
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse event line {line_num + 1}: {stripped}")
                continue
                
        else:
            # This should be a phase pick line if we have a current event
            if current_event_id is not None and current_event_info is not None and len(parts) >= 4:
                try:
                    phase_info = {
                        'event_id': current_event_id,
                        'event_datetime': current_event_info['datetime_str'],
                        'event_lat': current_event_info['lat'],
                        'event_lon': current_event_info['lon'],
                        'event_depth': current_event_info['depth'],
                        'event_magnitude': current_event_info['magnitude'],
                        'station': parts[0],
                        'arrival_time': float(parts[1]),
                        'weight': float(parts[2]) if parts[2] != '-1.000' else np.nan,
                        'phase_type': parts[3],
                        'quality': parts[4] if len(parts) > 4 else ''
                    }
                    
                    phases.append(phase_info)
                    
                except (ValueError, IndexError) as e:
                    # Skip problematic phase lines
                    continue
    
    # Convert to DataFrames
    events_df = pd.DataFrame(events)
    phases_df = pd.DataFrame(phases)
    
    # Convert event_id to numeric if possible
    if len(events_df) > 0:
        try:
            events_df['event_id'] = pd.to_numeric(events_df['event_id'])
            phases_df['event_id'] = pd.to_numeric(phases_df['event_id'])
        except:
            pass  # Keep as string if conversion fails
    
    return events_df, phases_df

def create_organized_catalog(events_df, phases_df):
    """
    Create organized catalog where each row represents one event at one station
    with both P- and S-wave picks and arrival times.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        Event information from phase file
    phases_df : pandas.DataFrame
        Phase picks from phase file
    
    Returns:
    --------
    pandas.DataFrame
        Organized catalog with events × stations structure
    """
    
    print("Creating organized catalog with events × stations structure...")
    
    # Get all unique stations and events
    all_stations = phases_df['station'].unique()
    all_events = events_df['event_id'].unique()
    
    print(f"Events: {len(all_events)}, Stations: {len(all_stations)}")
    print(f"Stations in network: {sorted(all_stations)}")
    
    final_rows = []
    
    for event_id in all_events:
        # Get event metadata
        event_info = events_df[events_df['event_id'] == event_id].iloc[0]
        
        # Get all phase picks for this event
        event_phases = phases_df[phases_df['event_id'] == event_id]
        
        # Group phase picks by station
        for station in all_stations:
            station_phases = event_phases[event_phases['station'] == station]
            
            # Initialize row with event information
            row = {
                'id': event_id,
                'year': event_info['year'],
                'datetime': event_info['datetime_str'],
                'lat': event_info['lat'],
                'lon': event_info['lon'],
                'dep': event_info['depth'],  # Use 'dep' to match old catalog format
                'mag': event_info['magnitude'],
                'station': station,
                'total_picks': len(station_phases),
                'p_arrival_time': np.nan,
                'p_weight': np.nan,
                'p_quality': '',
                's_arrival_time': np.nan,
                's_weight': np.nan,
                's_quality': ''
            }
            
            # Fill in phase pick information if available
            if len(station_phases) > 0:
                # P-wave information
                p_phases = station_phases[station_phases['phase_type'] == 'P']
                if len(p_phases) > 0:
                    p_pick = p_phases.iloc[0]  # Take first P pick if multiple
                    row['p_arrival_time'] = p_pick['arrival_time']
                    row['p_weight'] = p_pick['weight']
                    row['p_quality'] = p_pick['quality']
                
                # S-wave information
                s_phases = station_phases[station_phases['phase_type'] == 'S']
                if len(s_phases) > 0:
                    s_pick = s_phases.iloc[0]  # Take first S pick if multiple
                    row['s_arrival_time'] = s_pick['arrival_time']
                    row['s_weight'] = s_pick['weight']
                    row['s_quality'] = s_pick['quality']
            
            final_rows.append(row)
    
    catalog_df = pd.DataFrame(final_rows)
    
    # Filter to only keep rows with both P and S picks
    has_both_picks = (~catalog_df['p_arrival_time'].isna()) & (~catalog_df['s_arrival_time'].isna())
    catalog_filtered = catalog_df[has_both_picks].copy()
    
    print(f"Total event-station pairs: {len(catalog_df)}")
    print(f"Pairs with both P and S picks: {len(catalog_filtered)}")

def organize_stream_by_events(stream, catalog, time_tolerance=5.0):
    """
    Organize the stream data by matching traces to catalog events.
    
    Parameters:
    -----------
    stream : obspy.Stream
        The full stream containing all waveform data
    catalog : pandas.DataFrame
        Catalog of events with datetime and station information
    time_tolerance : float
        Tolerance in seconds for matching trace start times to event times
        
    Returns:
    --------
    dict
        Dictionary mapping event indices to their corresponding stream data
    """
    event_streams = {}
    
    print(f"Organizing stream data for {len(catalog)} events...")
    
    for i, (idx, event) in enumerate(catalog.iterrows()):
        event_time = UTCDateTime(event['datetime'])
        station_id = event['station']
        
        # Find traces that match this event's time and station
        matching_traces = []
        
        for tr in stream.select(station=str(station_id)):
            # Check if trace start time is close to event time
            time_diff = abs(tr.stats.starttime - event_time)
            
            if time_diff <= time_tolerance:
                matching_traces.append(tr.copy())
        
        if len(matching_traces) >= 3:  # Need at least Z, N, E components
            event_stream = obspy.Stream(matching_traces)
            event_streams[idx] = event_stream
            print(f"  Event {i+1}: Found {len(matching_traces)} traces for {station_id} at {event_time}")
        else:
            print(f"  Event {i+1}: Only {len(matching_traces)} traces found for {station_id} at {event_time}")
            event_streams[idx] = None
    
    return event_streams
    
def compute_snr_for_event(event_stream, event_row, component='E'):
    """
    Calculate SNR for a specific component using S-wave windows.
    
    Parameters:
    -----------
    event_stream : obspy.Stream or list
        Stream or list of traces for the event
    event_row : dict or pandas.Series
        Event metadata including S and P arrival times
    component : str
        Component to calculate SNR for ('E', 'N', or 'Z')
        
    Returns:
    --------
    float
        Signal-to-noise ratio for S-wave window
    """
    try:
        # Convert to stream if it's a list
        if isinstance(event_stream, list):
            event_stream = obspy.Stream(event_stream)
        
        # Check if S-arrival time exists
        if 's_arrival_time' not in event_row or pd.isna(event_row['s_arrival_time']):
            print(f"  No S-arrival time for {component}")
            return np.nan
        
        event_time = UTCDateTime(event_row['datetime'])
        s_pick = event_time + float(event_row['s_arrival_time'])
        print(f"  S-pick time: {s_pick}")
        
        # Get S-P delay for adaptive noise window sizing
        sp_delay = None
        if 'p_arrival_time' in event_row and not pd.isna(event_row['p_arrival_time']):
            sp_delay = event_row['s_arrival_time'] - event_row['p_arrival_time']
            print(f"  S-P delay: {sp_delay:.2f}s")
        
        # Find component trace
        trace = None
        for tr in event_stream:
            if tr.stats.channel[-1].upper() == component.upper():
                trace = tr.copy()
                break
        
        if trace is None:
            print(f"  No {component} component trace found")
            return np.nan
        
        print(f"  Found {component} trace: {trace.stats.starttime} to {trace.stats.endtime}")
        
        # Apply filtering
        trace.filter("bandpass", freqmin=5, freqmax=40)
        trace.taper(type="hann", max_percentage=0.05)
        
        # Define adaptive time windows
        noise_window = min(0.4, sp_delay / 2) if sp_delay else 0.4
        signal_window = 0.2
        
        noise_start = s_pick - noise_window  # BEFORE S-arrival
        noise_end = s_pick                   # UP TO S-arrival
        signal_start = s_pick                # FROM S-arrival
        signal_end = s_pick + signal_window  # AFTER S-arrival
        
        print(f"  Noise window: {noise_start} to {noise_end} ({noise_window:.2f}s)")
        print(f"  Signal window: {signal_start} to {signal_end} ({signal_window:.2f}s)")
        
        # Check trace coverage
        if (trace.stats.starttime > noise_start or trace.stats.endtime < signal_end):
            print(f"  Trace doesn't cover required windows")
            print(f"  Trace: {trace.stats.starttime} to {trace.stats.endtime}")
            return np.nan
        
        # Extract data for noise and signal windows
        noise_data = trace.slice(noise_start, noise_end).data
        signal_data = trace.slice(signal_start, signal_end).data
        
        print(f"  Noise data points: {len(noise_data)}")
        print(f"  Signal data points: {len(signal_data)}")
        
        if len(noise_data) == 0 or len(signal_data) == 0:
            print(f"  Empty noise or signal window")
            return np.nan
        
        # Calculate RMS amplitudes
        noise_rms = np.sqrt(np.mean(noise_data**2))
        signal_rms = np.sqrt(np.mean(signal_data**2))
        
        print(f"  Noise RMS: {noise_rms:.6f}")
        print(f"  Signal RMS: {signal_rms:.6f}")
        
        if noise_rms == 0:
            print(f"  Zero noise level")
            return np.nan
        
        snr = signal_rms / noise_rms
        print(f"  SNR: {snr:.2f}")
        return float(snr)
        
    except Exception as e:
        print(f"  Error calculating SNR for {component}: {e}")
        return np.nan

def calculate_snr_for_organized_waveforms(organized_waveforms):
    """
    Calculate SNR for all events in organized_waveforms and add to the dataset.
    
    All required metadata (S/P arrival times, datetime) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
        
    Returns:
    --------
    dict
        Updated organized_waveforms with SNR values added to each event
    """
    
    print(f"Calculating SNR for {len(organized_waveforms)} events in organized_waveforms...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\n{'='*60}")
        print(f"Processing event {event_id}...")
        print(f"{'='*60}")
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['snr_e'] = np.nan
            event_data['snr_n'] = np.nan
            event_data['snr_horizontal'] = np.nan
            continue
        
        print(f"  Found {len(event_traces)} traces for this event")
        
        # Calculate SNR for horizontal components using event_data directly
        print("\n  Calculating SNR for E component:")
        snr_e = compute_snr_for_event(event_traces, event_data, component='E')
        
        print("\n  Calculating SNR for N component:")
        snr_n = compute_snr_for_event(event_traces, event_data, component='N')
        
        # Calculate horizontal average
        if not np.isnan(snr_e) and not np.isnan(snr_n):
            snr_horizontal = (snr_e + snr_n) / 2
        elif not np.isnan(snr_e):
            snr_horizontal = snr_e
        elif not np.isnan(snr_n):
            snr_horizontal = snr_n
        else:
            snr_horizontal = np.nan
        
        # Add SNR values to event_data
        event_data['snr_e'] = snr_e
        event_data['snr_n'] = snr_n
        event_data['snr_horizontal'] = snr_horizontal
        
        # Print summary
        print(f"\n  Final SNR Results:")
        if not np.isnan(snr_e):
            print(f"    E-component: {snr_e:.2f}")
        else:
            print("    E-component: N/A")

        if not np.isnan(snr_n):
            print(f"    N-component: {snr_n:.2f}")
        else:
            print("    N-component: N/A")

        if not np.isnan(snr_horizontal):
            print(f"    Horizontal average: {snr_horizontal:.2f}")
        else:
            print("    Horizontal average: N/A")

        if not np.isnan(snr_horizontal):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("SNR Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid SNR: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    snr_values = [data.get('snr_horizontal', np.nan) for data in organized_waveforms.values()]
    valid_snr = [v for v in snr_values if not np.isnan(v)]
    
    if valid_snr:
        print(f"SNR range: {min(valid_snr):.2f} to {max(valid_snr):.2f}")
        print(f"Mean SNR: {np.mean(valid_snr):.2f}")
        print(f"Median SNR: {np.median(valid_snr):.2f}")
    
    return organized_waveforms

def calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon):
    """
    Calculate back-azimuth from station to earthquake (0-360°)
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    sta_lat, sta_lon : float
        Station latitude and longitude in degrees
        
    Returns:
    --------
    float
        Back-azimuth in degrees (0-360°)
    """
    # Convert to radians
    eq_lat_rad = np.radians(eq_lat)
    eq_lon_rad = np.radians(eq_lon)
    sta_lat_rad = np.radians(sta_lat)
    sta_lon_rad = np.radians(sta_lon)
    
    # Calculate back-azimuth
    dlon = eq_lon_rad - sta_lon_rad
    y = np.sin(dlon) * np.cos(eq_lat_rad)
    x = (np.cos(sta_lat_rad) * np.sin(eq_lat_rad) - 
         np.sin(sta_lat_rad) * np.cos(eq_lat_rad) * np.cos(dlon))
    
    back_az = np.degrees(np.arctan2(y, x))
    
    # Normalize to 0-360°
    back_az = (back_az + 360) % 360
    
    return back_az


def calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df):
    """
    Calculate back-azimuth for all events in organized_waveforms and add to the dataset.
    
    All required event metadata (lat, lon, station) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    stations_df : pandas.DataFrame
        Station information with coordinates (station, latitude, longitude)
        
    Returns:
    --------
    dict
        Updated organized_waveforms with back-azimuth values added to each event
    """
    
    print(f"Calculating back-azimuth for {len(organized_waveforms)} events...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get event location from event_data
        event_lat = event_data.get('latitude')
        event_lon = event_data.get('longitude')
        
        if event_lat is None or event_lon is None:
            print(f"  No event location found in event_data")
            event_data['back_azimuth'] = np.nan
            continue
        
        # Get station name from event data
        station_name = event_data.get('station')
        
        if station_name is None:
            print(f"  No station information for event {event_id}")
            event_data['back_azimuth'] = np.nan
            continue
        
        # Find station coordinates
        station_info = stations_df[stations_df['Station ID'] == station_name]
        
        if station_info.empty:
            print(f"  No coordinates found for station {station_name}")
            event_data['back_azimuth'] = np.nan
            continue
        
        station_lat = station_info.iloc[0]['Latitude (°N)']
        station_lon = station_info.iloc[0]['Longitude (°W)']
        
        # Calculate back-azimuth (from station to event)
        back_az = calculate_back_azimuth(event_lat, event_lon, station_lat, station_lon)
        
        # Add to event_data
        event_data['back_azimuth'] = back_az
        
        print(f"  Station: {station_name}")
        print(f"  Event location: ({event_lat:.4f}, {event_lon:.4f})")
        print(f"  Station location: ({station_lat:.4f}, {station_lon:.4f})")
        print(f"  Back-azimuth: {back_az:.2f}°")
        
        success_count += 1
    
    print(f"\n{'='*60}")
    print("Back-Azimuth Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid back-azimuth: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    back_az_values = [data.get('back_azimuth', np.nan) for data in organized_waveforms.values()]
    valid_back_az = [v for v in back_az_values if not np.isnan(v)]
    
    if valid_back_az:
        print(f"Back-azimuth range: {min(valid_back_az):.2f}° to {max(valid_back_az):.2f}°")
        print(f"Mean back-azimuth: {np.mean(valid_back_az):.2f}°")
    
    return organized_waveforms

import numpy as np
from obspy.signal.spectral_estimation import PPSD

# Check for optional mtspec library
try:
    from mtspec import mtspec
    MTSPEC_AVAILABLE = True
except ImportError:
    MTSPEC_AVAILABLE = False
    print("mtspec not available, using ObsPy PPSD method")

def estimate_dominant_period(trace, method='obspy'):
    """
    Estimate the dominant period of a seismic trace using spectral analysis.
    
    Parameters:
    -----------
    trace : obspy.Trace
        Input seismic trace
    method : str
        'obspy' for ObsPy PPSD or 'mtspec' for multitaper method
        
    Returns:
    --------
    float
        Dominant period in seconds
    """
    if trace is None or len(trace.data) == 0:
        return np.nan
    
    # Preprocess trace
    trace_copy = trace.copy()
    trace_copy.detrend("linear")
    trace_copy.taper(type="hann", max_percentage=0.05)
    
    try:
        if method == 'mtspec' and MTSPEC_AVAILABLE:
            # Multitaper spectral estimation
            spec, freq = mtspec(
                data=trace_copy.data, 
                delta=trace_copy.stats.delta, 
                time_bandwidth=4, 
                number_of_tapers=7
            )
            dominant_freq = freq[np.argmax(spec)]
        else:
            # ObsPy PPSD method (fallback)
            df = trace_copy.stats.sampling_rate
            nfft = 2 ** int(np.ceil(np.log2(len(trace_copy.data))))
            freq, power = PPSD(trace_copy.data, NFFT=nfft, Fs=df, method="multitaper")
            dominant_freq = freq[np.argmax(power)]
        
        # Convert to dominant period
        t_dom = 1.0 / dominant_freq if dominant_freq > 0 else np.nan
        return t_dom
    
    except Exception as e:
        print(f"Error estimating dominant period: {e}")
        return np.nan

def calculate_dynamic_parameters(event_data, s_arrival_buffer=1.0):
    """
    Calculate dynamic analysis parameters for an event using spectral analysis.
    
    All required data (traces, datetime, s_arrival_time) is in the event_data from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing traces and metadata
    s_arrival_buffer : float
        Buffer around S-arrival for analysis window (seconds)
        
    Returns:
    --------
    dict
        Dictionary containing dynamic parameters
    """
    try:
        # Get traces from event_data
        event_traces = event_data.get('traces', [])
        if not event_traces:
            raise ValueError("No traces in event_data")
        
        # Convert to stream if needed
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        # Get S-arrival time
        event_time = UTCDateTime(event_data['datetime'])
        s_pick = event_time + float(event_data['s_arrival_time'])
        
        # Extract traces around S-arrival for spectral analysis
        analysis_start = s_pick - s_arrival_buffer
        analysis_end = s_pick + s_arrival_buffer
        
        # Get horizontal components for analysis
        trace_e = None
        trace_n = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component in ['E', '2']:
                trace_e = tr.slice(analysis_start, analysis_end)
            elif component in ['N', '1']:
                trace_n = tr.slice(analysis_start, analysis_end)
        
        # Calculate dominant periods for both horizontal components
        t_dom_e = estimate_dominant_period(trace_e) if trace_e and len(trace_e.data) > 10 else np.nan
        t_dom_n = estimate_dominant_period(trace_n) if trace_n and len(trace_n.data) > 10 else np.nan
        
        # Use average or the valid one
        if not np.isnan(t_dom_e) and not np.isnan(t_dom_n):
            t_dom = (t_dom_e + t_dom_n) / 2
        elif not np.isnan(t_dom_e):
            t_dom = t_dom_e
        elif not np.isnan(t_dom_n):
            t_dom = t_dom_n
        else:
            t_dom = 0.1  # Default fallback (10 Hz dominant frequency)
        
        # Calculate dynamic window parameters
        # Window should be several dominant periods long, but within reasonable bounds
        dynamic_window_length = max(0.5, min(2.0, 4 * t_dom))  # 4 periods, but between 0.5-2.0s
        
        # Calculate optimal filter frequencies
        # Center frequency around 1/T_dom, with reasonable bandwidth
        center_freq = 1.0 / t_dom
        freq_min = max(1.0, center_freq * 0.5)  # Lower bound at least 1 Hz
        freq_max = min(50.0, center_freq * 2.5)  # Upper bound at most 50 Hz
        
        return {
            't_dom': t_dom,
            't_dom_e': t_dom_e,
            't_dom_n': t_dom_n,
            'dynamic_window_length': dynamic_window_length,
            'optimal_freq_min': freq_min,
            'optimal_freq_max': freq_max,
            'center_frequency': center_freq
        }
        
    except Exception as e:
        print(f"Error calculating dynamic parameters: {e}")
        return {
            't_dom': 0.1,
            'dynamic_window_length': 1.0,
            'optimal_freq_min': 5.0,
            'optimal_freq_max': 40.0,
            'center_frequency': 10.0
        }
    
def create_splitting_analysis(event_data, use_dynamic_params=True):
    """
    Create a SWSPy splitting object from organized_waveforms event data.
    
    All required data (traces, station, back_azimuth, incidence, s_arrival_time) is 
    contained in event_data from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with waveform data
        - station: Station name/ID
        - back_azimuth: Back-azimuth from event to station (degrees)
        - incidence: P-wave incidence angle (degrees from vertical)
        - s_arrival_time: S-wave arrival time (seconds from origin)
        - datetime: Event origin time (UTCDateTime compatible string)
    use_dynamic_params : bool, optional
        Whether to calculate and use dynamic windowing/filtering parameters
        based on spectral analysis (default=True)
        
    Returns:
    --------
    swspy.splitting object
        Splitting object ready for analysis with optimized parameters
    """
    
    # Get traces from event_data
    event_traces = event_data.get('traces', [])
    if not event_traces:
        raise ValueError("No traces in event_data")
    
    # Convert to stream if needed
    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()
    
    # Calculate dynamic parameters if requested
    if use_dynamic_params:
        print(f"  Calculating dynamic parameters...")
        dynamic_params = calculate_dynamic_parameters(event_data)
        print(f"    Dominant period: {dynamic_params['t_dom']:.3f} s")
        print(f"    Dynamic window length: {dynamic_params['dynamic_window_length']:.3f} s")
        print(f"    Optimal frequency range: {dynamic_params['optimal_freq_min']:.1f}-{dynamic_params['optimal_freq_max']:.1f} Hz")
        
        freq_min = dynamic_params['optimal_freq_min']
        freq_max = dynamic_params['optimal_freq_max']
        window_length = dynamic_params['dynamic_window_length']
        t_dom = dynamic_params['t_dom']
    else:
        # Use default fixed parameters
        freq_min = 5.0
        freq_max = 40.0
        window_length = 1.0
        t_dom = 0.1
        dynamic_params = None
    
    # Apply optimal filtering to the stream
    print(f"  Applying bandpass filter: {freq_min:.1f}-{freq_max:.1f} Hz")
    stream_filtered = stream.copy()
    stream_filtered.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    
    # Extract required metadata from event_data
    station_name = event_data.get('station', 'UNKNOWN')
    back_azimuth = event_data.get('back_azimuth')
    incidence_angle = event_data.get('incidence')
    
    # Validate required fields
    if back_azimuth is None or np.isnan(back_azimuth):
        raise ValueError(f"Missing or invalid back_azimuth for station {station_name}")
    if incidence_angle is None or np.isnan(incidence_angle):
        raise ValueError(f"Missing or invalid incidence angle for station {station_name}")
    
    # Calculate S-arrival absolute time
    event_time = UTCDateTime(event_data['datetime'])
    s_arrival_time = float(event_data['s_arrival_time'])
    s_arrival_absolute = event_time + s_arrival_time
    
    print(f"  Creating SWSPy splitting object...")
    print(f"    Station: {station_name}")
    print(f"    Back-azimuth: {back_azimuth:.2f}°")
    print(f"    Incidence: {incidence_angle:.2f}°")
    print(f"    S-arrival: {s_arrival_absolute}")
    
    # Create splitting object with SWSPy using exact pattern from user
    splitting_event = swspy.splitting.create_splitting_object(
        stream_filtered, 
        stations_in=[station_name],
        back_azis_all_stations=[back_azimuth],
        receiver_inc_angles_all_stations=[incidence_angle],
        S_phase_arrival_times=[s_arrival_absolute]
    )
    
    # Set dynamic analysis parameters on the splitting object
    if use_dynamic_params:
        # Dynamic windowing based on dominant period
        window_half_length = window_length / 2.0
        
        splitting_event.overall_win_start_pre_fast_S_pick = window_half_length
        splitting_event.win_S_pick_tolerance = min(0.1, t_dom / 4.0)  # Quarter period tolerance
        splitting_event.overall_win_start_post_fast_S_pick = window_half_length
        splitting_event.rotate_step_deg = 1.0
        splitting_event.max_t_shift_s = min(0.2, t_dom)  # Max shift = dominant period, capped at 0.2s
        splitting_event.n_win = max(5, int(20 / t_dom))  # More windows for higher frequency signals
        
        print(f"  Dynamic analysis parameters set:")
        print(f"    Window pre/post S-pick: {window_half_length:.3f}s")
        print(f"    Max time shift: {splitting_event.max_t_shift_s:.3f}s")
        print(f"    Number of windows: {splitting_event.n_win}")
    else:
        # Use default fixed parameters
        splitting_event.overall_win_start_pre_fast_S_pick = 0.3
        splitting_event.win_S_pick_tolerance = 0.1
        splitting_event.overall_win_start_post_fast_S_pick = 0.2
        splitting_event.rotate_step_deg = 1.0
        splitting_event.max_t_shift_s = 0.1
        splitting_event.n_win = 10
    
    # Store dynamic parameters in the splitting object for reference
    if dynamic_params is not None:
        splitting_event.dynamic_params = dynamic_params
    
    return splitting_event

def plot_three_component_seismogram(event_data, event_id=None, figsize=(12, 8), 
                                   time_window=(0, 5), filter_data=True, 
                                   lowfreq=5.0, highfreq=40.0):
    """
    Plot 3-component seismogram with P- and S-wave arrival markers for an event from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with Z/N/E components
        - station: Station name
        - p_arrival_time: P-wave arrival time (seconds from origin)
        - s_arrival_time: S-wave arrival time (seconds from origin)
        - magnitude: Event magnitude (optional)
        - datetime: Event datetime (optional)
    event_id : str or int, optional
        Event ID for plot title (extracted from event_data if not provided)
    figsize : tuple, optional
        Figure size (width, height) in inches, default=(12, 8)
    time_window : tuple, optional
        Time window to display (start, end) in seconds, default=(0, 5)
    filter_data : bool, optional
        Whether to apply bandpass filter, default=True
    lowfreq : float, optional
        Low-frequency corner for bandpass filter (Hz), default=5.0
    highfreq : float, optional
        High-frequency corner for bandpass filter (Hz), default=40.0
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    
    # Get traces from event_data
    event_traces = event_data.get('traces', [])
    if not event_traces:
        print("No traces found in event_data")
        return None
    
    # Convert to stream if needed
    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()
    
    # Apply filter if requested
    if filter_data:
        stream.filter("bandpass", freqmin=lowfreq, freqmax=highfreq)
    
    # Get event metadata
    station = event_data.get('station', 'UNKNOWN')
    if event_id is None:
        event_id = event_data.get('event_id', 'UNKNOWN')
    
    p_arrival_time = event_data.get('p_arrival_time', np.nan)
    s_arrival_time = event_data.get('s_arrival_time', np.nan)
    magnitude = event_data.get('magnitude', np.nan)
    datetime_str = event_data.get('datetime', '')
    
    # Setup plot
    fig = plt.figure(figsize=figsize)
    
    components = {'Z': 'Vertical', 'N': 'North-South', 'E': 'East-West'}
    colors = {'Z': 'black', 'N': 'black', 'E': 'black'}
    
    # Plot each component
    for i, comp in enumerate(components.keys(), 1):
        # Find the component (handle both N/E and 1/2 naming)
        comp_trace = None
        for tr in stream:
            channel_comp = tr.stats.channel[-1].upper()
            if comp == 'Z' and channel_comp == 'Z':
                comp_trace = tr
                break
            elif comp == 'N' and channel_comp in ['N', '1']:
                comp_trace = tr
                break
            elif comp == 'E' and channel_comp in ['E', '2']:
                comp_trace = tr
                break
        
        plt.subplot(3, 1, i)
        
        if comp_trace is not None:
            # Plot the trace
            times = comp_trace.times()
            plt.plot(times, comp_trace.data, color=colors.get(comp, 'black'), linewidth=1)
            
            # Set time limits
            plt.xlim(time_window[0], time_window[1])
            
            # Add P-arrival marker if available
            if not np.isnan(p_arrival_time):
                if time_window[0] <= p_arrival_time <= time_window[1]:
                    plt.axvline(x=p_arrival_time, color='blue', linestyle='--', 
                               linewidth=1.5, label='P-arrival', alpha=0.7)
                    # Position text slightly above the data
                    y_pos = comp_trace.data.max() * 0.85
                    plt.text(p_arrival_time, y_pos, 'P', color='blue', 
                            fontsize=12, fontweight='bold', ha='center')
            
            # Add S-arrival marker if available
            if not np.isnan(s_arrival_time):
                if time_window[0] <= s_arrival_time <= time_window[1]:
                    plt.axvline(x=s_arrival_time, color='red', linestyle='--', 
                               linewidth=1.5, label='S-arrival', alpha=0.7)
                    # Position text slightly above the data
                    y_pos = comp_trace.data.max() * 0.85
                    plt.text(s_arrival_time, y_pos, 'S', color='red', 
                            fontsize=12, fontweight='bold', ha='center')
            
            plt.title(f"{components[comp]} Component ({comp})", fontsize=11)
            plt.ylabel("Amplitude", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add legend on first subplot
            if i == 1:
                handles = []
                labels = []
                if not np.isnan(p_arrival_time):
                    handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5))
                    labels.append('P-arrival')
                if not np.isnan(s_arrival_time):
                    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
                    labels.append('S-arrival')
                if handles:
                    plt.legend(handles, labels, loc='upper right', fontsize=9)
        else:
            plt.xlim(time_window[0], time_window[1])
            plt.text(0.5, 0.5, f"No {components[comp]} component data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=10)
            plt.ylabel("Amplitude", fontsize=10)
            plt.grid(True, alpha=0.3)
        
        # Only add x-label to bottom plot
        if i == 3:
            plt.xlabel("Time (s)", fontsize=10)
    
    # Create title with event information
    title_parts = [f"Event {event_id} at Station {station}"]
    if not np.isnan(magnitude):
        title_parts.append(f"M{magnitude:.1f}")
    if datetime_str:
        title_parts.append(datetime_str)
    
    title = " | ".join(title_parts)
    if filter_data:
        title += f" | Filtered {lowfreq}-{highfreq} Hz"
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    return fig


def plot_multiple_events_seismograms(organized_waveforms, event_ids=None, max_events=4, **kwargs):
    """
    Plot 3-component seismograms for multiple events from organized_waveforms.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary of events from organized_waveforms
    event_ids : list, optional
        List of specific event IDs to plot. If None, plots first max_events
    max_events : int, optional
        Maximum number of events to plot if event_ids not specified, default=4
    **kwargs : dict
        Additional keyword arguments passed to plot_three_component_seismogram()
        
    Returns:
    --------
    list
        List of figure objects created
    """
    
    # Determine which events to plot
    if event_ids is None:
        # Plot first max_events
        event_ids = list(organized_waveforms.keys())[:max_events]
    
    figures = []
    
    for event_id in event_ids:
        if event_id not in organized_waveforms:
            print(f"Warning: Event {event_id} not found in organized_waveforms")
            continue
        
        event_data = organized_waveforms[event_id]
        
        # Add event_id to event_data if not present
        if 'event_id' not in event_data:
            event_data['event_id'] = event_id
        
        print(f"\nPlotting event {event_id}...")
        fig = plot_three_component_seismogram(event_data, event_id=event_id, **kwargs)
        
        if fig is not None:
            figures.append(fig)
            plt.show()
        else:
            print(f"Failed to create plot for event {event_id}")
    
    print(f"\nCreated {len(figures)} seismogram plots")
    return figures