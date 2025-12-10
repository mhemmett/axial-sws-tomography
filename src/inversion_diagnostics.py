"""
Inversion Diagnostics and Quality Assessment Module

This module provides comprehensive diagnostic tools for evaluating global anisotropy
inversion results, including resolution analysis, checkerboard tests, uncertainty
quantification, and model validation metrics.

The diagnostics help assess:
- Spatial resolution and parameter trade-offs
- Model reliability and robustness
- Data adequacy and coverage
- Regularization parameter selection
- Synthetic recovery tests

Author: Axial Seamount Research Team
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass

from global_anisotropy_inversion import GlobalAnisotropyInverter


@dataclass
class ResolutionMetrics:
    """Container for resolution analysis results."""
    resolution_matrix: np.ndarray
    diagonal_resolution: np.ndarray
    spread_function: np.ndarray
    model_covariance: np.ndarray
    parameter_uncertainties: np.ndarray
    correlation_lengths: np.ndarray


class InversionDiagnostics:
    """
    Comprehensive diagnostics for global anisotropy inversion results.
    
    This class provides tools for assessing inversion quality, model resolution,
    and parameter reliability through various analytical and synthetic tests.
    """
    
    def __init__(self, inverter: GlobalAnisotropyInverter):
        """
        Initialize diagnostics with a configured inverter.
        
        Parameters:
        -----------
        inverter : GlobalAnisotropyInverter
            Configured inversion object with mesh and observations
        """
        self.inverter = inverter
        self.resolution_cache = {}
        
    def compute_resolution_matrix(self, alpha_damping: float, 
                                alpha_smoothing: float) -> ResolutionMetrics:
        """
        Compute full resolution matrix and associated metrics.
        
        The resolution matrix R describes how well each model parameter
        can be resolved from the data: m_est = R * m_true
        
        Parameters:
        -----------
        alpha_damping : float
            Damping regularization parameter
        alpha_smoothing : float
            Smoothing regularization parameter
            
        Returns:
        --------
        metrics : ResolutionMetrics
            Comprehensive resolution analysis results
        """
        print("Computing resolution matrix...")
        
        # Build system matrices
        G = self.inverter._build_design_matrix()
        L_damp, L_smooth = self.inverter._build_regularization_matrix()
        
        # Combined regularization matrix
        n_params = G.shape[1]
        L = csr_matrix((0, n_params))
        
        if alpha_damping > 0:
            L = L_damp * alpha_damping
        if alpha_smoothing > 0:
            if L.shape[0] == 0:
                L = L_smooth * alpha_smoothing
            else:
                from scipy.sparse import vstack
                L = vstack([L, L_smooth * alpha_smoothing])
        
        # Generalized inverse
        GTG = G.T @ G
        LTL = L.T @ L
        
        try:
            # Resolution matrix: R = (G^T G + L^T L)^(-1) G^T G
            A_inv = spsolve(GTG + LTL, np.eye(GTG.shape[0]))
            if A_inv.ndim == 1:
                A_inv = A_inv.reshape(-1, 1)
            R = A_inv @ GTG
            
        except Exception as e:
            print(f"Direct resolution computation failed: {e}")
            # Fallback: approximate using pseudoinverse
            from scipy.linalg import pinv
            G_dense = G.toarray() if hasattr(G, 'toarray') else G
            L_dense = L.toarray() if hasattr(L, 'toarray') else L
            
            # Augmented system
            G_aug = np.vstack([G_dense, L_dense])
            target_aug = np.vstack([G_dense, np.zeros_like(L_dense)])
            
            R = pinv(G_aug) @ target_aug
        
        # Convert to dense for analysis
        if hasattr(R, 'toarray'):
            R = R.toarray()
        
        # Compute metrics
        diagonal_resolution = np.diag(R)
        
        # Spread function (off-diagonal energy)
        spread_function = np.zeros(R.shape[0])
        for i in range(R.shape[0]):
            spread_function[i] = np.sum(R[i, :]**2) - R[i, i]**2
        
        # Model covariance matrix (for uncertainties)
        try:
            # C_m = (G^T G + L^T L)^(-1)
            model_covariance = A_inv
            parameter_uncertainties = np.sqrt(np.diag(model_covariance))
        except:
            # Approximate from resolution matrix
            model_covariance = np.eye(R.shape[0]) * 0.1  # Placeholder
            parameter_uncertainties = np.ones(R.shape[0]) * 0.1
        
        # Correlation lengths (spatial scale of averaging)
        correlation_lengths = self._compute_correlation_lengths(R)
        
        return ResolutionMetrics(
            resolution_matrix=R,
            diagonal_resolution=diagonal_resolution,
            spread_function=spread_function,
            model_covariance=model_covariance,
            parameter_uncertainties=parameter_uncertainties,
            correlation_lengths=correlation_lengths
        )
    
    def _compute_correlation_lengths(self, resolution_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate spatial correlation lengths from resolution matrix.
        
        This gives the characteristic length scale over which the inversion
        averages model parameters.
        """
        n_cells = len(self.inverter.mesh_centers)
        correlation_lengths = np.zeros(n_cells)
        
        for i in range(n_cells):
            # Get resolution kernel for this cell
            kernel = resolution_matrix[i, :]
            
            # Weight by resolution values
            weights = kernel**2
            
            if np.sum(weights) > 0:
                # Compute weighted average distance
                center_i = self.inverter.mesh_centers[i]
                distances = cdist([center_i], self.inverter.mesh_centers)[0]
                
                # Weighted RMS distance
                correlation_lengths[i] = np.sqrt(np.sum(weights * distances**2) / np.sum(weights))
            else:
                correlation_lengths[i] = np.nan
        
        return correlation_lengths
    
    def checkerboard_test(self, alpha_damping: float, alpha_smoothing: float,
                         checkerboard_size: float = 5.0, amplitude: float = 0.1,
                         noise_level: float = 0.0) -> Dict:
        """
        Perform checkerboard resolution test.
        
        Creates synthetic checkerboard pattern, generates synthetic data,
        and inverts to test spatial resolution capabilities.
        
        Parameters:
        -----------
        alpha_damping : float
            Damping regularization parameter
        alpha_smoothing : float  
            Smoothing regularization parameter
        checkerboard_size : float
            Size of checkerboard squares (km)
        amplitude : float
            Amplitude of checkerboard pattern
        noise_level : float
            Gaussian noise level to add to synthetic data
            
        Returns:
        --------
        results : dict
            Checkerboard test results including recovery metrics
        """
        print("Running checkerboard resolution test...")
        
        # Create checkerboard model
        true_model = self._create_checkerboard_model(checkerboard_size, amplitude)
        
        # Generate synthetic data
        G = self.inverter._build_design_matrix()
        synthetic_data = G @ true_model
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(synthetic_data), 
                                   len(synthetic_data))
            synthetic_data += noise
        
        # Store original observations and replace with synthetic
        original_obs = self.inverter.observations.copy()
        original_weights = self.inverter.observation_weights.copy()
        
        self.inverter.observations = synthetic_data
        self.inverter.observation_weights = np.ones(len(synthetic_data))
        
        try:
            # Invert synthetic data
            recovered_model = self.inverter.solve_inversion(
                alpha_damping, alpha_smoothing, 
                solver='spsolve', max_iterations=1000
            )['model_parameters']
            
            # Restore original observations
            self.inverter.observations = original_obs
            self.inverter.observation_weights = original_weights
            
            # Compute recovery metrics
            recovery_metrics = self._compute_recovery_metrics(true_model, recovered_model)
            
            results = {
                'true_model': true_model,
                'recovered_model': recovered_model,
                'synthetic_data': synthetic_data,
                'checkerboard_size': checkerboard_size,
                'amplitude': amplitude,
                'noise_level': noise_level,
                **recovery_metrics
            }
            
            return results
            
        except Exception as e:
            # Restore original observations
            self.inverter.observations = original_obs
            self.inverter.observation_weights = original_weights
            raise e
    
    def _create_checkerboard_model(self, square_size: float, amplitude: float) -> np.ndarray:
        """Create checkerboard pattern for resolution testing."""
        n_cells = len(self.inverter.mesh_centers)
        model = np.zeros(n_cells)
        
        for i, center in enumerate(self.inverter.mesh_centers):
            x, y = center[0], center[1]
            
            # Determine checkerboard square indices
            x_idx = int(x / square_size)
            y_idx = int(y / square_size)
            
            # Checkerboard pattern: alternate +/- amplitude
            if (x_idx + y_idx) % 2 == 0:
                model[i] = amplitude
            else:
                model[i] = -amplitude
        
        return model
    
    def _compute_recovery_metrics(self, true_model: np.ndarray, 
                                recovered_model: np.ndarray) -> Dict:
        """Compute quantitative recovery assessment metrics."""
        # Variance reduction
        ss_tot = np.sum((true_model - np.mean(true_model))**2)
        ss_res = np.sum((true_model - recovered_model)**2)
        variance_reduction = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Correlation coefficient
        correlation = np.corrcoef(true_model, recovered_model)[0, 1]
        
        # RMS error
        rms_error = np.sqrt(np.mean((true_model - recovered_model)**2))
        
        # Normalized RMS error
        model_range = np.max(true_model) - np.min(true_model)
        normalized_rms = rms_error / model_range if model_range > 0 else np.inf
        
        return {
            'variance_reduction': variance_reduction,
            'correlation': correlation,
            'rms_error': rms_error,
            'normalized_rms': normalized_rms,
            'recovery_quality': 'excellent' if variance_reduction > 0.8 else
                              'good' if variance_reduction > 0.6 else
                              'moderate' if variance_reduction > 0.4 else 'poor'
        }
    
    def synthetic_recovery_test(self, true_anisotropy: Dict, alpha_damping: float,
                              alpha_smoothing: float, noise_level: float = 0.05) -> Dict:
        """
        Test recovery of realistic synthetic anisotropy model.
        
        Parameters:
        -----------
        true_anisotropy : dict
            Dictionary with 'fast_axes' and 'delay_times' arrays
        alpha_damping : float
            Damping regularization parameter  
        alpha_smoothing : float
            Smoothing regularization parameter
        noise_level : float
            Relative noise level for synthetic data
            
        Returns:
        --------
        results : dict
            Synthetic recovery test results
        """
        print("Running synthetic recovery test...")
        
        # Convert anisotropy parameters to (Mc, Ms) model
        fast_axes = true_anisotropy['fast_axes']
        delay_times = true_anisotropy['delay_times']
        
        true_model = np.zeros(len(self.inverter.mesh_centers) * 2)
        
        for i in range(len(self.inverter.mesh_centers)):
            phi = fast_axes[i]
            A = delay_times[i]  # Anisotropy strength
            
            # Convert to (Mc, Ms) parameterization
            mc = A * np.cos(2 * phi)
            ms = A * np.sin(2 * phi)
            
            true_model[2*i] = mc
            true_model[2*i + 1] = ms
        
        # Generate synthetic observations
        G = self.inverter._build_design_matrix()
        synthetic_data = G @ true_model
        
        # Add realistic noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(synthetic_data),
                                   len(synthetic_data))
            synthetic_data += noise
        
        # Store and replace observations
        original_obs = self.inverter.observations.copy()
        original_weights = self.inverter.observation_weights.copy()
        
        self.inverter.observations = synthetic_data
        self.inverter.observation_weights = np.ones(len(synthetic_data))
        
        try:
            # Perform inversion
            result = self.inverter.solve_inversion(
                alpha_damping, alpha_smoothing,
                solver='spsolve', max_iterations=1000
            )
            
            recovered_model = result['model_parameters']
            
            # Restore original observations
            self.inverter.observations = original_obs
            self.inverter.observation_weights = original_weights
            
            # Convert recovered model back to anisotropy parameters
            recovered_anisotropy = self._convert_model_to_anisotropy(recovered_model)
            
            # Compute recovery metrics
            recovery_metrics = self._assess_anisotropy_recovery(
                true_anisotropy, recovered_anisotropy
            )
            
            return {
                'true_anisotropy': true_anisotropy,
                'recovered_anisotropy': recovered_anisotropy,
                'true_model': true_model,
                'recovered_model': recovered_model,
                'synthetic_data': synthetic_data,
                'inversion_result': result,
                **recovery_metrics
            }
            
        except Exception as e:
            # Restore original observations
            self.inverter.observations = original_obs
            self.inverter.observation_weights = original_weights
            raise e
    
    def _convert_model_to_anisotropy(self, model: np.ndarray) -> Dict:
        """Convert (Mc, Ms) model parameters to anisotropy parameters."""
        n_cells = len(self.inverter.mesh_centers)
        fast_axes = np.zeros(n_cells)
        delay_times = np.zeros(n_cells)
        
        for i in range(n_cells):
            mc = model[2*i]
            ms = model[2*i + 1]
            
            # Convert back to (φ, δt) 
            phi = 0.5 * np.arctan2(ms, mc)
            A = np.sqrt(mc**2 + ms**2)
            
            fast_axes[i] = phi
            delay_times[i] = A
        
        return {
            'fast_axes': fast_axes,
            'delay_times': delay_times
        }
    
    def _assess_anisotropy_recovery(self, true_anisotropy: Dict, 
                                  recovered_anisotropy: Dict) -> Dict:
        """Assess quality of anisotropy parameter recovery."""
        true_phi = true_anisotropy['fast_axes']
        true_dt = true_anisotropy['delay_times']
        
        recovered_phi = recovered_anisotropy['fast_axes']
        recovered_dt = recovered_anisotropy['delay_times']
        
        # Angular differences (accounting for π periodicity)
        phi_diff = np.abs(true_phi - recovered_phi)
        phi_diff = np.minimum(phi_diff, np.pi - phi_diff)  # Wrap to [0, π/2]
        
        # Statistics
        mean_angular_error = np.mean(phi_diff)
        rms_angular_error = np.sqrt(np.mean(phi_diff**2))
        
        dt_correlation = np.corrcoef(true_dt, recovered_dt)[0, 1]
        dt_rms_error = np.sqrt(np.mean((true_dt - recovered_dt)**2))
        
        return {
            'mean_angular_error': np.degrees(mean_angular_error),
            'rms_angular_error': np.degrees(rms_angular_error), 
            'delay_time_correlation': dt_correlation,
            'delay_time_rms_error': dt_rms_error,
            'angular_differences': np.degrees(phi_diff)
        }
    
    def data_coverage_analysis(self) -> Dict:
        """
        Analyze spatial and angular coverage of the data.
        
        Returns:
        --------
        coverage : dict
            Data coverage metrics and statistics
        """
        print("Analyzing data coverage...")
        
        # Ray path analysis
        ray_paths = []
        ray_azimuths = []
        
        for obs_type, data in self.inverter.splitting_observations.items():
            for station in data:
                for measurement in data[station]:
                    if 'ray_path' in measurement:
                        ray_paths.append(measurement['ray_path'])
                    if 'back_azimuth' in measurement:
                        ray_azimuths.append(measurement['back_azimuth'])
        
        # Cell hit counts
        n_cells = len(self.inverter.mesh_centers)
        cell_hits = np.zeros(n_cells)
        
        G = self.inverter._build_design_matrix()
        for i in range(G.shape[0]):
            # Count non-zero elements per cell (both Mc and Ms)
            for j in range(0, G.shape[1], 2):
                if G[i, j] != 0 or G[i, j+1] != 0:
                    cell_hits[j//2] += 1
        
        # Azimuthal coverage
        if ray_azimuths:
            azimuth_bins = np.linspace(0, 360, 37)  # 10-degree bins
            azimuth_counts, _ = np.histogram(ray_azimuths, bins=azimuth_bins)
            azimuth_coverage = np.sum(azimuth_counts > 0) / len(azimuth_counts)
        else:
            azimuth_coverage = 0.0
            azimuth_counts = np.zeros(36)
        
        return {
            'total_observations': G.shape[0],
            'total_cells': n_cells,
            'cells_with_data': np.sum(cell_hits > 0),
            'cell_coverage_fraction': np.sum(cell_hits > 0) / n_cells,
            'cell_hit_counts': cell_hits,
            'mean_hits_per_cell': np.mean(cell_hits[cell_hits > 0]) if np.any(cell_hits > 0) else 0,
            'azimuth_coverage_fraction': azimuth_coverage,
            'azimuth_distribution': azimuth_counts,
            'azimuth_bins': azimuth_bins[:-1]
        }
    
    def plot_resolution_diagnostics(self, resolution_metrics: ResolutionMetrics,
                                  save_path: Optional[str] = None) -> None:
        """
        Create comprehensive resolution diagnostic plots.
        
        Parameters:
        -----------
        resolution_metrics : ResolutionMetrics
            Results from compute_resolution_matrix()
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Diagonal resolution
        self._plot_spatial_field(
            self.inverter.mesh_centers, 
            resolution_metrics.diagonal_resolution,
            ax=axes[0, 0], title='Diagonal Resolution',
            colormap='viridis', vmin=0, vmax=1
        )
        
        # 2. Spread function
        self._plot_spatial_field(
            self.inverter.mesh_centers,
            resolution_metrics.spread_function,
            ax=axes[0, 1], title='Spread Function',
            colormap='plasma', vmin=0
        )
        
        # 3. Parameter uncertainties
        self._plot_spatial_field(
            self.inverter.mesh_centers,
            resolution_metrics.parameter_uncertainties[::2],  # Mc uncertainties
            ax=axes[0, 2], title='Parameter Uncertainties',
            colormap='Reds', vmin=0
        )
        
        # 4. Correlation lengths
        self._plot_spatial_field(
            self.inverter.mesh_centers,
            resolution_metrics.correlation_lengths,
            ax=axes[1, 0], title='Correlation Lengths (km)',
            colormap='Blues', vmin=0
        )
        
        # 5. Resolution matrix structure
        R = resolution_metrics.resolution_matrix
        im = axes[1, 1].imshow(R[:50, :50], aspect='auto', cmap='RdBu_r',
                              vmin=-0.5, vmax=1.0)
        axes[1, 1].set_title('Resolution Matrix (first 50×50)')
        axes[1, 1].set_xlabel('Model Parameter Index')
        axes[1, 1].set_ylabel('Model Parameter Index')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Resolution statistics
        axes[1, 2].hist(resolution_metrics.diagonal_resolution, bins=30, 
                       alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].axvline(np.mean(resolution_metrics.diagonal_resolution), 
                          color='red', linestyle='--', label='Mean')
        axes[1, 2].set_xlabel('Diagonal Resolution')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Resolution Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_spatial_field(self, centers: np.ndarray, values: np.ndarray,
                           ax, title: str, colormap: str = 'viridis',
                           vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
        """Plot spatial field on mesh."""
        # Filter out NaN values
        valid = ~np.isnan(values)
        if not np.any(valid):
            ax.set_title(f'{title} (No valid data)')
            return
        
        centers_valid = centers[valid]
        values_valid = values[valid]
        
        scatter = ax.scatter(centers_valid[:, 0], centers_valid[:, 1], 
                           c=values_valid, s=50, cmap=colormap,
                           vmin=vmin, vmax=vmax, alpha=0.8)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax)
    
    def plot_checkerboard_results(self, checkerboard_results: Dict,
                                save_path: Optional[str] = None) -> None:
        """Plot checkerboard test results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        true_model = checkerboard_results['true_model']
        recovered_model = checkerboard_results['recovered_model']
        
        # True model
        self._plot_spatial_field(
            self.inverter.mesh_centers, true_model,
            ax=axes[0], title='True Checkerboard Model',
            colormap='RdBu_r', vmin=-np.max(np.abs(true_model)),
            vmax=np.max(np.abs(true_model))
        )
        
        # Recovered model
        self._plot_spatial_field(
            self.inverter.mesh_centers, recovered_model,
            ax=axes[1], title='Recovered Model',
            colormap='RdBu_r', vmin=-np.max(np.abs(true_model)),
            vmax=np.max(np.abs(true_model))
        )
        
        # Residuals
        residuals = true_model - recovered_model
        self._plot_spatial_field(
            self.inverter.mesh_centers, residuals,
            ax=axes[2], title='Residuals (True - Recovered)',
            colormap='seismic', vmin=-np.max(np.abs(residuals)),
            vmax=np.max(np.abs(residuals))
        )
        
        # Add metrics to title
        metrics = checkerboard_results
        fig.suptitle(f"Checkerboard Test | VR: {metrics['variance_reduction']:.2f} | "
                    f"Corr: {metrics['correlation']:.2f} | "
                    f"Quality: {metrics['recovery_quality']}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_comprehensive_diagnostics_report(inverter: GlobalAnisotropyInverter,
                                          alpha_damping: float, alpha_smoothing: float,
                                          output_dir: str = './diagnostics/') -> Dict:
    """
    Generate comprehensive diagnostics report for inversion results.
    
    Parameters:
    -----------
    inverter : GlobalAnisotropyInverter
        Configured inversion object
    alpha_damping : float
        Damping regularization parameter
    alpha_smoothing : float
        Smoothing regularization parameter  
    output_dir : str
        Directory to save diagnostic plots and results
        
    Returns:
    --------
    report : dict
        Complete diagnostics report
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    diagnostics = InversionDiagnostics(inverter)
    
    print("Generating comprehensive diagnostics report...")
    
    # 1. Resolution analysis
    resolution_metrics = diagnostics.compute_resolution_matrix(alpha_damping, alpha_smoothing)
    diagnostics.plot_resolution_diagnostics(
        resolution_metrics, save_path=os.path.join(output_dir, 'resolution_analysis.png')
    )
    
    # 2. Checkerboard test
    checkerboard_results = diagnostics.checkerboard_test(
        alpha_damping, alpha_smoothing, checkerboard_size=5.0, amplitude=0.1
    )
    diagnostics.plot_checkerboard_results(
        checkerboard_results, save_path=os.path.join(output_dir, 'checkerboard_test.png')
    )
    
    # 3. Data coverage analysis
    coverage_analysis = diagnostics.data_coverage_analysis()
    
    # 4. Compile report
    report = {
        'regularization_parameters': {
            'alpha_damping': alpha_damping,
            'alpha_smoothing': alpha_smoothing
        },
        'resolution_metrics': resolution_metrics,
        'checkerboard_results': checkerboard_results,
        'coverage_analysis': coverage_analysis,
        'diagnostics_summary': {
            'mean_diagonal_resolution': np.mean(resolution_metrics.diagonal_resolution),
            'checkerboard_variance_reduction': checkerboard_results['variance_reduction'],
            'checkerboard_correlation': checkerboard_results['correlation'],
            'data_coverage_fraction': coverage_analysis['cell_coverage_fraction'],
            'azimuth_coverage_fraction': coverage_analysis['azimuth_coverage_fraction']
        }
    }
    
    print(f"Diagnostics report saved to {output_dir}")
    return report


# Example usage
if __name__ == "__main__":
    print("Inversion Diagnostics Module")
    print("Provides comprehensive quality assessment tools for global anisotropy inversion.")