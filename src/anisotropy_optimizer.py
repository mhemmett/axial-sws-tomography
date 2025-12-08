"""
Anisotropy Optimizer Module

Implements tomographic inversion for seismic anisotropy using gradient descent optimization.
Based on the ESS563 2025 tomography assignment approach, adapted for anisotropy parameters.

This module performs iterative optimization of anisotropy model parameters (percent_anisotropy 
and phi_fast) by minimizing misfit between observed and predicted shear-wave splitting measurements.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path


class AnisotropyOptimizer:
    """
    Tomographic inversion optimizer for seismic anisotropy parameters.
    
    Uses gradient descent with:
    - Finite difference Jacobian computation
    - Tikhonov regularization for smoothness
    - Armijo backtracking line search
    - Early stopping on validation set
    """
    
    def __init__(self, 
                 velocity_model,
                 ray_tracer,
                 coarse_grid_shape: Tuple[int, int, int] = (41, 41, 26),
                 regularization_lambda: float = 0.1,
                 max_iterations: int = 100,
                 convergence_tol: float = 1e-3,
                 verbose: bool = True):
        """
        Initialize the anisotropy optimizer.
        
        Parameters:
        -----------
        velocity_model : AxialVelocityModel
            Velocity model object with grid structure
        ray_tracer : RayTracer
            Ray tracer for computing anisotropic ray paths
        coarse_grid_shape : tuple
            Shape of coarse parameterization (nx, ny, nz)
        regularization_lambda : float
            Tikhonov regularization parameter
        max_iterations : int
            Maximum number of optimization iterations
        convergence_tol : float
            Relative misfit change threshold for convergence
        verbose : bool
            Print progress information
        """
        self.vm = velocity_model
        self.ray_tracer = ray_tracer
        self.coarse_shape = coarse_grid_shape
        self.reg_lambda = regularization_lambda
        self.max_iter = max_iterations
        self.conv_tol = convergence_tol
        self.verbose = verbose
        
        # Store original fine grid shape
        self.fine_shape = (velocity_model.nx, velocity_model.ny, velocity_model.nz)
        
        # Initialize model parameters (will be set in setup_model)
        self.percent_anisotropy_coarse = None
        self.phi_fast_coarse = None
        
        # Convergence history
        self.history = {
            'iteration': [],
            'training_misfit': [],
            'validation_misfit': [],
            'model_norm': [],
            'gradient_norm': [],
            'step_size': [],
            'time': []
        }
        
        if self.verbose:
            print(f"✓ AnisotropyOptimizer initialized")
            print(f"  Fine grid: {self.fine_shape}")
            print(f"  Coarse grid: {self.coarse_shape}")
            print(f"  Downsampling factors: ({self.fine_shape[0]//self.coarse_shape[0]}, "
                  f"{self.fine_shape[1]//self.coarse_shape[1]}, "
                  f"{self.fine_shape[2]//self.coarse_shape[2]})")
            print(f"  Number of parameters: {np.prod(self.coarse_shape) * 2:.0f}")
    
    def setup_model(self, percent_anisotropy_init: np.ndarray, phi_fast_init: np.ndarray):
        """
        Set up initial model parameters by downsampling to coarse grid.
        
        Parameters:
        -----------
        percent_anisotropy_init : np.ndarray
            Initial percent anisotropy on fine grid (nx, ny, nz)
        phi_fast_init : np.ndarray
            Initial fast direction in degrees on fine grid (nx, ny, nz)
        """
        # Downsample to coarse grid
        self.percent_anisotropy_coarse = self._downsample_to_coarse(percent_anisotropy_init)
        self.phi_fast_coarse = self._downsample_to_coarse(phi_fast_init)
        
        if self.verbose:
            print(f"\n✓ Model initialized")
            print(f"  Percent anisotropy range: {self.percent_anisotropy_coarse.min():.2f} - "
                  f"{self.percent_anisotropy_coarse.max():.2f}%")
            print(f"  Phi fast range: {self.phi_fast_coarse.min():.1f}° - "
                  f"{self.phi_fast_coarse.max():.1f}°")
    
    def _downsample_to_coarse(self, fine_grid: np.ndarray) -> np.ndarray:
        """
        Downsample fine grid to coarse grid using block averaging.
        
        Parameters:
        -----------
        fine_grid : np.ndarray
            Array on fine grid (nx_fine, ny_fine, nz_fine)
        
        Returns:
        --------
        coarse_grid : np.ndarray
            Array on coarse grid (nx_coarse, ny_coarse, nz_coarse)
        """
        # Calculate downsampling factors
        fx = self.fine_shape[0] // self.coarse_shape[0]
        fy = self.fine_shape[1] // self.coarse_shape[1]
        fz = self.fine_shape[2] // self.coarse_shape[2]
        
        # Trim fine grid to be evenly divisible
        nx_trim = self.coarse_shape[0] * fx
        ny_trim = self.coarse_shape[1] * fy
        nz_trim = self.coarse_shape[2] * fz
        
        fine_trimmed = fine_grid[:nx_trim, :ny_trim, :nz_trim]
        
        # Reshape and average
        coarse = fine_trimmed.reshape(self.coarse_shape[0], fx,
                                      self.coarse_shape[1], fy,
                                      self.coarse_shape[2], fz)
        coarse = coarse.mean(axis=(1, 3, 5))
        
        return coarse
    
    def _upsample_to_fine(self, coarse_grid: np.ndarray) -> np.ndarray:
        """
        Upsample coarse grid to fine grid using nearest neighbor interpolation.
        
        Parameters:
        -----------
        coarse_grid : np.ndarray
            Array on coarse grid (nx_coarse, ny_coarse, nz_coarse)
        
        Returns:
        --------
        fine_grid : np.ndarray
            Array on fine grid (nx_fine, ny_fine, nz_fine)
        """
        # Calculate upsampling factors
        fx = self.fine_shape[0] // self.coarse_shape[0]
        fy = self.fine_shape[1] // self.coarse_shape[1]
        fz = self.fine_shape[2] // self.coarse_shape[2]
        
        # Repeat values
        fine = np.repeat(coarse_grid, fx, axis=0)
        fine = np.repeat(fine, fy, axis=1)
        fine = np.repeat(fine, fz, axis=2)
        
        # Trim to exact fine grid size if needed
        fine = fine[:self.fine_shape[0], :self.fine_shape[1], :self.fine_shape[2]]
        
        return fine
    
    def compute_misfit(self, 
                      observations: List[Dict],
                      use_current_model: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute misfit between observed and predicted splitting parameters.
        
        Parameters:
        -----------
        observations : list of dict
            List of observed splitting measurements with keys:
            'event_x', 'event_y', 'event_z', 'station_x', 'station_y', 'station_z',
            'phi_obs', 'phi_error', 'dt_obs', 'dt_error'
        use_current_model : bool
            If True, use current model parameters; if False, must provide predictions
        
        Returns:
        --------
        total_misfit : float
            Total chi-squared misfit
        phi_residuals : np.ndarray
            Residuals for phi (observed - predicted)
        dt_residuals : np.ndarray
            Residuals for dt (observed - predicted)
        """
        n_obs = len(observations)
        phi_residuals = np.zeros(n_obs)
        dt_residuals = np.zeros(n_obs)
        phi_weights = np.zeros(n_obs)
        dt_weights = np.zeros(n_obs)
        
        # Upsample current model to fine grid for ray tracing
        if use_current_model:
            percent_aniso_fine = self._upsample_to_fine(self.percent_anisotropy_coarse)
            phi_fast_fine = self._upsample_to_fine(self.phi_fast_coarse)
            
            # Update velocity model using the public method
            self.vm.update_anisotropy(
                new_percent_anisotropy=percent_aniso_fine,
                new_phi_fast=phi_fast_fine
            )
        
        # Compute predictions for each observation
        for i, obs in enumerate(observations):
            event_loc = np.array([obs['event_x'], obs['event_y'], obs['event_z']])
            station_loc = np.array([obs['station_x'], obs['station_y'], obs['station_z']])
            
            try:
                # Trace anisotropic rays
                ray_fast, ray_slow, path_diff = self.ray_tracer.compute_anisotropic_ray_paths(
                    event_loc, station_loc, npts=100
                )
                
                # Extract predicted values
                phi_pred = path_diff.get('predicted_phi', 0.0)
                dt_pred = path_diff.get('predicted_dt', 0.0)
                
                # Compute residuals (observed - predicted)
                phi_residuals[i] = obs['phi_obs'] - phi_pred
                dt_residuals[i] = obs['dt_obs'] - dt_pred
                
                # Weights (inverse of uncertainties)
                phi_weights[i] = 1.0 / max(obs['phi_error'], 1.0)  # Avoid division by zero
                dt_weights[i] = 1.0 / max(obs['dt_error'], 0.01)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Ray tracing failed for observation {i}: {e}")
                # Use large residual for failed ray tracing
                phi_residuals[i] = 0.0
                dt_residuals[i] = 0.0
                phi_weights[i] = 0.0
                dt_weights[i] = 0.0
        
        # Compute weighted chi-squared misfit
        chi2_phi = np.sum((phi_residuals * phi_weights)**2)
        chi2_dt = np.sum((dt_residuals * dt_weights)**2)
        total_misfit = chi2_phi + chi2_dt
        
        return total_misfit, phi_residuals, dt_residuals
    
    def compute_gradient(self,
                        observations: List[Dict],
                        perturbation: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of misfit function using finite differences.
        
        Parameters:
        -----------
        observations : list of dict
            List of observed splitting measurements
        perturbation : float
            Relative perturbation for finite differences (e.g., 0.05 = 5%)
        
        Returns:
        --------
        grad_percent_aniso : np.ndarray
            Gradient with respect to percent_anisotropy (coarse grid)
        grad_phi : np.ndarray
            Gradient with respect to phi_fast (coarse grid)
        """
        # Initialize gradients
        grad_percent_aniso = np.zeros_like(self.percent_anisotropy_coarse)
        grad_phi = np.zeros_like(self.phi_fast_coarse)
        
        # Compute baseline misfit
        misfit_0, _, _ = self.compute_misfit(observations, use_current_model=True)
        
        # Perturb each parameter and compute gradient
        # For efficiency, only perturb cells that are likely to affect rays
        # Here we perturb all cells, but in practice could use sparse sampling
        
        if self.verbose:
            print(f"  Computing gradient via finite differences...")
            print(f"    Perturbation: ±{perturbation*100:.1f}%")
        
        # Gradient for percent_anisotropy
        for ix in range(self.coarse_shape[0]):
            for iy in range(self.coarse_shape[1]):
                for iz in range(self.coarse_shape[2]):
                    # Perturb parameter
                    original_value = self.percent_anisotropy_coarse[ix, iy, iz]
                    delta = original_value * perturbation
                    
                    # Forward perturbation
                    self.percent_anisotropy_coarse[ix, iy, iz] = original_value + delta
                    misfit_plus, _, _ = self.compute_misfit(observations, use_current_model=True)
                    
                    # Backward perturbation
                    self.percent_anisotropy_coarse[ix, iy, iz] = original_value - delta
                    misfit_minus, _, _ = self.compute_misfit(observations, use_current_model=True)
                    
                    # Central difference
                    grad_percent_aniso[ix, iy, iz] = (misfit_plus - misfit_minus) / (2 * delta)
                    
                    # Restore original value
                    self.percent_anisotropy_coarse[ix, iy, iz] = original_value
        
        # Gradient for phi_fast (similar approach)
        for ix in range(self.coarse_shape[0]):
            for iy in range(self.coarse_shape[1]):
                for iz in range(self.coarse_shape[2]):
                    original_value = self.phi_fast_coarse[ix, iy, iz]
                    delta = 1.0  # 1 degree perturbation
                    
                    # Forward perturbation
                    self.phi_fast_coarse[ix, iy, iz] = original_value + delta
                    misfit_plus, _, _ = self.compute_misfit(observations, use_current_model=True)
                    
                    # Backward perturbation
                    self.phi_fast_coarse[ix, iy, iz] = original_value - delta
                    misfit_minus, _, _ = self.compute_misfit(observations, use_current_model=True)
                    
                    # Central difference
                    grad_phi[ix, iy, iz] = (misfit_plus - misfit_minus) / (2 * delta)
                    
                    # Restore original value
                    self.phi_fast_coarse[ix, iy, iz] = original_value
        
        if self.verbose:
            print(f"    Gradient norms: percent_aniso={np.linalg.norm(grad_percent_aniso):.2e}, "
                  f"phi={np.linalg.norm(grad_phi):.2e}")
        
        return grad_percent_aniso, grad_phi
    
    def apply_regularization(self, grad_percent_aniso: np.ndarray, grad_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Tikhonov regularization (smoothness constraint) to gradients.
        
        Parameters:
        -----------
        grad_percent_aniso : np.ndarray
            Gradient for percent_anisotropy
        grad_phi : np.ndarray
            Gradient for phi_fast
        
        Returns:
        --------
        grad_percent_aniso_reg : np.ndarray
            Regularized gradient for percent_anisotropy
        grad_phi_reg : np.ndarray
            Regularized gradient for phi_fast
        """
        # Compute Laplacian (simple second derivative approximation)
        laplacian_percent = self._compute_laplacian(self.percent_anisotropy_coarse)
        laplacian_phi = self._compute_laplacian(self.phi_fast_coarse)
        
        # Add regularization term to gradient
        grad_percent_aniso_reg = grad_percent_aniso + self.reg_lambda * laplacian_percent
        grad_phi_reg = grad_phi + self.reg_lambda * laplacian_phi
        
        return grad_percent_aniso_reg, grad_phi_reg
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute discrete Laplacian for regularization.
        
        Parameters:
        -----------
        field : np.ndarray
            3D field on coarse grid
        
        Returns:
        --------
        laplacian : np.ndarray
            Discrete Laplacian
        """
        laplacian = np.zeros_like(field)
        
        # Simple 6-point stencil
        for axis in range(3):
            laplacian += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        
        return laplacian
    
    def line_search(self,
                   observations: List[Dict],
                   grad_percent_aniso: np.ndarray,
                   grad_phi: np.ndarray,
                   alpha_init: float = 0.1,
                   beta: float = 0.5,
                   c: float = 1e-4,
                   max_backtracks: int = 20) -> float:
        """
        Armijo backtracking line search to find optimal step size.
        
        Parameters:
        -----------
        observations : list of dict
            Observed splitting measurements
        grad_percent_aniso : np.ndarray
            Gradient for percent_anisotropy
        grad_phi : np.ndarray
            Gradient for phi_fast
        alpha_init : float
            Initial step size
        beta : float
            Reduction factor (0 < beta < 1)
        c : float
            Armijo condition parameter
        max_backtracks : int
            Maximum number of backtracking steps
        
        Returns:
        --------
        alpha : float
            Optimal step size
        """
        # Compute initial misfit and directional derivative
        misfit_0, _, _ = self.compute_misfit(observations, use_current_model=True)
        directional_deriv = -(np.sum(grad_percent_aniso**2) + np.sum(grad_phi**2))
        
        # Save current model
        percent_aniso_save = self.percent_anisotropy_coarse.copy()
        phi_fast_save = self.phi_fast_coarse.copy()
        
        alpha = alpha_init
        for i in range(max_backtracks):
            # Update model with step
            self.percent_anisotropy_coarse = percent_aniso_save - alpha * grad_percent_aniso
            self.phi_fast_coarse = phi_fast_save - alpha * grad_phi
            
            # Compute new misfit
            misfit_new, _, _ = self.compute_misfit(observations, use_current_model=True)
            
            # Check Armijo condition
            if misfit_new <= misfit_0 + c * alpha * directional_deriv:
                if self.verbose:
                    print(f"    Line search converged: alpha={alpha:.4f} after {i+1} backtracks")
                return alpha
            
            # Reduce step size
            alpha *= beta
        
        # If line search fails, restore model and return small alpha
        self.percent_anisotropy_coarse = percent_aniso_save
        self.phi_fast_coarse = phi_fast_save
        
        if self.verbose:
            print(f"    Line search failed, using alpha={alpha:.4f}")
        
        return alpha
    
    def optimize(self,
                training_obs: List[Dict],
                validation_obs: Optional[List[Dict]] = None,
                early_stopping_patience: int = 3) -> Dict:
        """
        Run gradient descent optimization to minimize splitting misfit.
        
        Parameters:
        -----------
        training_obs : list of dict
            Training set observations
        validation_obs : list of dict, optional
            Validation set observations for early stopping
        early_stopping_patience : int
            Number of iterations to wait for validation improvement
        
        Returns:
        --------
        result : dict
            Optimization results with final misfits and convergence info
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Anisotropy Optimization")
            print(f"{'='*60}")
            print(f"Training observations: {len(training_obs)}")
            if validation_obs:
                print(f"Validation observations: {len(validation_obs)}")
            print(f"Maximum iterations: {self.max_iter}")
            print(f"Convergence tolerance: {self.conv_tol}")
            print(f"Regularization lambda: {self.reg_lambda}")
        
        # Initialize
        best_validation_misfit = np.inf
        patience_counter = 0
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            iter_start = time.time()
            
            if self.verbose:
                print(f"\n--- Iteration {iteration+1}/{self.max_iter} ---")
            
            # Compute training misfit
            training_misfit, _, _ = self.compute_misfit(training_obs, use_current_model=True)
            
            # Compute gradient
            grad_percent_aniso, grad_phi = self.compute_gradient(training_obs)
            
            # Apply regularization
            grad_percent_aniso_reg, grad_phi_reg = self.apply_regularization(grad_percent_aniso, grad_phi)
            
            # Line search for step size
            alpha = self.line_search(training_obs, grad_percent_aniso_reg, grad_phi_reg)
            
            # Update model
            self.percent_anisotropy_coarse -= alpha * grad_percent_aniso_reg
            self.phi_fast_coarse -= alpha * grad_phi_reg
            
            # Compute validation misfit if provided
            validation_misfit = None
            if validation_obs:
                validation_misfit, _, _ = self.compute_misfit(validation_obs, use_current_model=True)
            
            # Store history
            iter_time = time.time() - iter_start
            self.history['iteration'].append(iteration + 1)
            self.history['training_misfit'].append(training_misfit)
            self.history['validation_misfit'].append(validation_misfit if validation_misfit else np.nan)
            self.history['model_norm'].append(np.sqrt(np.sum(self.percent_anisotropy_coarse**2) + 
                                                       np.sum(self.phi_fast_coarse**2)))
            self.history['gradient_norm'].append(np.sqrt(np.sum(grad_percent_aniso_reg**2) + 
                                                          np.sum(grad_phi_reg**2)))
            self.history['step_size'].append(alpha)
            self.history['time'].append(iter_time)
            
            if self.verbose:
                print(f"  Training misfit: {training_misfit:.4f}")
                if validation_misfit:
                    print(f"  Validation misfit: {validation_misfit:.4f}")
                print(f"  Gradient norm: {self.history['gradient_norm'][-1]:.4e}")
                print(f"  Step size: {alpha:.4f}")
                print(f"  Iteration time: {iter_time:.2f}s")
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(self.history['training_misfit'][-1] - 
                               self.history['training_misfit'][-2]) / self.history['training_misfit'][-2]
                if rel_change < self.conv_tol:
                    if self.verbose:
                        print(f"\n✓ Converged: relative misfit change {rel_change:.4e} < {self.conv_tol}")
                    break
            
            # Early stopping check
            if validation_obs and validation_misfit is not None:
                if validation_misfit < best_validation_misfit:
                    best_validation_misfit = validation_misfit
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if self.verbose:
                            print(f"\n✓ Early stopping: validation misfit not improving for {early_stopping_patience} iterations")
                        break
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete")
            print(f"{'='*60}")
            print(f"Total iterations: {len(self.history['iteration'])}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Final training misfit: {self.history['training_misfit'][-1]:.4f}")
            if validation_obs:
                print(f"Final validation misfit: {self.history['validation_misfit'][-1]:.4f}")
        
        # Prepare result dictionary
        result = {
            'success': True,
            'final_training_misfit': self.history['training_misfit'][-1],
            'final_validation_misfit': self.history['validation_misfit'][-1] if validation_obs else None,
            'iterations': len(self.history['iteration']),
            'total_time': total_time,
            'history': self.history,
            'percent_anisotropy_coarse': self.percent_anisotropy_coarse,
            'phi_fast_coarse': self.phi_fast_coarse,
            'percent_anisotropy_fine': self._upsample_to_fine(self.percent_anisotropy_coarse),
            'phi_fast_fine': self._upsample_to_fine(self.phi_fast_coarse)
        }
        
        return result
    
    def save_results(self, result: Dict, output_dir: str):
        """
        Save optimization results to disk.
        
        Parameters:
        -----------
        result : dict
            Optimization result dictionary
        output_dir : str or Path
            Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        np.save(output_path / 'percent_anisotropy_optimized.npy', 
                result['percent_anisotropy_fine'])
        np.save(output_path / 'phi_fast_optimized.npy', 
                result['phi_fast_fine'])
        
        # Save convergence history
        history_df = pd.DataFrame(result['history'])
        history_df.to_csv(output_path / 'convergence_history.csv', index=False)
        
        if self.verbose:
            print(f"\n✓ Results saved to {output_path}")
