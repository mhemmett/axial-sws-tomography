"""
Global Linear Least-Squares Anisotropy Inversion

This module implements a sophisticated global inversion method for seismic anisotropy
using both shear-wave splitting parameters and splitting intensity measurements.
The approach converts the cell-based averaging into a robust linear least-squares
problem with proper regularization.

Mathematical Framework:
- Parameterizes anisotropy using (Mc, Ms) = A*(cos(2ψ), sin(2ψ))  
- Linear forward operators for splitting (φ, δt) and splitting intensity (SI)
- Global matrix inversion with spatial regularization
- Uncertainty quantification and resolution analysis

Author: Axial Seamount Research Team
Date: December 2024
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class GlobalAnisotropyInverter:
    """
    Global linear least-squares inversion for seismic anisotropy structure.
    
    This class implements the mathematical formulation that jointly inverts
    shear-wave splitting parameters (φ, δt) and splitting intensity (SI)
    measurements to recover 3D anisotropy structure with proper regularization.
    """
    
    def __init__(self, velocity_model, regularization_params: Optional[Dict] = None):
        """
        Initialize the global anisotropy inverter.
        
        Parameters:
        -----------
        velocity_model : AxialVelocityModel
            The 3D velocity model providing grid geometry
        regularization_params : dict, optional
            Regularization parameters including smoothing weights
        """
        self.vm = velocity_model
        self.nx, self.ny, self.nz = velocity_model.nx, velocity_model.ny, velocity_model.nz
        self.n_cells = self.nx * self.ny * self.nz
        self.n_params = 2 * self.n_cells  # Mc and Ms for each cell
        
        # Default regularization parameters
        default_reg = {
            'lambda_smooth': 1.0,    # Spatial smoothing weight
            'lambda_damp': 0.1,      # Damping weight  
            'smooth_horizontal': 1.0, # Horizontal smoothing relative weight
            'smooth_vertical': 0.5,   # Vertical smoothing relative weight
        }
        self.reg_params = default_reg if regularization_params is None else {**default_reg, **regularization_params}
        
        # Initialize arrays
        self.observations = []
        self.design_matrix = None
        self.data_vector = None
        self.weight_matrix = None
        self.regularization_matrix = None
        
        # Results storage
        self.model_estimate = None
        self.model_uncertainty = None
        self.resolution_matrix = None
        self.inversion_stats = {}
        
    def convert_anisotropy_params(self, anisotropy_strength: np.ndarray, 
                                fast_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert (A, ψ) to linear parameters (Mc, Ms).
        
        Parameters:
        -----------
        anisotropy_strength : np.ndarray
            Anisotropy strength A for each cell
        fast_direction : np.ndarray  
            Fast direction ψ in degrees for each cell
            
        Returns:
        --------
        Mc, Ms : np.ndarray
            Linear anisotropy parameters
        """
        psi_rad = np.deg2rad(fast_direction)
        Mc = anisotropy_strength * np.cos(2 * psi_rad)
        Ms = anisotropy_strength * np.sin(2 * psi_rad)
        return Mc, Ms
    
    def convert_linear_params(self, Mc: np.ndarray, Ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert linear parameters (Mc, Ms) back to (A, ψ).
        
        Parameters:
        -----------
        Mc, Ms : np.ndarray
            Linear anisotropy parameters
            
        Returns:
        --------
        anisotropy_strength : np.ndarray
            Anisotropy strength A
        fast_direction : np.ndarray
            Fast direction ψ in degrees
        """
        anisotropy_strength = np.sqrt(Mc**2 + Ms**2)
        fast_direction = 0.5 * np.rad2deg(np.arctan2(Ms, Mc))
        
        # Ensure fast direction is in [0, 180) degrees
        fast_direction = fast_direction % 180
        
        return anisotropy_strength, fast_direction
    
    def add_splitting_observations(self, splitting_data: List[Dict]):
        """
        Add shear-wave splitting observations to the inversion.
        
        Parameters:
        -----------
        splitting_data : list of dict
            Each dict contains:
            - 'delta_t': delay time in seconds
            - 'phi': fast direction in degrees  
            - 'azimuth': ray azimuth in degrees
            - 'Lij': dict of {cell_idx: path_length}
            - 'sigma_dt': uncertainty in delay time
            - 'sigma_phi': uncertainty in fast direction (optional)
        """
        for obs in splitting_data:
            obs_record = {
                'type': 'splitting',
                'delta_t': obs['delta_t'],
                'phi': obs['phi'], 
                'azimuth': obs['azimuth'],
                'Lij': obs['Lij'],
                'sigma_dt': obs.get('sigma_dt', 0.01),  # Default 10ms uncertainty
                'sigma_phi': obs.get('sigma_phi', 5.0),  # Default 5° uncertainty
            }
            self.observations.append(obs_record)
    
    def add_splitting_intensity_observations(self, si_data: List[Dict]):
        """
        Add splitting intensity observations to the inversion.
        
        Parameters:
        -----------
        si_data : list of dict
            Each dict contains:
            - 'SI': splitting intensity value
            - 'azimuth': ray azimuth in degrees
            - 'Lij': dict of {cell_idx: path_length}  
            - 'sigma_si': uncertainty in SI
            - 's_factor': sensitivity scaling factor (default 1.0)
        """
        for obs in si_data:
            obs_record = {
                'type': 'si',
                'SI': obs['SI'],
                'azimuth': obs['azimuth'],
                'Lij': obs['Lij'],
                'sigma_si': obs.get('sigma_si', 0.1),
                's_factor': obs.get('s_factor', 1.0),
            }
            self.observations.append(obs_record)
    
    def build_design_matrix(self) -> sp.csr_matrix:
        """
        Build the global sparse design matrix G.
        
        Returns:
        --------
        G : scipy.sparse.csr_matrix
            Design matrix mapping model parameters to observations
        """
        # Count total rows: 2 per splitting obs + 1 per SI obs  
        n_splitting = sum(1 for obs in self.observations if obs['type'] == 'splitting')
        n_si = sum(1 for obs in self.observations if obs['type'] == 'si')
        n_rows = 2 * n_splitting + n_si
        
        # Initialize sparse matrix components
        rows, cols, vals = [], [], []
        data_vector = []
        uncertainties = []
        
        row_idx = 0
        
        for obs in self.observations:
            azimuth_rad = np.deg2rad(obs['azimuth'])
            c2a = np.cos(2 * azimuth_rad)
            s2a = np.sin(2 * azimuth_rad)
            
            if obs['type'] == 'splitting':
                # Convert (δt, φ) to two linear components
                dt = obs['delta_t']
                phi_rad = np.deg2rad(obs['phi'])
                c2p = np.cos(2 * phi_rad)
                s2p = np.sin(2 * phi_rad)
                
                # First row: δt * cos(2φ) component
                data_vector.append(dt * c2p)
                uncertainties.append(obs['sigma_dt'])
                
                # Second row: δt * sin(2φ) component  
                data_vector.append(dt * s2p)
                uncertainties.append(obs['sigma_dt'])
                
                # Fill design matrix entries for both rows
                for cell_idx, Lij in obs['Lij'].items():
                    col_mc = 2 * cell_idx      # Mc component
                    col_ms = 2 * cell_idx + 1  # Ms component
                    
                    # Get velocity for this cell to make G dimensionally correct
                    # G entries should be Lij/V (seconds) so model is dimensionless
                    cell_i = cell_idx // (self.ny * self.nz)
                    remainder = cell_idx % (self.ny * self.nz)
                    cell_j = remainder // self.nz
                    cell_k = remainder % self.nz
                    velocity_cell = self.vm.velocity[cell_i, cell_j, cell_k]  # km/s
                    
                    G_entry = Lij / velocity_cell  # Path length / velocity = traveltime (seconds)
                    
                    # First row contributions
                    rows.extend([row_idx, row_idx])
                    cols.extend([col_mc, col_ms]) 
                    vals.extend([G_entry * c2a * c2p, G_entry * s2a * c2p])
                    
                    # Second row contributions
                    rows.extend([row_idx + 1, row_idx + 1])
                    cols.extend([col_mc, col_ms])
                    vals.extend([G_entry * c2a * s2p, G_entry * s2a * s2p])
                
                row_idx += 2
                
            elif obs['type'] == 'si':
                # Single row for SI measurement
                si_value = obs['SI']
                s_factor = obs['s_factor']
                
                data_vector.append(si_value)
                uncertainties.append(obs['sigma_si'])
                
                # Fill design matrix entries
                for cell_idx, Lkj in obs['Lij'].items():
                    col_mc = 2 * cell_idx
                    col_ms = 2 * cell_idx + 1
                    
                    rows.extend([row_idx, row_idx])
                    cols.extend([col_mc, col_ms])
                    vals.extend([Lkj * s_factor * c2a, Lkj * s_factor * s2a])
                
                row_idx += 1
        
        # Build sparse matrix
        G = sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, self.n_params))
        
        # Store data vector and uncertainties
        self.data_vector = np.array(data_vector)
        self.uncertainties = np.array(uncertainties)
        
        return G
    
    def build_regularization_matrix(self) -> sp.csr_matrix:
        """
        Build spatial regularization matrix for smoothness constraints.
        
        Returns:
        --------
        R : scipy.sparse.csr_matrix
            Regularization matrix penalizing roughness
        """
        rows, cols, vals = [], [], []
        row_idx = 0
        
        # Helper to convert 3D indices to cell index
        def cell_index(i, j, k):
            return i * self.ny * self.nz + j * self.nz + k
        
        # Horizontal smoothing (X and Y directions)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    center_idx = cell_index(i, j, k)
                    
                    # X-direction smoothing
                    if i < self.nx - 1:
                        neighbor_idx = cell_index(i + 1, j, k)
                        weight = self.reg_params['smooth_horizontal']
                        
                        # Smoothing for both Mc and Ms components
                        for comp in [0, 1]:  # 0=Mc, 1=Ms
                            center_col = 2 * center_idx + comp
                            neighbor_col = 2 * neighbor_idx + comp
                            
                            rows.extend([row_idx, row_idx])
                            cols.extend([center_col, neighbor_col])
                            vals.extend([weight, -weight])
                            row_idx += 1
                    
                    # Y-direction smoothing  
                    if j < self.ny - 1:
                        neighbor_idx = cell_index(i, j + 1, k)
                        weight = self.reg_params['smooth_horizontal']
                        
                        for comp in [0, 1]:
                            center_col = 2 * center_idx + comp
                            neighbor_col = 2 * neighbor_idx + comp
                            
                            rows.extend([row_idx, row_idx])
                            cols.extend([center_col, neighbor_col])
                            vals.extend([weight, -weight])
                            row_idx += 1
        
        # Vertical smoothing (Z direction)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz - 1):
                    center_idx = cell_index(i, j, k)
                    neighbor_idx = cell_index(i, j, k + 1)
                    weight = self.reg_params['smooth_vertical']
                    
                    for comp in [0, 1]:
                        center_col = 2 * center_idx + comp
                        neighbor_col = 2 * neighbor_idx + comp
                        
                        rows.extend([row_idx, row_idx])
                        cols.extend([center_col, neighbor_col])
                        vals.extend([weight, -weight])
                        row_idx += 1
        
        # Add damping terms (diagonal regularization)
        lambda_damp = self.reg_params['lambda_damp']
        for param_idx in range(self.n_params):
            rows.append(row_idx)
            cols.append(param_idx)
            vals.append(lambda_damp)
            row_idx += 1
        
        n_reg_rows = row_idx
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_reg_rows, self.n_params))
        
        return R
    
    def solve_inversion(self, lambda_reg: Optional[float] = None, 
                       solver: str = 'spsolve') -> np.ndarray:
        """
        Solve the regularized linear least-squares problem.
        
        Parameters:
        -----------
        lambda_reg : float, optional
            Regularization parameter (auto-selected if None)
        solver : str
            Solver type: 'spsolve', 'cg', 'lsqr'
            
        Returns:
        --------
        model : np.ndarray
            Estimated model parameters [Mc_1, Ms_1, Mc_2, Ms_2, ...]
        """
        print("Building design matrix...")
        G = self.build_design_matrix()
        self.design_matrix = G
        
        print("Building regularization matrix...")  
        R = self.build_regularization_matrix()
        self.regularization_matrix = R
        
        print("Building weight matrix...")
        # Weight matrix with normalization for different observation types
        # Splitting observations contribute 2 rows, SI observations contribute 1 row
        # Normalize so each observation (not row) is weighted equally
        
        n_splitting = sum(1 for obs in self.observations if obs['type'] == 'splitting')
        n_si = sum(1 for obs in self.observations if obs['type'] == 'si')
        
        # Base weights from uncertainties
        W_diag = 1.0 / np.maximum(self.uncertainties, 1e-12)
        
        # Normalize by observation type
        # Splitting: 2 rows per observation, so divide by sqrt(2) to normalize
        # SI: 1 row per observation, keep as is
        row_idx = 0
        for obs in self.observations:
            if obs['type'] == 'splitting':
                # Two rows for this observation - normalize by sqrt(2)
                W_diag[row_idx] /= np.sqrt(2.0)
                W_diag[row_idx + 1] /= np.sqrt(2.0)
                row_idx += 2
            elif obs['type'] == 'si':
                # One row for this observation - no additional normalization needed
                row_idx += 1
        
        W = sp.diags(W_diag)
        self.weight_matrix = W
        
        print(f"  Splitting observations: {n_splitting} (2 rows each, normalized by √2)")
        print(f"  SI observations: {n_si} (1 row each)")
        print(f"  Total data rows: {len(W_diag)}")
        
        # Auto-select regularization parameter if not provided
        if lambda_reg is None:
            print("Auto-selecting regularization parameter...")
            lambda_reg = self._select_regularization_parameter(G, R, W)
            print(f"Selected λ = {lambda_reg:.2e}")
        
        # Form normal equations: (G^T W^2 G + λ^2 R^T R) m = G^T W^2 d
        print("Assembling normal equations...")
        GTW = G.T.dot(W)
        GTWG = GTW.dot(W.dot(G))
        RTR = R.T.dot(R)
        
        A = GTWG + (lambda_reg**2) * RTR
        b = GTW.dot(W.dot(self.data_vector))
        
        print(f"System size: {A.shape[0]} parameters, {len(self.data_vector)} observations")
        print(f"Matrix sparsity: {A.nnz / A.shape[0]**2 * 100:.1f}%")
        
        # Solve system
        print(f"Solving with {solver}...")
        if solver == 'spsolve':
            try:
                model = spla.spsolve(A, b)
            except Exception as e:
                print(f"Direct solver failed: {e}")
                print("Falling back to iterative solver...")
                model, info = spla.cg(A, b, tol=1e-8, maxiter=5000)
                if info != 0:
                    warnings.warn(f"Iterative solver convergence issue: info={info}")
        elif solver == 'cg':
            model, info = spla.cg(A, b, tol=1e-8, maxiter=5000)
            if info != 0:
                warnings.warn(f"CG solver convergence issue: info={info}")
        elif solver == 'lsqr':
            model = spla.lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=5000)[0]
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        self.model_estimate = model
        
        # Compute inversion statistics
        residuals = G.dot(model) - self.data_vector
        weighted_residuals = W.dot(residuals)
        rms_residual = np.sqrt(np.mean(weighted_residuals**2))
        
        self.inversion_stats = {
            'lambda_reg': lambda_reg,
            'rms_residual': rms_residual,
            'n_observations': len(self.data_vector),
            'n_parameters': len(model),
            'solver': solver,
        }
        
        print(f"Inversion complete!")
        print(f"  RMS weighted residual: {rms_residual:.3f}")
        print(f"  Regularization λ: {lambda_reg:.2e}")
        
        return model
    
    def _select_regularization_parameter(self, G: sp.csr_matrix, R: sp.csr_matrix, 
                                       W: sp.csr_matrix) -> float:
        """
        Auto-select regularization parameter using L-curve method.
        
        Parameters:
        -----------
        G, R, W : scipy.sparse matrices
            Design, regularization, and weight matrices
            
        Returns:
        --------
        lambda_opt : float
            Optimal regularization parameter
        """
        # Test range of lambda values (log scale)
        lambda_range = np.logspace(-4, 2, 20)
        misfit_norms = []
        model_norms = []
        
        GTW = G.T.dot(W)
        GTWG = GTW.dot(W.dot(G))
        RTR = R.T.dot(R)
        b = GTW.dot(W.dot(self.data_vector))
        
        for lam in lambda_range:
            try:
                A = GTWG + (lam**2) * RTR
                model = spla.spsolve(A, b)
                
                # Compute norms
                residuals = G.dot(model) - self.data_vector
                misfit_norm = np.linalg.norm(W.dot(residuals))
                model_norm = np.linalg.norm(R.dot(model))
                
                misfit_norms.append(misfit_norm)
                model_norms.append(model_norm)
            except:
                misfit_norms.append(np.inf)
                model_norms.append(np.inf)
        
        # Find L-curve corner (maximum curvature)
        valid_idx = np.isfinite(misfit_norms) & np.isfinite(model_norms)
        if not np.any(valid_idx):
            return 1.0  # Default fallback
        
        x = np.log10(np.array(misfit_norms)[valid_idx])
        y = np.log10(np.array(model_norms)[valid_idx])
        
        # Compute curvature (simplified method)
        if len(x) < 3:
            return lambda_range[len(lambda_range)//2]  # Middle value
        
        curvature = []
        for i in range(1, len(x)-1):
            dx1, dy1 = x[i] - x[i-1], y[i] - y[i-1]
            dx2, dy2 = x[i+1] - x[i], y[i+1] - y[i]
            
            # Curvature approximation
            k = abs(dx1*dy2 - dy1*dx2) / ((dx1**2 + dy1**2)**1.5)
            curvature.append(k)
        
        if curvature:
            max_curv_idx = np.argmax(curvature) + 1
            valid_lambdas = lambda_range[valid_idx]
            lambda_opt = valid_lambdas[max_curv_idx]
        else:
            lambda_opt = lambda_range[len(lambda_range)//2]
        
        return lambda_opt
    
    def get_anisotropy_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract anisotropy strength and fast direction from model estimate.
        
        Returns:
        --------
        anisotropy_strength : np.ndarray  
            3D array of anisotropy strength (percentage)
        fast_direction : np.ndarray
            3D array of fast directions (degrees)
        """
        if self.model_estimate is None:
            raise ValueError("Must run inversion first")
        
        # Extract Mc and Ms components
        Mc = self.model_estimate[0::2].reshape(self.nx, self.ny, self.nz)
        Ms = self.model_estimate[1::2].reshape(self.nx, self.ny, self.nz)
        
        # Convert to physical parameters
        # A is dimensionless (fractional velocity difference), no velocity scaling needed
        anisotropy_strength, fast_direction = self.convert_linear_params(Mc, Ms)
        
        return anisotropy_strength, fast_direction
    
    def compute_resolution_matrix(self) -> np.ndarray:
        """
        Compute model resolution matrix for assessing parameter recovery.
        
        Returns:
        --------
        resolution : np.ndarray
            Model resolution matrix R = (G^T G + λ^2 R^T R)^(-1) G^T G
        """
        if self.design_matrix is None:
            raise ValueError("Must run inversion first")
        
        G = self.design_matrix
        R = self.regularization_matrix
        W = self.weight_matrix
        lambda_reg = self.inversion_stats['lambda_reg']
        
        # Build matrices
        GTW = G.T.dot(W)
        GTWG = GTW.dot(W.dot(G))
        RTR = R.T.dot(R)
        
        A = GTWG + (lambda_reg**2) * RTR
        
        try:
            # Resolution matrix: R = A^(-1) * G^T W^2 G
            A_inv_GTWG = spla.spsolve(A, GTWG.toarray())
            resolution = A_inv_GTWG
        except:
            print("Warning: Could not compute full resolution matrix (memory/computational limits)")
            # Compute diagonal only as approximation
            A_diag = A.diagonal()
            GTWG_diag = GTWG.diagonal()
            resolution_diag = GTWG_diag / A_diag
            resolution = np.diag(resolution_diag)
        
        self.resolution_matrix = resolution
        return resolution
    
    def synthetic_test(self, true_anisotropy: np.ndarray, true_fast_dir: np.ndarray,
                      noise_level: float = 0.05) -> Dict:
        """
        Perform synthetic recovery test.
        
        Parameters:
        -----------
        true_anisotropy : np.ndarray
            True anisotropy strength field
        true_fast_dir : np.ndarray
            True fast direction field (degrees)
        noise_level : float
            Relative noise level for synthetic data
            
        Returns:
        --------
        results : dict
            Synthetic test results including recovery statistics
        """
        # Convert true model to linear parameters
        Mc_true, Ms_true = self.convert_anisotropy_params(true_anisotropy.flatten(), 
                                                         true_fast_dir.flatten())
        true_model = np.zeros(self.n_params)
        true_model[0::2] = Mc_true
        true_model[1::2] = Ms_true
        
        # Generate synthetic data
        G = self.design_matrix
        if G is None:
            raise ValueError("Must set up observations first")
        
        synthetic_data = G.dot(true_model)
        
        # Add noise
        noise_std = noise_level * np.abs(synthetic_data)
        noise_std = np.maximum(noise_std, 0.01 * np.max(np.abs(synthetic_data)))
        noise = np.random.normal(0, noise_std)
        noisy_data = synthetic_data + noise
        
        # Store original data and replace with synthetic
        original_data = self.data_vector.copy()
        original_uncertainties = self.uncertainties.copy()
        
        self.data_vector = noisy_data
        self.uncertainties = noise_std
        
        # Run inversion
        recovered_model = self.solve_inversion()
        
        # Restore original data
        self.data_vector = original_data
        self.uncertainties = original_uncertainties
        
        # Compute recovery statistics
        model_error = recovered_model - true_model
        rms_error = np.sqrt(np.mean(model_error**2))
        correlation = np.corrcoef(recovered_model, true_model)[0, 1]
        
        # Convert back to physical parameters
        Mc_recovered = recovered_model[0::2].reshape(self.nx, self.ny, self.nz)
        Ms_recovered = recovered_model[1::2].reshape(self.nx, self.ny, self.nz)
        aniso_recovered, fast_recovered = self.convert_linear_params(Mc_recovered, Ms_recovered)
        
        results = {
            'true_model': true_model,
            'recovered_model': recovered_model,
            'rms_error': rms_error,
            'correlation': correlation,
            'noise_level': noise_level,
            'anisotropy_recovered': aniso_recovered,
            'fast_direction_recovered': fast_recovered,
        }
        
        return results


def create_checkerboard_model(nx: int, ny: int, nz: int, 
                             checker_size: int = 3,
                             max_anisotropy: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic checkerboard anisotropy model for testing.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Grid dimensions
    checker_size : int
        Size of checkerboard squares  
    max_anisotropy : float
        Maximum anisotropy percentage
        
    Returns:
    --------
    anisotropy : np.ndarray
        3D anisotropy strength field
    fast_direction : np.ndarray
        3D fast direction field (degrees)
    """
    anisotropy = np.zeros((nx, ny, nz))
    fast_direction = np.zeros((nx, ny, nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Checkerboard pattern
                check_i = (i // checker_size) % 2
                check_j = (j // checker_size) % 2
                check_k = (k // checker_size) % 2
                
                if (check_i + check_j + check_k) % 2 == 0:
                    anisotropy[i, j, k] = max_anisotropy
                    fast_direction[i, j, k] = 45.0  # NE-SW
                else:
                    anisotropy[i, j, k] = max_anisotropy * 0.5
                    fast_direction[i, j, k] = 135.0  # NW-SE
    
    return anisotropy, fast_direction


# Example usage and testing functions
if __name__ == "__main__":
    print("Global Anisotropy Inversion Module")
    print("This module provides sophisticated linear least-squares inversion")
    print("for seismic anisotropy using splitting parameters and intensity.")