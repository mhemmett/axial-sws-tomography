"""
Ray Tracing for Axial Seamount

This module provides ray tracing functionality using PyKonal for 3D velocity models.
Used in shear-wave splitting analysis to compute ray paths and path lengths.

Author: Michael Hemmett
Date: December 2025
"""

import numpy as np
import pykonal

if not hasattr(np, "infty"):
    np.infty = float("inf")


class RayTracer:
    """
    Ray tracing for computing ray paths through velocity models.
    
    This class computes seismic ray paths using PyKonal's eikonal solver.
    Used in shear-wave splitting analysis to compute path lengths through
    each model cell for building the design matrix.
    
    Parameters
    ----------
    velocity_model : AxialVelocityModel
        3D velocity model
    """
    
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model
        # No persistent solver - create fresh instance for each ray trace
    
    def _create_fresh_solver(self):
        """Create a fresh PyKonal solver instance for ray tracing.
        
        This avoids state contamination from previous solve() calls
        that can cause NaN values in subsequent ray traces.
        
        Returns
        -------
        solver : pykonal.EikonalSolver
            Fresh solver instance configured with velocity model
        """
        vm = self.velocity_model
        
        if vm.velocity is None:
            raise ValueError("Velocity model not created.")
        
        # Create fresh PyKonal solver
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        
        # Define grid
        solver.velocity.min_coords = vm.x_min, vm.y_min, vm.z_min
        solver.velocity.node_intervals = (
            (vm.x_max - vm.x_min) / (vm.nx - 1),
            (vm.y_max - vm.y_min) / (vm.ny - 1),
            (vm.z_max - vm.z_min) / (vm.nz - 1)
        )
        solver.velocity.npts = vm.nx, vm.ny, vm.nz
        solver.velocity.values = vm.velocity
        
        return solver
    
    def compute_ray_path(self, source_loc, receiver_loc, npts=100):
        """
        Compute ray path from source to receiver using PyKonal's trace_ray.
        
        Traces the ray by following the gradient of the traveltime field
        from receiver back to source, then resamples for smooth visualization.
        
        IMPORTANT: PyKonal's methods require different coordinate systems:
        - trial.push() expects grid indices (integers)
        - trace_ray() expects physical coordinates (floats in km)
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km (physical coordinates)
        receiver_loc : array-like
            Receiver location [x, y, z] in km (physical coordinates)
        npts : int
            Number of points for final resampled ray path (default: 100)
        
        Returns
        -------
        ray_path : ndarray
            Array of shape (npts, 3) with [x, y, z] along ray from source to receiver
        """
        vm = self.velocity_model
        
        # Convert source to grid indices for solver initialization
        src_idx = (
            int((source_loc[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((source_loc[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((source_loc[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Ensure source indices within bounds
        src_idx = (
            max(0, min(src_idx[0], vm.nx - 1)),
            max(0, min(src_idx[1], vm.ny - 1)),
            max(0, min(src_idx[2], vm.nz - 1))
        )
        
        # Convert receiver to numpy array with physical coordinates for trace_ray
        # NOTE: trace_ray expects physical coordinates (km), not grid indices!
        rec_coords = np.array(receiver_loc, dtype=np.float64)
        
        # Create fresh solver to avoid state contamination
        solver = self._create_fresh_solver()
        
        # Initialize solver state
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        
        # Set source location (uses grid indices)
        solver.traveltime.values[src_idx] = 0.0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)
        solver.solve()
        
        try:
            # Use PyKonal's native trace_ray method
            # IMPORTANT: Pass physical coordinates (km), not grid indices!
            ray = solver.traveltime.trace_ray(rec_coords)
            
            # Validate ray
            if len(ray) < 2:
                raise ValueError(f"Ray tracing returned too few points: {len(ray)}")
            
            if not np.all(np.isfinite(ray)):
                raise ValueError("Ray path contains NaN or infinite values")
            
            # PyKonal traces from receiver to source, so reverse it
            ray = ray[::-1]
            
            # Resample to exactly npts points with uniform spacing along the ray
            distances = np.zeros(len(ray))
            for i in range(1, len(ray)):
                distances[i] = distances[i-1] + np.linalg.norm(ray[i] - ray[i-1])
            
            total_distance = distances[-1]
            
            if total_distance > 0:
                # Create uniform spacing
                t_new = np.linspace(0, total_distance, npts)
                
                # Interpolate ray coordinates
                ray_resampled = np.column_stack([
                    np.interp(t_new, distances, ray[:, 0]),
                    np.interp(t_new, distances, ray[:, 1]),
                    np.interp(t_new, distances, ray[:, 2])
                ])
                
                return ray_resampled
            else:
                return ray
                
        except (ValueError, RuntimeError, IndexError, AttributeError) as e:
            # Fallback to straight line if ray tracing fails
            print(f"Warning: Ray tracing failed ({str(e)}), using straight line")
            t = np.linspace(0, 1, npts)
            ray = np.outer(1-t, source_loc) + np.outer(t, receiver_loc)
            return ray
    
    def compute_travel_time(self, source_loc, receiver_loc):
        """
        Compute travel time from source to receiver.
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km
        receiver_loc : array-like
            Receiver location [x, y, z] in km
        
        Returns
        -------
        travel_time : float
            Travel time in seconds
        """
        vm = self.velocity_model
        
        # Convert source to grid indices
        src_idx = (
            int((source_loc[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((source_loc[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((source_loc[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Ensure indices within bounds
        src_idx = (
            max(0, min(src_idx[0], vm.nx - 1)),
            max(0, min(src_idx[1], vm.ny - 1)),
            max(0, min(src_idx[2], vm.nz - 1))
        )
        
        # Convert receiver to grid indices
        rec_idx = (
            int((receiver_loc[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((receiver_loc[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((receiver_loc[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Ensure indices within bounds
        rec_idx = (
            max(0, min(rec_idx[0], vm.nx - 1)),
            max(0, min(rec_idx[1], vm.ny - 1)),
            max(0, min(rec_idx[2], vm.nz - 1))
        )
        
        # Create fresh solver
        solver = self._create_fresh_solver()
        
        # Initialize solver state
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        
        # Set source
        solver.traveltime.values[src_idx] = 0.0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)
        
        # Solve
        solver.solve()
        
        # Get travel time at receiver
        travel_time = solver.traveltime.values[rec_idx]
        
        return travel_time

def compute_cell_path_lengths(ray_path, vm):
    """
    Compute actual path lengths (km) through each cell from ray coordinates.
    
    Parameters:
    -----------
    ray_path : np.ndarray
        Array of shape (npts, 3) with [x, y, z] coordinates in km
    vm : AxialVelocityModel
        Velocity model with grid information
    
    Returns:
    --------
    Lij : dict
        Dictionary mapping cell_idx -> path_length_km
    """
    Lij = {}
    
    # Process ray segments between consecutive points
    for i in range(len(ray_path) - 1):
        p1, p2 = ray_path[i], ray_path[i+1]
        
        # Segment midpoint for cell identification
        mid = (p1 + p2) / 2.0
        
        # Convert to grid indices
        cell_i = int((mid[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1))
        cell_j = int((mid[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1))
        cell_k = int((mid[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        
        # Ensure within bounds
        if (0 <= cell_i < vm.nx and 0 <= cell_j < vm.ny and 0 <= cell_k < vm.nz):
            cell_idx = cell_i * vm.ny * vm.nz + cell_j * vm.nz + cell_k
            
            # Compute segment length in km
            segment_length = np.linalg.norm(p2 - p1)
            
            # Accumulate path length for this cell
            Lij[cell_idx] = Lij.get(cell_idx, 0.0) + segment_length
    
    return Lij

print("✓ Helper function compute_cell_path_lengths() defined")
print("  Computes actual path lengths (km) from ray coordinates")