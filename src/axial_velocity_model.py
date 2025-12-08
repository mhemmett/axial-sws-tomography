"""
3D Velocity Model for Axial Seamount

This module creates realistic 3D heterogeneous velocity models for Axial Seamount,
including geological features such as the caldera, magma chamber, ring faults, and
hydrothermal outflow zones.

Author: Michael Hemmett
Date: December 2025
"""

import numpy as np


class AxialVelocityModel:
    """
    3D velocity model for Axial Seamount with realistic geological structure.
    
    Parameters
    ----------
    nx, ny, nz : int
        Number of grid points in x, y, z directions
    x_range : tuple
        (x_min, x_max) in km, centered on caldera
    y_range : tuple
        (y_min, y_max) in km, centered on caldera
    z_range : tuple
        (z_min, z_max) in km, depth positive downward
    
    Attributes
    ----------
    x, y, z : ndarray
        1D coordinate arrays
    X, Y, Z : ndarray
        3D meshgrid arrays
    velocity : ndarray
        3D velocity model in km/s
    """
    
    def __init__(self, nx=101, ny=101, nz=51,
                 x_range=(-16.0, 16.0), y_range=(-16.0, 16.0), z_range=(0.0, 10.0)):
        
        self.nx, self.ny, self.nz = nx, ny, nz
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        # Create coordinate arrays
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.y = np.linspace(self.y_min, self.y_max, ny)
        self.z = np.linspace(self.z_min, self.z_max, nz)
        
        # Create 3D meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Initialize velocity model
        self.velocity = None
        
        # Initialize anisotropic velocity models
        self.velocity_fast = None
        self.velocity_slow = None
        self.phi_fast = None
        self.percent_anisotropy = None
        
    def create_model(self, v0_seafloor=2.0, velocity_gradient=0.3,
                    caldera_center=(0.0, 0.0), caldera_width=3.0, caldera_length=8.0,
                    caldera_angle=15.0, magma_chamber_depth=(1.1, 2.3),
                    mmr_east_offset=0.7):
        """
        Create Axial Seamount velocity model with geological features based on
        recent seismic tomography and geological studies (Yang et al. 2024, 
        Lee et al. 2024, Carbotte et al. 2020).
        
        Key features:
        - Oblong caldera with N-NE trending orientation
        - Eastern-offset main magma reservoir (MMR) with prolate-spheroid geometry
        - High-velocity ponded lava core beneath caldera floor
        - Inward and outward-dipping ring fault systems
        - Stacked sills extending to ~5 km depth
        
        Parameters
        ----------
        v0_seafloor : float
            Base seafloor velocity in km/s (typical < 3 km/s)
        velocity_gradient : float
            Velocity increase with depth (km/s per km)
        caldera_center : tuple
            (x, y) position of caldera center in km
        caldera_width : float
            E-W width of caldera in km
        caldera_length : float
            N-S length of caldera in km
        caldera_angle : float
            Rotation angle of caldera from N in degrees (N-NE trending, ~15°)
        magma_chamber_depth : tuple
            (top, bottom) depth range of main magma reservoir in km
            Based on seismic imaging: top at 1.1-2.3 km, thickness 0.6-1 km
        mmr_east_offset : float
            Eastward offset of MMR center from caldera center in km
        
        Returns
        -------
        velocity : ndarray
            3D velocity model in km/s
        
        References
        ----------
        Yang et al. (2024): FWI imaging showing asymmetric magma plumbing
        Lee et al. (2024): Ambient noise tomography and MMR location
        Carbotte et al. (2020): Stacked sills as melt-mush feeder conduit
        """
        
        # Base velocity structure (increases with depth)
        # Most velocities < 3 km/s in shallow layers (Yang et al. 2024)
        velocity_3d = v0_seafloor + velocity_gradient * self.Z
        
        # Caldera parameters
        cx, cy = caldera_center
        theta = np.radians(caldera_angle)
        
        # MMR parameters (eastern offset from caldera center)
        mmr_cx = cx + mmr_east_offset * np.cos(theta)
        mmr_cy = cy + mmr_east_offset * np.sin(theta)
        mag_top, mag_bot = magma_chamber_depth
        mag_center = (mag_top + mag_bot) / 2
        mag_thickness = mag_bot - mag_top
        
        # Add geological features
        for i in range(self.nx):
            for j in range(self.ny):
                # Rotate coordinates to align with caldera orientation
                x_rot = (self.X[i, j, 0] - cx) * np.cos(theta) + (self.Y[i, j, 0] - cy) * np.sin(theta)
                y_rot = -(self.X[i, j, 0] - cx) * np.sin(theta) + (self.Y[i, j, 0] - cy) * np.cos(theta)
                
                # Elliptical distance from caldera center
                caldera_dist = np.sqrt((x_rot / (caldera_width/2))**2 + (y_rot / (caldera_length/2))**2)
                
                # Rotate coordinates relative to MMR center (eastern offset)
                x_mmr = (self.X[i, j, 0] - mmr_cx) * np.cos(theta) + (self.Y[i, j, 0] - mmr_cy) * np.sin(theta)
                y_mmr = -(self.X[i, j, 0] - mmr_cx) * np.sin(theta) + (self.Y[i, j, 0] - mmr_cy) * np.cos(theta)
                
                # Distance from MMR center
                mmr_horizontal_dist = np.sqrt(x_mmr**2 + y_mmr**2)
                
                for k in range(self.nz):
                    depth = self.Z[i, j, k]
                    
                    # HIGH-VELOCITY PONDED LAVA CORE (Yang et al. 2024)
                    # 1.2 km wide × 0.2 km thick at 246m depth (Vp ~3.2 km/s)
                    ponded_lava_depth = 0.246
                    ponded_lava_thickness = 0.2
                    ponded_lava_radius = 0.6  # half-width = 0.6 km
                    
                    if (caldera_dist < 0.5 and 
                        depth >= ponded_lava_depth and 
                        depth <= ponded_lava_depth + ponded_lava_thickness):
                        # High-velocity anomaly: +0.4 km/s above background
                        ponded_effect = 0.4 * np.exp(-((depth - ponded_lava_depth - ponded_lava_thickness/2)/(ponded_lava_thickness/2))**2)
                        velocity_3d[i, j, k] += ponded_effect
                    
                    # INWARD-DIPPING RING FAULTS (narrow low-velocity zones)
                    # Bound the high-velocity core, dip inward to caldera center
                    # Located at caldera walls, reach ~1.6 km depth
                    ring_fault_inner = 0.85  # Inner edge of ring fault zone
                    ring_fault_outer = 1.05  # Outer edge of ring fault zone
                    ring_fault_width = 0.1   # Narrow zone (~100-200m)
                    
                    if (ring_fault_inner <= caldera_dist <= ring_fault_outer and 
                        depth <= 1.6 and y_rot > -caldera_length/4):
                        # Inward dip: velocity reduction increases slightly toward center
                        inward_dip_factor = 1.0 - 0.3 * (caldera_dist - ring_fault_inner) / ring_fault_width
                        fault_reduction = 0.6 * inward_dip_factor * np.exp(-depth/1.2)
                        velocity_3d[i, j, k] -= fault_reduction
                    
                    # OUTWARD-DIPPING RING FAULTS (eastern and western caldera walls)
                    # Dip outward, accommodate inflation/deflation
                    if (ring_fault_outer <= caldera_dist <= 1.2 and 
                        depth <= 1.6 and 
                        (abs(x_rot) > caldera_width/3)):  # Eastern and western walls
                        # Outward dip: velocity reduction decreases outward
                        outward_dip_factor = np.exp(-((caldera_dist - ring_fault_outer)/0.3)**2)
                        fault_reduction = 0.5 * outward_dip_factor * np.exp(-depth/1.2)
                        velocity_3d[i, j, k] -= fault_reduction
                    
                    # CENTRAL CALDERA FLOOR (fractured, subsided)
                    if caldera_dist < 0.7 and depth > ponded_lava_depth + ponded_lava_thickness and depth <= 2.0:
                        # Moderately fractured below ponded lava
                        velocity_reduction = 0.8 * np.exp(-depth/1.5)
                        velocity_3d[i, j, k] -= velocity_reduction
                    
                    # MAIN MAGMA RESERVOIR (MMR) - Prolate spheroid shape
                    # Eastern offset, slight northwestward dip
                    # Thickness: 0.6-1 km (mag_thickness)
                    # Top: 1.1-2.3 km depth
                    # Melt fraction: 39-65% → strong velocity reduction
                    
                    if depth >= mag_top and depth <= mag_bot:
                        # Prolate spheroid: elongated vertically, elliptical horizontally
                        # Semi-axes: horizontal ~1.5 km, vertical ~mag_thickness/2
                        horizontal_extent = 1.5  # km
                        vertical_half_height = mag_thickness / 2.0
                        
                        # Distance from MMR center in prolate spheroid coordinates
                        normalized_horizontal = mmr_horizontal_dist / horizontal_extent
                        normalized_vertical = (depth - mag_center) / vertical_half_height
                        spheroid_dist = np.sqrt(normalized_horizontal**2 + normalized_vertical**2)
                        
                        if spheroid_dist <= 1.0:
                            # Strong velocity reduction for high melt fraction
                            # Peak reduction at center, decreases toward edges
                            melt_effect = 1.8 * (1.0 - spheroid_dist**2)
                            velocity_3d[i, j, k] -= melt_effect
                    
                    # STACKED SILLS (Carbotte et al. 2020)
                    # Extend from MMR (~2 km) to ~5 km depth
                    # Vertically stacked melt lenses with ~300-450m spacing
                    # Act as pathways for magma supply to MMR
                    if depth > mag_bot and depth <= 5.0 and mmr_horizontal_dist < 2.5:
                        # Create stacked sills with quasi-periodic spacing
                        sill_spacing = 0.375  # km (average of 300-450m)
                        sill_thickness = 0.1   # km (~100m)
                        
                        # Compute which sill layer we're in
                        depth_below_mmr = depth - mag_bot
                        sill_index = depth_below_mmr / sill_spacing
                        distance_to_sill_center = abs((sill_index - np.round(sill_index)) * sill_spacing)
                        
                        # If near a sill center, apply velocity reduction
                        if distance_to_sill_center < sill_thickness:
                            # Velocity reduction decreases with distance from MMR
                            distance_factor = np.exp(-(mmr_horizontal_dist/2.0)**2)
                            thickness_factor = np.exp(-(distance_to_sill_center/(sill_thickness/2))**2)
                            sill_effect = 0.8 * distance_factor * thickness_factor
                            velocity_3d[i, j, k] -= sill_effect
                    
                    # HYDROTHERMAL OUTFLOW ZONES
                    # Northern outflow zone
                    north_dist = np.sqrt((x_rot/1.0)**2 + ((y_rot - caldera_length/3)/4.0)**2)
                    if north_dist <= 1.0 and depth <= 3.0:
                        outflow_effect = 0.6 * np.exp(-depth/2.0) * np.exp(-north_dist**2)
                        velocity_3d[i, j, k] -= outflow_effect
                    
                    # Southern outflow (through horseshoe opening)
                    south_dist = np.sqrt((x_rot/0.8)**2 + ((y_rot + caldera_length/4)/3.0)**2)
                    if south_dist <= 1.0 and depth <= 2.0 and y_rot < -caldera_length/6:
                        outflow_effect = 0.4 * np.exp(-depth/1.5) * np.exp(-south_dist**2)
                        velocity_3d[i, j, k] -= outflow_effect
        
        # Add regional volcanic structure (broader low-velocity zone)
        for i in range(self.nx):
            for j in range(self.ny):
                regional_dist = np.sqrt(self.X[i, j, 0]**2 + self.Y[i, j, 0]**2)
                if regional_dist <= 12.0:
                    regional_reduction = 0.5 * np.exp(-self.Z[i, j, :]/5.0) * np.exp(-(regional_dist/12.0)**2)
                    velocity_3d[i, j, :] -= regional_reduction
        
        # Add seafloor layer (constant low velocity)
        seafloor_layer_thickness = 0.5  # km
        seafloor_velocity = 1.8  # km/s
        for i in range(self.nx):
            for j in range(self.ny):
                regional_dist = np.sqrt(self.X[i, j, 0]**2 + self.Y[i, j, 0]**2)
                if regional_dist > 12.0:
                    for k in range(self.nz):
                        depth = self.Z[i, j, k]
                        if depth <= seafloor_layer_thickness:
                            velocity_3d[i, j, k] = seafloor_velocity
        
        self.velocity = velocity_3d
        return velocity_3d
    
    def get_velocity_at_point(self, x, y, z):
        """
        Get velocity at a specific point via interpolation.
        
        Parameters
        ----------
        x, y, z : float or array-like
            Coordinates in km
        
        Returns
        -------
        velocity : float or ndarray
            Velocity at the specified point(s) in km/s
        """
        from scipy.interpolate import RegularGridInterpolator
        
        if self.velocity is None:
            raise ValueError("Velocity model not created. Call create_model() first.")
        
        interpolator = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.velocity,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)])
        return interpolator(points)
    
    def get_slice(self, axis='z', value=0.0):
        """
        Extract a 2D slice from the 3D velocity model.
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        value : float
            Position along the axis for the slice
        
        Returns
        -------
        slice_data : ndarray
            2D slice of velocity model
        coords : tuple
            Coordinate arrays for the slice
        """
        if self.velocity is None:
            raise ValueError("Velocity model not created. Call create_model() first.")
        
        if axis == 'z':
            idx = np.argmin(np.abs(self.z - value))
            return self.velocity[:, :, idx], (self.x, self.y)
        elif axis == 'y':
            idx = np.argmin(np.abs(self.y - value))
            return self.velocity[:, idx, :], (self.x, self.z)
        elif axis == 'x':
            idx = np.argmin(np.abs(self.x - value))
            return self.velocity[idx, :, :], (self.y, self.z)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
    
    def create_anisotropic_model(self, base_percent_anisotropy=4.0, phi_base=37.0,
                                vary_with_structure=True, caldera_center=(0.0, 0.0),
                                mmr_east_offset=0.7, caldera_width=3.0, caldera_length=8.0, 
                                caldera_angle=15.0):
        """
        Create anisotropic velocity model with fast and slow S-wave components.
        
        This method creates two velocity models (fast and slow) that differ by a
        percentage representing crustal anisotropy from aligned microcracks. The
        anisotropy strength can vary spatially, with enhanced anisotropy near
        ring faults where fracturing is intense.
        
        Parameters
        ----------
        base_percent_anisotropy : float
            Base percent velocity difference between fast and slow directions (typical 3-5%)
        phi_base : float
            Base fast polarization direction in degrees from North (clockwise)
            Based on literature values (e.g., 37° at Axial Seamount)
        vary_with_structure : bool
            If True, anisotropy strength varies with geological structure
            (stronger near ring faults, weaker elsewhere)
        caldera_center : tuple
            (x, y) position of caldera center in km
        mmr_east_offset : float
            Eastern offset of MMR center from caldera center in km (default 0.7)
        caldera_width : float
            E-W width of caldera in km
        caldera_length : float
            N-S length of caldera in km
        caldera_angle : float
            Rotation angle of caldera from N in degrees
        
        Returns
        -------
        velocity_fast : ndarray
            3D velocity model for fast S-wave component in km/s
        velocity_slow : ndarray
            3D velocity model for slow S-wave component in km/s
        phi_fast : ndarray
            3D array of fast polarization directions in degrees
        percent_anisotropy : ndarray
            3D array of percent anisotropy at each point
        
        Notes
        -----
        The fast and slow velocities are computed as:
        v_fast = v_isotropic * (1 + percent_anisotropy/200)
        v_slow = v_isotropic * (1 - percent_anisotropy/200)
        
        This ensures the average velocity remains close to the isotropic velocity.
        """
        if self.velocity is None:
            raise ValueError("Isotropic velocity model not created. Call create_model() first.")
        
        # Initialize anisotropy strength field
        percent_anisotropy = np.ones_like(self.velocity) * base_percent_anisotropy
        
        # Initialize fast direction field with spatial variation
        phi_fast = np.ones_like(self.velocity) * phi_base
        
        # Compute azimuthal angle from MMR center (with eastern offset) for tangential orientation
        cx, cy = caldera_center
        mmr_cx = cx + mmr_east_offset  # MMR center is offset eastward
        theta_rot = np.radians(caldera_angle)
        
        if vary_with_structure:
            # Enhance anisotropy near ring faults and fractured zones
            # Use caldera center for ring fault geometry, MMR center for radial patterns
            theta = np.radians(caldera_angle)
            
            for i in range(self.nx):
                for j in range(self.ny):
                    # Rotate coordinates to align with caldera (for ring fault geometry)
                    x_rot = (self.X[i, j, 0] - cx) * np.cos(theta) + (self.Y[i, j, 0] - cy) * np.sin(theta)
                    y_rot = -(self.X[i, j, 0] - cx) * np.sin(theta) + (self.Y[i, j, 0] - cy) * np.cos(theta)
                    
                    # Elliptical distance from caldera center (for ring faults)
                    caldera_dist = np.sqrt((x_rot / (caldera_width/2))**2 + (y_rot / (caldera_length/2))**2)
                    
                    # Distance from MMR center (for radial patterns)
                    x_mmr_rot = (self.X[i, j, 0] - mmr_cx) * np.cos(theta) + (self.Y[i, j, 0] - cy) * np.sin(theta)
                    y_mmr_rot = -(self.X[i, j, 0] - mmr_cx) * np.sin(theta) + (self.Y[i, j, 0] - cy) * np.cos(theta)
                    mmr_dist = np.sqrt((x_mmr_rot / (caldera_width/2))**2 + (y_mmr_rot / (caldera_length/2))**2)
                    
                    for k in range(self.nz):
                        depth = self.Z[i, j, k]
                        
                        # High-velocity ponded lava core (r < 0.6, 0.2-0.4 km): Suppressed anisotropy
                        # Dense solidified lava has reduced microcrack density
                        if caldera_dist < 0.6 and 0.2 <= depth <= 0.4:
                            percent_anisotropy[i, j, k] = base_percent_anisotropy * 0.3  # 1-2% anisotropy
                        
                        # Inward-dipping ring fault zone (0.85 < r < 1.05): Enhanced anisotropy
                        elif 0.85 <= caldera_dist <= 1.05 and depth <= 1.6:
                            # 6-7% anisotropy at inner ring faults (extend to top of MMR at 1.6 km)
                            depth_factor = np.exp(-depth/1.5)  # Decreases with depth
                            percent_anisotropy[i, j, k] = base_percent_anisotropy + 2.5 * depth_factor
                        
                        # Outward-dipping ring fault zone (1.05 < r < 1.2): Enhanced anisotropy
                        # Primarily at eastern and western walls
                        elif 1.05 < caldera_dist <= 1.2 and depth <= 1.6:
                            # Check if near eastern or western walls (within 30° of E-W axis)
                            azimuth_from_caldera = np.degrees(np.arctan2(y_rot, x_rot))
                            near_ew_walls = (abs(azimuth_from_caldera) < 30) or (abs(abs(azimuth_from_caldera) - 180) < 30)
                            if near_ew_walls:
                                # 5-6% anisotropy at outer ring faults
                                depth_factor = np.exp(-depth/1.5)
                                percent_anisotropy[i, j, k] = base_percent_anisotropy + 1.5 * depth_factor
                            else:
                                # Reduced anisotropy elsewhere
                                percent_anisotropy[i, j, k] = base_percent_anisotropy * 0.8
                        
                        # Inside caldera (r < 0.85): Moderate anisotropy
                        elif caldera_dist < 0.85 and depth <= 2.3:
                            depth_factor = np.exp(-depth/1.5)
                            percent_anisotropy[i, j, k] = base_percent_anisotropy * (0.8 + 0.4 * depth_factor)
                        
                        # Outside caldera but near (1.2 < r < 2.0): Reduced anisotropy
                        elif 1.2 < caldera_dist < 2.0 and depth <= 5.0:
                            distance_factor = np.exp(-((caldera_dist - 1.2)/1.0)**2)
                            percent_anisotropy[i, j, k] = base_percent_anisotropy * (0.6 + 0.4 * distance_factor)
                        
                        # Deep or far from caldera: Base anisotropy
                        elif depth > 5.0 or caldera_dist >= 2.0:
                            percent_anisotropy[i, j, k] = base_percent_anisotropy * 0.7
                        
                        # Update fast direction based on structure
                        # Compute azimuth from caldera center (for ring faults)
                        azimuth_caldera = np.degrees(np.arctan2(y_rot, x_rot))
                        # Compute azimuth from MMR center (for radial patterns)
                        azimuth_mmr = np.degrees(np.arctan2(y_mmr_rot, x_mmr_rot))
                        
                        # Inward-dipping ring faults: tangential orientation (perpendicular to radial)
                        if 0.85 <= caldera_dist <= 1.05 and depth <= 1.6:
                            phi_fast[i, j, k] = azimuth_caldera + 90.0  # Tangential to caldera
                            # Normalize to 0-180 range
                            while phi_fast[i, j, k] < 0:
                                phi_fast[i, j, k] += 180.0
                            while phi_fast[i, j, k] >= 180:
                                phi_fast[i, j, k] -= 180.0
                        
                        # Outward-dipping ring faults: tangential orientation at E/W walls
                        elif 1.05 < caldera_dist <= 1.2 and depth <= 1.6:
                            near_ew_walls = (abs(azimuth_caldera) < 30) or (abs(abs(azimuth_caldera) - 180) < 30)
                            if near_ew_walls:
                                phi_fast[i, j, k] = azimuth_caldera + 90.0  # Tangential to caldera
                                # Normalize to 0-180 range
                                while phi_fast[i, j, k] < 0:
                                    phi_fast[i, j, k] += 180.0
                                while phi_fast[i, j, k] >= 180:
                                    phi_fast[i, j, k] -= 180.0
                        
                        # Near magma chamber (shallow): radial pattern from MMR center
                        elif mmr_dist < 1.5 and depth <= 2.3:
                            # Radial orientation from MMR center (accounts for eastern offset)
                            phi_fast[i, j, k] = azimuth_mmr
                            # Normalize to 0-180 range
                            while phi_fast[i, j, k] < 0:
                                phi_fast[i, j, k] += 180.0
                            while phi_fast[i, j, k] >= 180:
                                phi_fast[i, j, k] -= 180.0
                        
                        # Transition zone: blend between radial (from MMR) and regional
                        elif 1.5 <= mmr_dist < 2.5:
                            radial_phi = azimuth_mmr  # Use MMR-centered azimuth
                            while radial_phi < 0:
                                radial_phi += 180.0
                            while radial_phi >= 180:
                                radial_phi -= 180.0
                            
                            # Weight: 1.0 at r=1.5 (radial), 0.0 at r=2.5 (regional)
                            weight = (2.5 - mmr_dist) / 1.0
                            weight = np.clip(weight, 0.0, 1.0)
                            
                            # Circular mean for angles
                            rad1 = np.radians(radial_phi * 2)  # Double for 0-180 range
                            rad2 = np.radians(phi_base * 2)
                            mean_x = weight * np.cos(rad1) + (1-weight) * np.cos(rad2)
                            mean_y = weight * np.sin(rad1) + (1-weight) * np.sin(rad2)
                            phi_fast[i, j, k] = np.degrees(np.arctan2(mean_y, mean_x)) / 2.0
                            
                            # Normalize to 0-180
                            while phi_fast[i, j, k] < 0:
                                phi_fast[i, j, k] += 180.0
                            while phi_fast[i, j, k] >= 180:
                                phi_fast[i, j, k] -= 180.0
                        
                        # Far from caldera or deep: regional stress (phi_base)
                        # Already initialized, no change needed
        
        # Create fast and slow velocity models
        # Division by 200 (not 100) keeps average velocity ~ isotropic velocity
        self.velocity_fast = self.velocity * (1.0 + percent_anisotropy / 200.0)
        self.velocity_slow = self.velocity * (1.0 - percent_anisotropy / 200.0)
        self.phi_fast = phi_fast
        self.percent_anisotropy = percent_anisotropy
        
        return self.velocity_fast, self.velocity_slow, phi_fast, percent_anisotropy
    
    def _update_anisotropic_velocities(self):
        """
        Update fast and slow velocities based on current percent_anisotropy values.
        
        This method should be called whenever percent_anisotropy is modified
        to keep velocity_fast and velocity_slow in sync with the anisotropy model.
        It's primarily used during optimization when the anisotropy parameters
        are being iteratively updated.
        
        Notes
        -----
        Division by 200 (not 100) ensures the average velocity remains close
        to the isotropic velocity:
        - v_fast = v * (1 + A/200)
        - v_slow = v * (1 - A/200)
        - average = v * [(1 + A/200) + (1 - A/200)] / 2 = v
        """
        if self.velocity is None:
            raise ValueError("Isotropic velocity model not created. Call create_model() first.")
        
        if self.percent_anisotropy is None:
            raise ValueError("Anisotropy not initialized. Call create_anisotropic_model() first.")
        
        # Update fast and slow velocities based on current anisotropy
        self.velocity_fast = self.velocity * (1.0 + self.percent_anisotropy / 200.0)
        self.velocity_slow = self.velocity * (1.0 - self.percent_anisotropy / 200.0)
    
    def update_anisotropy(self, new_percent_anisotropy, new_phi_fast=None):
        """
        Update the anisotropy model with new parameters and recalculate velocities.
        
        This is the public interface for updating anisotropy during optimization.
        
        Parameters
        ----------
        new_percent_anisotropy : ndarray
            New percent anisotropy values (must match model grid shape)
        new_phi_fast : ndarray, optional
            New fast direction values in degrees (must match model grid shape)
            If None, phi_fast is not updated
        
        Raises
        ------
        ValueError
            If shapes don't match the model grid
        """
        # Validate shape
        if new_percent_anisotropy.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Shape mismatch: new_percent_anisotropy has shape {new_percent_anisotropy.shape}, "
                f"expected ({self.nx}, {self.ny}, {self.nz})"
            )
        
        # Update percent anisotropy
        self.percent_anisotropy = new_percent_anisotropy.copy()
        
        # Update fast direction if provided
        if new_phi_fast is not None:
            if new_phi_fast.shape != (self.nx, self.ny, self.nz):
                raise ValueError(
                    f"Shape mismatch: new_phi_fast has shape {new_phi_fast.shape}, "
                    f"expected ({self.nx}, {self.ny}, {self.nz})"
                )
            self.phi_fast = new_phi_fast.copy()
        
        # Update velocities to match new anisotropy
        self._update_anisotropic_velocities()
 
    def get_anisotropy_slice(self, axis='z', value=0.0, field='percent'):
        """
        Extract a 2D slice from the anisotropy field.
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        value : float
            Position along the axis for the slice
        field : str
            'percent' for percent_anisotropy or 'phi' for phi_fast
        
        Returns
        -------
        slice_data : ndarray
            2D slice of anisotropy field
        coords : tuple
            Coordinate arrays for the slice
        """
        if field == 'percent':
            data = self.percent_anisotropy
            if data is None:
                raise ValueError("Anisotropic model not created. Call create_anisotropic_model() first.")
        elif field == 'phi':
            data = self.phi_fast
            if data is None:
                raise ValueError("Anisotropic model not created. Call create_anisotropic_model() first.")
        else:
            raise ValueError("field must be 'percent' or 'phi'")
        
        if axis == 'z':
            idx = np.argmin(np.abs(self.z - value))
            return data[:, :, idx], (self.x, self.y)
        elif axis == 'y':
            idx = np.argmin(np.abs(self.y - value))
            return data[:, idx, :], (self.x, self.z)
        elif axis == 'x':
            idx = np.argmin(np.abs(self.x - value))
            return data[idx, :, :], (self.y, self.z)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
