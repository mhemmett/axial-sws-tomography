# Axial Seamount Shear-Wave Splitting Tomography

A complete workflow for analyzing seismic anisotropy at Axial Seamount through shear-wave splitting tomography using global linear least-squares inversion.

## Overview

This repository provides a comprehensive analysis framework for shear-wave splitting at Axial Seamount, a submarine volcano on the Juan de Fuca Ridge. The workflow implements global linear inversion tomography to map 3D subsurface anisotropy structure from observed splitting parameters (φ, δt) and splitting intensity (SI) measurements.

### Key Features

- **Complete Workflow**: From raw earthquake catalogs to 3D anisotropy models
- **Quality Control Pipeline**: Automated filtering for SNR, P-wave rectilinearity, and incidence angles  
- **Global Linear Inversion**: Regularized least-squares inversion with proper dimensional analysis
- **Dual Observation Types**: Combines shear-wave splitting (φ, δt) and splitting intensity (SI) measurements
- **3D Velocity Modeling**: Realistic geological structures including magma chamber and caldera
- **Ray Tracing**: PyKonal-based ray path computation through 3D velocity models
- **Physically Accurate**: Proper path length computation, velocity scaling, and normalized weighting
- **Comprehensive Visualization**: 3D plots of velocity models, anisotropy structure, and fast direction vectors

## Repository Structure

```
axial-sws-tomography/
├── notebooks/
│   └── axial_sws_tomography.ipynb    # Main analysis workflow
├── src/
│   ├── axial_velocity_model.py       # 3D velocity model construction
│   ├── obs_array.py                  # Ocean bottom seismometer array
│   ├── earthquake_location.py        # Ray tracing through velocity models
│   ├── global_anisotropy_inversion.py # Global linear least-squares inversion
│   ├── splitting_intensity.py        # Splitting intensity (Chevrot 2000)
│   ├── inversion_diagnostics.py      # Resolution and uncertainty analysis
│   ├── get_all_traces.py            # Waveform data retrieval
│   └── splitting_functions.py       # Shear-wave splitting analysis & QC
├── swspy/                           # Local shear-wave splitting library
├── data/
│   ├── axial_seamount_stations.csv  # Station coordinates and metadata
│   ├── ax.hinv.pha.shots_erup       # Raw phase pick data
│   ├── 2018_eq_catalog.csv         # Processed earthquake catalog
│   └── 2015_jun_dec_eq_catalog.csv # Filtered catalog subset
├── results/                         # Output directory for results
├── figures/                         # Generated plots and figures
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/axial-sws-tomography.git
   cd axial-sws-tomography
   ```

2. **Create a conda environment** (recommended):
   ```bash
   conda create -n axial-sws python=3.9
   conda activate axial-sws
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages**:
   ```bash
   conda install -c conda-forge obspy pykonal
   ```

## Usage

### Quick Start

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   Navigate to `notebooks/axial_sws_tomography.ipynb`

3. **Run the workflow**:
   Execute cells sequentially to perform the complete analysis

### Workflow Steps

The notebook implements the following analysis pipeline:

1. **Data Loading & Organization** - Import earthquake catalogs and station metadata
2. **Extended Time Window Creation** - Generate proper time windows for waveform analysis
3. **Waveform Data Retrieval** - Download and organize seismic traces
4. **Quality Control Pipeline** - Filter data based on SNR, rectilinearity, and geometry
5. **Shear-Wave Splitting Analysis** - Silver & Chan (1991) method with Teanby (2004) clustering
6. **Splitting Intensity Calculation** - Chevrot (2000) formulation from rotated components
7. **3D Velocity Model Construction** - Build realistic velocity structure (2.4-2.9 km/s range)
8. **Ray Tracing** - PyKonal-based path computation through heterogeneous media
9. **Global Linear Inversion** - Regularized least-squares with proper physical units
10. **Visualization & Analysis** - 3D plots of anisotropy strength and fast direction patterns

### Key Parameters

The analysis can be customized by modifying these parameters in the notebook:

```python
# Quality control thresholds
QC_THRESHOLDS = {
    'min_snr': 2.0,               # Minimum S-wave signal-to-noise ratio
    'min_rectilinearity': 0.7,    # Minimum P-wave rectilinearity
    'max_incidence': 30.0,        # Maximum incidence angle (degrees)
    'min_magnitude': 0.0,         # Minimum event magnitude
}
# Velocity model parameters
vm = AxialVelocityModel(
    nx=51, ny=51, nz=26,          # Grid dimensions
    x_range=(-3.0, 3.0),          # X extent (km)
    y_range=(-3.0, 3.0),          # Y extent (km)  
    z_range=(0.0, 3.0)            # Depth range (km)
)

# Regularization parameters
regularization_params = {
    'lambda_smooth': 1.0,         # Spatial smoothing weight
    'lambda_damp': 0.1,           # Damping weight  
    'smooth_horizontal': 1.0,     # Horizontal smoothing
    'smooth_vertical': 0.5        # Vertical smoothing
}   z_range=(0.0, 10.0)           # Depth range (km)
)
```

## Scientific Background
### Global Linear Least-Squares Inversion

This workflow implements a physically accurate global inversion approach following Nataf (1986) and Chevrot (2000):

1. **Forward Model**: **d** = **G** · **m** where **m** = [M_c, M_s] with M_c = A cos(2ψ), M_s = A sin(2ψ)
2. **Design Matrix G**: Built from ray path lengths L_ij (km) divided by velocity V (km/s) for dimensional consistency
   - G entries have units of seconds: L_ij / V
   - Model parameters m are dimensionless (fractional velocity difference)
3. **Observations d**: Combined φ, δt, and splitting intensity (SI) measurements  
   - Splitting: 2 rows per observation with normalized weighting (÷√2)
   - SI: 1 row per observation with incidence angle sensitivity sin(2θ)
4. **Regularization**: Spatial smoothing + damping with L-curve parameter selection
5. **Solution**: Solve (G^T W² G + λ² R^T R) **m** = G^T W² **d**
6. **Recovery**: Anisotropy strength A = √(M_c² + M_s²), fast direction ψ = 0.5 · arctan2(M_s, M_c)
   - A is dimensionless (fractional velocity difference, typically 0.01-0.15 for 1-15%)
   - Only multiply by 100 for percentage display in visualizations
This workflow implements a global inversion approach following Nataf (1986) and Chevrot (2000):

1. **Forward Model**: d = G · m where m = [M_c, M_s] with M_c = A cos(2ψ), M_s = A sin(2ψ)
2. **Design Matrix G**: Built from isotropic ray path lengths L and sensitivity terms
## Results

The analysis produces several key outputs:

### Splitting Measurements
- High-quality splitting parameters (φ, δt) for event-station pairs
- Splitting intensity (SI) measurements from rotated waveforms
- Comprehensive quality control metrics (SNR, rectilinearity, incidence angle)
- Statistical uncertainty estimates via clustering

### 3D Anisotropy Model  
- Anisotropy strength (A) distribution throughout the upper crust (1-15% expected)
- Fast direction (ψ) patterns showing structural alignment
- Model resolution and data coverage assessment
- Regularized solution balancing data fit and smoothness

### Visualization Products
- 3D scatter plots of anisotropy strength and fast direction
- Horizontal depth slices with vector field overlays
- Velocity model slices showing geological features
- Station locations and ray path coverage
- Quality control and diagnostic plots

### Technical Improvements
- **Physical Accuracy**: Proper dimensional analysis with L_ij/V scaling in G matrix
- **Path Lengths**: Actual ray segment lengths (km) computed from coordinate arrays
- **Normalized Weighting**: Equal contribution per observation regardless of type
- **Incidence Angle**: sin(2θ) sensitivity factor for SI observations
- High-quality splitting parameters for event-station pairs
- Comprehensive quality control metrics
- Statistical uncertainty estimates

### 3D Anisotropy Model  
- Fast direction (φ) distribution throughout the crust
- Percent anisotropy structure 
- Data coverage and resolution assessment

### Visualization Products
- 3D velocity and anisotropy models
- Example ray paths (fast vs slow)
- Splitting parameter distributions
- Quality control diagnostics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis method'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{axial_sws_tomography,
  title={Axial Seamount Shear-Wave Splitting Tomography},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/axial-sws-tomography}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ocean Observatories Initiative (OOI) for continuous seismic monitoring
- SWSPy library for shear-wave splitting analysis
- ObsPy community for seismic data processing tools
- Axial Seamount research community for geological constraints

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your.email@institution.edu]

---

*This research contributes to understanding magmatic processes and crustal structure at submarine volcanoes through seismic anisotropy analysis.*