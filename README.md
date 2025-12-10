# Axial Seamount Shear-Wave Splitting Tomography

A complete workflow for analyzing seismic anisotropy at Axial Seamount through shear-wave splitting tomography using direct inversion methods.

## Overview

This repository provides a comprehensive analysis framework for shear-wave splitting at Axial Seamount, a submarine volcano on the Juan de Fuca Ridge. The workflow implements direct inversion tomography to map subsurface anisotropy structure from observed splitting measurements.

### Key Features

- **Complete Workflow**: From raw earthquake catalogs to 3D anisotropy models
- **Quality Control Pipeline**: Automated filtering for signal quality, P-wave rectilinearity, and incidence angles  
- **Direct Inversion Method**: Data-driven tomographic inversion without iterative optimization
- **3D Velocity Modeling**: Incorporates realistic geological structures (magma chamber, caldera)
- **Ray Tracing**: Anisotropic ray path computation through heterogeneous media
- **Comprehensive Visualization**: 3D plotting of velocity models, ray paths, and splitting parameters

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
5. **Shear-Wave Splitting Analysis** - Automated splitting parameter estimation
6. **3D Velocity Model Construction** - Build realistic velocity structure with geological features
7. **Ray Tracing** - Compute anisotropic ray paths through the velocity model
8. **Direct Inversion** - Map splitting measurements to 3D anisotropy structure
9. **Visualization & Analysis** - Generate comprehensive plots and summaries

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
    nx=101, ny=101, nz=51,        # Grid dimensions
    x_range=(-16.0, 16.0),        # X extent (km)
    y_range=(-16.0, 16.0),        # Y extent (km)  
    z_range=(0.0, 10.0)           # Depth range (km)
)
```

## Scientific Background

### Shear-Wave Splitting

Shear-wave splitting occurs when S-waves propagate through anisotropic media, splitting into fast and slow components with different velocities and polarizations. The splitting parameters (φ, δt) characterize:

- **φ (phi)**: Fast direction azimuth relative to north
- **δt (dt)**: Time delay between fast and slow arrivals

### Global Linear Least-Squares Inversion

This workflow implements a global inversion approach following Nataf (1986) and Chevrot (2000):

1. **Forward Model**: d = G · m where m = [M_c, M_s] with M_c = A cos(2ψ), M_s = A sin(2ψ)
2. **Design Matrix G**: Built from isotropic ray path lengths L and sensitivity terms
3. **Observations d**: Stack φ, δt, and splitting intensity (SI) measurements  
4. **Regularization**: Spatial smoothing + damping with L-curve parameter selection
5. **Solution**: Solve (G^T W² G + λ² R^T R) m = G^T W² d
6. **Recovery**: Anisotropy strength A = √(M_c² + M_s²), fast direction ψ = 0.5 arctan2(M_s, M_c)

### Geological Context

Axial Seamount provides an ideal natural laboratory for studying volcanic processes and magmatic anisotropy due to:

- Active volcanism and frequent seismicity
- Well-instrumented with ocean bottom seismometers (OBS)
- Known geological structure (caldera, magma chamber, lava flows)
- Previous geophysical studies for validation

## Results

The analysis produces several key outputs:

### Splitting Measurements
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