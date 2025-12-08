# Setup script for Axial Seamount Shear-Wave Splitting Tomography

"""
Setup script for the axial-sws-tomography package.

This package provides tools for analyzing seismic anisotropy at Axial Seamount
through shear-wave splitting tomography using direct inversion methods.
"""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="axial-sws-tomography",
    version="1.0.0",
    author="Michael Hemmett",
    author_email="mhemmett@seismic.edu",
    description="Shear-wave splitting tomography analysis for Axial Seamount",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/axial-sws-tomography",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Geophysics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=0.5"],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yaml", "*.yml"],
    },
)