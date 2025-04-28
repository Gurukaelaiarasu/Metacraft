import streamlit as st
import numpy as np
from utils.materials_database import get_material_list, get_material_properties

def create_sidebar():
    """
    Create the sidebar for simulation parameters
    
    Returns:
    --------
    dict : Dictionary of simulation parameters
    """
    st.sidebar.title("Simulation Parameters")
    
    # Environment parameters
    st.sidebar.header("Environment")
    environment_index = st.sidebar.slider(
        "Environment Refractive Index",
        min_value=1.0,
        max_value=3.0,
        value=1.0,
        step=0.01,
        help="Refractive index of the surrounding medium"
    )
    
    # Wavelength range
    st.sidebar.header("Wavelength Range")
    wavelength_col1, wavelength_col2 = st.sidebar.columns(2)
    with wavelength_col1:
        wavelength_min = st.number_input(
            "Min Wavelength (nm)",
            min_value=300,
            max_value=2000,
            value=400,
            step=10,
            help="Minimum wavelength for analysis"
        )
    with wavelength_col2:
        wavelength_max = st.number_input(
            "Max Wavelength (nm)",
            min_value=300,
            max_value=2000,
            value=800,
            step=10,
            help="Maximum wavelength for analysis"
        )
    
    # Adjust if min > max
    if wavelength_min > wavelength_max:
        wavelength_max = wavelength_min + 10
    
    # Visualization parameters
    st.sidebar.header("Visualization")
    
    # Field type
    field_type = st.sidebar.selectbox(
        "Field Type",
        options=["Electric", "Magnetic", "Power"],
        index=0,
        help="Type of field to visualize"
    )
    
    # Field component
    field_component = st.sidebar.selectbox(
        "Field Component",
        options=["Magnitude", "X Component", "Y Component", "Z Component"],
        index=0,
        help="Field component to visualize"
    )
    
    # Slice axis
    slice_axis = st.sidebar.selectbox(
        "Slice Axis",
        options=["X", "Y", "Z"],
        index=2,
        help="Axis along which to take a slice for field visualization"
    )
    
    # Slice position
    slice_position = st.sidebar.slider(
        "Slice Position",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Position along the slice axis"
    )
    
    # Advanced options (collapsible)
    with st.sidebar.expander("Advanced Options"):
        resolution = st.slider(
            "Simulation Resolution",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Resolution level for simulation (higher values increase accuracy but take longer)"
        )
        
        use_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=False,
            help="Use GPU for faster simulation (if available)"
        )
        
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "JSON", "XLSX"],
            index=0,
            help="File format for exporting results"
        )
    
    # Return parameters as a dictionary
    return {
        'environment_index': environment_index,
        'wavelength_min': wavelength_min,
        'wavelength_max': wavelength_max,
        'field_type': field_type.lower(),
        'field_component': field_component.split()[0].lower(),
        'slice_axis': slice_axis.lower(),
        'slice_position': slice_position,
        'resolution': resolution,
        'use_gpu': use_gpu,
        'export_format': export_format
    }
