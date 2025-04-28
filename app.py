import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

from components.sidebar import create_sidebar
from components.results_display import display_results
from components.structure_builder import create_structure_builder
from models.dl_models import predict_optical_properties
from utils.materials_database import get_material_properties
from utils.visualization import plot_field_distribution
from utils.simulation import calculate_optical_properties

# Set page config
st.set_page_config(
    page_title="Plasmonic Metamaterial Simulator",
    page_icon="🔬",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'field_data' not in st.session_state:
    st.session_state.field_data = None
if 'wavelength_range' not in st.session_state:
    st.session_state.wavelength_range = (400, 800)
if 'current_wavelength' not in st.session_state:
    st.session_state.current_wavelength = 550

# App title and description
st.title("Deep Learning Plasmonic Metamaterial Designer")
st.markdown("""
    Design and simulate plasmonic metamaterials, photonic crystals, and various 
    nanostructures using deep learning models. Analyze optical properties and 
    visualize field distributions across wavelength ranges.
""")

# Create structure builder and sidebar
structure_params = create_structure_builder()
sidebar_params = create_sidebar()

# Combine all parameters
sim_params = {**structure_params, **sidebar_params}

# Run simulation button
col1, col2 = st.columns([3, 1])
with col1:
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)
with col2:
    export_button = st.button("Export Results", type="secondary", use_container_width=True, 
                            disabled=st.session_state.simulation_results is None)

if run_button:
    with st.spinner("Running simulation... This may take a moment."):
        # Progress bar for visual feedback
        progress_bar = st.progress(0)
        
        # Calculate optical properties using deep learning model
        progress_bar.progress(25)
        time.sleep(0.5)  # Simulate computation time
        
        wavelength_range = np.linspace(
            sim_params['wavelength_min'], 
            sim_params['wavelength_max'], 
            100
        )
        
        # Get optical properties from the model
        optical_properties = predict_optical_properties(
            structure_type=sim_params['structure_type'],
            materials=sim_params['materials'],
            dimensions=sim_params['dimensions'],
            has_substrate=sim_params['has_substrate'],
            substrate_material=sim_params['substrate_material'] if sim_params['has_substrate'] else None,
            environment_index=sim_params['environment_index'],
            wavelength_range=wavelength_range
        )
        
        progress_bar.progress(75)
        time.sleep(0.5)  # Simulate computation time
        
        # Calculate field distributions for the current wavelength
        field_data = calculate_optical_properties(
            wavelength=st.session_state.current_wavelength,
            structure_type=sim_params['structure_type'],
            materials=sim_params['materials'],
            dimensions=sim_params['dimensions'],
            has_substrate=sim_params['has_substrate'],
            substrate_material=sim_params['substrate_material'] if sim_params['has_substrate'] else None,
            environment_index=sim_params['environment_index']
        )
        
        progress_bar.progress(100)
        
        # Save results to session state
        st.session_state.simulation_results = {
            'wavelength': wavelength_range,
            'reflectance': optical_properties['reflectance'],
            'transmittance': optical_properties['transmittance'],
            'absorption': optical_properties['absorption'],
            'params': sim_params
        }
        st.session_state.field_data = field_data
        st.session_state.wavelength_range = (sim_params['wavelength_min'], sim_params['wavelength_max'])
        
        # Clear the progress bar after completion
        progress_bar.empty()
        
        # Rerun to update the UI
        st.rerun()

if export_button and st.session_state.simulation_results is not None:
    results_df = pd.DataFrame({
        'Wavelength (nm)': st.session_state.simulation_results['wavelength'],
        'Reflectance': st.session_state.simulation_results['reflectance'],
        'Transmittance': st.session_state.simulation_results['transmittance'],
        'Absorption': st.session_state.simulation_results['absorption']
    })
    
    # Convert DataFrame to CSV string
    csv = results_df.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="metamaterial_simulation_results.csv",
        mime="text/csv",
    )

# Display results if they exist
if st.session_state.simulation_results is not None:
    display_results(
        st.session_state.simulation_results,
        st.session_state.field_data,
        st.session_state.current_wavelength
    )

# Footer
st.markdown("---")
st.markdown("""
    **About**: This application uses deep learning models to predict optical properties of plasmonic metamaterials
    and photonic structures. The models have been trained on extensive simulation data.
""")
