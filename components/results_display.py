import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.visualization import plot_optical_spectra, plot_field_distribution, create_wavelength_selector

def display_results(simulation_results, field_data, current_wavelength):
    """
    Display simulation results and visualizations
    
    Parameters:
    -----------
    simulation_results : dict
        Dictionary containing simulation results
    field_data : dict
        Dictionary containing field distribution data
    current_wavelength : float
        Currently selected wavelength for field visualization
    """
    st.header("Simulation Results")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Optical Properties", 
        "Field Distribution", 
        "3D Visualization",
        "Numerical Data"
    ])
    
    # Tab 1: Optical Properties
    with tab1:
        # Display optical spectra
        st.subheader("Optical Properties vs Wavelength")
        spectra_fig = plot_optical_spectra(
            simulation_results['wavelength'],
            simulation_results['reflectance'],
            simulation_results['transmittance'],
            simulation_results['absorption']
        )
        st.plotly_chart(spectra_fig, use_container_width=True)
        
        # Display additional information
        col1, col2, col3 = st.columns(3)
        with col1:
            max_absorption = np.max(simulation_results['absorption'])
            max_absorption_wavelength = simulation_results['wavelength'][np.argmax(simulation_results['absorption'])]
            st.metric(
                "Peak Absorption", 
                f"{max_absorption:.2f}", 
                f"at {max_absorption_wavelength:.0f} nm"
            )
        
        with col2:
            max_reflectance = np.max(simulation_results['reflectance'])
            max_reflectance_wavelength = simulation_results['wavelength'][np.argmax(simulation_results['reflectance'])]
            st.metric(
                "Peak Reflectance", 
                f"{max_reflectance:.2f}", 
                f"at {max_reflectance_wavelength:.0f} nm"
            )
            
        with col3:
            max_transmittance = np.max(simulation_results['transmittance'])
            max_transmittance_wavelength = simulation_results['wavelength'][np.argmax(simulation_results['transmittance'])]
            st.metric(
                "Peak Transmittance", 
                f"{max_transmittance:.2f}", 
                f"at {max_transmittance_wavelength:.0f} nm"
            )
    
    # Tab 2: Field Distribution
    with tab2:
        st.subheader("Field Distribution Visualization")
        
        # Create wavelength selector
        selected_wavelength = create_wavelength_selector(
            (simulation_results['wavelength'][0], simulation_results['wavelength'][-1]),
            current_wavelength
        )
        
        # Update current wavelength in session state
        if selected_wavelength != st.session_state.current_wavelength:
            st.session_state.current_wavelength = selected_wavelength
            st.rerun()
        
        # Field visualization controls
        col1, col2, col3 = st.columns(3)
        with col1:
            field_type = st.selectbox(
                "Field Type",
                options=["Electric", "Magnetic", "Power"],
                index=0,
                key="field_type_select"
            )
        
        with col2:
            field_component = st.selectbox(
                "Component",
                options=["Magnitude", "X", "Y", "Z"],
                index=0,
                key="field_component_select"
            )
            
        with col3:
            slice_axis = st.selectbox(
                "Slice Axis",
                options=["X", "Y", "Z"],
                index=2,
                key="slice_axis_select"
            )
        
        # Slice position slider
        slice_position = st.slider(
            "Slice Position",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="slice_position_slider"
        )
        
        # Display field distribution
        if field_data:
            field_fig = plot_field_distribution(
                field_data,
                field_type.lower(),
                field_component.lower(),
                slice_axis.lower(),
                slice_position
            )
            st.plotly_chart(field_fig, use_container_width=True)
        else:
            st.info("Field data is not yet available. Run the simulation to visualize field distributions.")
    
    # Tab 3: 3D Visualization
    with tab3:
        st.subheader("3D Field Visualization")
        
        # Field type selection for 3D view
        field_type_3d = st.selectbox(
            "Field Type",
            options=["Electric", "Magnetic", "Power"],
            index=0,
            key="field_type_3d_select"
        )
        
        # Create 3D visualization
        if field_data:
            # Extract the appropriate field data
            if field_type_3d.lower() == 'electric':
                field_3d = field_data['electric_field']
            elif field_type_3d.lower() == 'magnetic':
                field_3d = field_data['magnetic_field']
            else:  # power
                field_3d = field_data['power_field']
            
            # Calculate magnitude
            x_comp = field_3d.get('x', np.zeros(field_3d['shape']))
            y_comp = field_3d.get('y', np.zeros(field_3d['shape']))
            z_comp = field_3d.get('z', np.zeros(field_3d['shape']))
            magnitude = np.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
            
            # Threshold for isosurface
            threshold = st.slider(
                "Isosurface Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="threshold_slider"
            )
            
            # Create a 3D isosurface visualization
            grid_size = field_3d['shape'][0]
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            z = np.linspace(-1, 1, grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Normalize magnitude to 0-1 range
            magnitude_norm = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-10)
            
            # Create isosurface
            fig = go.Figure(data=go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=magnitude_norm.flatten(),
                isomin=threshold,
                isomax=1.0,
                surface_count=2,
                colorscale='Viridis',
                opacity=0.6,
                caps=dict(x_show=False, y_show=False, z_show=False)
            ))
            
            # Add a colorbar
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                width=700,
                height=700,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Field data is not yet available. Run the simulation to visualize 3D field distributions.")
    
    # Tab 4: Numerical Data
    with tab4:
        st.subheader("Numerical Data")
        
        # Create a DataFrame with the simulation results
        data = pd.DataFrame({
            'Wavelength (nm)': simulation_results['wavelength'],
            'Reflectance': simulation_results['reflectance'],
            'Transmittance': simulation_results['transmittance'],
            'Absorption': simulation_results['absorption']
        })
        
        # Display the DataFrame with pagination
        st.dataframe(data, use_container_width=True, hide_index=True)
        
        # Show simulation parameters
        st.subheader("Simulation Parameters")
        param_data = pd.DataFrame({
            'Parameter': list(simulation_results['params'].keys()),
            'Value': list(simulation_results['params'].values())
        })
        st.dataframe(param_data, use_container_width=True, hide_index=True)
