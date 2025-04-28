import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

def plot_optical_spectra(wavelength, reflectance, transmittance, absorption):
    """
    Create an interactive plot of optical spectra (R, T, A)
    
    Parameters:
    -----------
    wavelength : array
        Array of wavelengths
    reflectance : array
        Array of reflectance values
    transmittance : array
        Array of transmittance values
    absorption : array
        Array of absorption values
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Add traces for reflectance, transmittance, and absorption
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=reflectance,
        mode='lines',
        name='Reflectance',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=transmittance,
        mode='lines',
        name='Transmittance',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=absorption,
        mode='lines',
        name='Absorption',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Optical Properties vs Wavelength',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Intensity',
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def plot_field_distribution(field_data, field_type='electric', component='magnitude', slice_axis='z', slice_position=0):
    """
    Create a heatmap visualization of field distribution
    
    Parameters:
    -----------
    field_data : dict
        Dictionary containing field data
    field_type : str
        Type of field to plot ('electric', 'magnetic', 'power')
    component : str
        Field component to plot ('x', 'y', 'z', 'magnitude')
    slice_axis : str
        Axis along which to take a slice ('x', 'y', 'z')
    slice_position : int
        Position along the slice axis
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Extract field data
    if field_type == 'electric':
        field = field_data.get('electric_field', {})
    elif field_type == 'magnetic':
        field = field_data.get('magnetic_field', {})
    else:  # power
        field = field_data.get('power_field', {})
    
    # Get grid dimensions
    grid_shape = field.get('shape', (50, 50, 50))
    grid_x = np.linspace(-1, 1, grid_shape[0])
    grid_y = np.linspace(-1, 1, grid_shape[1])
    grid_z = np.linspace(-1, 1, grid_shape[2])
    
    # Extract the requested component
    if component == 'x':
        field_values = field.get('x', np.zeros(grid_shape))
    elif component == 'y':
        field_values = field.get('y', np.zeros(grid_shape))
    elif component == 'z':
        field_values = field.get('z', np.zeros(grid_shape))
    else:  # magnitude
        x_comp = field.get('x', np.zeros(grid_shape))
        y_comp = field.get('y', np.zeros(grid_shape))
        z_comp = field.get('z', np.zeros(grid_shape))
        field_values = np.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
    
    # Create slice based on selected axis
    if slice_axis == 'x':
        pos_index = min(max(0, int((slice_position + 1) * grid_shape[0] / 2)), grid_shape[0] - 1)
        slice_data = field_values[pos_index, :, :]
        x_coords, y_coords = grid_y, grid_z
        x_label, y_label = 'Y Position', 'Z Position'
    elif slice_axis == 'y':
        pos_index = min(max(0, int((slice_position + 1) * grid_shape[1] / 2)), grid_shape[1] - 1)
        slice_data = field_values[:, pos_index, :]
        x_coords, y_coords = grid_x, grid_z
        x_label, y_label = 'X Position', 'Z Position'
    else:  # z
        pos_index = min(max(0, int((slice_position + 1) * grid_shape[2] / 2)), grid_shape[2] - 1)
        slice_data = field_values[:, :, pos_index]
        x_coords, y_coords = grid_x, grid_y
        x_label, y_label = 'X Position', 'Y Position'
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        x=x_coords,
        y=y_coords,
        colorscale='Viridis',
        colorbar=dict(title=f"{field_type.capitalize()} Field {component.capitalize()}")
    ))
    
    # Update layout
    title_text = f"{field_type.capitalize()} Field {component.capitalize()} ({slice_axis.upper()}={slice_position:.2f})"
    fig.update_layout(
        title=title_text,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_structure_preview(structure_type, dimensions, materials):
    """
    Create a simple visualization of the structure
    
    Parameters:
    -----------
    structure_type : str
        Type of structure
    dimensions : list
        List of dimensional parameters
    materials : list
        List of materials
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    if structure_type == '2D_Array':
        # Create a simple 2D array visualization
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X * np.pi) * np.sin(Y * np.pi)
        
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        
    elif structure_type == 'Thin_Film':
        # Create a simple thin film visualization
        layers = len(materials)
        heights = np.cumsum([1] * layers)
        
        # Create colored rectangles for each layer
        for i in range(layers):
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[heights[i]-1, heights[i]-1, heights[i], heights[i], heights[i]-1],
                fill="toself",
                fillcolor=color,
                line=dict(color='black', width=1),
                name=f"Layer {i+1}: {materials[i]}"
            ))
            
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(range=[0, heights[-1]]),
            showlegend=True
        )
        
    elif structure_type == 'Particle':
        # Create a simple particle visualization (2D circle)
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.5
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill="toself",
            fillcolor=px.colors.qualitative.Plotly[0],
            line=dict(color='black'),
            name=f"Particle: {materials[0]}"
        ))
        
        fig.update_layout(
            xaxis=dict(range=[-0.6, 0.6]),
            yaxis=dict(range=[-0.6, 0.6], scaleanchor="x", scaleratio=1),
            showlegend=True
        )
        
    else:  # 3D_Structure or default
        # Create a simple 3D structure visualization
        x, y, z = np.indices((5, 5, 5))
        cube = (x < 3) & (y < 3) & (z < 3)
        
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=cube.flatten(),
            opacity=0.4,
            surface_count=17,
            colorscale='Blues'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{structure_type} Preview",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def create_wavelength_selector(wavelength_range, current_wavelength):
    """
    Create an interactive wavelength selector widget
    
    Parameters:
    -----------
    wavelength_range : tuple
        Min and max wavelength values
    current_wavelength : float
        Currently selected wavelength
        
    Returns:
    --------
    float : Selected wavelength
    """
    # Create a slider for wavelength selection
    selected_wavelength = st.slider(
        "Select Wavelength for Field Visualization",
        min_value=float(wavelength_range[0]),
        max_value=float(wavelength_range[1]),
        value=float(current_wavelength),
        step=1.0,
        format="%.0f nm"
    )
    
    return selected_wavelength
