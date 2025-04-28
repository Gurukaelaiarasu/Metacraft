import streamlit as st
import numpy as np
from utils.materials_database import get_material_list, get_material_properties
from utils.visualization import plot_structure_preview

def create_structure_builder():
    """
    Create the structure builder interface
    
    Returns:
    --------
    dict : Dictionary of structure parameters
    """
    st.header("Structure Builder")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Structure type selection
        structure_type = st.selectbox(
            "Structure Type",
            options=["2D_Array", "3D_Structure", "Thin_Film", "Particle"],
            index=0,
            help="Type of metamaterial structure to simulate"
        )
        
        # Material selection
        available_materials = get_material_list()
        materials_db = get_material_properties()
        
        if structure_type == "Thin_Film":
            # For thin films, allow multiple material layers
            num_layers = st.slider(
                "Number of Layers",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of material layers in the thin film stack"
            )
            
            materials = []
            for i in range(num_layers):
                material = st.selectbox(
                    f"Layer {i+1} Material",
                    options=available_materials,
                    index=min(i, len(available_materials)-1),
                    key=f"layer_{i}_material"
                )
                materials.append(material)
        else:
            # For other structures, select primary material
            primary_material = st.selectbox(
                "Primary Material",
                options=available_materials,
                index=0,
                help="Main material for the structure"
            )
            
            # Option for additional materials in complex structures
            if structure_type in ["2D_Array", "3D_Structure"]:
                add_secondary = st.checkbox(
                    "Add Secondary Material",
                    value=False,
                    help="Add an additional material to the structure"
                )
                
                if add_secondary:
                    secondary_material = st.selectbox(
                        "Secondary Material",
                        options=[m for m in available_materials if m != primary_material],
                        index=0,
                        help="Additional material for the structure"
                    )
                    materials = [primary_material, secondary_material]
                else:
                    materials = [primary_material]
            else:
                # For particles, just use the primary material
                materials = [primary_material]
        
        # Substrate options
        has_substrate = st.checkbox(
            "Include Substrate",
            value=True,
            help="Include a substrate beneath the structure"
        )
        
        if has_substrate:
            substrate_material = st.selectbox(
                "Substrate Material",
                options=available_materials,
                index=min(5, len(available_materials)-1),  # Default to silicon dioxide
                help="Material for the substrate"
            )
        else:
            substrate_material = None
            
    with col2:
        # Dimension controls based on structure type
        st.subheader("Dimensions")
        
        dimensions = []
        if structure_type == "2D_Array":
            period = st.slider(
                "Period (nm)",
                min_value=100,
                max_value=1000,
                value=500,
                step=10,
                help="Period of the array (center-to-center distance)"
            )
            
            element_size = st.slider(
                "Element Size (nm)",
                min_value=50,
                max_value=500,
                value=200,
                step=10,
                help="Size of each array element"
            )
            
            height = st.slider(
                "Height (nm)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Height of the structure"
            )
            
            # Add a shape selector
            shape = st.selectbox(
                "Element Shape",
                options=["Circle", "Square", "Triangle", "Hexagon"],
                index=0,
                help="Shape of each array element"
            )
            
            dimensions = [period, element_size, height]
            
        elif structure_type == "3D_Structure":
            x_size = st.slider(
                "X Size (nm)",
                min_value=100,
                max_value=2000,
                value=500,
                step=10,
                help="Size in X direction"
            )
            
            y_size = st.slider(
                "Y Size (nm)",
                min_value=100,
                max_value=2000,
                value=500,
                step=10,
                help="Size in Y direction"
            )
            
            z_size = st.slider(
                "Z Size (nm)",
                min_value=100,
                max_value=2000,
                value=500,
                step=10,
                help="Size in Z direction"
            )
            
            # Add a pattern selector
            pattern = st.selectbox(
                "3D Pattern",
                options=["Woodpile", "Inverse Opal", "Hollow Shell", "Multilayer"],
                index=0,
                help="Type of 3D pattern"
            )
            
            dimensions = [x_size, y_size, z_size]
            
        elif structure_type == "Thin_Film":
            # For thin films, define thickness for each layer
            layer_thicknesses = []
            for i in range(len(materials)):
                thickness = st.slider(
                    f"Layer {i+1} Thickness (nm)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    key=f"layer_{i}_thickness",
                    help=f"Thickness of layer {i+1}"
                )
                layer_thicknesses.append(thickness)
            
            dimensions = layer_thicknesses
            
        else:  # Particle
            diameter = st.slider(
                "Diameter (nm)",
                min_value=10,
                max_value=500,
                value=100,
                step=5,
                help="Diameter of the particle"
            )
            
            # Add a shape selector for particles
            shape = st.selectbox(
                "Particle Shape",
                options=["Sphere", "Rod", "Cube", "Disk", "Star"],
                index=0,
                help="Shape of the particle"
            )
            
            # Add aspect ratio for non-spherical particles
            if shape != "Sphere":
                aspect_ratio = st.slider(
                    "Aspect Ratio",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                    help="Ratio of longest to shortest dimension"
                )
                dimensions = [diameter, aspect_ratio]
            else:
                dimensions = [diameter]
        
        # Structure preview
        st.subheader("Structure Preview")
        preview_fig = plot_structure_preview(structure_type, dimensions, materials)
        st.plotly_chart(preview_fig, use_container_width=True)
        
        # Display material properties
        with st.expander("Material Properties"):
            for material in materials:
                properties = materials_db.get(material, {})
                st.write(f"**{material}**")
                
                # Display key properties
                if 'description' in properties:
                    st.write(properties['description'])
                
                props_text = []
                if 'type' in properties:
                    props_text.append(f"Type: {properties['type'].capitalize()}")
                if 'n_real' in properties:
                    props_text.append(f"Refractive index: {properties['n_real']}")
                if 'permittivity' in properties:
                    props_text.append(f"Permittivity: {properties['permittivity']}")
                
                st.write(", ".join(props_text))
                st.write("---")
    
    # Return parameters as a dictionary
    return {
        'structure_type': structure_type,
        'materials': materials,
        'dimensions': dimensions,
        'has_substrate': has_substrate,
        'substrate_material': substrate_material if has_substrate else None
    }
