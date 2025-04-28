import numpy as np

def encode_materials(materials_list, material_db):
    """
    Encode materials into numerical features for model input
    
    Parameters:
    -----------
    materials_list : list
        List of material names
    material_db : dict
        Dictionary mapping material names to properties
        
    Returns:
    --------
    numpy.ndarray : Encoded material features
    """
    encoded_features = []
    
    for material in materials_list:
        # Get material properties
        properties = material_db.get(material, {})
        
        # Extract key properties for the model
        n_real = properties.get('n_real', 1.0)  # Real part of refractive index
        n_imag = properties.get('n_imag', 0.0)  # Imaginary part of refractive index
        permittivity = properties.get('permittivity', 1.0)
        permeability = properties.get('permeability', 1.0)
        plasma_freq = properties.get('plasma_freq', 0.0)
        
        # Create feature vector for this material
        material_features = [n_real, n_imag, permittivity, permeability, plasma_freq]
        encoded_features.append(material_features)
    
    return np.array(encoded_features)

def normalize_dimensions(dimensions, structure_type):
    """
    Normalize dimensional parameters based on structure type
    
    Parameters:
    -----------
    dimensions : list
        List of dimensional parameters
    structure_type : str
        Type of structure
        
    Returns:
    --------
    numpy.ndarray : Normalized dimensions
    """
    dimensions = np.array(dimensions, dtype=float)
    
    if structure_type == '2D_Array':
        # For 2D arrays, normalize by typical period (~500nm)
        dimensions /= 500.0
    elif structure_type == '3D_Structure':
        # For 3D structures, normalize by typical size (~1µm)
        dimensions /= 1000.0
    elif structure_type == 'Thin_Film':
        # For thin films, normalize by typical thickness (~100nm)
        dimensions /= 100.0
    else:  # 'Particle' or default
        # For particles, normalize by typical particle size (~50nm)
        dimensions /= 50.0
    
    return dimensions

def prepare_model_input(structure_type, materials, dimensions, has_substrate, 
                       substrate_material, environment_index, wavelength,
                       material_db):
    """
    Prepare input data for deep learning models
    
    Parameters:
    -----------
    structure_type : str
        Type of structure
    materials : list
        List of material names
    dimensions : list
        List of dimensional parameters
    has_substrate : bool
        Whether the structure has a substrate
    substrate_material : str or None
        Substrate material name/id if has_substrate is True
    environment_index : float
        Refractive index of the environment
    wavelength : float
        Wavelength for prediction
    material_db : dict
        Dictionary of material properties
        
    Returns:
    --------
    numpy.ndarray : Preprocessed input ready for model
    """
    # Encode materials
    material_features = encode_materials(materials, material_db)
    material_features = material_features.flatten()  # Flatten for concatenation
    
    # Normalize dimensions
    normalized_dimensions = normalize_dimensions(dimensions, structure_type)
    
    # Encode substrate information
    if has_substrate and substrate_material:
        substrate_features = encode_materials([substrate_material], material_db).flatten()
        substrate_present = np.array([1.0])  # 1 for substrate present
    else:
        # Use zeros for substrate features if no substrate
        substrate_features = np.zeros(5)  # Same length as material encoding
        substrate_present = np.array([0.0])  # 0 for no substrate
    
    # Normalize wavelength (typical range: 300-1000nm)
    normalized_wavelength = np.array([wavelength / 1000.0])
    
    # Normalize environment index (typical range: 1.0-3.0)
    normalized_env_index = np.array([environment_index / 3.0])
    
    # Create a mapping for structure type (one-hot encoding)
    structure_mapping = {
        '2D_Array': [1, 0, 0, 0],
        '3D_Structure': [0, 1, 0, 0],
        'Thin_Film': [0, 0, 1, 0],
        'Particle': [0, 0, 0, 1]
    }
    structure_encoding = np.array(structure_mapping.get(structure_type, [0, 0, 0, 1]))
    
    # Combine all features into a single input vector
    model_input = np.concatenate([
        material_features,
        normalized_dimensions,
        substrate_present,
        substrate_features,
        normalized_env_index,
        normalized_wavelength,
        structure_encoding
    ])
    
    # Reshape for model input (batch size of 1)
    return np.reshape(model_input, (1, -1))

def postprocess_output(raw_predictions):
    """
    Process raw model output into physically meaningful results
    
    Parameters:
    -----------
    raw_predictions : numpy.ndarray
        Raw output from the model
        
    Returns:
    --------
    dict : Dictionary with processed optical properties
    """
    # Extract primary outputs (typically in range [0,1] due to sigmoid activation)
    reflectance = np.clip(raw_predictions[0, 0], 0, 1)
    transmittance = np.clip(raw_predictions[0, 1], 0, 1)
    
    # Ensure physical constraints are met (R + T <= 1)
    if reflectance + transmittance > 1:
        # Scale down proportionally to maintain ratio
        total = reflectance + transmittance
        reflectance = reflectance / total
        transmittance = transmittance / total
    
    # Calculate absorption (A = 1 - R - T)
    absorption = 1 - reflectance - transmittance
    
    return {
        'reflectance': reflectance,
        'transmittance': transmittance,
        'absorption': absorption
    }
