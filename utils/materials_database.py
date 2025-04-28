import numpy as np

def get_material_properties():
    """
    Get a dictionary of available materials and their optical properties
    
    Returns:
    --------
    dict : Dictionary mapping material names to property dictionaries
    """
    # Create a dictionary of materials and their properties
    # For a real application, this would be a database or loaded from a file
    materials_db = {
        # Metals
        'Gold': {
            'type': 'metal',
            'n_model': 'drude-lorentz',
            'plasma_freq': 9.03,  # eV
            'damping_freq': 0.026,  # eV
            'interband': True,
            'permittivity': -10.0 + 1.5j,  # at 550nm
            'permeability': complex(1.0, 0.0),
            'color': '#FFD700',
            'description': 'Excellent plasmonic material with resonance in visible range'
        },
        'Silver': {
            'type': 'metal',
            'n_model': 'drude-lorentz',
            'plasma_freq': 9.01,  # eV
            'damping_freq': 0.018,  # eV
            'interband': True,
            'permittivity': -14.0 + 0.5j,  # at 550nm
            'permeability': complex(1.0, 0.0),
            'color': '#C0C0C0',
            'description': 'Low-loss plasmonic material'
        },
        'Aluminum': {
            'type': 'metal',
            'n_model': 'drude-lorentz',
            'plasma_freq': 15.3,  # eV
            'damping_freq': 0.1,  # eV
            'interband': True,
            'permittivity': -40.0 + 10.0j,  # at 550nm
            'permeability': complex(1.0, 0.0),
            'color': '#848789',
            'description': 'UV plasmonic material'
        },
        'Copper': {
            'type': 'metal',
            'n_model': 'drude-lorentz',
            'plasma_freq': 8.8,  # eV
            'damping_freq': 0.09,  # eV
            'interband': True,
            'permittivity': -13.0 + 1.0j,  # at 550nm
            'permeability': complex(1.0, 0.0),
            'color': '#B87333',
            'description': 'Plasmonic material with interband transitions'
        },
        
        # Dielectrics
        'Silicon': {
            'type': 'semiconductor',
            'n_model': 'sellmeier',
            'n_real': 3.48,  # at 1000nm
            'n_imag': 0.0,
            'bandgap': 1.11,  # eV
            'permittivity': 12.1,
            'permeability': 1.0,
            'color': '#5C5C5C',
            'description': 'High-index semiconductor for photonic crystals'
        },
        'Silicon Dioxide': {
            'type': 'dielectric',
            'n_model': 'sellmeier',
            'n_real': 1.46,
            'n_imag': 0.0,
            'permittivity': 2.13,
            'permeability': 1.0,
            'color': '#F0F8FF',
            'description': 'Common dielectric material for substrates'
        },
        'Titanium Dioxide': {
            'type': 'dielectric',
            'n_model': 'sellmeier',
            'n_real': 2.49,
            'n_imag': 0.0,
            'permittivity': 6.2,
            'permeability': 1.0,
            'color': '#F0F8FF',
            'description': 'High-index dielectric for photonic applications'
        },
        'Gallium Arsenide': {
            'type': 'semiconductor',
            'n_model': 'sellmeier',
            'n_real': 3.3,
            'n_imag': 0.0,
            'bandgap': 1.42,  # eV
            'permittivity': 10.9,
            'permeability': 1.0,
            'color': '#808080',
            'description': 'III-V semiconductor for optoelectronics'
        },
        
        # Other materials
        'ITO': {
            'type': 'tco',
            'n_model': 'drude',
            'n_real': 1.9,
            'n_imag': 0.05,
            'plasma_freq': 1.5,  # eV
            'permittivity': 3.8 + 0.1j,
            'permeability': 1.0,
            'color': '#D3D3D3',
            'description': 'Transparent conductive oxide'
        },
        'Graphene': {
            'type': '2d-material',
            'n_model': 'drude',
            'permittivity': 2.5 + 1.0j,  # effective
            'permeability': 1.0,
            'color': '#808080',
            'description': '2D material with tunable properties'
        },
        'Water': {
            'type': 'liquid',
            'n_model': 'sellmeier',
            'n_real': 1.33,
            'n_imag': 0.0,
            'permittivity': 1.77,
            'permeability': 1.0,
            'color': '#ADD8E6',
            'description': 'Common liquid for sensing applications'
        }
    }
    
    return materials_db

def get_material_list():
    """
    Get a list of available material names
    
    Returns:
    --------
    list : List of material names
    """
    materials_db = get_material_properties()
    return list(materials_db.keys())

def get_refractive_index(material, wavelength):
    """
    Get the complex refractive index of a material at a specific wavelength
    
    Parameters:
    -----------
    material : str
        Material name
    wavelength : float
        Wavelength in nm
        
    Returns:
    --------
    complex : Complex refractive index
    """
    materials_db = get_material_properties()
    material_props = materials_db.get(material, {})
    
    # Get model type
    model_type = material_props.get('n_model', 'constant')
    
    if model_type == 'constant':
        # Return constant values
        n_real = material_props.get('n_real', 1.0)
        n_imag = material_props.get('n_imag', 0.0)
        return complex(n_real, n_imag)
    
    elif model_type == 'drude':
        # Simple Drude model
        plasma_freq = material_props.get('plasma_freq', 9.0)  # eV
        damping_freq = material_props.get('damping_freq', 0.05)  # eV
        
        # Convert wavelength to eV
        photon_energy = 1240.0 / wavelength  # eV
        
        # Calculate permittivity
        eps_real = 1.0 - (plasma_freq**2) / (photon_energy**2 + damping_freq**2)
        eps_imag = (plasma_freq**2 * damping_freq) / (photon_energy * (photon_energy**2 + damping_freq**2))
        
        # Calculate refractive index
        eps_complex = complex(eps_real, eps_imag)
        n_complex = np.sqrt(eps_complex)
        
        return n_complex
    
    elif model_type == 'drude-lorentz':
        # Drude-Lorentz model (simplified)
        plasma_freq = material_props.get('plasma_freq', 9.0)  # eV
        damping_freq = material_props.get('damping_freq', 0.05)  # eV
        has_interband = material_props.get('interband', False)
        
        # Convert wavelength to eV
        photon_energy = 1240.0 / wavelength  # eV
        
        # Drude term
        eps_drude_real = 1.0 - (plasma_freq**2) / (photon_energy**2 + damping_freq**2)
        eps_drude_imag = (plasma_freq**2 * damping_freq) / (photon_energy * (photon_energy**2 + damping_freq**2))
        
        # Add Lorentz terms for interband transitions if applicable
        if has_interband:
            # Simplified interband contribution
            eps_lorentz_real = 1.5 * np.exp(-(photon_energy - 2.0)**2 / 0.5)
            eps_lorentz_imag = 0.5 * np.exp(-(photon_energy - 2.5)**2 / 0.5)
        else:
            eps_lorentz_real = 0.0
            eps_lorentz_imag = 0.0
        
        # Combine Drude and Lorentz terms
        eps_real = eps_drude_real + eps_lorentz_real
        eps_imag = eps_drude_imag + eps_lorentz_imag
        
        # Calculate refractive index
        eps_complex = complex(eps_real, eps_imag)
        n_complex = np.sqrt(eps_complex)
        
        return n_complex
    
    elif model_type == 'sellmeier':
        # Simplified Sellmeier model
        n_real = material_props.get('n_real', 1.0)
        wavelength_um = wavelength / 1000.0  # Convert to micrometers
        
        # Add wavelength dependence
        n_wavelength = n_real + 0.01 * np.sin(2 * np.pi * wavelength_um / 10.0)
        
        # Add very small imaginary component for absorption
        n_imag = material_props.get('n_imag', 0.0) + 0.001 * wavelength_um
        
        return complex(n_wavelength, n_imag)
    
    # Default: return a default value
    return complex(1.5, 0.0)
