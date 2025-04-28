import numpy as np
from utils.field_calculator import calculate_electric_field, calculate_magnetic_field, calculate_power_field

def calculate_optical_properties(wavelength, structure_type, materials, dimensions, 
                                has_substrate, substrate_material, environment_index):
    """
    Calculate optical properties and field distributions for a given wavelength
    
    Parameters:
    -----------
    wavelength : float
        Wavelength for calculation
    structure_type : str
        Type of structure (2D_Array, 3D_Structure, Thin_Film, Particle)
    materials : list
        List of material names/ids
    dimensions : list
        List of dimensional parameters
    has_substrate : bool
        Whether the structure has a substrate
    substrate_material : str or None
        Substrate material name/id if has_substrate is True
    environment_index : float
        Refractive index of the environment
        
    Returns:
    --------
    dict : Dictionary containing field distributions
    """
    # Set up the grid for field calculations
    grid_size = 50  # Number of grid points in each dimension
    
    # Calculate field distributions
    electric_field = calculate_electric_field(
        wavelength, structure_type, materials, dimensions, 
        has_substrate, substrate_material, environment_index, grid_size
    )
    
    magnetic_field = calculate_magnetic_field(
        wavelength, structure_type, materials, dimensions, 
        has_substrate, substrate_material, environment_index, grid_size
    )
    
    power_field = calculate_power_field(electric_field, magnetic_field)
    
    # Return all field data
    return {
        'electric_field': electric_field,
        'magnetic_field': magnetic_field,
        'power_field': power_field,
        'wavelength': wavelength,
        'structure_type': structure_type
    }

def calculate_dispersive_properties(structure_type, materials, dimensions, 
                                   has_substrate, substrate_material, environment_index,
                                   wavelength_range):
    """
    Calculate wavelength-dependent optical properties
    
    Parameters:
    -----------
    structure_type : str
        Type of structure
    materials : list
        List of material names/ids
    dimensions : list
        List of dimensional parameters
    has_substrate : bool
        Whether the structure has a substrate
    substrate_material : str or None
        Substrate material name/id if has_substrate is True
    environment_index : float
        Refractive index of the environment
    wavelength_range : array
        Array of wavelengths for calculation
        
    Returns:
    --------
    dict : Dictionary with wavelength-dependent properties
    """
    # Initialize arrays for results
    n_wavelengths = len(wavelength_range)
    reflectance = np.zeros(n_wavelengths)
    transmittance = np.zeros(n_wavelengths)
    absorption = np.zeros(n_wavelengths)
    
    # Calculate properties for each wavelength
    for i, wavelength in enumerate(wavelength_range):
        # For this simplified implementation, we'll use a model based on
        # typical optical responses for different structure types
        
        # Base oscillation frequency determined by structure type
        if structure_type == '2D_Array':
            freq_factor = 0.015
        elif structure_type == '3D_Structure':
            freq_factor = 0.01
        elif structure_type == 'Thin_Film':
            freq_factor = 0.02
        else:  # 'Particle' or default
            freq_factor = 0.025
            
        # Material-dependent phase shift
        material_factor = sum([hash(str(m)) % 10 for m in materials]) / 10.0
        
        # Calculate optical properties using simplified models
        # (In a real implementation, this would use actual physics-based models)
        reflectance[i] = 0.3 + 0.2 * np.sin(freq_factor * wavelength + material_factor)
        transmittance[i] = 0.4 + 0.3 * np.cos(freq_factor * wavelength + material_factor + 1.5)
        
        # Ensure physical constraints
        if reflectance[i] + transmittance[i] > 1:
            # Scale down to ensure R + T <= 1
            total = reflectance[i] + transmittance[i]
            reflectance[i] /= total
            transmittance[i] /= total
        
        # Calculate absorption
        absorption[i] = 1 - reflectance[i] - transmittance[i]
    
    # Return results
    return {
        'wavelength': wavelength_range,
        'reflectance': reflectance,
        'transmittance': transmittance,
        'absorption': absorption
    }

def calculate_effective_parameters(structure_type, materials, dimensions, 
                                  has_substrate, substrate_material, environment_index,
                                  wavelength):
    """
    Calculate effective medium parameters of the metamaterial
    
    Parameters:
    -----------
    structure_type : str
        Type of structure
    materials : list
        List of material names/ids
    dimensions : list
        List of dimensional parameters
    has_substrate : bool
        Whether the structure has a substrate
    substrate_material : str or None
        Substrate material name/id if has_substrate is True
    environment_index : float
        Refractive index of the environment
    wavelength : float
        Wavelength for calculation
        
    Returns:
    --------
    dict : Dictionary with effective medium parameters
    """
    # Calculate effective parameters using simplified models
    # (In a real implementation, this would use actual effective medium theory)
    
    # Effective refractive index
    n_eff_real = 1.5 + 0.3 * np.sin(wavelength / 200) + 0.2 * environment_index
    n_eff_imag = 0.1 + 0.05 * np.cos(wavelength / 300)
    
    # Effective permittivity and permeability
    eps_eff_real = n_eff_real**2 - n_eff_imag**2
    eps_eff_imag = 2 * n_eff_real * n_eff_imag
    
    mu_eff_real = 1.0 + 0.1 * np.sin(wavelength / 250)
    mu_eff_imag = 0.05 + 0.02 * np.cos(wavelength / 350)
    
    # Return effective parameters
    return {
        'n_eff': complex(n_eff_real, n_eff_imag),
        'eps_eff': complex(eps_eff_real, eps_eff_imag),
        'mu_eff': complex(mu_eff_real, mu_eff_imag),
        'wavelength': wavelength
    }
