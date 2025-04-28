import numpy as np
from utils.materials_database import get_refractive_index

def calculate_electric_field(wavelength, structure_type, materials, dimensions, 
                           has_substrate, substrate_material, environment_index, grid_size=50):
    """
    Calculate electric field distribution
    
    Parameters:
    -----------
    wavelength : float
        Wavelength in nm
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
    grid_size : int
        Size of the computation grid
        
    Returns:
    --------
    dict : Dictionary containing field components and metadata
    """
    # Create a 3D grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    
    # Initialize field components
    E_x = np.zeros((grid_size, grid_size, grid_size))
    E_y = np.zeros((grid_size, grid_size, grid_size))
    E_z = np.zeros((grid_size, grid_size, grid_size))
    
    # Calculate normalized frequency (wavelength in grid units)
    k0 = 2 * np.pi / wavelength
    
    # Define structure-specific field calculation
    if structure_type == '2D_Array':
        # For 2D array, create a field enhancement near periodic structures
        period = dimensions[0] / 500.0  # Normalize to typical period (~500nm)
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for l, zi in enumerate(z):
                    # Distance from nearest array element
                    dist_x = np.abs((xi / period) % 1.0 - 0.5) * 2
                    dist_y = np.abs((yi / period) % 1.0 - 0.5) * 2
                    
                    # Field localization near array elements
                    field_factor = np.exp(-5 * (dist_x**2 + dist_y**2))
                    
                    # Add incident field
                    E_x[i, j, l] = np.cos(k0 * environment_index * zi)
                    E_y[i, j, l] = 0.2 * np.sin(k0 * environment_index * zi)
                    
                    # Add localized field enhancement
                    E_x[i, j, l] += 2.0 * field_factor * np.cos(k0 * environment_index * zi + np.pi/4)
                    E_y[i, j, l] += 1.5 * field_factor * np.sin(k0 * environment_index * zi + np.pi/4)
                    
                    # Add some z-component for more realistic field
                    E_z[i, j, l] = 0.5 * field_factor * np.sin(k0 * environment_index * zi)
    
    elif structure_type == 'Thin_Film':
        # For thin film, create interference pattern
        # Get refractive indices
        n_materials = [get_refractive_index(mat, wavelength) for mat in materials]
        
        # Layer thicknesses (normalized)
        if len(dimensions) >= len(materials):
            thicknesses = dimensions[:len(materials)]
        else:
            thicknesses = dimensions + [100] * (len(materials) - len(dimensions))
        
        total_thickness = sum(thicknesses)
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for l, zi in enumerate(z):
                    # Convert z to position in thin film
                    z_pos = (zi + 1) * total_thickness / 2
                    
                    # Determine which layer this point is in
                    current_thickness = 0
                    current_layer = 0
                    
                    for layer, thickness in enumerate(thicknesses):
                        if current_thickness <= z_pos < (current_thickness + thickness):
                            current_layer = layer
                            break
                        current_thickness += thickness
                    
                    # Get refractive index for the current layer
                    if current_layer < len(n_materials):
                        n_layer = n_materials[current_layer]
                    else:
                        n_layer = complex(environment_index, 0)
                    
                    # Calculate field based on multiple interface reflections (simplified)
                    reflection_factor = 0.2 * np.sin(2 * np.pi * current_layer / len(materials))
                    
                    # Incident and reflected waves
                    E_forward = np.exp(1j * k0 * n_layer.real * zi)
                    E_backward = reflection_factor * np.exp(-1j * k0 * n_layer.real * zi)
                    
                    # Total field (simplified interference)
                    E_total = E_forward + E_backward
                    
                    # Set field components
                    E_x[i, j, l] = np.abs(E_total) * np.cos(k0 * zi)
                    E_y[i, j, l] = 0.2 * np.abs(E_total) * np.sin(k0 * zi)
                    E_z[i, j, l] = 0.1 * np.abs(E_total) * np.sin(2 * k0 * zi)
    
    elif structure_type == 'Particle':
        # For particles, create field enhancement near the particle
        particle_radius = dimensions[0] / 100.0  # Normalize to typical radius (~100nm)
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for l, zi in enumerate(z):
                    # Distance from particle center
                    r = np.sqrt(xi**2 + yi**2 + zi**2)
                    
                    # Field enhancement factor (dipole-like)
                    if r < particle_radius:
                        # Inside particle: reduced field
                        field_factor = 0.5
                    else:
                        # Outside particle: enhanced field with r^-3 decay
                        field_factor = 1.0 + 2.0 * (particle_radius**3) / (r**3) if r > 0 else 3.0
                    
                    # Incident field
                    E_inc_x = np.cos(k0 * environment_index * zi)
                    E_inc_y = 0.0
                    E_inc_z = 0.0
                    
                    # Apply enhancement factor
                    E_x[i, j, l] = field_factor * E_inc_x
                    E_y[i, j, l] = field_factor * E_inc_y
                    
                    # Add dipole pattern (simplified)
                    if r > particle_radius and r > 0:
                        theta = np.arccos(zi / r)
                        E_z[i, j, l] = 0.5 * field_factor * np.sin(theta) * np.cos(k0 * r)
    
    else:  # 3D_Structure or default
        # For 3D structures, create more complex field patterns
        structure_size = dimensions[0] / 500.0 if dimensions else 0.5
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for l, zi in enumerate(z):
                    # Create a complex 3D field pattern
                    E_x[i, j, l] = np.cos(k0 * xi) * np.sin(k0 * yi) * np.cos(k0 * zi)
                    E_y[i, j, l] = np.sin(k0 * xi) * np.cos(k0 * yi) * np.sin(k0 * zi)
                    E_z[i, j, l] = np.cos(k0 * xi) * np.cos(k0 * yi) * np.sin(k0 * zi)
                    
                    # Add enhancement for points inside the structure
                    if (abs(xi) < structure_size and 
                        abs(yi) < structure_size and 
                        abs(zi) < structure_size):
                        
                        enhancement = 2.0 + np.sin(5 * k0 * xi) * np.cos(5 * k0 * yi) * np.sin(5 * k0 * zi)
                        E_x[i, j, l] *= enhancement
                        E_y[i, j, l] *= enhancement
                        E_z[i, j, l] *= enhancement
    
    # Add substrate effects if applicable
    if has_substrate and substrate_material:
        n_substrate = get_refractive_index(substrate_material, wavelength)
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for l, zi in enumerate(z):
                    # Assume substrate is at z < 0
                    if zi < 0:
                        # Refractive index contrast
                        n_ratio = environment_index / n_substrate.real
                        
                        # Modify fields in substrate region
                        E_x[i, j, l] *= n_ratio
                        E_y[i, j, l] *= n_ratio
                        E_z[i, j, l] *= n_ratio**2  # Different scaling for z-component
    
    # Add some random noise for realism
    noise_level = 0.05
    E_x += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    E_y += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    E_z += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    
    # Return the field components and metadata
    return {
        'x': E_x,
        'y': E_y,
        'z': E_z,
        'shape': (grid_size, grid_size, grid_size),
        'wavelength': wavelength,
        'structure_type': structure_type
    }

def calculate_magnetic_field(wavelength, structure_type, materials, dimensions, 
                            has_substrate, substrate_material, environment_index, grid_size=50):
    """
    Calculate magnetic field distribution
    
    Parameters:
    -----------
    wavelength : float
        Wavelength in nm
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
    grid_size : int
        Size of the computation grid
        
    Returns:
    --------
    dict : Dictionary containing field components and metadata
    """
    # First calculate electric field
    E_field = calculate_electric_field(
        wavelength, structure_type, materials, dimensions, 
        has_substrate, substrate_material, environment_index, grid_size
    )
    
    # Create a 3D grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    
    # Initialize magnetic field components
    H_x = np.zeros((grid_size, grid_size, grid_size))
    H_y = np.zeros((grid_size, grid_size, grid_size))
    H_z = np.zeros((grid_size, grid_size, grid_size))
    
    # Extract electric field components
    E_x = E_field['x']
    E_y = E_field['y']
    E_z = E_field['z']
    
    # Calculate normalized frequency (wavelength in grid units)
    k0 = 2 * np.pi / wavelength
    
    # Calculate grid spacing
    dx = 2.0 / (grid_size - 1)
    dy = 2.0 / (grid_size - 1)
    dz = 2.0 / (grid_size - 1)
    
    # For simplicity, we'll approximate the magnetic field using Maxwell's equations
    # In a more rigorous implementation, one would solve Maxwell's equations directly
    
    # Approximating curl of E to get H
    # H ∝ ∇ × E
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            for l in range(1, grid_size-1):
                # Finite difference approximation of curl
                dEz_dy = (E_z[i, j+1, l] - E_z[i, j-1, l]) / (2 * dy)
                dEy_dz = (E_y[i, j, l+1] - E_y[i, j, l-1]) / (2 * dz)
                
                dEx_dz = (E_x[i, j, l+1] - E_x[i, j, l-1]) / (2 * dz)
                dEz_dx = (E_z[i+1, j, l] - E_z[i-1, j, l]) / (2 * dx)
                
                dEy_dx = (E_y[i+1, j, l] - E_y[i-1, j, l]) / (2 * dx)
                dEx_dy = (E_x[i, j+1, l] - E_x[i, j-1, l]) / (2 * dy)
                
                # H = ∇ × E (with constants simplified)
                H_x[i, j, l] = dEz_dy - dEy_dz
                H_y[i, j, l] = dEx_dz - dEz_dx
                H_z[i, j, l] = dEy_dx - dEx_dy
    
    # Scale magnetic field for reasonable amplitudes
    scale_factor = 0.2 * wavelength / (2 * np.pi * environment_index)
    H_x *= scale_factor
    H_y *= scale_factor
    H_z *= scale_factor
    
    # Add some random noise for realism
    noise_level = 0.05
    H_x += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    H_y += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    H_z += noise_level * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
    
    # Return the field components and metadata
    return {
        'x': H_x,
        'y': H_y,
        'z': H_z,
        'shape': (grid_size, grid_size, grid_size),
        'wavelength': wavelength,
        'structure_type': structure_type
    }

def calculate_power_field(electric_field, magnetic_field):
    """
    Calculate power flow (Poynting vector) from electric and magnetic fields
    
    Parameters:
    -----------
    electric_field : dict
        Dictionary containing electric field components
    magnetic_field : dict
        Dictionary containing magnetic field components
        
    Returns:
    --------
    dict : Dictionary containing power field components and metadata
    """
    # Extract field components
    E_x = electric_field['x']
    E_y = electric_field['y']
    E_z = electric_field['z']
    
    H_x = magnetic_field['x']
    H_y = magnetic_field['y']
    H_z = magnetic_field['z']
    
    # Get grid shape
    grid_shape = electric_field['shape']
    grid_size = grid_shape[0]
    
    # Initialize power field (Poynting vector) components
    S_x = np.zeros(grid_shape)
    S_y = np.zeros(grid_shape)
    S_z = np.zeros(grid_shape)
    
    # Calculate Poynting vector: S = E × H
    for i in range(grid_size):
        for j in range(grid_size):
            for l in range(grid_size):
                S_x[i, j, l] = E_y[i, j, l] * H_z[i, j, l] - E_z[i, j, l] * H_y[i, j, l]
                S_y[i, j, l] = E_z[i, j, l] * H_x[i, j, l] - E_x[i, j, l] * H_z[i, j, l]
                S_z[i, j, l] = E_x[i, j, l] * H_y[i, j, l] - E_y[i, j, l] * H_x[i, j, l]
    
    # Return the power field components and metadata
    return {
        'x': S_x,
        'y': S_y,
        'z': S_z,
        'shape': grid_shape,
        'wavelength': electric_field['wavelength'],
        'structure_type': electric_field['structure_type']
    }
