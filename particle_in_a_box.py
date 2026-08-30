import numpy as np 
import matplotlib.pyplot as plt

# Constructing Hamiltonian matrix
def hamiltonian_matrix(n, dx, potential_values):
    
    """ To construct the hamiltonian matrix we need:
        n : (int) number of interior grid points
        dx : (float) grid spacing
        potential_values : (array like) potential energy value at every grid point
    """
    
    """ Kinetic Matrix """
    kinetic_matrix = np.zeros((n, n))
    for i in range(n):
        kinetic_matrix[i,i] = -2 
        if i+1 < n:
            kinetic_matrix[i+1,i]= kinetic_matrix[i,i+1] = 1 
    kinetic_energy = -0.5*kinetic_matrix/(dx**2)
    
    """ Potential Energy Matrix """
    potential_energy = np.diag(potential_values)
    
    hamiltonian_matrix = kinetic_energy + potential_energy
    
    return hamiltonian_matrix

# Infinite Potential Well
def infinite_well(L, interior_points, state_number):
    """ Solving particle in a box problem in 1-D
        L : width of the box
        interior_points : the number of grid points inside the wall
        state_number : quantum state to plot; starts from 0
    """
    
    if interior_points <= 0:
        raise ValueError("interior_points must be positive.")

    if state_number < 0 or state_number >= interior_points:
        raise ValueError("Invalid state_number.")
    
    
    dx = L/(interior_points+1)
    x_full = np.linspace(0, L, interior_points + 2)
    
    # Infinite well:  V = 0  inside the walls
    potential_values = np.zeros(interior_points)
    
    hamiltonian = hamiltonian_matrix(interior_points, dx, potential_values)
    
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    quantum_number = np.arange(1, len(energies)+1)
    analytical_energies = quantum_number**2 * np.pi**2 / (2 * L**2)
    
    # Adding the end points
    phi = np.concatenate(([0], wavefunctions[:, state_number], [0]))
    
    # Normalization of the wavefunction
    norm = np.sum(np.abs(phi)**2)*dx
    phi_norm = phi / np.sqrt(norm)
    
    # Probability density
    probability_density = np.abs(phi_norm)**2
    
    # Plotting the normalized wavefunctions
    plt.figure(figsize=(10,6))
    plt.plot(x_full, phi_norm, label=f'Wavefunction (n={state_number+1}) ' , color='blue')
    plt.title(f' n= {state_number+1} , Energy= {energies[state_number+1]}')
    plt.xlabel('Position (x)')
    plt.ylabel('Wavefunction (ψ)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # plotting the probability density 
    plt.plot(x_full, probability_density)
    plt.title(f' Probability density ')
    plt.xlabel('Position (x)')
    plt.ylabel(' (|ψ|^2)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotting the energy vs quantum number graph
    plt.figure(figsize=(10, 6))
    plt.plot(quantum_number,energies,label="Numerical energy")
    plt.plot(quantum_number,analytical_energies,label="Analytical energy")
    plt.xlabel("Quantum Number (n)")
    plt.ylabel("Energy")
    plt.title("Numerical vs Analytical Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

infinite_well(2,500,0)
"""
    later work on step potential.
"""

