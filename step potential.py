import numpy as np 
import matplotlib.pyplot as plt

def step_potential(x, V0=1.0, step_position=2):

    x = np.array(x, dtype=float)
    V = np.where(x < step_position, 0.0, V0)
    return V 

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

def solving_plotting(L, V0, x0, interior_points, state_number):
    
    if interior_points <= 0:
        raise ValueError("interior_points must be positive.")

    if state_number < 0 or state_number >= interior_points:
        raise ValueError("Invalid state_number.")
    if L < 0:
        raise ValueError("length can not be negative.")
    
    dx = L/(interior_points+1)
    x = np.linspace(dx, L-dx, interior_points)
    x_full = np.linspace(0, L, interior_points+2)
    
    potential_values = step_potential(x, V0, x0)
    hamiltonian = hamiltonian_matrix(interior_points, dx, potential_values)
    
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    
    phi_interior = eigenvectors[:, state_number]
    energies = eigenvalues[state_number]
    #normalisation:
    phi = np.concatenate(([0], phi_interior, [0]))
    norm = np.sum(np.abs(phi)**2)*dx
    phi_norm = phi/np.sqrt(norm)
    
    plt.plot(x_full, phi_norm, label=f'Wavefunction (n={state_number+1}) ' , color='blue')
    plt.title(f' n= {state_number+1} , Energy= {energies}')
    plt.xlabel('Position (x)')
    plt.ylabel('Wavefunction (ψ)')
    plt.legend()
    plt.grid(True)
    plt.show()

solving_plotting(4, 10, 2, 1000, 1)