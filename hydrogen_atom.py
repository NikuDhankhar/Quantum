import numpy as np 
import matplotlib.pyplot as plt
from hamiltonian_matrix import hamiltonian_matrix

#a few constants:
hbar = 1.054571817 * 10**(-34)
m_e = 9.11 * 10**(-31)
epsilon = 8.85 * 10**(-12)
e = 1.6 * 10**(-19)
A = -hbar**2/(2*m_e)

import numpy as np 
import matplotlib.pyplot as plt

# Constructing Hamiltonian matrix
def hamiltonian_matrix(n, dr, potential_values):
    
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
    kinetic_energy = A*kinetic_matrix/dr**2
    
    """ Potential Energy Matrix """
    potential_energy = np.diag(potential_values)
    
    hamiltonian = kinetic_energy + potential_energy
    
    return hamiltonian

def hydrogen_atom(r,l):
    potential = -e**2/(4*np.pi*epsilon*r) - A*l*(l+1)/r**2
    return potential

def solve_and_plot(r,l,interior_points, state_number,nl):
    if interior_points <= 0:
        raise ValueError("interior_points must be positive.")

    if state_number < 0 or state_number >= interior_points:
        raise ValueError("Invalid state_number.")
    
    dr = r/(interior_points+1)
    interior_grid = np.linspace(dr, r-dr, interior_points)
    r_full = np.linspace(0, r, interior_points+2)
    potential = hydrogen_atom(interior_grid, l)
    
    hamiltonian = hamiltonian_matrix(interior_points, dr, potential)
    
    energies, wavevector = np.linalg.eigh(hamiltonian)
    
    # Plotting the normalized wavefunctions
    plt.figure(figsize=(10,6))
    for i in range(nl):
        phi =np.concatenate(([0],wavevector[:, i],[0]))
        if phi[np.argmax(np.abs(phi))] < 0:
            phi = -phi
        norm = np.sum(np.abs(phi)**2)*dr
        phi_norm = phi / np.sqrt(norm)
        plt.plot(r_full, phi_norm, label=f'Wavefunction (n={i+1}) ')
    
    plt.title(f' n= {state_number+1} , Energy= {energies[state_number]:.2e}Joules')
    plt.xlabel('radius (r)')
    plt.ylabel('radial wavefunction u(r)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.grid(True)
    plt.show()
r = 20*10**(-10)
solve_and_plot(r,0,1000,0,1)