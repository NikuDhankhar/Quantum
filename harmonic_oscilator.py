import numpy as np 
import matplotlib.pyplot as plt
from hamiltonian_matrix import hamiltonian_matrix

def harmonic_osci(x):
    potential = 0.5 * x**2
    return potential

def solve_and_plot(L,interior_points,state_number):
    if interior_points <= 0:
        raise ValueError("interior_points must be positive.")
    
    if state_number < 0 or state_number >= interior_points:
        raise ValueError("Invalid state_number.")
    if L < 0:
        raise ValueError("length can not be negative.")
        
    dx = 2*L/(interior_points+1)
    x = np.linspace(-L+dx,L-dx, interior_points)
    x_full = np.linspace(-L, L, interior_points+2)
    
    potential_values = harmonic_osci(x)
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
    
solve_and_plot(8,1000,0)