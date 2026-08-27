import numpy as np 
import matplotlib.pyplot as plt

#variables used:
# L = Length of the well
# interior_points = Number of interior points in the well
# dx = Spacing between points
# n = Number of interior points
# 
# hamiltonian matrix and potential energy matrix
def matrix(n, dx, potential_values):
    #hamiltonian matrix
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i,i] = -2 
        if i+1 < n:
            mat[i+1,i]=mat[i,i+1] = 1 
    hamilt = -0.5*mat/(dx**2)
    
    # potential energy matrix
    potential_energy = np.diag(potential_values)
    print(potential_energy)
    
    matrix = hamilt + potential_energy
    
    return matrix

#inifinite potential well
def infinite_well(L, interior_points, state_number):
    dx = L/(interior_points+1)
    x = np.linspace(dx, L-dx, interior_points)
    x_full = np.linspace(0, L, interior_points + 2)
    potential_values = np.zeros(interior_points)  
    mat = matrix(interior_points, dx, potential_values)
    
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    
    #sorting eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    energies = eigenvalues[idx]
    wavefunctions = eigenvectors[:, idx]
    phi = np.concatenate(([0], wavefunctions[:, state_number], [0]))
    
    #normalization of the wavefunction
    norm = np.sum(np.abs(phi)**2)*dx
    phi = phi / np.sqrt(norm)
    
    #area under the curve
    area = np.trapz(np.abs(phi)**2, x_full)
    print(f"Area under the curve for state {state_number+1}: {area}")
    
    #plotting the wavefunctions
    plt.figure(figsize=(10,6))
    plt.plot(x_full, phi, label=f'Wavefunction (n={state_number+1}) ' , color='blue')
    plt.title(f'State {state_number+1} , Area: {area:.4f}')
    plt.xlabel('Position (x)')
    plt.ylabel('Wavefunction (ψ)')
    plt.legend()
    plt.grid(True)
    plt.show()
    


infinite_well(1, 200, 0)