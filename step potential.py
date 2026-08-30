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

