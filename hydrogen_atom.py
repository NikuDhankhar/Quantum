import numpy as np 
import matplotlib.pyplot as plt

#a few constants:
hbar = 1.054571817 * 10**(-34) #joules second
m_e = 9.11 * 10**(-31)  # kg
epsilon_0 = 8.85 * 10**(-12)  # C2 N-1 m-2 OR Farad/metre
e = 1.6 * 10**(-19)     # coulombs
A = -hbar**2/(2*m_e)

# Bohr radius
a0 = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2)

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

def hydrogen_potential(r, l):
    # Coulomb potential
    coulomb = -e**2 / (4 * np.pi * epsilon_0 * r)
    # Centrifugal potential
    centrifugal = hbar**2 * l * (l + 1) / (2 * m_e * r**2)
    return coulomb + centrifugal

def solve_and_plot(r_max,l,interior_points, state_number,nl):
    if interior_points <= 0:
        raise ValueError("interior_points must be positive.")

    if state_number < 0 or state_number >= interior_points:
        raise ValueError("Invalid state_number.")
    
    dr = r_max/(interior_points+1)

    r = np.linspace(dr, r_max-dr, interior_points)
    potential = hydrogen_potential(r, l)
    
    hamiltonian = hamiltonian_matrix(interior_points, dr, potential)
    
    energies, wavevector = np.linalg.eigh(hamiltonian)
    
    # Plotting the normalized wavefunctions
    plt.figure(figsize=(10,6))
    for i in range(nl):
        u = wavevector[:, i]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        norm = np.sum(np.abs(u)**2)*dr
        u_norm = u / np.sqrt(norm)
        
        #calculating R
        R = u_norm/r
        
        plt.plot(r/a0, R,label=r"$R(r)$" )
    plt.title(f' n= {state_number+1} , Energy= {energies[state_number]:.2e}Joules')
    plt.xlabel('r/a0')
    plt.ylabel('radial wavefunction R(r)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.grid(True)
    plt.show()
r = 20*10**(-10)
solve_and_plot(r,0,1000,0,1)