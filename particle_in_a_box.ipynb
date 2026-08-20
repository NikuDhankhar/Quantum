import numpy as np
import matplotlib.pyplot as plt

L = 2.0 #length of the box
n = 200 #number of grid points
dx = L/(n+1) #grid spacing
x_inner = np.linspace(dx ,L-dx ,n) #inner grid points

x_full = np.linspace(0, L, n + 2)

#creating matrix of n*n order
mat = np.zeros((n,n))

# filling the matrix with values
for i in range(n):
    mat[i,i] = -2
    if i+1 < n:
        mat[i,i+1] = mat[i+1,i] = 1
    else:
        break

# defining the hamiltonian
hamiltonian = -0.5*mat/(dx**2)

#eigevalues are energies and eigenvectors are wavefunctiona
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)

idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# normalizing the wavefunctions
for i in range(3):
    psi = eigenvectors[:,i]
    eigenvectors[:,i] /= np.sqrt(np.sum(eigenvectors[:,i]**2)*dx)
    
    psi_full = np.concatenate(([0], psi, [0]))
    if psi_full[2] < 0: 
        psi_full = -psi_full
    plt.plot(x_full, psi_full, label=f'n={i+1}')


plt.xlabel("x")
plt.ylabel("wavefunction")
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()
