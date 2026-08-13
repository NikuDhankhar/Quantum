import numpy as np
import matplotlib.pyplot as plt

L = 2.0 #length of the box
n = 200
x = np.linspace(0,L,n)
dx = x[1] - x[0]
mat = np.zeros((n,n))

for i in range(n):
    mat[i,i] = -2
    if i+1 < n:
        mat[i,i+1] = mat[i+1,i] = 1
    else:
        break
    
hamiltonian = -0.5*mat/(dx**2)

eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)

idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

plt.plot(x, eigenvectors[:,0], label='n=1')
plt.plot(x, eigenvectors[:,1], label='n=2')
plt.xlabel("x")
plt.ylabel("wavefunction")
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()
