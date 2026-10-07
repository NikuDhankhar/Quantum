import numpy as np
import matplotlib.pyplot as plt

def RK4(f,U0,t):
    n = len(t)
    m = len(U0)
    U =np.zeros((n,m))
    U[0]=U0
    
    for i in range(n-1):
        h = t[i+1]-t[i]
        
        k1 = f(U[i],t[i])
        k2 = f(U[i]+h*k1/2,t[i]+h/2)
        k3 = f(U[i]+h* k2/2,t[i]+h/2)
        k4 = f(U[i]+h*k3,t[i]+h)       
        U[i+1]=U[i]+h*(k1+2*(k2+k3)+k4)/6
    return U