import numpy as np
from scipy.special import gamma
from scipy.sparse import diags

def gaussian_ensemble_density(lambdas, beta):
    """
    Given a numpy array of eigenvalues, lambdas, will compute joint density for beta ensemble
    """
    n = len(lambdas)
    constant = (2*np.pi)**(n/2) * np.prod([gamma(1+k*beta/2)/gamma(1+beta/2) for k in range(1,n+1)])
    return 1/constant * np.exp(-.5*np.sum(lambdas**2)) * np.prod(np.abs(np.diff(lambdas))**beta)

def Generate_GOE(n):
    """Creates nxn GOE"""
    A = np.random.normal(size=[n,n])
    G = (A+A.T)/2
    return G

def Generate_GUE(n):
    """Creates nxn GUE"""
    i = complex(0,1)
    Lambda_real = np.random.normal(size=[n,n])
    Lambda_im = np.random.normal(size=[n,n])
    Lambda = Lambda_real + Lambda_im * i
    G = (Lambda+Lambda.T.conjugate())/2
    return G

def Generate_Ginibre(n):
    """Creates nxn Ginibre matrix"""
    G_real = np.random.normal(scale= np.sqrt(1/(2*n)), size=[n,n])
    G_im = np.random.normal(scale= np.sqrt(1/(2*n)), size=[n,n]) * complex(0,1)
    G = G_real + G_im
    return G

def Generate_Hermite(n, beta):
    """Creates nxn Hermite matrix"""
    main_diag = np.sqrt(2) * np.random.normal(size=n)
    off_diag = [np.sqrt(np.random.chisquare(beta * (n-i))) for i in range(1,n)]
    H = diags([off_diag, main_diag, off_diag], [-1,0,1]).toarray()/np.sqrt(2)
    return H

def Generate_Custom(f, n, m):
    """Create a nxm matrix A such that A_ij = f(i,j)"""
    return np.fromfunction(np.vectorize(f, otypes=[float]), (n,m))