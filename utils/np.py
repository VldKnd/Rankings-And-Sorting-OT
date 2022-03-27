import numpy as np
import matplotlib.pyplot as plt

def get_distr(u, v, K):
    """
    Given a, b functions retunrs the distribution of mass
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
        Vector a computed by Sinkhorn
        
    b : np.array, dims = (N_y)
        Vector b computed by Sinkhorn

    K : np.arrya, dims = (N_x, N_y)
        The kernel function of pairs of objects
        
    """
    
    return ((np.diag(u).dot(K)).dot(np.diag(v)))

def Sinkhorn(a, b, X , Y, eps, h, nu):
    """
    The Sinkhorn algorithm to compute u, v functions
    
    This is the numericaly unstable version, does not work with eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    X : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    Y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float,
        Strength of regularisation 
        
    h : function,
        Function representing distance between two elements. Its expected to be convex
        
    nu : float,
        Error to tolerate for convergance
    """
    
    size_x = X.shape[0]
    size_y = Y.shape[0]
    C = np.empty((size_x, size_y))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            C[i, j] = h(x, y)
            
    K = np.exp(-C/eps)
    u = np.ones_like(X)

    v = b/(K.T@u)
    u = a/(K@v)

    while np.abs(v*(K.T@u) - b).sum() > nu:
        v = b/(K.T@u)
        u = a/(K@v)
        
    return u, v, K
    
def soft_min(M, eps):
    """
    The soft minimum function
    
    Parameters
    ----------
    M : np.array, dims = (N_x, N_y)
         Input array to perform minimization line wise
    
    eps : float,
        Strength of regularisation 
        
    """
    
    return -eps*np.log(np.exp(-M/eps).sum(axis=1, keepdims=True))
    
def log_Sinkhorn(a, b, X , Y, eps, h, nu):
    """
    The Sinkhorn algorithm to compute u, v functions
    
    This is the numericaly stable version. To be applied if eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    X : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    Y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float,
        Strength of regularisation 
        
    h : function,
        Function representing distance between two elements. Its expected to be convex
        
    nu : float, optional
        Error to tolerate for convergance
    """
    
    size_x = X.shape[0]
    size_y = Y.shape[0]
    
    C = np.empty((size_x, size_y))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            C[i, j] = h(x, y)
    
    
    alpha = np.zeros((size_x, 1))
    betta = np.zeros((size_y, 1))
    
    ones_x = np.ones((size_x, 1))
    ones_y = np.ones((size_y, 1))
    
    alpha = eps*np.log(a) + soft_min(C - alpha@ones_y.T - ones_x@betta.T, eps) + alpha
    betta = eps*np.log(b) + soft_min(C.T - ones_y@alpha.T - betta@ones_x.T, eps) + betta
    
    while np.abs(
            np.exp(C.T - ones_y@alpha.T - betta@ones_x.T)@ones_x \
            - b).sum() > nu:

        alpha = eps*np.log(a) + soft_min(C - alpha@ones_y.T - ones_x@betta.T, eps) + alpha
        betta = eps*np.log(b) + soft_min(C.T - ones_y@alpha.T - betta@ones_x.T, eps) + betta

    return alpha, betta, C

def Rank_Sort_log(a, x, b, y, u, v, eps, h, nu):
    """
    Function to get Ranking and Sorted values
    
    This is the numericaly stable version. To be applied if eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    x : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float, optional
        Strength of regularisation (default is 0.1)
        
    h : function, optional
        Function representing distance between two elements. Its expected to be convex (default is L2)
        
    nu : float, optional
        Error to tolerate for convergance (default is 1e-5)
    """
    
    alpha, betta, C = Sinkhorn(a, b, x, y, eps, h, nu)
    b_hat = np.cumsum(b)
    n = x.shape[0]
    
    R_tilda = (a**-1)*np.exp(C - alpha@np.ones_like(betta).T - np.ones_like(alpha)@betta.T)@b_hat
    S_tilda = (b**-1)*np.exp(C.T - np.ones_like(betta)@alpha.T - betta@np.ones_like(alpha).T)@x
    
    return R_tilda, S_tilda

def squash(x):
    """
    Normalisation 
    
    Parameters
    ----------
    x : np.array, dims = (N)
    """
    
    n = x.shape[0]
    x_sum = np.ones_like(x)*x.sum()
    d = (1/(n**1/2))*np.sum((x - x_sum)**2)
    return (x - x_sum)/d, x_sum, d
    
def unsquash(triple):
    """
    Inverse of normalisation 
    
    Parameters
    ----------
    triple : (np.array, np.array, float)
        (x, x_sum, d),
        x - vector to be normalised
        x_sum - vector of sums, pre-computed during application of squash function
        d - normalising constant that were computed during application of squash function
    """
    
    x, x_sum, d = triple
    return x*d + x_sum

def sigmoid(x):
    """
    Sigmoid mapping 
    
    Parameters
    ----------
    x : np.array, dims = (N)
    """
    
    return 1/(1+np.exp(-x))

def sigmoid_inv(x):
    """
    Inverse of sigmoid mapping 
    
    Parameters
    ----------
    x : np.array, dims = (N)
    """
    
    return np.log(x/(1-x))

def L2(x, y):
    """
    L2 distance
    
    Parameters
    ----------
    x : np.array, dims = (N)
    
    y : np.array, dims = (N)
    """
    
    return np.sum( (x - y)**2 )

def Rank_Sort(a, x, b, y, eps=0.1, h=L2, nu=1e-5):
    """
    Function to get Ranking and Sorted values
    
    This is the numericaly unstable version, does not work with eps <= 0.01.
    
    Parameters
    ----------
    a : np.array, dims = (N_x)
         Marginal distribution of x's
        
    x : np.array, dims = (N_x)
        Numbers to be sorted,  (default is 0.1)
        
    b : np.array, dims = (N_y)
        Marginal distribution of y's
        
    y : np.array, dims = (N_y)
        Milestones to be compared against, its expected to be equaly space from 0 to 1
        
    eps : float, optional
        Strength of regularisation (default is 0.1)
        
    h : function, optional
        Function representing distance between two elements. Its expected to be convex (default is L2)
        
    nu : float, optional
        Error to tolerate for convergance (default is 1e-5)
    """
        
    u, v, K = Sinkhorn(a, b, x, y, eps, h, nu)
    b_hat = np.cumsum(b)
    n = x.shape[0]
    
    R_tilda = (n*(a**-1))*u*(K@(v*b_hat))
    S_tilda = (b**-1)*v*(K.T@(u*x))
    
    return R_tilda, S_tilda