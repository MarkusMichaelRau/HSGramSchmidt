import numpy as np
from hs import *

def inner_product(v1, v2):
    """Calculate the inner (dot) product of two vectors."""
    return hs.inner_prod(v1, v2)

def scalar_multiply(scalar, vector):
    """Multiply a vector by a scalar."""
    return hs.scal_mult(scalar, vector)

def add(u, v):
    return hs.add(u, v)

def proj(v, u): 
    nom = inner_product(v, u)
    denom = inner_product(u, u)
    return scalar_multiply(nom/denom, u)

def substract(v, u):
    minus_u = scalar_multiply(-1, u)
    return add(v, minus_u)

def gram_schmidt_step(v, projection):

    minus_proj = scalar_multiply(-1, projection)

def gram_schmidt(vectors):
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors.
    
    Parameters:
        list of vectors: List or array of Vectors of the hilbert space class.
        
    Returns:
        list of vectors: Orthogonalized vectors as rows in a list.
    """
    
    n = len(vectors)

    orthogonal = [vectors[0]]*n

    for i in range(1, n):
        proj = proj(vectors[i], orthogonal[i-1])
        orthogonal[i] = substract(vectors[i], proj)
        
    return orthogonal



