import numpy as np
from scipy.integrate import simpson as simps

def normalize(grid, pz): 
    return pz/simps(pz, grid)

class Vector:
    def __init__(self, grid, pz):
        assert grid[0] == -grid[-1]
        self.grid = grid
        self.pz = normalize(grid, pz)
        self.l = (grid[-1] - grid[0])/2.
        self.normalize()

def C(grid, g):   
    return Vector(grid, normalize(grid, g.pz))

def scal_mult(alpha, g):
    return C(g.grid, np.power(g.pz, alpha))

def add(f, g):
    assert g.grid == f.grid
    return C(f.grid, f.pz * g.pz)

def inner_prod(f, g):
    assert f.grid == g.grid
    first_term = simps(np.log(f.pz) * np.log(g.pz), x=grid)
    second_term = (1. / (2 * l)) * simps(np.log(f.pz), x=f.grid) * simps(np.log(g.pz), x=grid)
    return first_term - second_term

def norm(f):
        
    first_term = simps(np.log(f.pz) * np.log(f.pz), x=f.grid)
    second_term = simps(np.log(f.pz), x=grid)**2
    completed = first_term - (1. / (2 * l)) * second_term
    return np.sqrt(completed)

def distance(f, g):
    assert f.grid == g.grid
    return np.sqrt(inner_prod(f.pz, f.pz) - 2. * inner_prod(f.pz, g.pz) + inner_prod(g.pz, g.pz))


