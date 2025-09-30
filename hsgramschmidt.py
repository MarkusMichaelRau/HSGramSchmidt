import numpy as np 
from matplotlib import pyplot as plt 
from scipy.integrate import simpson as simps
from scipy.stats import norm, lognorm
from scipy.stats import cauchy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numdifftools as nd
from scipy.special import eval_legendre
from scipy.stats import uniform


class HilbertSpace(object):
    
    def __init__(self, num):
        self.l = 1
        self.grid = np.linspace(-self.l, self.l, num=num)
    
    def C(self,g):
    
        return g/simps(g, x=self.grid)
    
    def scal_mult(self, alpha, g):
        return np.pow(g, alpha)

    def add(self, f, g):
        return self.C(f, g)
    
    def inner_prod(self, f, g):
    
        first_term = simps(np.log(f)*np.log(g), x=self.grid) 
        second_term = 1./(2*self.l)*simps(np.log(f), x=self.grid)* simps(np.log(g), x=self.grid)
        return first_term - second_term
    
    def norm(self, f):
        print(f)
        first_term = simps(np.log(f)*np.log(f), x=self.grid)
        second_term= simps(np.log(f), x=self.grid)**2
        completed = first_term - (1./(2*self.l))*second_term
        return np.sqrt(completed)
    
    def distance(self, f, g):
        return np.sqrt(self.inner_prod(f, f) - self.inner_prod(f, g) - self.inner_prod(g, f) + self.inner_prod(g, g))
        
        
    def check(self, f):
        if not np.isclose(np.sum(f.grid - self.grid),0.):
            raise ValueError('Incompatible pdf')

        else:
            return 0
        
    def legendre_basis(self, n):
        x = self.grid
        return self.C(np.exp(np.sqrt((2*n+1)/(2*self.l)) * eval_legendre(n, x)))
    
    def legendre_aich(self, n):
        x = self.grid
        return np.sqrt((2*n+1)/(2*self.l)) * eval_legendre(n, x)


class PDF(object):
    def __init__(self, grid, pdf):
        if len(grid)%2 > 0:
            raise ValueError('need odd number of grid points')
            
        self.mids = grid
        self.pdf= pdf/np.trapz(pdf, self.mids)
                
        self.hs = HilbertSpace(len(self.mids))

    def represent(self, num_coeff):
        loss = lambda x: np.mean((np.sum([x[j]*self.hs.legendre_aich(j) for j in range(num_coeff)], axis=0) - np.log(self.pdf))**2)

        res = minimize(loss, x0=[0.1, 0.1, 0.1, 0.1, 0.1])
        return res

    def evaluate(self, coefs): 

        output_pdf = np.exp(np.sum([coefs[j]*self.hs.legendre_aich(j) for j in range(len(coefs))], axis=0))
        
        return output_pdf/np.trapz(output_pdf, self.mids)
