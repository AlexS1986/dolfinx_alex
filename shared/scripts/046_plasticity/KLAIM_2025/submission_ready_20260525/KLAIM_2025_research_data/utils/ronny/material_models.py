import util.aux
from typing import Protocol
import ufl
import numpy as np
import abc
import inspect

#general strain measures

def eps_lin(u): # Linearized strain tensor
    #return ufl.sym(ufl.grad(u))
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    
def trace_eps_lin(u): # Volume dilatation
    return ufl.tr(eps_lin(u))

def trace_eps_lin_neg(u):
    return ufl.conditional(ufl.le(trace_eps_lin(u), 0.0), trace_eps_lin(u), 0.0)

def trace_eps_lin_pos(u):
    return ufl.conditional(ufl.gt(trace_eps_lin(u), 0.0), trace_eps_lin(u), 0.0)

def trace_eps_lin_neg_pp(u):
    return 0.5 * (trace_eps_lin(u) - abs(trace_eps_lin(u)))

def trace_eps_lin_pos_pp(u):
    return 0.5 * (trace_eps_lin(u) + abs(trace_eps_lin(u)))

def epsD_lin(u): # Deviatoric part of linearized strain tensor
    return ufl.dev(eps_lin(u))

#Incremental strain energy densities

def delta_psi_el_D(u, u_old, sig_D):
    delta_epsD = epsD_lin(u) - epsD_lin(u_old)
    delta_eps = eps_lin(u) - eps_lin(u_old)
    #dpsi_el_D_undegraded = ufl.inner((sig_D_undegraded(u_old) + sig_D_undegraded(u))/2, depsD)
    delta_psi_el_D = (ufl.inner(sig_D(u_old), delta_eps) + ufl.inner(sig_D(u), delta_eps))/ 2
    return delta_psi_el_D

def delta_psi_el_V_pos(u, u_old, sig_V_pos):
    #depsV_pos = aux.pos(trace_eps_pos(u) - trace_eps_pos(u_old))
    delta_epsV = trace_eps_lin(u) - trace_eps_lin(u_old)
    delta_eps = eps_lin(u) - eps_lin(u_old)
    # dpsi_el_V_pos_undegraded = (sig_V_pos_scalar_undegraded(u_old) + sig_V_pos_scalar_undegraded(u))/2 * depsV
    delta_psi_el_V_pos = 0.5*(ufl.inner(sig_V_pos(u_old), delta_eps) + ufl.inner(sig_V_pos(u), delta_eps))
    return delta_psi_el_V_pos

def delta_psi_el_V_neg(u, u_old, sig_V_neg):
    #depsV_neg = aux.neg(trace_eps_neg(u) - trace_eps_neg(u_old))
    delta_epsV = trace_eps_lin(u) - trace_eps_lin(u_old)
    delta_eps = eps_lin(u) - eps_lin(u_old)
    ##dpsi_el_V_neg_undegraded = ufl.inner((sig_V_neg(u_old) + sig_V_neg(u))/2, depsV_neg)
    #dpsi_el_V_neg_undegraded = (sig_V_neg_scalar_undegraded(u_old) + sig_V_neg_scalar_undegraded(u))/2 * depsV
    delta_psi_el_V_neg = 0.5*(ufl.inner(sig_V_neg(u_old), delta_eps) + ufl.inner(sig_V_neg(u), delta_eps))
    return delta_psi_el_V_neg


#conversions

def lambda_from_E_G(E=None, G=None, dim=None):
    if dim == 2:
        return 2*G*(E-2*G)/(4*G-E)
    elif dim== 3:
        return G*(E-2*G)/(3*G-E)

def lambda_from_E_nu(E=None, nu=None, dim=None):
    if dim == 2:
        return E*nu/((1+nu)*(1-nu))
    elif dim == 3:
        return E*nu/((1+nu)*(1-2*nu))

def mu_from_E_nu(E=None, nu=None):
    return E/(2*(1+nu))

def Kmod_from_E_nu(E=None, nu=None, dim=None):
    if dim == 2:
        return E/(2*(1-nu))
    elif dim == 3:
        return E/(3*(1-2*nu))

def Kmod_from_lambda_mu(lmbda=None, mu=None, dim=None):
    return lmbda + 2*mu/dim



# class MaterialModelIsotropy(abc.ABC):
#     def G_mod(u): ...
#     def K_mod(u): ...
#     def eqeps(u): ...
#     def sig_D(u): ...
#     def sig_V(u): ...
#     def sig_V_scalar(u): ...
#     def sig_V_neg(u): ...
#     def sig_V_pos(u): ...
#     def sig(u): ...
#     #def sig_V_neg_pp(u): ...
#     #def sig_V_pos_pp(u): ...
    

class LinearElasticityIso():

    def __init__(self, dim = int, E = callable, G = callable, K = callable, nu = callable):
        self.dim = dim
        self.E = E
        self.G = G
        self.K = K
        self.nu = nu
            
    def G_mod(self):
        return self.G

    def K_mod(self):
        return self.K
    
    def eqeps(self, u):
        if self.dim == 2:
            return ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u)))
        else:
            return ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u))))
    
    def sig_D(self, u): #Deviatoric stress
       return 2*self.G_mod() * epsD_lin(u)
    
    def sig_V(self, u): #Volumetric stress
        return self.K_mod()*trace_eps_lin(u)*ufl.Identity(self.dim)
        
    def sig_V_scalar(self, u):
        return self.K_mod()*trace_eps_lin(u)
    
    def sig_V_neg_scalar(self, u):
        return self.K_mod()*trace_eps_lin_neg(u)
    
    def sig_V_pos_scalar(self, u):
        return self.K_mod()*trace_eps_lin_pos(u)

    def sig_V_neg(self, u): #Volumetric compressive stress
        return self.K_mod()*trace_eps_lin_neg(u)*ufl.Identity(self.dim)
            
    def sig_V_pos(self, u): #Volumetric tensile stress
        return self.K_mod()*trace_eps_lin_pos(u)*ufl.Identity(self.dim)
           
    def sig(self, u):
        return self.sig_V(u) + self.sig_D(u)
    
    def sig_V_neg_pp(self, u): #Volumetric compressive stress
        return self.K_mod()*trace_eps_lin_neg_pp(u)*ufl.Identity(self.dim)
            
    def sig_V_pos_pp(self, u): #Volumetric tensile stress
        return self.K_mod()*trace_eps_lin_pos_pp(u)*ufl.Identity(self.dim)
    
    def psi_el_V_neg(self, u, u_old): #Volumetric elastic strain energy density (compressive part)
        return 0.5*self.K_mod()*trace_eps_lin_neg(u)**2
        
    def psi_el_V_pos(self, u, u_old): #Volumetric elastic strain energy density (tensile part)
        return 0.5*self.K_mod()*trace_eps_lin_pos(u)**2
        
    def psi_el_V(self, u, u_old): #Total volumetric elastic strain energy density
        return 0.5*self.K_mod()*trace_eps_lin(u)**2
    
    def psi_el_D(self, u, u_old): #Deviatoric elastic strain energy density
        return self.G_mod()*ufl.inner(epsD_lin(u), epsD_lin(u))
    



    # find caller frame coming from sim.py
    # caller = None
    # for frm in inspect.stack():
    #     if frm.filename.endswith('sim.py'):
    #         caller = frm
    #         break
    # if caller:
    #     print(f"LinearElasticityIso initialized — called from {caller.filename}:{caller.lineno} in {caller.function}")
    # else:
    #     # fallback to immediate caller if sim.py not found on the stack
    #     frm = inspect.stack()[1]
    #     print(f"LinearElasticityIso initialized — called from {frm.filename}:{frm.lineno} in {frm.function}")
    # input()
    

class RambergOsgoodIso_full(): #non-linear both in deviatoric and volumetric part
    
    def __init__(self, dim = int, E = callable, G = callable, K = callable, nu = callable, b = callable, r = callable, eps_y = callable):
        self.dim = dim
        self.E = E
        self.G = G
        self.K = K
        self.nu = nu
        self.b = b
        self.r = r
        self.eps_y = eps_y
            
    def eqeps(self, u):
        if self.dim == 2:
            return ufl.conditional(ufl.lt(ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))))
        else:
            return ufl.conditional(ufl.lt(ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))))
        
    def G_mod(self, u): #modified shear modulus
        eqeps_rel = ufl.sqrt((self.eqeps(u)/self.eps_y)**2)
        A = (1+eqeps_rel**self.r)**(1/self.r)
        return self.G*(self.b+(1-self.b)/A)
    
    def K_mod(self, u): ##modified bulk modulus
        eqeps_rel = ufl.sqrt((self.eqeps(u)/self.eps_y)**2)
        A = (1+eqeps_rel**self.r)**(1/self.r)
        return self.K*(self.b+(1-self.b)/A)
            
    def sig_D(self, u): #Deviatoric stress
        return 2*self.G_mod(u) * epsD_lin(u)
    
    def sig_V(self, u): #Volumetric stress
        return self.K_mod(u)*trace_eps_lin(u)*ufl.Identity(self.dim)
        
    def sig_V_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin(u)
    
    def sig_V_neg_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin_neg(u)
    
    def sig_V_pos_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin_pos(u)

    def sig_V_neg(self, u): #Volumetric compressive stress
        return self.K_mod(u)*trace_eps_lin_neg(u)*ufl.Identity(self.dim)
            
    def sig_V_pos(self, u): #Volumetric tensile stress
        return self.K_mod(u)*trace_eps_lin_pos(u)*ufl.Identity(self.dim)
           
    def sig(self, u):
        return self.sig_V(u) + self.sig_D(u)
    
    # find caller frame coming from sim.py
    # caller = None
    # for frm in inspect.stack():
    #     if frm.filename.endswith('sim.py'):
    #         caller = frm
    #         break
    # if caller:
    #     print(f"RambergOsgoodIso_full initialized — called from {caller.filename}:{caller.lineno} in {caller.function}")
    # else:
    #     # fallback to immediate caller if sim.py not found on the stack
    #     frm = inspect.stack()[1]
    #     print(f"RambergOsgoodIso_full initialized — called from {frm.filename}:{frm.lineno} in {frm.function}")
    # input()
    
class RambergOsgoodIso(): #non-linear in deviatoric part only
    
    def __init__(self, dim = int, E = callable, G = callable, K = callable, nu = callable, b = callable, r = callable, eps_y = callable):
        self.dim = dim
        self.E = E
        self.G = G
        self.K = K
        self.nu = nu
        self.b = b
        self.r = r
        self.eps_y = eps_y
            
    def eqeps(self, u):
        if self.dim == 2:
            return ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u)) + np.finfo(np.float64).eps)  #ufl.conditional(ufl.lt(ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))))
        else:
            return ufl.sqrt(2/3*ufl.inner(epsD_lin(u), epsD_lin(u))+ np.finfo(np.float64).eps)  #ufl.conditional(ufl.lt(ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u))))) #ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))) + np.finfo(np.float64).eps #
        
    def G_mod(self, u): #modified shear modulus
        eqeps_rel = ufl.sqrt((self.eqeps(u)/self.eps_y)**2)
        A = (1+eqeps_rel**self.r)**(1/self.r)
        #return 0.5*self.E*(self.b+(1-self.b)/A) 
        return self.G*(self.b+(1-self.b)/A)
    
    def K_mod(self): ##modified bulk modulus
        return self.K
            
    def sig_D(self, u): #Deviatoric stress
        return 2*self.G_mod(u) * epsD_lin(u)
    
    def sig_V(self, u): #Volumetric stress
        return self.K_mod()*trace_eps_lin(u)*ufl.Identity(self.dim)
        
    def sig_V_scalar(self, u):
        return self.K_mod()*trace_eps_lin(u)

    def sig_V_neg_scalar(self, u):
        return self.K_mod()*trace_eps_lin_neg(u)

    def sig_V_pos_scalar(self, u):
        return self.K_mod()*trace_eps_lin_pos(u)

    def sig_V_neg(self, u): #Volumetric compressive stress
        return self.K_mod()*trace_eps_lin_neg(u)*ufl.Identity(self.dim)

    def sig_V_pos(self, u): #Volumetric tensile stress
        return self.K_mod()*trace_eps_lin_pos(u)*ufl.Identity(self.dim)

    def sig(self, u):
        return self.sig_V(u) + self.sig_D(u)
    
    def psi_el_V_incremental(self, u, u_old, psi_el_V_old, sig_V_pos, sig_V_neg):
        return psi_el_V_old + delta_psi_el_V_pos(u, u_old, sig_V_pos) + delta_psi_el_V_neg(u, u_old, sig_V_neg)

    def psi_el_V_pos_incremental(self, u, u_old, psi_el_V_pos_old, sig_V_pos):
        return psi_el_V_pos_old + delta_psi_el_V_pos(u, u_old, sig_V_pos)

    def psi_el_D_incremental(self, u, u_old, psi_el_D_old, sig_D):
        return psi_el_D_old + delta_psi_el_D(u, u_old, sig_D)

    def psi_el_incremental(self, u, u_old, psi_el_old, sig_D, sig_V_pos, sig_V_neg):
        return psi_el_old + delta_psi_el_D(u, u_old, sig_D) + delta_psi_el_V_pos(u, u_old, sig_V_pos) + delta_psi_el_V_neg(u, u_old, sig_V_neg)


    # find caller frame coming from sim.py
    # caller = None
    # for frm in inspect.stack():
    #     if frm.filename.endswith('sim.py'):
    #         caller = frm
    #         break
    # if caller:
    #     print(f"RambergOsgoodIso initialized — called from {caller.filename}:{caller.lineno} in {caller.function}")
    # else:
    #     # fallback to immediate caller if sim.py not found on the stack
    #     frm = inspect.stack()[1]
    #     print(f"RambergOsgoodIso initialized — called from {frm.filename}:{frm.lineno} in {frm.function}")
    # input()

class RambergOsgoodIso_nu(): #non-linear in deviatoric part only, non-linearity enforced in nu
    
    def __init__(self, dim = int, E = callable, G = callable, K = callable, nu = callable, b = callable, r = callable, eps_y = callable):
        self.dim = dim
        self.E = E
        self.G = G
        self.K = K
        self.nu = nu
        self.b = b
        self.r = r
        self.eps_y = eps_y
            
    def eqeps(self, u):
        if self.dim == 2:
            return ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u)) + np.finfo(np.float64).eps)  #ufl.conditional(ufl.lt(ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u))))
        else:
            return ufl.sqrt(2/3*ufl.inner(epsD_lin(u), epsD_lin(u))+ np.finfo(np.float64).eps)  #ufl.conditional(ufl.lt(ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))), 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u))))) #ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u)))) + np.finfo(np.float64).eps #

    def nu_mod(self, u):
        eqeps_rel = ufl.sqrt((self.eqeps(u)/self.eps_y)**2)
        A = (1+eqeps_rel**self.r)**(1/self.r)
        RO_factor = self.b+(1-self.b)/A
        nu_mod = (1+self.nu)/RO_factor -1
        return nu_mod   

    def G_mod(self, u): #modified shear modulus
        eqeps_rel = ufl.sqrt((self.eqeps(u)/self.eps_y)**2)
        A = (1+eqeps_rel**self.r)**(1/self.r)
        #return 0.5*self.E*(self.b+(1-self.b)/A) 
        return self.G*(self.b+(1-self.b)/A)
    
    def K_mod(self, u): ##modified bulk modulus
        #return self.K
        return ufl.conditional(ufl.lt(self.E/(3*(1-2*self.nu_mod(u))), self.E/(3*(1-2*0.49))), self.E/(3*(1-2*self.nu_mod(u))), self.E/(3*(1-2*0.49)))
            
    def sig_D(self, u): #Deviatoric stress
        #return 2*self.G_mod(u) * epsD_lin(u)
        return self.E/(1+self.nu_mod(u)) * epsD_lin(u)
    
    def sig_V(self, u): #Volumetric stress
        return self.K_mod(u)*trace_eps_lin(u)*ufl.Identity(self.dim)
        #return self.K*trace_eps_lin(u)*ufl.Identity(self.dim)
        
    def sig_V_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin(u)
        #return self.K*trace_eps_lin(u)
    
    def sig_V_neg_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin_neg(u)
        #return self.K*trace_eps_lin_neg(u)
    
    def sig_V_pos_scalar(self, u):
        return self.K_mod(u)*trace_eps_lin_pos(u)
        #return self.K*trace_eps_lin_pos(u)

    def sig_V_neg(self, u): #Volumetric compressive stress
        return self.K_mod(u)*trace_eps_lin_neg(u)*ufl.Identity(self.dim)
        #return self.K*trace_eps_lin_neg(u)*ufl.Identity(self.dim)
            
    def sig_V_pos(self, u): #Volumetric tensile stress
        return self.K_mod(u)*trace_eps_lin_pos(u)*ufl.Identity(self.dim)
        #return self.K*trace_eps_lin_pos(u)*ufl.Identity(self.dim)

    def sig(self, u):
        return self.sig_V(u) + self.sig_D(u)




class LinearElasticityIsoLame():

    def __init__(self, dim = int, lmbda = callable, mu = callable):
        self.dim = dim
        self.lmbda = lmbda
        self.mu = mu
        self.nu = lmbda/(2*(lmbda + mu)) # Poisson's ratio             
    
    
    def K_mod(self):
        return self.lmbda + 2*self.mu/self.dim
    
    def eqeps(self, u):
        if self.dim == 2:
            return ufl.sqrt(ufl.inner(epsD_lin(u), epsD_lin(u)))
        else:
            return ufl.sqrt((3/(2*(1+self.nu)**2))*(ufl.inner(epsD_lin(u), epsD_lin(u))))
    
    def sig_D(self, u): #Deviatoric stress
        return 2*self.mu * epsD_lin(u)
    
    def sig_V(self, u): #Volumetric stress
        return self.K_mod()*trace_eps_lin(u)*ufl.Identity(self.dim)
        
    def sig_V_scalar(self, u):
        return self.K_mod()*trace_eps_lin(u)
    
    def sig_V_neg_scalar(self, u):
        return self.K_mod()*trace_eps_lin_neg(u)
    
    def sig_V_pos_scalar(self, u):
        return self.K_mod()*trace_eps_lin_pos(u)

    def sig_V_neg(self, u): #Volumetric compressive stress
        return self.K_mod()*trace_eps_lin_neg(u)*ufl.Identity(self.dim)
            
    def sig_V_pos(self, u): #Volumetric tensile stress
        return self.K_mod()*trace_eps_lin_pos(u)*ufl.Identity(self.dim)
           
    def sig(self, u):
        return self.sig_V(u) + self.sig_D(u)
    
    def sig_V_neg_pp(self, u): #Volumetric compressive stress
        return self.K_mod()*trace_eps_lin_neg_pp(u)*ufl.Identity(self.dim)
            
    def sig_V_pos_pp(self, u): #Volumetric tensile stress
        return self.K_mod()*trace_eps_lin_pos_pp(u)*ufl.Identity(self.dim)
    
    def psi_el_V_neg(self, u, u_old): #Volumetric elastic strain energy density (compressive part)
        return 0.5*self.K_mod()*trace_eps_lin_neg(u)**2
        
    def psi_el_V_pos(self, u, u_old): #Volumetric elastic strain energy density (tensile part)
        return 0.5*self.K_mod()*trace_eps_lin_pos(u)**2
        
    def psi_el_V(self, u, u_old): #Total volumetric elastic strain energy density
        return 0.5*self.K_mod()*trace_eps_lin(u)**2
    
    def psi_el_D(self, u, u_old): #Deviatoric elastic strain energy density
        return self.mu*ufl.inner(epsD_lin(u), epsD_lin(u))

   







######################################################################################################################

# if ModelDim == 2:
#     if args.mat_law == 'l_nl':
#         def dev_fact(u): 
#             return 2*Gmod/(1+2*(Gmod*bconst)*eqeps(u)**(1-1/nexp))
#     elif args.mat_law == 'nl': #TODO broken!
#         def dev_fact(u): # Non-linearity factor for deviatoric stress
#             return (1/(1+nu))**2*((1+nu)/2)**(1/nexp)*2*bconst*(eqeps(u))**((1-nexp)/nexp)
#     elif args.mat_law == 'RO':
#         def G_mod(u): #modified shear modulus
#             eqeps_rel = ufl.sqrt((eqeps(u)/eps_y)**2)
#             A = (1+eqeps_rel**r_RO)**(1/r_RO)
#             return Gmod*(b_RO+(1-b_RO)/A)
#         def K_mod(u): ##modified bulk modulus
#             eqeps_rel = ufl.sqrt((eqeps(u)/eps_y)**2)
#             A = (1+eqeps_rel**r_RO)**(1/r_RO)
#             return Kmod*(b_RO+(1-b_RO)/A)
# else:
#     if args.mat_law == 'lin_el':
#         def G_mod(u):
#             return 2*Gmod
#         def K_mod(u):
#             return Kmod
#     if args.mat_law == 'l_nl':
#         def dev_fact(u):
#             #return 2*Gmod/(1+3*(Gmod/bconst)*eqeps(u)**(1-1/nexp))
#             return 2*Gmod/(1+3*(Gmod*bconst)*eqeps(u)**(1-1/nexp))
#     elif args.mat_law == 'nl': #TODO broken!
#         def dev_fact(u): # Non-linearity factor for deviatoric stress #TODO
#             #return (1+nu)**((nexp+1)/nexp)*bconst*(2*eqeps(u)/3)**((1-nexp)/nexp)
#             ###return (1+nu)**((nexp+1)/nexp)*(1/bconst)*(2*eqeps(u)/3)**((1-nexp)/nexp) #TODO
#             #return (1/bconst)*(2/3*(1+nu)*eqeps(u))**(1/nexp-1)
#             ###return 1/bconst*(2/3*(1+nu)*eqeps(u))**(1/nexp-1)
#             return 1/bconst*(2/3*(1+nu)*eqeps(u))**(1/nexp-1)
#     elif args.mat_law == 'RO':
#         def G_mod(u): #modififed shear modulus
#             eqeps_rel = ufl.sqrt((eqeps(u)/eps_y)**2)
#             A = (1+eqeps_rel**r_RO)**(1/r_RO)
#             return Gmod*(b_RO+(1-b_RO)/A)
#         def K_mod(u): ##modififed bulk modulus
#             eqeps_rel = ufl.sqrt((eqeps(u)/eps_y)**2)
#             A = (1+eqeps_rel**r_RO)**(1/r_RO)
#             return Kmod*(b_RO+(1-b_RO)/A)

