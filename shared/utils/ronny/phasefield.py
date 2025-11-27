import sys
sys.path.append('..')
import dolfinx as dlfx
import ufl
import numpy as np
import math
import util.aux as aux

class Phasefield():
    def __init__(self, DG0=None, degrad_type='cubic', split_type='vol_dev', incremental=True):
        self.DG0 = DG0
        self.degrad_type = degrad_type
        self.split_type = split_type
        self.incremental = incremental

        self.crack_driving_energy_old = dlfx.fem.Function(self.DG0)
        self.crack_driving_energy_old.x.array[:] = np.zeros_like(self.crack_driving_energy_old.x.array)
        self.crack_driving_energy_old.x.scatter_forward()
    
    @staticmethod
    def degrad_quadratic(s: any, eta: dlfx.fem.Constant) -> any:
        deg = s**2+eta
        return deg

    @staticmethod
    def diff_degrad_quadratic(s: any) -> any:
        degds = 2.0 * s
        return degds

    @staticmethod
    def degrad_cubic(s: any, eta: dlfx.fem.Constant, beta=0.2) -> any:
        deg = beta * ((s ** 2) * s - (s ** 2)) + 3.0 * (s ** 2) - 2.0*(s ** 2) * s + eta
        return deg

    @staticmethod
    def diff_degrad_cubic(s: any, beta=0.2) -> any:
        degds = beta * (3.0*(s ** 2)  - 2.0*s) + 6.0 * s - 6.0 * s ** 2
        return degds

    @staticmethod
    def psi_s(s: any, Gc: any, epsilon: any) -> any: #Fracture surface energy density
        return  Gc * (((1 - s) ** 2) / (4 * epsilon) + epsilon * (ufl.dot(ufl.grad(s), ufl.grad(s))))

    @staticmethod
    def crack_driving_energy_vol_dev(u: any, u_old: any, Emod: any, nu: any, eps: any, sig_V_pos_undegraded: any, sig_D_undegraded: any, sig_undegraded: any, psi_el_V_pos_undegraded: any, psi_el_D_undegraded: any, psi_el_V_pos_old: any, psi_el_D_old: any, delta_psi_el_V_pos_undegraded: any, delta_psi_el_D_undegraded: any, crack_driving_energy_old: any):
        return psi_el_V_pos_undegraded(u, u_old, psi_el_V_pos_old, sig_V_pos_undegraded) + psi_el_D_undegraded(u, u_old, psi_el_D_old, sig_D_undegraded)

    @staticmethod
    def crack_driving_energy_vol_dev_incremental(u: any, u_old: any, Emod: any, nu: any, eps: any, sig_V_pos_undegraded: any, sig_D_undegraded: any, sig_undegraded: any, psi_el_V_pos_undegraded: any, psi_el_D_undegraded: any, psi_el_V_pos_old: any, psi_el_D_old: any, delta_psi_el_V_pos_undegraded: any, delta_psi_el_D_undegraded: any, crack_driving_energy_old: any):
        return crack_driving_energy_old + delta_psi_el_V_pos_undegraded(u, u_old, sig_V_pos_undegraded) + delta_psi_el_D_undegraded(u, u_old, sig_D_undegraded)

    @staticmethod
    def crack_driving_energy_spectral(u: any, u_old: any, Emod: any, nu: any, eps: any, sig_V_pos_undegraded: any, sig_D_undegraded: any, sig_undegraded: any, psi_el_V_pos_undegraded: any, psi_el_D_undegraded: any, psi_el_V_pos_old: any, psi_el_D_old: any, delta_psi_el_V_pos_undegraded: any, delta_psi_el_D_undegraded: any, crack_driving_energy_old: any):
        s1, s2, s3 = aux.Eigenvalues_3x3(sig_undegraded(u))
        tr_diag = s1+s2+s3
        return (((1+nu)/(2*Emod))*(aux.pos(s1)**2 + aux.pos(s2)**2 + aux.pos(s3)**2) 
            - (nu/(2*Emod))*(aux.pos(tr_diag))**2)
    
    # @staticmethod
    # def crack_driving_energy_spectral_incremental(u: any, u_old: any, Emod: any, nu: any, eps: any, sig_V_pos_undegraded: any, sig_D_undegraded: any, sig_undegraded: any, psi_el_V_pos_undegraded: any, psi_el_D_undegraded: any, psi_el_V_pos_old: any, psi_el_D_old: any, delta_psi_el_V_pos_undegraded: any, delta_psi_el_D_undegraded: any, crack_driving_energy_old: any):
    #     s1, s2, s3 = aux.Eigenvalues_3x3(sig_undegraded(u))
    #     tr_diag = s1+s2+s3
    #     return aux.pos(s1)**2

    @staticmethod
    def crack_driving_energy_spectral_incremental(u: any, u_old: any, Emod: any, nu: any, eps: any, sig_V_pos_undegraded: any, sig_D_undegraded: any, sig_undegraded: any, psi_el_V_pos_undegraded: any, psi_el_D_undegraded: any, psi_el_V_pos_old: any, psi_el_D_old: any, delta_psi_el_V_pos_undegraded: any, delta_psi_el_D_undegraded: any, crack_driving_energy_old: any):
        e1_old, e2_old, e3_old = aux.Eigenvalues_3x3(eps(u_old))
        e1, e2, e3 = aux.Eigenvalues_3x3(eps(u)) 
        s1_old, s2_old, s3_old = aux.Eigenvalues_3x3(sig_undegraded(u_old))
        s1, s2, s3 = aux.Eigenvalues_3x3(sig_undegraded(u))
        dpsi_el_spectral_1 = ufl.conditional(ufl.lt(s1, 0.0), 0.0, 0.5*(s1_old + s1)*(e1 - e1_old))
        dpsi_el_spectral_2 = ufl.conditional(ufl.lt(s2, 0.0), 0.0, 0.5*(s2_old + s2)*(e2 - e2_old))
        dpsi_el_spectral_3 = ufl.conditional(ufl.lt(s3, 0.0), 0.0, 0.5*(s3_old + s3)*(e3 - e3_old))
        dpsi_el_spectral = dpsi_el_spectral_1 + dpsi_el_spectral_2 + dpsi_el_spectral_3
        return crack_driving_energy_old + dpsi_el_spectral

    degrad_dict = {
        'quadratic': (degrad_quadratic, diff_degrad_quadratic),
        'cubic': (degrad_cubic, diff_degrad_cubic)
    }
    
    crack_driving_energy_dict = {
        'vol_dev': {'regular': crack_driving_energy_vol_dev,
                    'incremental': crack_driving_energy_vol_dev_incremental},

        'spectral': {'regular': crack_driving_energy_spectral,
                        'incremental': crack_driving_energy_spectral_incremental}
    }

    
           


# def degrad_quadratic(s: any, eta: dlfx.fem.Constant) -> any:
#     degrad = s**2+eta
#     return degrad

# def diff_degrad_quadratic(s: any) -> any:
#     degds = 2.0 * s
#     return degds

# def degrad_cubic(s: any, eta: dlfx.fem.Constant, beta=0.2) -> any:
#     degrad = beta * ((s ** 2) * s - (s ** 2)) + 3.0 * (s ** 2) - 2.0*(s ** 2) * s + eta
#     return degrad

# def diff_degrad_cubic(s: any, beta=0.2) -> any:
#     degds = beta * (3.0*(s ** 2)  - 2.0*s) + 6.0 * s - 6.0 * s ** 2
#     return degds

def sig_c_quadr_deg(Gc, mu, epsilon):
    return 9.0/16.0 * math.sqrt(Gc*2.0*mu/(6.0*epsilon))

def sig_c_cubic_deg(Gc, mu, epsilon):
    return 81.0/50.0 * math.sqrt(Gc*2.0*mu/(15.0*epsilon))

def get_Gc_for_given_sig_c_quadr(sig_c, mu, epsilon):
    return (256.0 * epsilon / (27.0 * mu)) * sig_c**2

def get_Gc_for_given_sig_c_cub(sig_c, mu, epsilon):
    return (15.0 * epsilon / (2.0 * mu)) * (50.0/81.0*sig_c)**2

#crack driving energies for different split schemes
def crack_driving_energy_3D_spectral_incremental(u, u_old, eps, sig_undegraded, crack_driving_energy_old):
    e1_old, e2_old, e3_old = aux.Eigenvalues_3x3(eps(u_old))
    e1, e2, e3 = aux.Eigenvalues_3x3(eps(u)) #TODO: verify that ei and si are work complements
    s1_old, s2_old, s3_old = aux.Eigenvalues_3x3(sig_undegraded(u_old))
    s1, s2, s3 = aux.Eigenvalues_3x3(sig_undegraded(u))
    dpsi_el_spectral_1 = ufl.conditional(ufl.lt(s1, 0.0), 0.0, 0.5*(s1_old + s1)*(e1 - e1_old))
    dpsi_el_spectral_2 = ufl.conditional(ufl.lt(s2, 0.0), 0.0, 0.5*(s2_old + s2)*(e2 - e2_old))
    dpsi_el_spectral_3 = ufl.conditional(ufl.lt(s3, 0.0), 0.0, 0.5*(s3_old + s3)*(e3 - e3_old))
    dpsi_el_spectral = dpsi_el_spectral_1 + dpsi_el_spectral_2 + dpsi_el_spectral_3
    return crack_driving_energy_old + dpsi_el_spectral

def crack_driving_energy_2D_spectral_incremental(u, u_old, eps, sig_undegraded, crack_driving_energy_old):
    e1_old, e2_old = aux.Eigenvalues_2x2(eps(u_old))
    e1, e2 = aux.Eigenvalues_2x2(eps(u)) #TODO: verify that ei and si are work complements
    s1_old, s2_old = aux.Eigenvalues_2x2(sig_undegraded(u_old))
    s1, s2 = aux.Eigenvalues_2x2(sig_undegraded(u))
    dpsi_el_spectral_1 = ufl.conditional(ufl.lt(s1, 0.0), 0.0, 0.5*(s1_old + s1)*(e1 - e1_old))
    dpsi_el_spectral_2 = ufl.conditional(ufl.lt(s2, 0.0), 0.0, 0.5*(s2_old + s2)*(e2 - e2_old))
    dpsi_el_spectral = dpsi_el_spectral_1 + dpsi_el_spectral_2
    return crack_driving_energy_old + dpsi_el_spectral

def crack_driving_energy_vol_dev_incremental(u, u_old, eps, sig_undegraded, crack_driving_energy_old):
    return crack_driving_energy_old + dpsi_el_V_pos_undegraded(u, u_old) + dpsi_el_D_undegraded(u, u_old)