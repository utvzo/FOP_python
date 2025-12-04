# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:45:21 2025

@author: Julius
"""
# IMPORTS AND FS SELLMEIER CONSTANTS %%

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.constants import speed_of_light

CHI_01 = 0.6962
CHI_02 = 0.4079
CHI_03 = 0.8975
LAMBDA_1 = 0.06840
LAMBDA_2 = 0.1162
LAMBDA_3 = 9.8962

# %%
#because the original function of course just shit itself to death...
def sellmeier_dispersion(lambda_um):
    c0 = speed_of_light  # m/s

    # sellmeier coeff for FS 
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2

    lam_um = np.array(lambda_um)
    lam_m = lam_um * 1e-6  # um to m

    #n2 formula
    n_sq = (
        1
        + B1 * lam_um**2 / (lam_um**2 - C1)
        + B2 * lam_um**2 / (lam_um**2 - C2)
        + B3 * lam_um**2 / (lam_um**2 - C3)
        )
    n = np.sqrt(n_sq)
    # numerical gradients
    dn_dlam = np.gradient(n, lam_m)
    d2n_dlam2 = np.gradient(dn_dlam, lam_m)

    # MAT DISP
    D_SI = -(lam_m / c0) * d2n_dlam2   # s/m

    # conversion magic vol. 34674
    D_ps = D_SI * 1e12 / 1e9 * 1e-3  
    D_ps = D_SI * 1e6

    return n, D_ps


def sellmeier_fs(lambda_um):
    c_0 = 3e8  # speed of light
    z = 1000   # meters

    n_sq = (
        1
        + (CHI_01 * lambda_um**2) / (lambda_um**2 - LAMBDA_1**2)
        + (CHI_02 * lambda_um**2) / (lambda_um**2 - LAMBDA_2**2)
        + (CHI_03 * lambda_um**2) / (lambda_um**2 - LAMBDA_3**2)
    )
    n = np.sqrt(n_sq)

    dn_dlambda = np.gradient(n, lambda_um)
    n_g = n - lambda_um * dn_dlambda
    t_g = z * n_g / c_0
    d2n_dlambda2 = np.gradient(dn_dlambda, lambda_um)
    M_lambda = -(lambda_um / c_0) * d2n_dlambda2 * 1e6

    return n, t_g, M_lambda

def plot_refractive_index(x, y):
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="n(λ)", linewidth=2, color='blue')
    
    #optional VIS marker
    #plt.axvspan(0.4, 0.7, color='orange', alpha=0.2, label="VIS (0.4–0.7 µm)")
    
    plt.xlabel("Wavelength λ in [µm]")
    plt.ylabel("Refractive index n(λ)")
    plt.title("Refractive Index of Fused Silica (Sellmeier Equation)")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xlim(x.min(), x.max())
    plt.tight_layout()
    
    plt.show()
    
def plot_group_delay(x, y):
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="t_g(λ)", linewidth=2, color='blue')
    
    #optional VIS marker
    #plt.axvspan(0.4, 0.7, color='orange', alpha=0.2, label="VIS (0.4–0.7 µm)")
    
    plt.xlabel("Wavelength λ in [µm]")
    plt.ylabel("Arrival time t_g(λ) [s]")
    plt.title("Arrival time t_g")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xlim(x.min(), x.max())
    plt.tight_layout()
    
    plt.show()
    
def plot_material_dispersion(x, y):
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="M_(λ)", linewidth=2, color='blue')
    
    #optional VIS marker
    #plt.axvspan(0.4, 0.7, color='orange', alpha=0.2, label="VIS (0.4–0.7 µm)")
    
    plt.xlabel("Wavelength λ in [µm]")
    plt.ylabel("Material Dispersion M_λ(λ) [s]")
    plt.title("Material dispersion M_λ")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xlim(x.min(), x.max())
    #plt.xlim(0.5, x.max())
    #plt.ylim(-500,250)
    plt.tight_layout()
    
    plt.show()
    

def main():
    #VAR for range
    lower_bound_lambda = 0.2
    upper_bound_lambda = 3.7
    
    #LINSPACE
    lambda_x = np.linspace(lower_bound_lambda, upper_bound_lambda, 5000)
    
    #FUNC CALLS
    n_lambda, t_g, M_lambda = sellmeier_fs(lambda_x)
    plot_refractive_index(lambda_x, n_lambda)
    plot_group_delay(lambda_x, t_g)
    plot_material_dispersion(lambda_x, M_lambda*1e12)   #ps/nmkm
    #print(t_g)

if __name__ == "__main__":
    main()