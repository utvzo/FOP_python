import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# CONST
a = 250e-9       # WAVEGUIDE HALF-WIDTH
n1 = 3.5         # CORE 
n2 = 1.44        # CLADDING 
n3 = 1.0         # SUBSTRATE
c = 3e8          # LIGHT SPEED

# FUNC

def u_func(V,B):
    u = V*np.sqrt(1-B)
    return(u)

def w_func(V,B):
    w = V*np.sqrt(B)
    return(w)

def w_prime_func(V, GAMMA, B):
    w_prime = V*np.sqrt(GAMMA+B)
    return(w_prime)

def GAMMA_func(n1, n2, n3):
    GAMMA = (n2**2-n3**2)/(n1**2-n2**2)
    return(GAMMA)

def V_func(a, k0, n1, n2):
    V = a*k0*np.sqrt(n1**2-n2**2)
    return(V)

# TM Moden B
def eq_B(B, V, GAMMA, n1, n2, n3, m):
    u = u_func(V,B)
    w = w_func(V,B)
    w_prime = w_prime_func(V, GAMMA, B)
    return 0.5*np.arctan((n1**2 * w)/(n2**2 * u)) + \
           0.5*np.arctan((n1**2 * w_prime)/(n3**2 * u)) + \
           0.5*m*np.pi - u

# B sovler
def solver_B(V, GAMMA, n1, n2, n3, m=0):
    B_min = 1e-10
    B_max = 1-1e-10
    
    #boundaries --> no zero div errors
    f_min = eq_B(B_min, V, GAMMA, n1, n2, n3, m)
    f_max = eq_B(B_max, V, GAMMA, n1, n2, n3, m)
    
    if np.sign(f_min) == np.sign(f_max):
        # scan if sign changes
        B_vals = np.linspace(B_min, B_max, 500)
        f_vals = [eq_B(B, V, GAMMA, n1, n2, n3, m) for B in B_vals]
        
        sign_change = np.where(np.array(f_vals[:-1])*np.array(f_vals[1:])<0)[0]
        if len(sign_change) == 0:
            return np.nan
        
        B_min = B_vals[sign_change[0]]
        B_max = B_vals[sign_change[0]+1]
    
    sol = root_scalar(eq_B, args=(V,GAMMA,n1,n2,n3,m), bracket=[B_min,B_max])
    return sol.root

# Field of TM
def field_TM(x, u, w, w_prime, a, beta, n1, n2, n3, omega=1):
    Hy = np.zeros_like(x)
    Ex = np.zeros_like(x)
    Ez = np.zeros_like(x)
    
    # CORE
    mask_core = np.abs(x) <= a
    Hy[mask_core] = np.cos(u*x[mask_core]/a)
    Ex[mask_core] = -beta/(omega*n1**2)*Hy[mask_core]
    Ez[mask_core] = 1/(omega*n1**2)*np.gradient(Hy[mask_core], x[mask_core])
    
    # SUB
    mask_sub = x < -a
    Hy[mask_sub] = np.cos(u)*np.exp(w_prime*(x[mask_sub]+a)/a)
    Ex[mask_sub] = -beta/(omega*n3**2)*Hy[mask_sub]
    Ez[mask_sub] = 1/(omega*n3**2)*np.gradient(Hy[mask_sub], x[mask_sub])
    
    # CLAD
    mask_clad = x > a
    Hy[mask_clad] = np.cos(u)*np.exp(-w*(x[mask_clad]-a)/a)
    Ex[mask_clad] = -beta/(omega*n2**2)*Hy[mask_clad]
    Ez[mask_clad] = 1/(omega*n2**2)*np.gradient(Hy[mask_clad], x[mask_clad])
    
    return Hy, Ex, Ez

# Plot B / n_eff vs Frequenz
def plot_modes_vs_freq(lambda_range, B_list, n1, n2, a):

    
    # Frequenz in THz
    f_range_THz = c / lambda_range / 1e12
    
    # Normalisierte Frequenz V
    V_range = 2 * np.pi * a / lambda_range * np.sqrt(n1**2 - n2**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # -------- B vs f --------
    ax1 = axes[0]
    for m, B_mode in enumerate(B_list):
        B_mode = np.array(B_mode)
        mask = ~np.isnan(B_mode)
        ax1.plot(f_range_THz[mask], B_mode[mask], label=f"m={m}")
    ax1.set_xlabel("Frequency f [THz]")
    ax1.set_ylabel("Normalized propagation constant B")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("B vs f")
    ax1.set_xlim(f_range_THz.min(), f_range_THz.max())
    
    # Obere Achse für V
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    
    # V-Ticks und die entsprechenden Frequenzen
    V_ticks = np.linspace(V_range.min(), V_range.max(), 5)
    f_ticks = V_ticks * c / (2 * np.pi * a * np.sqrt(n1**2 - n2**2)) / 1e12  # in THz
    f_ticks = f_ticks[(f_ticks >= f_range_THz.min()) & (f_ticks <= f_range_THz.max())]
    ax1_top.set_xticks(f_ticks)
    ax1_top.set_xticklabels([f"{V:.2f}" for V in V_ticks[:len(f_ticks)]])
    ax1_top.set_xlabel("Normalized frequency V")
    
    # -------- n_eff vs f --------
    ax2 = axes[1]
    for m, B_mode in enumerate(B_list):
        B_mode = np.array(B_mode)
        mask = ~np.isnan(B_mode)
        n_eff_mode = np.sqrt(B_mode*(n1**2 - n2**2) + n2**2)
        ax2.plot(f_range_THz[mask], n_eff_mode[mask], label=f"m={m}")
    ax2.set_xlabel("Frequency f [THz]")
    ax2.set_ylabel("Effective index n_eff")
    ax2.legend()
    ax2.set_title("n_eff vs f")
    ax2.grid(True)
    ax2.set_xlim(f_range_THz.min(), f_range_THz.max())
    
    # Obere Achse für V
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    f_ticks = V_ticks * c / (2 * np.pi * a * np.sqrt(n1**2 - n2**2)) / 1e12
    f_ticks = f_ticks[(f_ticks >= f_range_THz.min()) & (f_ticks <= f_range_THz.max())]
    ax2_top.set_xticks(f_ticks)
    ax2_top.set_xticklabels([f"{V:.2f}" for V in V_ticks[:len(f_ticks)]])
    ax2_top.set_xlabel("Normalized frequency V")
    
    plt.tight_layout()
    plt.show()



# Feldplot für einen Mode
def plot_fields(x, Hy, Ex, Ez, mode_number):
    plt.figure(figsize=(10,5))
    plt.plot(x*1e9, Hy, label="Hy")
    plt.plot(x*1e9, Ex, label="Ex")
    plt.plot(x*1e9, Ez, label="Ez")
    plt.xlabel("x [nm]")
    plt.ylabel("Field amplitude (a. u.)")
    plt.title(f"TM{mode_number} Field Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

# MAIN
if __name__ == "__main__":
    # wavelength range argument
    lambda_range = np.linspace(1.12e-6, 3.7e-6, 50) #OG RANGE
    #lambda_range = np.linspace(0.1e-6, 20e-6, 150) #TESTING
    
    GAMMA = GAMMA_func(n1,n2,n3)
    
    # Loops for no of modes wanted --> kind of guessing right how many
    B_all = []
    for m in range(3):  # no. of miodes
        B_mode = []
        for lam in lambda_range:
            k0 = 2*np.pi/lam
            V = V_func(a,k0,n1,n2)
            B = solver_B(V, GAMMA, n1, n2, n3, m)
            B_mode.append(B)
        B_all.append(B_mode)
    
    B_all = np.array(B_all)
    
    #aaaaa
    plot_modes_vs_freq(lambda_range, B_all, n1, n2, a)
    
    # ex plot for lam = 1.55um 
    lam = 1.55e-6
    k0 = 2*np.pi/lam
    V = V_func(a,k0,n1,n2)
    B0 = solver_B(V, GAMMA, n1, n2, n3, m=0)
    beta0 = k0*np.sqrt(B0*(n1**2 - n2**2) + n2**2)
    u0 = u_func(V,B0)
    w0 = w_func(V,B0)
    w_prime0 = w_prime_func(V,GAMMA,B0)
    
    x = np.linspace(-2*a, 2*a, 1000)
    Hy, Ex, Ez = field_TM(x, u0, w0, w_prime0, a, beta0, n1, n2, n3)
    plot_fields(x, Hy, Ex, Ez, mode_number=0)
