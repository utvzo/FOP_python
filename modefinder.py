# %% HEADER AND CONSTANTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

import n2_fused_silica as fs

# CONST ASSIGNMENT D
a = 250e-9       # WAVEGUIDE HALF-WIDTH
n1 = 3.5         # CORE 
n2 = 1.44        # CLADDING 
n3 = 1.0         # SUBSTRATE
c = 3e8          # LIGHT SPEED

#CONST ASSIGNMENT E
a_2 = 2e-6
n2_2 = 1.45
delta_n = 0.004
# %% HELPER FUNCTIONS

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
           
def n_eg(lambda_range, B_list, n1, n2, a):
    # scale f to thz
    
    #normalized freq V
    n_eff = []
    lamrange = []
    #V_range = 2 * np.pi * a / lambda_range * np.sqrt(n1**2 - n2**2)
    for m, B_mode in enumerate(B_list):
        B_mode = np.array(B_mode)
        mask = ~np.isnan(B_mode)
        n_eff_mode = np.sqrt(B_mode*(n1**2 - n2**2) + n2**2)
        n_eff.append(n_eff_mode[mask])
        lamrange.append(lambda_range[mask]) 
        
    #returns lists of arrays --> useful if more than m=0 modes are needed 
    #same procedure as in plots from c)
    
    m0_neff = n_eff[0]
    m0_lamrange = lamrange[0]
    
    if len(m0_neff) == len(m0_lamrange):
        n_eg = []
        dn_dlam = np.gradient(m0_neff, m0_lamrange)
        for i in range(len(m0_neff)):
            n_eg.append(m0_neff[i]-m0_lamrange[i]*dn_dlam[i])        
    else:   
        print("ERR:lenghts do not match")
        
    return(m0_neff,m0_lamrange, np.array(n_eg))
    
    
def W_lambda(B, delta_n, lambda_array, a, n1, n2, c, m=0):
    V = V_func(a, 2*np.pi/lambda_array, n1, n2)
    W_lam = -1*delta_n/(c*lambda_array)*V*np.gradient(np.gradient(B[m]*V, V), V)
    
    return(W_lam)
    
    
     
    
# %% SOLVER AND FIELD FUNCTION
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
    
    #root scalar func for solving eq numerically
    sol = root_scalar(eq_B, args=(V,GAMMA,n1,n2,n3,m), bracket=[B_min,B_max])
    return sol.root

# Field of TM
def field_TM(x, u, w, w_prime, a, beta, n1, n2, n3, omega=1):
    #create empty arrays analogous to x
    Hy = np.zeros_like(x)
    Ex = np.zeros_like(x)
    Ez = np.zeros_like(x)
    
    # CORE
    #intervalcore is all x vals in +- core diameter range
    interval_core = np.abs(x) <= a
    Hy[interval_core] = np.cos(u*x[interval_core]/a)
    Ex[interval_core] = -beta/(omega*n1**2)*Hy[interval_core]
    Ez[interval_core] = 1/(omega*n1**2)*np.gradient(Hy[interval_core], x[interval_core])
    
    # SUB
    interval_sub = x < -a
    Hy[interval_sub] = np.cos(u)*np.exp(w_prime*(x[interval_sub]+a)/a)
    Ex[interval_sub] = -beta/(omega*n3**2)*Hy[interval_sub]
    Ez[interval_sub] = 1/(omega*n3**2)*np.gradient(Hy[interval_sub], x[interval_sub])
    
    # CLAD
    interval_clad = x > a
    Hy[interval_clad] = np.cos(u)*np.exp(-w*(x[interval_clad]-a)/a)
    Ex[interval_clad] = -beta/(omega*n2**2)*Hy[interval_clad]
    Ez[interval_clad] = 1/(omega*n2**2)*np.gradient(Hy[interval_clad], x[interval_clad])
    
    return Hy, Ex, Ez

# %% PLOTTER FUNCTIONS
# Plot B and n_eff vs frequency
def plot_B_n_f_V(lambda_range, B_list, n1, n2, a):

    
    # scale f to thz
    f_range_THz = c / lambda_range / 1e12
    
    #normalized freq V
    V_range = 2 * np.pi * a / lambda_range * np.sqrt(n1**2 - n2**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # B - f/V -plot
    ax1 = axes[0]
    for m, B_mode in enumerate(B_list):
        B_mode = np.array(B_mode)
        mask = ~np.isnan(B_mode)
        ax1.plot(f_range_THz[mask], B_mode[mask], label=f"m={m}")
    ax1.set_xlabel("Frequency f [THz]")
    ax1.set_ylabel("Normalized propagation constant B")
    ax1.set_ylim(0,1)
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("B vs f")
    ax1.set_xlim(f_range_THz.min(), f_range_THz.max())
    
    # second V axis
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    
    # ticks and further cosmetics
    V_ticks = np.linspace(V_range.min(), V_range.max(), 5)
    f_ticks = V_ticks * c / (2 * np.pi * a * np.sqrt(n1**2 - n2**2)) / 1e12  # in THz
    f_ticks = f_ticks[(f_ticks >= f_range_THz.min()) & (f_ticks <= f_range_THz.max())]
    ax1_top.set_xticks(f_ticks)
    ax1_top.set_xticklabels([f"{V:.2f}" for V in V_ticks[:len(f_ticks)]])
    ax1_top.set_xlabel("Normalized frequency V")
    
    # n_eff - f/V - plot
    ax2 = axes[1]
    #axes min max lists
    limarrmin = []
    limarrmax = []
    for m, B_mode in enumerate(B_list):
        B_mode = np.array(B_mode)
        mask = ~np.isnan(B_mode)
        n_eff_mode = np.sqrt(B_mode*(n1**2 - n2**2) + n2**2)
        ax2.plot(f_range_THz[mask], n_eff_mode[mask], label=f"m={m}")
        limarrmin.append(min(n_eff_mode[mask]))
        limarrmax.append(max(n_eff_mode[mask]))
    ax2.set_xlabel("Frequency f [THz]")
    ax2.set_ylabel("Effective index n_eff")
    #determines maximum and minimum for axes
    ax2.set_ylim(min(limarrmin),max(limarrmax))
    ax2.legend()
    ax2.set_title("n_eff vs f")
    ax2.grid(True)
    ax2.set_xlim(f_range_THz.min(), f_range_THz.max())
    
    # second V axis, ticks and further cosmetics
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    f_ticks = V_ticks * c / (2 * np.pi * a * np.sqrt(n1**2 - n2**2)) / 1e12
    f_ticks = f_ticks[(f_ticks >= f_range_THz.min()) & (f_ticks <= f_range_THz.max())]
    ax2_top.set_xticks(f_ticks)
    ax2_top.set_xticklabels([f"{V:.2f}" for V in V_ticks[:len(f_ticks)]])
    ax2_top.set_xlabel("Normalized frequency V")
    
    plt.tight_layout()
    plt.show()



# visualization to check for one field distr 
def plot_H_E(x, Hy, Ex, Ez, lam, mode_number):
    #Normalze to absmax of all
    nEx = np.linalg.norm(Ex, ord=np.inf)
    nHy = np.linalg.norm(Hy, ord=np.inf)
    nEz = np.linalg.norm(Ez, ord=np.inf)
    Nmax = max(nEx, nHy, nEz)
    
    plt.figure(figsize=(10,5))
    plt.plot(x*1e9, Hy/Nmax, label="Hy")
    plt.plot(x*1e9, Ex/Nmax, label="Ex")
    plt.plot(x*1e9, Ez/Nmax, label="Ez")
    plt.xlabel("x [nm]")
    plt.ylabel("Field amplitude (a. u.)")
    plt.xlim(min(x)*1e9,max(x)*1e9)
    plt.ylim(-1,1)
    plt.title(f"TM{mode_number} Field Distribution for λ = {lam} m")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% MAIN
if __name__ == "__main__":
    
    ################################ ASSIGNMENT D #############################
    # wavelength range argument
    lambda_range = np.linspace(1.12e-6, 3.7e-6, 50) #ORIG RANGE
    #lambda_range = np.linspace(0.1e-6, 20e-6, 150) #TESTING 
    
    GAMMA = GAMMA_func(n1,n2,n3)
    
    # Loops for no of modes wanted --> kind of guessing how many are reqired
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
    
    ############################## PLOTTING D ################################
    
    # double plot B and neff
    plot_B_n_f_V(lambda_range, B_all, n1, n2, a)
    
    # example plot for lam = 1.55um 
    lam = 3.0e-6
    k0 = 2*np.pi/lam
    V = V_func(a,k0,n1,n2)
    B0 = solver_B(V, GAMMA, n1, n2, n3, m=0)
    beta0 = k0*np.sqrt(B0*(n1**2 - n2**2) + n2**2)
    u0 = u_func(V,B0)
    w0 = w_func(V,B0)
    w_prime0 = w_prime_func(V,GAMMA,B0)
    
    x = np.linspace(-2*a, 2*a, 1000)
    Hy, Ex, Ez = field_TM(x, u0, w0, w_prime0, a, beta0, n1, n2, n3)
    plot_H_E(x, Hy, Ex, Ez, lam, mode_number=0)
    
    ################################ ASSIGNMENT E #############################

    lambda_range_2 = np.linspace(1e-6, 2.5e-6, 200)

    # 
    cases = [
        ("a, δn", a_2, delta_n),
        ("a/√10, 10δn", a_2/np.sqrt(10), 10*delta_n)
            ]
    results = {}
    
    for label, a_val, dn in cases:
        
        n1_val = n2_2 + dn
        GAMMA_val = GAMMA_func(n1_val, n2_2, n2_2)

        #solver, m=0
        B_list = []
        for lam in lambda_range_2:
            k0 = 2*np.pi/lam
            V = V_func(a_val, k0, n1_val, n2_2)
            B_val = solver_B(V, GAMMA_val, n1_val, n2_2, n2_2, 0)
            B_list.append(B_val)

        B_list = np.array([B_list]) 
        
        neff, lam_clean, n_g = n_eg(lambda_range_2, B_list, n1_val, n2_2, a_val)
        
        W_lam = W_lambda(B_list, dn, lam_clean, a_val, n1_val, n2_2, c, m=0)
        
        # resULT
        results[label] = {
            "B": B_list,
            "neff": neff,
            "lam": lam_clean,
            "n_g": n_g,
            "W": W_lam
            }
        

    ############################## PLOTTING E ################################
    
    plt.figure()
    for label in results:
        lam = results[label]["lam"]
        n_g = results[label]["n_g"]
        plt.plot(lam*1e6, n_g, label=label)
    plt.xlabel("λ [µm]")
    plt.ylabel("Group index n_g")
    plt.xlim(1.015,2.485)
    plt.ylim(1.45,1.495)
    plt.legend()
    plt.grid(True)
    plt.title("Group index n_g(λ)")
    plt.show()

    plt.figure()
    for label in results:
        lam = results[label]["lam"]
        Wlam = results[label]["W"]
        plt.plot(lam*1e6, Wlam*1e6, label=label)
    plt.xlabel("λ [µm]")
    plt.ylabel("W_λ [ps/(nm km)]")
    plt.xlim(1.015,2.485)
    plt.ylim(-55,0)
    plt.legend()
    plt.grid(True)
    plt.title("Waveguide dispersion W_λ(λ)")
    plt.show()
    
    
    ################################ ASSIGNMENT F ################################

    # wavelength um
    lambda_um = lambda_range_2 * 1e6
    
    # n2fs reference
    n_silica, M_ps = fs.sellmeier_dispersion(lambda_um)
    
    # CD 
    C_results = {}
    for label in results:
        lam = results[label]["lam"]             
        lam_um_local = lam * 1e6
        
        Wlam = results[label]["W"]               # WD
        _, M_ps_local = fs.sellmeier_dispersion(lam_um_local) #interpolate
        C_results[label] = Wlam*1e6 + M_ps_local     # CD
        
    #plot
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(lambda_um, M_ps, label="M_λ", linewidth=2)
    for label in C_results:
        lam_um_local = results[label]["lam"] * 1e6
        ax1.plot(lam_um_local, C_results[label], linewidth=2, label=f"C_λ {label}")

    ax1.set_xlabel("λ [µm]")
    ax1.set_ylabel("M_λ, C_λ [ps/(km·nm)]")
    ax1.set_xlim(1.015,2.485)
    ax1.set_ylim(-80,80)
    ax1.grid(True)

    #Wlambda
    ax2 = ax1.twinx()
    colorarray = ['orange', 'green']

    #fixes the color mapping (just dont ask pls...)
    label_colors = {label: colorarray[i % len(colorarray)] 
                for i, label in enumerate(results)}

    for label in results:
        lam_um_local = results[label]["lam"] * 1e6
        ax2.plot(
            lam_um_local,
            results[label]["W"] * 1e6,
            "--",
            label=f"W_λ {label}",
            color=label_colors[label]
        )
    ax2.set_ylabel("W_λ [ps/(km·nm)]")
    ax2.set_ylim(-80,80)
    #cosmetuics
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("Material / waveguide / chromatic dispersion")
    plt.tight_layout()
    plt.show()
    
    
    
    
    #TEST
    
    for label in results:
        W = np.array(results[label]['W'])
        print(f"{label}: W_min={np.nanmin(W):.4e}, max={np.nanmax(W):.4e}, nans={np.isnan(W).sum()} \n")

    
    for label in results:
        C = np.array(C_results[label])
        print(f"{label}: C_min={np.nanmin(C):.4e}, max={np.nanmax(C):.4e}, mean={np.nanmean(C):.4e} \n")
    
   
    labels = list(results.keys())
    if len(labels) == 2:
        C1 = C_results[labels[0]]
        C2 = C_results[labels[1]]
        lam_um = results[labels[0]]['lam'] * 1e6
        plt.figure()
        plt.plot(lam_um, C2 - C1)
        plt.xlabel("λ [µm]")
        plt.ylabel("C(smallcore)-C(bigcore) [ps/(nm·km)]")
        plt.title("Delta CD curve")
        plt.grid(True)
        plt.show()
    