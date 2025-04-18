import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd 

# Read input from excel
input_df = pd.read_excel('tff_param.xlsx', sheet_name='Input', index_col=0)

# =============================================================================
# PARAMETERS
# =============================================================================
# Membrane and fluid properties
L_p    = input_df.loc['L_p', 'value']    # Clean water hydraulic permeability [m/s/Pa] - Increased 10x
mu_0   = input_df.loc['mu_0', 'value']    # Solvent viscosity [Pa·s]
k_mu   = input_df.loc['k_mu', 'value']     # Viscosity coefficient [m³/kg]
n      = input_df.loc['n', 'value']      # Viscosity exponent
TMP    = input_df.loc['TMP', 'value']    # Transmembrane pressure [Pa] - Increased
iRT    = input_df.loc['iRT', 'value']   # Osmotic pressure constant [Pa·m³/mol]
S      = input_df.loc['S', 'value']      # Sieving coefficient (for mAbs, ~0)

k_drop = input_df.loc['k_drop', 'value']        # Pressure drop coefficient (reduced)

# Flow and module parameters
A_m    = input_df.loc['A_m', 'value']     # Membrane area [m²] - Increased for faster filtration
Q_f    = input_df.loc['Q_f', 'value']     # Feed flow rate [m³/s] - Increased
A_c    = input_df.loc['A_c', 'value']     # Channel cross-sectional area [m²]
D      = input_df.loc['D', 'value']    # Diffusion coefficient [m²/s]

# Recirculation (vessel) parameters
V_s    = input_df.loc['V_s', 'value']     # Active zone volume [m³] - Decreased for faster concentration
V_d    = input_df.loc['V_d', 'value']     # Dead zone volume [m³] - Decreased
k_d    = input_df.loc['k_d', 'value']      # Mass transfer coefficient between zones [s⁻¹] - Increased
beta   = input_df.loc['beta', 'value']     # Bypass fraction - Decreased
alpha  = V_d / (V_s + V_d)  # Dead zone fraction

# Fouling parameters
k_f    = input_df.loc['k_f', 'value']     # Fouling growth coefficient [s⁻¹] - Increased

# Intrinsic membrane resistance (computed from L_p and mu_0)
R_m    = 1/(L_p * mu_0)

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
C_s0  = input_df.loc['C_s0', 'value']      # Active zone concentration [kg/m³]
C_d0  = input_df.loc['C_d0', 'value']      # Dead zone concentration [kg/m³]
V_r0  = V_s + V_d # Total retentate volume [m³]
R_f0  = input_df.loc['R_f0', 'value']       # Initial fouling resistance
initial_state = [C_s0, C_d0, V_r0, R_f0]

# =============================================================================
# MODEL FUNCTION
# =============================================================================
def tff_model(t, y):
    # Unpack state variables:
    C_s, C_d, V_r, R_f = y
    
    # Check for very small volume and return zeros if nearly empty
    if V_r < 1e-4:
        return [0, 0, 0, 0]
    
    # Calculate actual volumes for active and dead zones
    V_s_actual = (1 - alpha) * V_r
    V_d_actual = alpha * V_r
    
    # Ensure volumes are positive
    if V_s_actual <= 0 or V_d_actual <= 0:
        return [0, 0, 0, 0]

    # Overall retentate concentration:
    C_r = (V_s_actual*C_s + V_d_actual*C_d) / V_r

    # Dynamic viscosity:
    mu = mu_0 * (1 + k_mu * (C_s**n))  # Use active zone concentration for viscosity
    
    # Calculate effective feed velocity in the active zone (excluding bypass):
    u_f = Q_f * (1 - beta) / A_c  # [m/s]
    # Mass transfer coefficient for turbulent flow:
    k_m = 0.036 * (D**(1/3)) * (u_f**(4/5))  # [m/s]
    
    # Solve for flux (J) using a more robust approach
    def compute_J(mu, R_f, TMP, C_b, k_drop, iRT, k_m, R_m):
        J_guess = TMP / ((R_m + R_f) * mu)  # Initial guess ignoring concentration polarization
        for _ in range(20):
            DeltaP = k_drop * mu * J_guess
            TMP_eff = max(0, TMP - DeltaP)  # Ensure non-negative effective TMP
            
            # Cap the concentration polarization factor for numerical stability
            pol_factor = min(20, J_guess / k_m)  # Limit to avoid numerical explosion
            C_w = C_b * np.exp(pol_factor)
            
            # Calculate osmotic pressure effect
            pi = iRT * C_w
            
            # Calculate new flux
            J_new = max(0, (TMP_eff - pi) / ((R_m + R_f) * mu))
            
            # Check convergence
            if abs(J_new - J_guess) < 1e-6 or J_new < 1e-10:
                break
            
            # Update guess with damping to improve convergence
            J_guess = 0.7 * J_new + 0.3 * J_guess
        
        return max(0, J_guess)  # Ensure non-negative flux

    # Calculate flux
    J = compute_J(mu, R_f, TMP, C_s, k_drop, iRT, k_m, R_m)
    
    # Calculate permeate flow rate
    Q_p = J * A_m
    
    # Ensure permeate flow doesn't exceed available volume
    Q_p = min(Q_p, V_r / 100)  # Limit permeate flow to avoid numerical instability
    
    # Mass balance for active zone:
    # Mass in from retentate - Mass out to permeate - Mass exchange with dead zone
    dC_s_dt = ((Q_f * (1-beta) * (C_r - C_s)) / V_s_actual) + \
              (k_d * (C_d - C_s)) + \
              ((C_s * Q_p) / V_s_actual)  # Concentration effect due to volume reduction
    
    # Mass balance for dead zone:
    dC_d_dt = k_d * (C_s - C_d) + ((C_d * Q_p) / V_d_actual)  # Concentration effect
    
    # Volume change - volume decreases with permeate flux
    dV_r_dt = -Q_p
    
    # Fouling resistance dynamics:
    dR_f_dt = k_f * J * C_s  # Make fouling dependent on concentration too
    
    return [dC_s_dt, dC_d_dt, dV_r_dt, dR_f_dt]

# =============================================================================
# SOLVE THE SYSTEM
# =============================================================================
t_span = (0, 1500)  # Reduced simulation time for faster concentration
t_eval = np.linspace(t_span[0], t_span[1], 500)

solution = solve_ivp(
    tff_model, 
    t_span, 
    initial_state, 
    t_eval=t_eval, 
    method='BDF',  # Better for stiff problems
    rtol=1e-4, 
    atol=1e-6
)

# =============================================================================
# CALCULATE DERIVED QUANTITIES
# =============================================================================
# Compute overall retentate concentration, C_r(t)
V_r_sol = solution.y[2]
V_s_actual = (1 - alpha) * V_r_sol
V_d_actual = alpha * V_r_sol
C_s_sol = solution.y[0]
C_d_sol = solution.y[1]

# Calculate overall concentration
C_r_sol = (V_s_actual * C_s_sol + V_d_actual * C_d_sol) / V_r_sol

# Calculate volume reduction factor
VRF = V_r0 / V_r_sol

# Calculate permeate flux over time
R_f_sol = solution.y[3]
mu_sol = mu_0 * (1 + k_mu * (C_s_sol**n))
J_sol = np.zeros_like(solution.t)

# Calculate flux at each time point
for i in range(len(solution.t)):
    # Skip calculation at very small volumes
    if V_r_sol[i] < 1e-4:
        J_sol[i] = 0
        continue
        
    def compute_J_point(mu, R_f, TMP, C_b):
        J_guess = TMP / ((R_m + R_f) * mu)
        for _ in range(20):
            DeltaP = k_drop * mu * J_guess
            TMP_eff = max(0, TMP - DeltaP)
            
            # Calculate mass transfer coefficient
            u_f = Q_f * (1 - beta) / A_c
            k_m = 0.036 * (D**(1/3)) * (u_f**(4/5))
            
            pol_factor = min(20, J_guess / k_m)
            C_w = C_b * np.exp(pol_factor)
            
            pi = iRT * C_w
            J_new = max(0, (TMP_eff - pi) / ((R_m + R_f) * mu))
            
            if abs(J_new - J_guess) < 1e-6:
                break
            J_guess = 0.7 * J_new + 0.3 * J_guess
        
        return max(0, J_guess)
    
    J_sol[i] = compute_J_point(mu_sol[i], R_f_sol[i], TMP, C_s_sol[i])

# Create datadrame from simulation result
output_df = pd.DataFrame({
    'time (s)': solution.t,
    'C_s': C_s_sol,
    'C_d': C_d_sol,
    'C_r': C_r_sol,
    'V_r': V_r_sol,
    'VRF': VRF,
    'R_f': R_f_sol,
    'mu': mu_sol,
    'J': J_sol*3600,
    'membrane' : R_f_sol/R_m
})

# write to excel sheet "output"
with pd.ExcelWriter('tff_param.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    output_df.to_excel(writer, sheet_name='Output', index=False)
    