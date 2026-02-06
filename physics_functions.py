# physics_functions.py
#
# Foundation Acknowledgment:
# The core physics models and calculation methods within this module
# are based on the work found in the following repository:
# octarine123/automotive-ICE-modelling
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

from typing import TYPE_CHECKING
import numpy as np
import constants as c
import math
import sys

if TYPE_CHECKING:
    from engine_model import CylinderState,  Valves, Valve


def eng_speed_rad(rpm):
    """Converts RPM (Revolutions Per Minute) to angular speed in radians per second."""
    return rpm * np.pi / 30.0


# --- Geometric Functions ---


def v_cyl(theta, A_piston, CV):
    """
    Docstring for v_cyl
    
    :param theta: np array of thetas 0-719
    :param A_piston: area of piston m^2
    :param CV: clearance volume m^3

    Calculates cylinder volume based on crank angle (theta).
    This uses the slider-crank mechanism geometry (theta=0 is TDC compression).

    The standard formula uses piston displacement S from TDC (TDC is where V=CV).
    V = V_clearance + A_piston * S units m^3 measured by degree.
    """
    R_stroke = c.RADIUS_CRANK  # Crank radius (half stroke)
    L_conrod = c.LEN_CONROD  # Connecting rod length

    # Piston displacement (S) from TDC compression (theta=0).
    # S is the distance the piston has traveled away from the cylinder head.
    S = R_stroke * (1 - np.cos(np.deg2rad(theta))) + L_conrod * (
        1 - np.sqrt(1 - (R_stroke / L_conrod) ** 2 * np.sin(np.deg2rad(theta)) ** 2)
    )

    # Cylinder Volume = Clearance Volume + Swept Volume
    # The original implementation's piston position calculation was over-complicated.
    # This corrected calculation uses the standard piston displacement formula.
    return CV + (A_piston * S)


# --- Thermal Cycle Functions (Heat & Work) ---

def calc_wiebe_fraction(theta, ignition_start_theta, burn_duration, a_vibe, m_vibe):
    """ this version is not sub-step friendly, so changed approach to fraction """
    
    delta_theta = theta - ignition_start_theta
    
    if delta_theta <= 0:
        return 0.0
    if delta_theta >= burn_duration:
        return 1.0
        
    # The standard Wiebe cumulative S-curve formula
    return 1.0 - np.exp(-a_vibe * (delta_theta / burn_duration) ** (m_vibe + 1))


def calc_woschni_heat_loss(CAD, rpm, cyl:CylinderState, valves:Valves):
    """   
    Calculates dQ_w/d_theta using the updated Woschni model with 
    instantaneous piston velocity and SI-standard coefficients.
    """
    # extract cylinder and valve parameters
    log_P = cyl.log_P
    log_T = cyl.log_T
    log_V = cyl.V_list
    IVC = valves.intake.close_angle
    EVO = valves.exhaust.open_angle
    spark = cyl.spark_event_theta
    
    # Constant setup
    P_curr_kPa = cyl.P_curr / 1000 # kPa
    V_curr = cyl.V_curr # m3
    T_curr = cyl.T_curr # T
    T_wall = cyl.T_wall
    V_clearance = cyl.V_clearance
    
    
    # Set the P,V,T references at IVC_Angle
    P_ref_kPa = log_P[int(IVC)] / 1000 # must be in kPa
    V_ref = log_V[int(IVC)] # m3
    T_ref = log_T[int(IVC)] # K
    
    
    # Set C1 and C2
    if spark <= CAD < EVO:
        C1, C2 =  2.28, 0.00324
    # Phase 2: Compression (from IVC to Spark)
    elif IVC <= CAD < spark:
        C1, C2 =  2.28, 0.0 
    # Phase 3: Gas Exchange (Exhaust and Intake)
    else:
        C1, C2 =  6.18, 0.0
        
    # 2. Safety Guards
    if T_curr <= 50.0 or P_curr_kPa <= 1.0 or np.isnan(T_curr):
        return 0.0
    
    # Calculate P_motoring for Wochni function.
    # Inside your crank-angle loop:
    if CAD >= IVC:
        # Calculate motored pressure based on isentropic compression from IVC
        # n is usually 1.32 for a PFI mixture
        P_motored_kPa = P_ref_kPa * (V_ref / V_curr)**1.32
    else:
        # During gas exchange, P_motored isn't used (C2=0), 
        # but you can set it to actual P for stability
        P_motored_kPa = P_curr_kPa

    # 3. Gas Velocity Calculation
    # mean piston speed (U means velocity, p is for piston)
    u_p_mean = 2.0 * c.STROKE * (rpm / 60.0)
    # Standard Woschni pressure term: C2 * (Vd * Tr / (Pr * Vr)) * (P - P_motored)
    delta_P = max(0, P_curr_kPa - P_motored_kPa)
    pressure_rise_term = C2 * ((c.V_DISPLACED * T_ref) / (P_ref_kPa * V_ref)) * delta_P
    # Final Velocity
    W_vel = (C1 * u_p_mean) + pressure_rise_term


    # 4. Heat Transfer Coefficient h_g
    # Using the standard SI Woschni correlation:
    h_g = 3.26 * (c.BORE**-0.2) * (P_curr_kPa**0.8) * (T_curr**-0.53) * (W_vel**0.8)

    # 5. Instantaneous Surface Area (A_w)
    # Cylinder wall area = pi * Bore * instantaneous_height
    # Instantaneous height x = V_curr / A_piston
    V_bore = V_curr - V_clearance
    dist_from_tdc = V_bore / c.A_PISTON
    A_w = (2 * c.A_PISTON) + (np.pi * c.BORE * dist_from_tdc)

    # 6. Heat Loss Rate & Conversion in Joules/sec
    # dQ/dt = h_g * A * (T_gas - T_wall)
    dQ_w_dt = h_g * A_w * (T_curr - T_wall)
    
    # Scale heat transfer based on RPM to allow starting
    # normall 0.8-1.2.  1.66 suggests T_wall is set too high or your combustion is too fast
    thermal_scaling = 1.2 # was 1.66, 

    # Convert J/s to J/deg: (dQ/dt) / (dtheta/dt) * (rad to deg)
    if rpm < 1.0: 
        return 0.0

    # Simplified conversion: (J/s) / (6 * RPM) = J/deg
    dQ_w_d_theta = (dQ_w_dt / (6.0 * rpm)) * thermal_scaling
    
    # if 349 <= CAD <= 362:
    #     print(f"DEBUG WOSCHNI  "
    #         f" CAD:{CAD} P_curr:{P_curr_kPa:.2f}kPa T_curr:{T_curr:.2f} W_vel:{W_vel:.2f} h_g:{h_g:.2f} "
    #         f"A_w:{A_w:.2f} dQ/dt:{dQ_w_dt:.2f}")

    return dQ_w_d_theta

def calc_piston_speed_factor(theta):
    """
    Calculates the instantaneous geometric factor (dS/dtheta) for piston speed.

    To get the actual piston speed (m/s):
    Vp = omega * (dS/dtheta)  (where omega is in rad/s and theta in rad)

    This function returns the dimensionless geometric factor based on theta in degrees.
    """
    R_stroke = c.RADIUS_CRANK
    L_conrod = c.LEN_CONROD

    # Piston velocity dS/dtheta (normalized by d(theta)/dt = omega)
    sin_t = np.sin(np.deg2rad(theta))
    cos_t = np.cos(np.deg2rad(theta))

    term1 = R_stroke * sin_t

    # Calculate the term under the square root in the displacement derivative
    lambda_sq_sin_sq = (R_stroke / L_conrod) ** 2 * sin_t**2
    if lambda_sq_sin_sq >= 1.0:
        # Avoid math domain error near BDC/TDC if R/L is close to 1
        lambda_sq_sin_sq = 0.9999

    denominator = np.sqrt(1 - lambda_sq_sin_sq)
    term2 = (R_stroke**2 / L_conrod) * sin_t * cos_t / denominator

    # The actual geometric factor dS/dtheta (if theta is in radians)
    return term1 + term2


# --- Valve Lift and Area Functions ---

# def calc_valve_lift(theta, L_peak, duration, phase_lag, zero_ref=-360.0):
#     """
#     Calculates instantaneous valve lift using a simple cosine-based approximation
#     of a cam profile. (No changes made here, calculation looks correct for approximation).
#     """

#     # Center the angle around the peak lift (phase_lag)
#     # The duration starts/ends at L=0.
#     theta_adjusted = (theta - phase_lag)

#     # Find the valve open range: start_angle to end_angle
#     start_angle = phase_lag - duration / 2.0
#     end_angle = phase_lag + duration / 2.0

#     # We must handle the 720-degree cycle wrap-around.
#     # The logic below attempts to handle the boundary condition of the cosine curve.

#     is_open = False

#     # Normalize theta to be in the range [0, 720)
#     theta_norm = theta % 720.0
#     start_norm = start_angle % 720.0
#     end_norm = end_angle % 720.0

#     if start_norm > end_norm:
#         # Overlap case (e.g., start at 500, end at 100)
#         if theta_norm >= start_norm or theta_norm <= end_norm:
#             is_open = True
#     else:
#         # Standard case (e.g., start at 100, end at 500)
#         if theta_norm >= start_norm and theta_norm <= end_norm:
#             is_open = True

#     if not is_open:
#         return 0.0

#     # Calculate the cosine lift profile (L = L_peak * (1 - cos(theta_adjusted * 360/duration)) / 2)
#     # The factor 360/duration ensures the argument of cos goes from -pi to pi over the duration.
#     lift_ratio = (1 - np.cos(np.deg2rad(theta_adjusted) * 360.0 / duration)) / 2.0

#     return max(0.0, L_peak * lift_ratio)


# def calc_valve_area(theta_list):
#     """
#     Calculates the instantaneous curtain flow area for intake and exhaust valves
#     over the entire crank angle range (theta_list).
#     """

#     # --- Intake Valve Constants (Index 0) ---
#     D_i = c.VALVE_DATA['D_valve'][0]
#     L_i_peak = c.VALVE_DATA['L_peak'][0]
#     Dur_i = c.VALVE_DATA['duration'][0]
#     PL_i = c.VALVE_DATA['phase_lag'][0]
#     Cd_i = c.VALVE_DATA['flow_coeff'][0]

#     ZERO_REF = c.THETA_MIN

#     # --- Exhaust Valve Constants (Index 1) ---
#     D_e = c.VALVE_DATA['D_valve'][1]
#     L_e_peak = c.VALVE_DATA['L_peak'][1]
#     Dur_e = c.VALVE_DATA['duration'][1]
#     PL_e = c.VALVE_DATA['phase_lag'][1]
#     Cd_e = c.VALVE_DATA['flow_coeff'][1]

#     A_in_list = []
#     A_ex_list = []

#     for theta in theta_list:

#         # Intake Valve (IV) lift calculation
#         L_i_act = calc_valve_lift(theta, L_i_peak, Dur_i, PL_i, ZERO_REF)
#         # Area = pi * D_valve * L_actual * flow_coeff (Curtain area approximation)
#         A_in = np.pi * D_i * L_i_act * Cd_i
#         A_in_list.append(float(A_in))

#         # Exhaust Valve (EV) lift calculation
#         L_e_act = calc_valve_lift(theta, L_e_peak, Dur_e, PL_e, ZERO_REF)
#         A_ex = np.pi * D_e * L_e_act * Cd_e
#         A_ex_list.append(float(A_ex))


#     return {
#         'A in': A_in_list,
#         'A ex': A_ex_list
#     }


# def theta_to_720(theta):
#     """Safe modulo 720 — works on scalars and arrays"""
#     return theta % 720.0


# def calc_valve_lift_vectorized(theta_array: np.ndarray, valve: Valve) -> np.ndarray:
#     """
#     Vectorized version of calc_valve_lift — returns array of lifts for entire theta_array
#     """

#     ca = theta_array % 720.0 # crank angle

#     open_ca = valve.open_angle
#     close_ca = valve.close_angle
    
    
#     # 1. Calculate duration and position
#     duration = (close_ca - open_ca) % 720
#     rel_angle = (ca - open_ca) % 720
#     mid_dur = duration / 2.0
    
#     lift = np.zeros_like(ca, dtype=float)
#     active = (rel_angle <= duration)
    
#     # 2. Use mm here (e.g., 9.5)
#     l_max = valve.max_lift

#     phi_deg = (rel_angle[active] - mid_dur) / mid_dur * 180.0
#     # p < 1.0 makes it 'fatter' (more area). p > 1.0 makes it 'pointier'.
#     p = 0.3 # was 0.4 and 0.7
#     lift[active] = l_max * ((1.0 + np.cos(np.radians(phi_deg))) / 2.0)**p
    
#     return lift


def calc_valve_lift_vectorized(theta: np.ndarray, valve: 'Valve') -> np.ndarray:
    """
    Calculates the valve lift with the effect of the Cam follower diameter
    """
    # 1. Generate the "Pointy" raw lift (what the cam lobe looks like)
    raw_lift = calc_cam_lift_vectorized(theta, valve)
    
    # 2. Define the follower influence (12mm radius)
    # On a ~28mm base circle, 12mm is roughly 45-50 crank degrees of contact!
    follower_radius = 12.0 
    base_circle_radius = 14.0
    deg_offset = int(np.degrees(follower_radius / base_circle_radius))
    
    # 3. Apply the "Flat Face" effect
    # The valve follows the maximum height currently touching any part of the 24mm face
    fat_lift = np.zeros_like(raw_lift)
    for i in range(len(raw_lift)):
        # Search the 'window' of the 24mm wide lifter
        window = np.take(raw_lift, range(i - deg_offset, i + deg_offset), mode='wrap')
        fat_lift[i] = np.max(window)
        
    return fat_lift

def calc_cam_lift_vectorized(theta: np.ndarray, valve: 'Valve', fatness: float = 0.5) -> np.ndarray:
    """
    1-degree resolution optimized smooth profile.
    Uses Gaussian tapering at the seats to prevent stepped artifacts.
    """
    # 1. Geometry extraction
    open_a = valve.open_angle
    close_a = valve.close_angle
    max_l = valve.max_lift
    
    # 2. Duration and Wrap Handling
    duration = (close_a - open_a) % 720
    midpoint = (open_a + (duration / 2)) % 720
    
    # 3. Shortest angular distance to midpoint
    dist = (theta - midpoint + 360) % 720 - 360
    
    # 4. Normalized radius (0 at peak, 1 at seats)
    # We add a tiny buffer to the duration to let the curve 'breathe'
    # without hitting a 1-degree hard edge.
    x = dist / (duration / 2)
    
    # 5. The Profile: Cosine with Gaussian decay
    # This formula is C-infinite; it never technically hits a 'step'
    # The 'fatness' term widens the peak.
    p = 1.0 + (fatness * 1.0)
    
    # Core Lift Calculation
    lift = max_l * np.exp(-5.0 * (np.abs(x)**(2 * p)))
    
    # 6. Hard-Floor Anti-Ghosting
    # Even though it's smooth, we don't want 0.0001mm lift all 720 degrees.
    # We clear anything outside the window + a 2-degree buffer for anti-aliasing.
    mask = np.abs(dist) < (duration / 2 + 2)
    return np.where(mask, lift, 0.0)


def calc_valve_area_vectorized(theta_array: np.ndarray, valve: Valve) -> np.ndarray:
    
    # Get lift in mm (0 to 9.5)
    lift_mm = calc_valve_lift_vectorized(theta_array, valve)
    
    # Get diameter in mm (e.g., 32.0)
    diam_mm = valve.diameter
    
    # Formula: Area = pi * D * L
    # We divide by 1,000,000 to get m^2
    area_m2 = (np.pi * diam_mm * lift_mm) / 1e6
    
    return area_m2


# --- Flow and Mass Functions ---

def calc_isentropic_flow(A_valve, lift, diameter, P_cyl, T_cyl, R_cyl, g_cyl, P_extern, T_extern, R_extern, g_extern, is_intake = True):
    """
    P_cyl, T_cyl, R_cyl, g_cyl: State INSIDE the cylinder
    P_manifold, T_manifold, R_manifold, g_manifold: State OUTSIDE (Manifold for intake, Atmosphere for exhaust)
    """
    if A_valve < 1e-9 or lift < 1e-5:
        return 0.0, 0.0

    # Identify Upstream (Source) and Downstream (Sink)
    if P_cyl >= P_extern:
        # Flowing OUT of cylinder (Exhaust or Intake Reversion)
        P_up, T_up, R_up, gamma = P_cyl, T_cyl, R_cyl, g_cyl
        P_down = P_extern
        direction = -1.0 # Negative means leaving cylinder
    else:
        # Flowing INTO cylinder (Normal Intake or Exhaust Backflow/EGR)
        P_up, T_up, R_up, gamma = P_extern, T_extern, R_extern, g_extern
        P_down = P_cyl
        direction = 1.0 # Positive means entering cylinder

    # --- Standard Isentropic Math ---
    pr = P_down / max(P_up, 1.0)
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    
    Cd = _calc_physics_Cd(lift, diameter)
    # Cd = 0.7 if not is_intake else Cd
    # Cd=1.0

    if pr <= pr_crit:
        # Choked Flow
        mdot = (Cd * A_valve * P_up * np.sqrt(gamma / (R_up * T_up)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
    else:
        # Subcritical Flow
        pr_eff = min(pr, 0.9999)
        mdot = (Cd * A_valve * P_up * np.sqrt(2 * gamma / (R_up * T_up * (gamma - 1)) * (pr_eff**(2/gamma) - pr_eff**((gamma+1)/gamma))))

    return mdot * direction, Cd

# def _calc_physics_Cd(lift, diameter):
#     L_D = lift / diameter
    
#     # Standard Taylor/Heywood Correlation for Poppet Valves
#     if L_D <= 0.05:
#         # Initial opening / viscous regime
#         return 0.30 + (4.0 * L_D) 
#     elif L_D <= 0.20:
#         # Transitional regime
#         return 0.50 + (1.33 * (L_D - 0.05))
#     elif L_D <= 0.25:
#         # Peak detachment efficiency
#         return 0.70
#     else:
#         # High-lift port restriction (Geometric drop)
#         # As L/D increases, the effective Cd must drop because 
#         # the port cross-section limits the mass flow.
#         return 0.70 - 0.8 * (L_D - 0.25)
    
def _calc_physics_Cd(lift, diameter):
    if lift <= 0: return 0.0
    ld_ratio = lift / diameter
    
    # Target: Stay slightly more efficient at mid-lift (0.15 - 0.25 L/D)
    # We increase the base_cd to 0.74 (a very healthy ported VW head)
    # and slow the decay rate slightly.
    base_cd = 0.74 
    efficiency_gain = 0.12
    decay_rate = 8.0 # Reduced from 10.0 to broaden the "sweet spot"
    
    cd = base_cd + efficiency_gain * np.exp(-decay_rate * ld_ratio)
    return cd

# def _calc_dynamic_discharge_coeff(rpm, lift, diameter, is_intake):
#     if lift <= 0.0 or diameter <= 0.0:
#         return 0.0
    
#     # lift and diameter are in meters
#     l_over_d = lift / diameter 

#     # 1. Low-lift penalty
#     if l_over_d < 0.03:
#         low_lift_factor = 0.20 + 6.67 * l_over_d
#     elif l_over_d < 0.10:
#         low_lift_factor = 0.40 + 3.0 * (l_over_d - 0.03)
#     else:
#         low_lift_factor = 0.70 + 0.8 * (l_over_d - 0.10)
    
#     # 2. Geometry boost
#     geo_boost = 1.0 + 0.2 * min(l_over_d / 0.25, 1.0)

#     # 3. Mach Index (Z) - Using Piston Speed
#     # sp = mean piston speed (m/s)
#     sp = (2 * c.STROKE * rpm) / 60.0 
    
#     # Z-index formula (all units in meters/seconds now)
#     z = ((c.BORE / diameter)**2) * (sp / 340.0)
#     # this is causes restrictive flow in combination withe cosine CAM profile.
#     # choke_factor = np.exp(-0.6 * (z / 0.5)**2)
#     choke_factor = np.exp(-0.2 * (z / 0.5)**2)

#     Cd = 0.70 * low_lift_factor * geo_boost * choke_factor
#     return np.clip(Cd, 0.15, 0.90)


# def calc_discharge_coeff(rpm):
#     """
#     Calculates a correction factor (multiplied by the base Cd) to simulate
#     volumetric efficiency drop at high RPM due to friction/flow losses. (No changes made here).
#     """
#     if rpm > c.RPM_MAX_FLOW_REF:
#         # Simple quadratic drop-off model
#         rpm_ratio = (rpm - c.RPM_MAX_FLOW_REF) / c.RPM_MAX_FLOW_REF
#         cd_corr = 1.0 - c.K_CD_RPM_CORR * (rpm_ratio**2)
#         return max(0.5, cd_corr)  # Clamp to 0.5 minimum
#     return 1.0



# def intake_mass_flow(A_valve, P_cyl, P_man, T_man, rpm):
#     """
#     Industry-standard NA intake flow with momentum override.
#     Absolutely no NaNs, no crashes, works from 250 rpm to 12 000 rpm.
#     """
    
#     if A_valve < 1e-8:
#         return 0.0

#     # Fixed effective back-pressure — this is what every OEM uses
#     # P_down_eff = 92_000.0  # 0.92 bar
#     # pr = P_down_eff / max(P_man, 1000.0)  # never divide by zero
#     pr = P_cyl / P_man

#     gamma = c.GAMMA_AIR
#     R = c.R_SPECIFIC_AIR
  
#     Cd = 0.68  # calibrated real-world value

#     # Critical pressure ratio for air
#     pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))  

#     if pr <= pr_crit:
#         # Choked flow — simple, robust, never NaN
#         mdot = (
#             Cd
#             * A_valve
#             * P_man
#             * np.sqrt(gamma / (R * T_man))
#             * (gamma + 1.0)
#             / 2.0 ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
#         )
#     else:
#         # Sub-critical — mathematically safe version
#         term1 = pr ** (2.0 / gamma)
#         term2 = pr ** ((gamma + 1.0) / gamma)
#         inside = 2.0 * gamma / (R * T_man) / (gamma - 1.0) * (term1 - term2)
#         inside = max(inside, 0.0)  # this kills the NaN
#         mdot = Cd * A_valve * P_man * np.sqrt(inside)

#     deg_per_sec = max(rpm, 50.0) * 6.0  # rpm → deg/s
#     return (mdot / deg_per_sec)  # kg/deg


# def exhaust_mass_flow(A_valve, P_cyl, T_cyl, rpm):
#     """Exhaust flow — always outward, no reverse"""
#     if A_valve < 1e-9:
#         return 0.0

#     return calc_mass_flow_base(
#         A_valve=A_valve,
#         P_up=P_cyl,
#         T_up=T_cyl,
#         P_down=c.P_ATM_PA * 0.98,
#         R_spec=c.R_SPECIFIC_MIX,
#         gamma=c.GAMMA_AIR,
#         rpm=rpm,
#         Cd=0.70,
#     )


# def calc_mass_flow_base(A_valve, P_up, T_up, P_down, R_spec, gamma, rpm, Cd):
#     """Pure isentropic flow — no clamps, no tricks"""
#     if P_up < 100.0 or T_up < 200.0:
#         return 0.0

#     P_up = max(P_up, 100.0)  # tiny numerical guard only
#     P_down = max(P_down, 10.0)

#     pr = P_down / P_up
#     crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

#     if pr <= crit:
#         # Choked
#         mdot = (
#             Cd
#             * A_valve
#             * P_up
#             * np.sqrt(gamma / (R_spec * T_up))
#             * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
#         )
#     else:
#         # Subcritical
#         pr_eff = min(pr, 0.9999)
#         mdot = (
#             Cd
#             * A_valve
#             * P_up
#             * np.sqrt(
#                 2
#                 * gamma
#                 / (R_spec * T_up * (gamma - 1))
#                 * (pr_eff ** (2 / gamma) * (1 - pr_eff ** ((gamma - 1) / gamma)))
#             )
#         )

#     deg_per_sec = max(rpm, 50.0) * 6.0
#     return mdot / deg_per_sec


# --- Combustion and Heat Functions ---


def is_combustion_phase(theta, t_spark, t_burn):
    """Checks if the current crank angle is within the combustion window. (No changes made here)."""
    # Combustion starts at (t_spark + burn_delay) ATDC
    t_start = -t_spark + c.BURN_DELAY
    t_end = t_start + t_burn

    return t_start <= theta < t_end


def calc_mass_burned_rate(theta, M_fuel_total):
    """
    Calculates the rate of mass burning (dMb/dtheta) using the Vibe function
    and the total fuel mass for the cycle. (No changes made here).
    """

    # 1. Define Burn Start and Duration (normalized to 0-1)
    x_start = -c.T_IGNITITION + c.BURN_DELAY  # ATDC
    burn_duration = c.T_BURN

    # 2. Normalized Crank Angle (x: fraction of burn duration)
    x = (theta - x_start) / burn_duration

    if x < 0.0 or x > 1.0:
        return 0.0, 0.0

    # 3. Vibe Parameters
    a = c.WEIBE_A
    m = c.WB_M

    # 4. Mass Fraction Burned (x_b)
    x_b = 1.0 - np.exp(-a * x ** (m + 1))

    # 5. Mass Burn Rate (dx_b/dtheta)
    # The term (a * (m + 1) / burn_duration) is d/dtheta of the exponent part
    dxb_d_theta = (a * (m + 1) / burn_duration) * x**m * np.exp(-a * x ** (m + 1))

    # Mass rate dMb/dtheta = M_fuel_total * dxb/dtheta
    dMb_d_theta = M_fuel_total * dxb_d_theta

    return dMb_d_theta, x_b


# --- Thermodynamic Integration ---

def integrate_first_law(
    P_curr, T_curr, M_curr, V_curr, Delta_M, Delta_Q_in, Delta_Q_loss, 
    dV_d_theta, gamma, theta_delta, T_manifold, R_spec,
    cycle=None, CAD=None, substep=None # for debug purposes only
):
    # 1. Physical Safeguards
    M_curr = max(M_curr, 1e-9)
    V_curr = max(V_curr, 1e-9)
    
    # 2. Rates
    dM_d_theta = Delta_M / theta_delta
    dQ_net_rate = (Delta_Q_in / theta_delta) - Delta_Q_loss 

    # 3. PREDICTOR STEP
    T_flow_pred = T_manifold if Delta_M >= 0 else T_curr
    
    term_heat = (gamma - 1.0) * dQ_net_rate
    term_work = -gamma * P_curr * dV_d_theta
    term_mass = gamma * R_spec * T_flow_pred * dM_d_theta
    
    dP_d_theta_pred = (term_heat + term_work + term_mass) / V_curr
    
    # Predict midpoint states
    P_mid = P_curr + dP_d_theta_pred * (0.5 * theta_delta)
    V_mid = V_curr + dV_d_theta * (0.5 * theta_delta)
    M_mid = M_curr + (0.5 * Delta_M)
    T_mid = (P_mid * V_mid) / (M_mid * R_spec)


    # 4. CORRECTOR STEP
    T_flow_corr = T_manifold if Delta_M >= 0 else T_mid
    
    term_work_mid = -gamma * P_mid * dV_d_theta
    term_mass_mid = gamma * R_spec * T_flow_corr * dM_d_theta
    
    dP_d_theta_final = (term_heat + term_work_mid + term_mass_mid) / V_mid
    
    # 5. FINAL STATE
    P_next = P_curr + dP_d_theta_final * theta_delta
    M_next = M_curr + Delta_M
    V_next = V_curr + dV_d_theta * theta_delta
    T_next = (P_next * V_next) / (M_next * R_spec)
    
    return P_next, T_next


# --- Engine Performance Functions ---


def calc_pumping_losses(P_cyl, V_list, theta_list):
    """This function is a placeholder; PMEP is implicitly calculated in W_indicated."""
    pass



# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     """
#     Calculations the friction associated with a single cylinder.  it excludes the firction of the oil pump, valve train etc
#     that is common across all cylinders.
#     """
#     # 1. Kinematics
#     geom_factor = calc_piston_speed_factor(theta)
#     omega = (rpm * 2 * np.pi) / 60.0
#     v_piston_inst = abs(omega * geom_factor)
    
#     # 2. Viscosity (Stays the same, very standard)
#     visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))

#     # 3. Dynamic Coefficients (Heavily tuned down)
#     # These represent only the piston assembly (rings/skirts)
#     f_linear = 0.08 * v_piston_inst          # was 0.12
#     f_quadratic = 0.001 * (v_piston_inst**2) # was 0.0015
#     f_pressure = 0.00001 * max(0, p_cyl)     # was 0.000015
    
#     total_f_friction = (f_linear + f_quadratic + f_pressure)
    
#     # 4. Correct Conversion to Torque
#     # We use Crank Radius * sin(theta) to represent the leverage of the friction
#     t_friction = total_f_friction * c.RADIUS_CRANK * abs(np.sin(np.deg2rad(theta)))
    
#     return t_friction * visc_factor

def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
    """
    Calculations the friction associated with a single cylinder.  it excludes the firction of the oil pump, valve train etc
    that is common across all cylinders.
    """
     # 1. Kinematics
    geom_factor = calc_piston_speed_factor(theta)
    omega = (rpm * 2 * np.pi) / 60.0
    v_piston_inst = abs(omega * geom_factor)
    
    # 2. Viscosity
    visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))

    # 3. Dynamic Coefficients 
    # Increasing f_linear slightly to raise the 3000rpm baseline
    f_linear = 1.65 * v_piston_inst # was 2.0
    
    # Increasing f_quadratic to ensure we hit the "doubling" target at 4500rpm
    f_quadratic = 0.015 * (v_piston_inst**2) # was 0.02
    
    # f_pressure remains a minor player during motoring but major during firing
    f_pressure = 0.00001 * max(0, p_cyl) 
    
    total_f_friction = (f_linear + f_quadratic + f_pressure)
    
    # 4. Correct Conversion to Torque
    t_friction = total_f_friction * c.RADIUS_CRANK * abs(np.sin(np.deg2rad(theta)))
    
    # Apply viscosity mostly to the speed-based shearing
    return t_friction * visc_factor

def calc_engine_core_friction(clt, rpm):
    """
    Calculated the Global parasitics: Oil pump, crank seals, and cam/valvetrain drag.
    """
    visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))
    
    # Base parasitic now scales slightly with RPM
    # (e.g., 2Nm at 1000rpm, 4Nm at 3000rpm)
    base_nm = 2.1 + (rpm/6000) # was 1.3 + rpm/3000
    
    return base_nm * visc_factor

    

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     """
#     calculates instantaneous friction for a cylinder based on 
#     speed of rotation, oil and piston ring load.
#     """
#     # 1. Get kinematics
#     geom_factor = calc_piston_speed_factor(theta)
#     omega = (rpm * 2 * np.pi) / 60.0
#     v_piston_inst = abs(omega * geom_factor) # Instantaneous speed m/s
    
#     # 2. Viscosity Factor (Your existing logic)
#     visc_factor = max(1.0, 2.5 * np.exp(-0.025 * (clt - 20)))
    
#     # 3. Calculate Forces (Newton) for single cylinder
#     C_VISCOUS = 5.0 # Ns/m
#     C_RING_LOAD = 0.00004 # m^2
#     f_viscous = C_VISCOUS * v_piston_inst * visc_factor
#     f_pressure = C_RING_LOAD * p_cyl # p_cyl in Pascals
    
#     # 4. Total Force to Torque
#     # Moment arm is geom_factor
#     t_friction = (f_viscous + f_pressure) * abs(geom_factor)
    
#     return t_friction


# """ this function over simplied Friction and return FMEP, the average for 720 cycle """
# def calc_friction_torque_per_degree(rpm, clt):
#     """
#     Generic Friction based on Mean Piston Speed and Displacement.
#     """
#     # 1. Mean Piston Speed (m/s)
#     # Sp = 2 * Stroke * RPM / 60
#     piston_speed_mean = (2 * c.STROKE * rpm) / 60.0
    
#     # 2. Friction Mean Effective Pressure (FMEP) in Bar
#     # Accessory drag + Hydrodynamic (bearings) + Windage (square of speed)
#     # These coefficients are standard for naturally aspirated SI engines.
#     fmep_bar = 0.35 + (0.09 * piston_speed_mean) + (0.005 * piston_speed_mean**2)
    
#     # 3. Adjust for Oil Viscosity (CLT effect)
#     # Based on your rising CLT, this factor will decrease from ~2.5 to 1.0.
#     visc_factor = max(1.0, 2.5 * np.exp(-0.025 * (clt - 20)))
#     fmep_bar *= visc_factor
    
#     # 4. Convert FMEP to Torque (Nm)
#     # Torque = (FMEP_Pa * Displacement_m3) / (4 * pi)
#     v_total_m3 = c.V_DISPLACED * c.NUM_CYL
#     t_friction_total = (fmep_bar * 1e5 * v_total_m3) / (4 * np.pi)
    
#     return t_friction_total / 720.0
#     # return t_friction_total



def calc_coolant_temperature_increment(
    brake_torque_nm: float,
    rpm: float,
    current_clt: float,
    dt_sec: float = 0.02,  # control loop period
    fraction_to_coolant: float = 0.28,
    coolant_thermal_mass_J_per_C: float = 35500.0,
) -> float:
    """
    Physics-based coolant warm-up.
    Returns dT in °C for this time step.
    """
    if brake_torque_nm <= 0:
        return 0.0

    # Total mechanical work per second (Watts)
    power_watts = brake_torque_nm * rpm * 2 * np.pi / 60.0

    # Heat to coolant (Watts)
    heat_to_coolant_watts = power_watts * fraction_to_coolant

    # Temperature rise
    dT = (heat_to_coolant_watts * dt_sec) / coolant_thermal_mass_J_per_C

    return dT


def calc_thermostat_cooling_rate(clt: float, t_ambient: float = c.T_AMBIENT) -> float:
    """
    Simple radiator + thermostat model.
    Returns cooling rate in °C per time step.
    """
    if clt <= 92.0:
        return 0.0
    # Exponential opening above 92 °C
    excess = clt - 92.0
    return 0.00015 * excess**1.8  # tuned to real data


def update_coolant_temp(current_clt, brake_torque_nm, rpm):
    dT = calc_coolant_temperature_increment(
        brake_torque_nm, rpm=rpm, current_clt=current_clt
    )
    cooling = calc_thermostat_cooling_rate(current_clt)
    return np.clip(current_clt + dT - cooling, -10, 115)

def detect_knock(peak_bar, clt, rpm, spark_advance, lambda_, fuel_octane=95.0):
    """
    Refined knock detection for higher-fidelity Wiebe physics.
    Now accounts for RPM and Fuel Octane.
    """
    # 1. Base threshold scaled for realistic Wiebe Pmax
    # A safe Pmax for a 10:1 CR N/A engine is around 100 bar.
    base_threshold = 95.0 
    
    # 2. Octane Correction (Every point of Octane is worth ~2 bar of tolerance)
    # Baseline is 95 RON. 
    octane_offset = (fuel_octane - 95.0) * 2.5
    
    # 3. RPM Sensitivity (High RPM reduces time for knock to occur)
    # Every 1000 RPM adds ~3 bar of pressure tolerance
    rpm_safety = (rpm / 1000.0) * 2.0

    # 4. Thermal & Chemistry Factors (Preserving your logic with better scaling)
    # CLT protection
    cold_protection = np.clip((90.0 - clt) * 0.5, 0, 15.0)
    
    # Rich mixture cooling (Lambda < 1.0)
    rich_safety = np.clip((1.0 - lambda_) * 30.0, 0, 10.0)
    
    # Spark Penalty (If you push way past typical MBT limits)
    advance_penalty = max(0.0, (spark_advance - 30.0) * 2.0)

    # 5. Calculate Final Threshold
    knock_threshold_bar = (
        base_threshold 
        + octane_offset 
        + rpm_safety 
        + cold_protection 
        + rich_safety 
        - advance_penalty
    )

    # 6. Result Calculation
    pressure_ratio = peak_bar / knock_threshold_bar

    if pressure_ratio > 1.0:
        knock_detected = True
        # Intensity scales exponentially (1.1 ratio = slight knock, 1.3 = engine damage)
        knock_intensity = (pressure_ratio - 1.0) * 20.0 
    else:
        knock_detected = False
        knock_intensity = 0.0
        
    return knock_detected, knock_intensity

def calc_indicated_torque_step(delta_work_J, stroke):
    """
    Calculates instantaneous torque for a single degree step.
    Ensures negative torque during the intake (pumping) stroke.
    """
    # Convert 1.0 degree to radians (based on c.THETA_DELTA)
    theta_delta_rad = c.THETA_DELTA * (np.pi / 180.0)
    torque_raw = delta_work_J / theta_delta_rad

    return torque_raw

def calc_wiebe_heat_rate(theta, theta_start, duration, total_heat_J):
    """
    Calculates the instantaneous heat release (Joules/degree) using the Wiebe function.
    """
    # a=5.0 and m=2.0 are standard for spark-ignition engines
    a = 5.0
    m = 2.0
    
    # Normalized progress through the burn (0.0 to 1.0)
    delta_theta = theta - theta_start
    
    if delta_theta < 0 or delta_theta > duration:
        # # Add this inside the active burn window logic
        # if total_heat_J > 0:
        #     print(f"DEBUG_BURN | θ:{theta:05.1f} | "
        #         f"Burn_Progress:{y:4.2f} | "
        #         f"dQ_this_deg:{delta_theta * c.THETA_DELTA:6.2f}J | "
        #         f"Total_Expected:{total_heat_J:6.2f}J")
        return 0.0
    
    # Normalized position
    y = delta_theta / duration
    
    # Wiebe Mass Fraction Burned (MFB) derivative: dXb/dtheta
    # This gives us the fraction of total heat released during THIS degree
    term1 = a * (m + 1) / duration
    term2 = y**m
    term3 = np.exp(-a * y**(m + 1))
    
    dxb_dtheta = term1 * term2 * term3
    
    return dxb_dtheta * total_heat_J


def update_cylinder_wall_temperature(
    current_clt_C, 
    cycle_Q_loss_joules, 
    rpm,
    previous_T_wall
):
    """
    Determines T_wall with thermal inertia to prevent unrealistic spikes.
    """
    if rpm < 100:
        return current_clt_C + 273.15

    cycle_time_sec = 120.0 / rpm
    heat_flux_watts = cycle_Q_loss_joules / cycle_time_sec
    
    # R_wall should be tuned. 0.002 is a better starting point for Watts.
    R_wall = 0.002 
    
    # The 'steady state' wall temp for this specific power level
    target_T_wall = (current_clt_C + 273.15) + (heat_flux_watts * R_wall)
    
    # Thermal Inertia: Alpha represents how much the wall can change per cycle.
    # 0.01 means it takes ~100 cycles to reach 63% of a temp change.
    alpha = 0.05 
    new_T_wall = (alpha * target_T_wall) + ((1 - alpha) * previous_T_wall)
    
    # Safety clamp: Wall can't be cooler than coolant or hotter than melting point
    return np.clip(new_T_wall, current_clt_C + 273.15, 600.0)

def get_burn_duration(rpm, lambda_):
    """
    Refined for VW WBX 2.1 (94mm Bore).
    Models the relationship between turbulence (RPM) and flame speed.
    """
    # Base duration in degrees for a 94mm bore at 3000 RPM
    # WBX heads are not high-swirl; 45-50 degrees is a realistic MBT duration.
    BASE_DURATION = 48.0 
    REF_RPM = 3000.0

    # 1. RPM Factor (Turbulence):
    # As RPM increases, turbulence increases, keeping the burn duration 
    # (in degrees) relatively stable, but it still widens slightly at high RPM.
    # At 1000 RPM: factor is (1000/3000)**-0.35 ≈ 1.47 (Slower burn)
    # At 6000 RPM: factor is (6000/3000)**-0.35 ≈ 0.78 (Faster burn)
    f_rpm = (max(rpm, 200) / REF_RPM)**-0.35

    # 2. Lambda Factor: 
    # Gasoline flame speed peaks around Lambda 0.9 and drops sharply when lean.
    # This quadratic penalty simulates the 'lean-stumble' of the Digifant system.
    f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

    # 3. Dynamic Clipping
    # Ensure we stay within physical limits (20 deg for instant bang, 100 deg for fire in exhaust)
    burn_duration = np.clip(BASE_DURATION * f_rpm * f_lambda, 25.0, 100.0)
    
    return burn_duration

# def get_burn_duration(rpm, lambda_):
#     # For a WBX 2.1 (Large bore, relatively slow flame path)
#     BASE_DURATION = 45.0  # Degrees at 3000 RPM
#     REF_RPM = 3000.0

#     # 1. RPM Factor: Lower RPM should result in fewer degrees.
#     # At 250 RPM, f_rpm will be approx (250/3000)**0.15 = 0.68
#     # f_rpm = (rpm / REF_RPM)**0.15   # FAILED OUT LOWER RPM AS SET A LONGER BURN DURATIAON.
#     f_rpm = (rpm / REF_RPM)**-0.2

#     # 2. Lambda Factor: Stays the same, very sensitive to lean mixtures
#     f_lambda = 1.0 + 1.5 * (lambda_ - 0.9)**2.0

#     # 3. Dynamic Clipping: Allow it to be much faster at cranking speeds
#     # A 25-30 degree burn at cranking is more realistic for a "catch"
#     burn_duration = max(25.0, min(80.0, BASE_DURATION * f_rpm * f_lambda))
    
#     return burn_duration

def update_intake_manifold_pressure(effective_tps, rpm):
    """
    Calculates MAP based on the Flow Coefficient (Cd) of the throttle
    vs the Volumetric Efficiency (VE) demand of the cylinders.
    """
    tps_fraction = np.clip(effective_tps / 100.0, 0.001, 1.0)
    
    # 1. Flow into manifold (Atmospheric pushing in)
    # Higher TPS = lower restriction
    flow_in_coef = tps_fraction * 1.5 
    
    # 2. Flow out of manifold (Pistons pulling out)
    # Scales linearly with RPM and displacement (2.1L)
    # 0.15 is a tuning constant for the 'pumping' strength of this specific engine
    flow_out_demand = (rpm / 3000.0) * 0.09
    
    # 3. The Balance
    # As flow_in_coef becomes much larger than flow_out_demand (WOT), 
    # the ratio approaches 1.0 (Atmospheric).
    map_ratio = flow_in_coef / (flow_in_coef + flow_out_demand)
    
    # 4. Clipping
    # A VW 2.1L rarely pulls below 15 kPa or goes above 101 kPa (NA)
    return c.P_ATM_PA * np.clip(map_ratio, 0.15, 1.0)

# def update_intake_manifold_pressure(effective_tps, rpm):
#     """
#     effective_tps: The sum of TPS + IACV (0 to 100)
#     """
#     # Simply convert the 0-100 command to a 0.0-1.0 fraction
#     tps_fraction = effective_tps / 100.0
    
#     # Engine suction scales with RPM
#     # We use a non-linear suction factor to model efficiency drop at high RPM
#     suction_demand = (rpm / 6000.0) * np.clip(rpm / 5000.0, 0.25, 0.4)
    
#     # If the ECU closes both valves (0.0), the MAP should drop to the floor
#     if tps_fraction <= 0:
#         return c.P_ATM_PA * 0.15 
        
#     # Pressure is the balance of Inflow (fraction) vs Outflow (suction)
#     map_ratio = tps_fraction / (tps_fraction + suction_demand)
    
#     # Return pressure in Pascals, clipped to realistic vacuum limits
#     return c.P_ATM_PA * np.clip(map_ratio, 0.08, 1.0)


def calculate_combustion_dq(cad, substep_idx, substep_size, cyl_state, lambda_):
    """
    Calculates the heat release (dQ) for a specific degree/substep.
    """
    # 1. Determine Wiebe Slice
    theta_start = cad + (substep_idx * substep_size)
    theta_next = theta_start + substep_size
    
    f1 = calc_wiebe_fraction(theta_start, cyl_state.ignition_start_theta, 
                             cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
    f2 = calc_wiebe_fraction(theta_next, cyl_state.ignition_start_theta, 
                             cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
    step_fraction = max(0.0, f2 - f1)

    # 2. Dynamic Efficiency
    # Increased to 0.97 for the lambda power window to help hit 65 bar target
    eff = 0.97 if 0.85 <= lambda_ <= 1.05 else 0.85
    
    # 3. Energy release limited by the limiting reactant
    # Proportional burn of the mass present at the time of spark
    dm_fuel_potential = cyl_state.fuel_mass_at_spark * step_fraction
    dm_air_potential  = cyl_state.air_mass_at_spark  * step_fraction

    # Clamp by remaining cylinder contents
    actual_fuel_burned = min(cyl_state.fuel_mass_kg, dm_fuel_potential)
    actual_air_burned  = min(cyl_state.air_mass_kg, dm_air_potential)

    # 4. Stoichiometric Heat Calculation (Rich vs Lean)
    if lambda_ < 1.0:
        # Rich: Limited by available Oxygen (Air)
        dq = (actual_air_burned / 14.7) * c.LHV_FUEL_GASOLINE * eff
    else:
        # Lean: Limited by available Fuel
        dq = actual_fuel_burned * c.LHV_FUEL_GASOLINE * eff

    return dq, actual_fuel_burned, actual_air_burned


def calc_specific_heat_cv(T):
    """
    Calculates temperature-dependent specific heat at constant volume (cv) 
    for air/combustion products.
    
    Based on a linear approximation of the JANAF tables for the 400K-3500K range.
    As T increases, cv increases, which naturally dampens peak temp and pressure.
    """
    # 718 J/kg.K is the standard room-temp value.
    # At 2500K, cv for air is closer to 950-1000 J/kg.K.
    # Linear approximation: cv = cv_base + slope * (T - T_ref)
    if T < 300:
        return 718.0
    
    # This slope represents the increasing energy required to vibrate/dissociate 
    # molecules at high temps.
    cv_temp = 718.0 + 0.115 * (T - 300.0)
    
    # Cap it to prevent non-physical extrapolation at extreme temps
    return min(cv_temp, 1100.0)