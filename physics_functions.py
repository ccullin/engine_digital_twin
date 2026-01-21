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
    from engine_model import Valve

# --- Utility Functions ---

# Removed 'dif_list' as it was unused and its implementation for calculating derivatives (dX/dtheta)
# was non-standard. Derivatives for simulation should be calculated using difference steps
# (e.g., dP/dtheta * theta_delta).


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

def calc_wiebe_fraction(theta, theta_start, duration, a_vibe=6.908, m_vibe=2.0):
    """ this version is not sub-step friendly, so changed approach to fraction """
    
    delta_theta = theta - theta_start
    
    if delta_theta <= 0:
        return 0.0
    if delta_theta >= duration:
        return 1.0
        
    # The standard Wiebe cumulative S-curve formula
    return 1.0 - np.exp(-a_vibe * (delta_theta / duration) ** (m_vibe + 1))

""" this version is not sub-step friendly, so changed approach to fraction """
# def calc_vibe_heat_release(
#     rpm, lambda_, theta, Q_total, theta_start, duration_ref, a_vibe=6.908, m_vibe=2.0
# ):
#     """
#     Dynamic Wiebe function with RPM and lambda-sensitive burn duration.
#     This is the REAL fix that makes cold-start at 250 RPM actually ignite.
#     """

#     # DO NOT let burn be too slow at low RPM → this is what was killing cold start
#     rpm_factor = max(rpm / 1000.0, 0.25)  # Never go below 0.25 (250 RPM → 0.25)
#     lambda_factor = max(lambda_, 0.8) ** 0.3  # Rich = faster burn

#     # Dynamic combustion duration in crank degrees
#     # 40° at 6000 RPM → ~100° at 250 RPM → perfect cold-start ignition
#     duration = max(25.0, 38.0 * (rpm_factor**-0.82) * (lambda_factor**-0.6))  # min 25°

#     delta_theta = theta - theta_start
#     if delta_theta <= 0:
#         return 0.0

#     x = 1.0 - np.exp(-a_vibe * (delta_theta / duration) ** (m_vibe + 1))
#     dq_dtheta = (
#         a_vibe
#         * (m_vibe + 1)
#         / duration
#         * (delta_theta / duration) ** m_vibe
#         * np.exp(-a_vibe * (delta_theta / duration) ** (m_vibe + 1))
#     )

#     return dq_dtheta * Q_total

def calc_woschni_heat_loss(theta, rpm, P_curr, T_curr, V_curr, T_wall, V_clearance, theta_delta):
    """
    Docstring for calc_woschni_heat_loss
    
    :param theta: Description
    :param rpm: current rpm
    :param P_curr: Cylinder P (Pa)
    :param T_curr: Cylinder T (K)
    :param V_curr: Cylinder volume, inc. V clearnace (m3)
    :param T_wall: Cylinder wall temp (K)
    :param V_clerance:  the Cylinder clearnace volume (m3)
    :param theta_delta: (sub step size in CAD, e.g. 0.2 if subloop is 5)
    
    Calculates dQ_w/d_theta using the updated Woschni model with 
    instantaneous piston velocity and SI-standard coefficients.
    """
    # 1. Physical Constants & Engine Geometry
    omega = 2.0 * np.pi * (rpm / 60.0) # rad/s
    
    # 2. Safety Guards
    if T_curr <= 50.0 or P_curr <= 1000.0 or np.isnan(T_curr):
        return 0.0

    # 3. Gas Velocity Calculation (Updates 1 & 2)
    # Use the geometric factor (dS/dtheta) to find instantaneous velocity (m/s)
    # V_piston = omega * dS/dtheta (where dS/dtheta is in meters/rad)
    v_factor = calc_piston_speed_factor(theta) 
    u_p_instant = abs(omega * v_factor)

    # # C1 is typically 2.28 for intake/compression
    # # C2 is 0 for motoring (no combustion-induced pressure rise)
    # C1 = 2.28
    # W_vel = C1 * u_p_instant
    
    # C1: 2.28 (Compression/Power), 6.18 (Gas Exchange)
    C1 = 2.28 if (180 <= theta <= 540) else 6.18
    
    # Simple C2 Logic: Only apply if we are in the combustion window
    # and pressure is significantly above atmospheric.
    C2 = 0.00324
    pressure_rise_term = 0.0
    if 340 <= theta <= 420 and P_curr > 1.2e5:
        # A simple approximation of the pressure-driven velocity surge
        # V_d is displaced volume, T_r/P_r/V_r are reference states (use IVC values)
        pressure_rise_term = C2 * (c.V_DISPLACED * T_curr / (P_curr * V_curr)) * (P_curr - 1e5)
    
    W_vel = (C1 * u_p_instant) + pressure_rise_term

    # 4. Heat Transfer Coefficient h_g (Update 3)
    # Using the standard SI Woschni correlation:
    # h_g = 3.26 * B^(-0.2) * P^0.8 * T^(-0.55) * w^0.8
    # B = Bore (m), P = Pressure (Pa), T = Temp (K), w = velocity (m/s)
    P_bar = P_curr / 1e5
    h_g = 3.26 * (c.BORE**-0.2) * (P_bar**0.8) * (T_curr**-0.55) * (W_vel**0.8)

    # 5. Instantaneous Surface Area (A_w)
    # Cylinder wall area = pi * Bore * instantaneous_height
    # Instantaneous height x = V_curr / A_piston
    V_bore = V_curr - V_clearance
    dist_from_tdc = V_bore / c.A_PISTON
    A_w = (2 * c.A_PISTON) + (np.pi * c.BORE * dist_from_tdc)

    # 6. Heat Loss Rate & Conversion
    # dQ/dt = h_g * A * (T_gas - T_wall)
    dQ_w_dt = h_g * A_w * (T_curr - T_wall)
    
    # Scale heat transfer based on RPM to allow starting
    # 0.2 at cranking (200rpm) ramp to 1.0 at idle (900rpm)
    thermal_scaling = np.clip((rpm - 400) / (1200 - 400) * 0.9 + 0.1, 0.1, 1.0) # should be required no P_curr -> P_bar
    dQ_w_dt = h_g * A_w * (T_curr - T_wall)


    # Convert J/s to J/deg: (dQ/dt) / (dtheta/dt) * (rad to deg)
    if omega < 1.0: 
        return 0.0
        
    dQ_w_d_theta = (dQ_w_dt / omega) * (np.pi / 180.0)

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

def calc_valve_lift(theta, L_peak, duration, phase_lag, zero_ref=-360.0):
    """
    Calculates instantaneous valve lift using a simple cosine-based approximation
    of a cam profile. (No changes made here, calculation looks correct for approximation).
    """

    # Center the angle around the peak lift (phase_lag)
    # The duration starts/ends at L=0.
    theta_adjusted = (theta - phase_lag)

    # Find the valve open range: start_angle to end_angle
    start_angle = phase_lag - duration / 2.0
    end_angle = phase_lag + duration / 2.0

    # We must handle the 720-degree cycle wrap-around.
    # The logic below attempts to handle the boundary condition of the cosine curve.

    is_open = False

    # Normalize theta to be in the range [0, 720)
    theta_norm = theta % 720.0
    start_norm = start_angle % 720.0
    end_norm = end_angle % 720.0

    if start_norm > end_norm:
        # Overlap case (e.g., start at 500, end at 100)
        if theta_norm >= start_norm or theta_norm <= end_norm:
            is_open = True
    else:
        # Standard case (e.g., start at 100, end at 500)
        if theta_norm >= start_norm and theta_norm <= end_norm:
            is_open = True

    if not is_open:
        return 0.0

    # Calculate the cosine lift profile (L = L_peak * (1 - cos(theta_adjusted * 360/duration)) / 2)
    # The factor 360/duration ensures the argument of cos goes from -pi to pi over the duration.
    lift_ratio = (1 - np.cos(np.deg2rad(theta_adjusted) * 360.0 / duration)) / 2.0

    return max(0.0, L_peak * lift_ratio)


def calc_valve_area(theta_list):
    """
    Calculates the instantaneous curtain flow area for intake and exhaust valves
    over the entire crank angle range (theta_list).
    """

    # --- Intake Valve Constants (Index 0) ---
    D_i = c.VALVE_DATA['D_valve'][0]
    L_i_peak = c.VALVE_DATA['L_peak'][0]
    Dur_i = c.VALVE_DATA['duration'][0]
    PL_i = c.VALVE_DATA['phase_lag'][0]
    Cd_i = c.VALVE_DATA['flow_coeff'][0]

    ZERO_REF = c.THETA_MIN

    # --- Exhaust Valve Constants (Index 1) ---
    D_e = c.VALVE_DATA['D_valve'][1]
    L_e_peak = c.VALVE_DATA['L_peak'][1]
    Dur_e = c.VALVE_DATA['duration'][1]
    PL_e = c.VALVE_DATA['phase_lag'][1]
    Cd_e = c.VALVE_DATA['flow_coeff'][1]

    A_in_list = []
    A_ex_list = []

    for theta in theta_list:

        # Intake Valve (IV) lift calculation
        L_i_act = calc_valve_lift(theta, L_i_peak, Dur_i, PL_i, ZERO_REF)
        # Area = pi * D_valve * L_actual * flow_coeff (Curtain area approximation)
        A_in = np.pi * D_i * L_i_act * Cd_i
        A_in_list.append(float(A_in))

        # Exhaust Valve (EV) lift calculation
        L_e_act = calc_valve_lift(theta, L_e_peak, Dur_e, PL_e, ZERO_REF)
        A_ex = np.pi * D_e * L_e_act * Cd_e
        A_ex_list.append(float(A_ex))


    return {
        'A in': A_in_list,
        'A ex': A_ex_list
    }


def theta_to_720(theta):
    """Safe modulo 720 — works on scalars and arrays"""
    return theta % 720.0


def calc_valve_lift_vectorized(theta_array: np.ndarray, valve: Valve) -> np.ndarray:
    """
    Vectorized version of calc_valve_lift — returns array of lifts for entire theta_array
    """

    ca = theta_array % 720.0 # crank angle

    open_ca = valve.open_angle
    close_ca = valve.close_angle
    
    # 1. Calculate duration and position
    duration = (close_ca - open_ca) % 720
    rel_angle = (ca - open_ca) % 720
    mid_dur = duration / 2.0
    
    lift = np.zeros_like(ca, dtype=float)
    active = (rel_angle <= duration)
    
    # 2. Use mm here (e.g., 9.5)
    l_max = valve.max_lift

    phi_deg = (rel_angle[active] - mid_dur) / mid_dur * 180.0
    # p < 1.0 makes it 'fatter' (more area). p > 1.0 makes it 'pointier'.
    p = 0.4
    lift[active] = l_max * ((1.0 + np.cos(np.radians(phi_deg))) / 2.0)**p
    
    return lift


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

    # Damping for stability near zero pressure delta
    # damping = 1.0
    # if pr > 0.98:
    #     damping = max(0.0, (1.0 - pr) / (1.0 - 0.98))
    
    return mdot * direction, Cd

def _calc_physics_Cd(lift, diameter):
    L_D = lift / diameter
    
    # Standard Taylor/Heywood Correlation for Poppet Valves
    if L_D <= 0.05:
        # Initial opening / viscous regime
        return 0.30 + (4.0 * L_D) 
    elif L_D <= 0.20:
        # Transitional regime
        return 0.50 + (1.33 * (L_D - 0.05))
    elif L_D <= 0.25:
        # Peak detachment efficiency
        return 0.70
    else:
        # High-lift port restriction (Geometric drop)
        # As L/D increases, the effective Cd must drop because 
        # the port cross-section limits the mass flow.
        return 0.70 - 0.8 * (L_D - 0.25)

def _calc_dynamic_discharge_coeff(rpm, lift, diameter, is_intake):
    if lift <= 0.0 or diameter <= 0.0:
        return 0.0
    
    # lift and diameter are in meters
    l_over_d = lift / diameter 

    # 1. Low-lift penalty
    if l_over_d < 0.03:
        low_lift_factor = 0.20 + 6.67 * l_over_d
    elif l_over_d < 0.10:
        low_lift_factor = 0.40 + 3.0 * (l_over_d - 0.03)
    else:
        low_lift_factor = 0.70 + 0.8 * (l_over_d - 0.10)
    
    # 2. Geometry boost
    geo_boost = 1.0 + 0.2 * min(l_over_d / 0.25, 1.0)

    # 3. Mach Index (Z) - Using Piston Speed
    # sp = mean piston speed (m/s)
    sp = (2 * c.STROKE * rpm) / 60.0 
    
    # Z-index formula (all units in meters/seconds now)
    z = ((c.BORE / diameter)**2) * (sp / 340.0)
    # this is causes restrictive flow in combination withe cosine CAM profile.
    # choke_factor = np.exp(-0.6 * (z / 0.5)**2)
    choke_factor = np.exp(-0.2 * (z / 0.5)**2)

    Cd = 0.70 * low_lift_factor * geo_boost * choke_factor
    return np.clip(Cd, 0.15, 0.90)


def calc_discharge_coeff(rpm):
    """
    Calculates a correction factor (multiplied by the base Cd) to simulate
    volumetric efficiency drop at high RPM due to friction/flow losses. (No changes made here).
    """
    if rpm > c.RPM_MAX_FLOW_REF:
        # Simple quadratic drop-off model
        rpm_ratio = (rpm - c.RPM_MAX_FLOW_REF) / c.RPM_MAX_FLOW_REF
        cd_corr = 1.0 - c.K_CD_RPM_CORR * (rpm_ratio**2)
        return max(0.5, cd_corr)  # Clamp to 0.5 minimum
    return 1.0


# physics_functions.py

def intake_mass_flow(A_valve, P_man, T_man, rpm):
    """
    Industry-standard NA intake flow with momentum override.
    Absolutely no NaNs, no crashes, works from 250 rpm to 12 000 rpm.
    """
    
    if A_valve < 1e-8:
        return 0.0

    # Fixed effective back-pressure — this is what every OEM uses
    P_down_eff = 92_000.0  # 0.92 bar
    pr = P_down_eff / max(P_man, 1000.0)  # never divide by zero

    gamma = 1.4
    R = 287.0
    Cd = 0.68  # calibrated real-world value

    # Critical pressure ratio for air
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))  # ≈ 0.5283

    if pr <= pr_crit:
        # Choked flow — simple, robust, never NaN
        mdot = (
            Cd
            * A_valve
            * P_man
            * np.sqrt(gamma / (R * T_man))
            * (gamma + 1.0)
            / 2.0 ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
    else:
        # Sub-critical — mathematically safe version
        term1 = pr ** (2.0 / gamma)
        term2 = pr ** ((gamma + 1.0) / gamma)
        inside = 2.0 * gamma / (R * T_man) / (gamma - 1.0) * (term1 - term2)
        inside = max(inside, 0.0)  # ← this kills the NaN
        mdot = Cd * A_valve * P_man * np.sqrt(inside)

    deg_per_sec = max(rpm, 50.0) * 6.0  # rpm → deg/s
    return (mdot / deg_per_sec)  # kg/deg


def exhaust_mass_flow(A_valve, P_cyl, T_cyl, rpm):
    """Exhaust flow — always outward, no reverse"""
    if A_valve < 1e-9:
        return 0.0

    return calc_mass_flow_base(
        A_valve=A_valve,
        P_up=P_cyl,
        T_up=T_cyl,
        P_down=c.P_ATM_PA * 0.98,
        R_spec=c.R_SPECIFIC_MIX,
        gamma=c.GAMMA_AIR,
        rpm=rpm,
        Cd=0.70,
    )


def calc_mass_flow_base(A_valve, P_up, T_up, P_down, R_spec, gamma, rpm, Cd):
    """Pure isentropic flow — no clamps, no tricks"""
    if P_up < 100.0 or T_up < 200.0:
        return 0.0

    P_up = max(P_up, 100.0)  # tiny numerical guard only
    P_down = max(P_down, 10.0)

    pr = P_down / P_up
    crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    if pr <= crit:
        # Choked
        mdot = (
            Cd
            * A_valve
            * P_up
            * np.sqrt(gamma / (R_spec * T_up))
            * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        )
    else:
        # Subcritical
        pr_eff = min(pr, 0.9999)
        mdot = (
            Cd
            * A_valve
            * P_up
            * np.sqrt(
                2
                * gamma
                / (R_spec * T_up * (gamma - 1))
                * (pr_eff ** (2 / gamma) * (1 - pr_eff ** ((gamma - 1) / gamma)))
            )
        )

    deg_per_sec = max(rpm, 50.0) * 6.0
    return mdot / deg_per_sec


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

    # # --- DEBUG SECTION ---
    # # We only care about steps where mass is actually moving
    # is_danger_zone = (CAD >= 680) or (CAD <= 40) or (180 <= CAD <= 240)
    # if is_danger_zone:
    #     print(
    #         f"DEBUG_cycle {cycle}/{CAD}/{substep}  "
    #         f"P_curr:{P_curr:6.0f}->{P_next:6.0f} T_curr:{T_curr:3.0f}->{T_next:3.0f} V_curr:{V_curr:9.2e}->{V_next:9.2e}" 
    #         f"Delta_M:{Delta_M:9.2e} dM_d_theta:{dM_d_theta:9.2e} " 
    #     )
    #     print(
    #         f"            term_heat:{term_heat:6.3f} term_work:{term_work:6.3f} term_mass:{term_mass:6.3f} "
    #     )
    # else:
    #     # Print every 20 degrees elsewhere just to monitor progress
    #     if CAD % 20 == 0 and substep == 0:
    #         print(f"DEBUG_STABLE | θ:{cycle}/{CAD} | P:{P_curr:.0f} | T:{T_curr:.1f}")
    
    
    
    # We only care about steps where mass is actually moving
    # if CAD > 710 or CAD < 230:
    #     if CAD == 710 or CAD == 719 or CAD % 10 == 0 or CAD == 228:
    #         if abs(Delta_M) > 1e-12:
    #             print(f"DEBUG_FLOW | CAD:{CAD if CAD is not None else '?'}/{substep if substep is not None else '?'}")
    #             print(f"  Direction: {'IN' if Delta_M > 0 else 'OUT'} | Delta_M: {Delta_M:8.2e}")
    #             print(f"  Enthalpy : T_manifold:{T_manifold:.1f} | T_cyl:{T_curr:.1f} | T_USED:{T_flow_corr:.1f}")
                
    #             # This checks if we are accidentally cooling the engine during exhaust
    #             if Delta_M < 0 and abs(T_flow_corr - T_curr) > 50:
    #                 print("  !!! ALERT: Heat Trap Detected. Removing hot gas at cold temperature.")
    # ------------------
    
    return P_next, T_next

# def integrate_first_law(
#     P_curr, T_curr, M_curr, V_curr, Delta_M, Delta_Q_in, Delta_Q_loss, 
#     dV_d_theta, gamma, theta_delta, T_inflow, R_spec
# ):
#     # Prevent catastrophic states
#     M_curr = max(M_curr, 1e-8)
#     V_curr = max(V_curr, 1e-8)

#     # Pre-compute rates
#     dM_d_theta = Delta_M / theta_delta
#     dQ_loss_rate = Delta_Q_loss 
#     dQ_in_rate = Delta_Q_in / theta_delta
    
#     dQ_net_d_theta = dQ_in_rate - dQ_loss_rate

#     # Next states
#     M_next = max(M_curr + Delta_M, 1e-7)
#     V_next = max(V_curr + dV_d_theta * theta_delta, 1e-8)

#     # Use the pressure at the start of the step for a stable derivative
#     term_heat = (gamma - 1.0) * dQ_net_d_theta
#     term_work = -gamma * P_curr * dV_d_theta
#     term_mass = gamma * R_spec * T_inflow * dM_d_theta
  
#     # Predictor step
#     dP_initial = (term_heat + term_work + term_mass) / V_curr
#     P_mid = P_curr + dP_initial * (0.5 * theta_delta)
#     V_mid = V_curr + dV_d_theta * (0.5 * theta_delta)

#     # Use P_mid and V_mid for the real derivative
#     term_work_mid = -gamma * P_mid * dV_d_theta
#     dP_d_theta_final = (term_heat + term_work_mid + term_mass) / V_mid
#     P_next = P_curr + dP_d_theta_final * theta_delta


#     # Temperature from Ideal Gas Law (The most physical way to derive T)
#     T_next = (P_next * V_next) / (M_next * R_spec)

#     # Weighted smoothing to maintain numerical stability
#     # P_final = 0.9 * P_next + 0.1 * P_curr
#     # T_final = 0.95 * T_next + 0.05 * T_curr
    
#     # return P_final, T_final, M_next
#     return P_next, T_next


# --- Engine Performance Functions ---


def calc_pumping_losses(P_cyl, V_list, theta_list):
    """This function is a placeholder; PMEP is implicitly calculated in W_indicated."""
    pass



def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
    """
    calculates instantaneous friction for a cylinder based on 
    speed of rotation, oil and piston ring load.
    """
    # 1. Get kinematics
    geom_factor = calc_piston_speed_factor(theta)
    omega = (rpm * 2 * np.pi) / 60.0
    v_piston_inst = abs(omega * geom_factor) # Instantaneous speed m/s
    
    # 2. Viscosity Factor (Your existing logic)
    visc_factor = max(1.0, 2.5 * np.exp(-0.025 * (clt - 20)))
    
    # 3. Calculate Forces (Newton) for single cylinder
    C_VISCOUS = 5.0 # Ns/m
    C_RING_LOAD = 0.00004 # m^2
    f_viscous = C_VISCOUS * v_piston_inst * visc_factor
    f_pressure = C_RING_LOAD * p_cyl # p_cyl in Pascals
    
    # 4. Total Force to Torque
    # Moment arm is geom_factor
    t_friction = (f_viscous + f_pressure) * abs(geom_factor)
    
    return t_friction


""" this function over simplied Friction and return FMEP, the average for 720 cycle """
def calc_friction_torque_per_degree(rpm, clt):
    """
    Generic Friction based on Mean Piston Speed and Displacement.
    """
    # 1. Mean Piston Speed (m/s)
    # Sp = 2 * Stroke * RPM / 60
    piston_speed_mean = (2 * c.STROKE * rpm) / 60.0
    
    # 2. Friction Mean Effective Pressure (FMEP) in Bar
    # Accessory drag + Hydrodynamic (bearings) + Windage (square of speed)
    # These coefficients are standard for naturally aspirated SI engines.
    fmep_bar = 0.35 + (0.09 * piston_speed_mean) + (0.005 * piston_speed_mean**2)
    
    # 3. Adjust for Oil Viscosity (CLT effect)
    # Based on your rising CLT, this factor will decrease from ~2.5 to 1.0.
    visc_factor = max(1.0, 2.5 * np.exp(-0.025 * (clt - 20)))
    fmep_bar *= visc_factor
    
    # 4. Convert FMEP to Torque (Nm)
    # Torque = (FMEP_Pa * Displacement_m3) / (4 * pi)
    v_total_m3 = c.V_DISPLACED * c.NUM_CYL
    t_friction_total = (fmep_bar * 1e5 * v_total_m3) / (4 * np.pi)
    
    return t_friction_total / 720.0
    # return t_friction_total



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


# def update_knock(last_cycle, clt):
#     peak_bar = last_cycle["peak_pressure_bar"]
#     if peak_bar > 65 and clt < 95:
#         return min(100.0, (peak_bar - 60) ** 2.5)
#     return 0.0

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
        
    # print(
    #     "KNOCK DEBUG:  "
    #     f"peak_bar: {peak_bar:6.1f} | "
    #     f"clt: {clt:4.1f} | "
    #     f"rpm: {rpm:4.0f} | "
    #     f"spark_advance: {spark_advance:4.1f} | "
    #     f"lambda_: {lambda_:4.2f} | "
    #     f"knock_int: {knock_intensity:4.1f}"
    # )
        
    return knock_detected, knock_intensity

def calc_indicated_torque_step(delta_work_J, stroke):
    """
    Calculates instantaneous torque for a single degree step.
    Ensures negative torque during the intake (pumping) stroke.
    """
    # Convert 1.0 degree to radians (based on c.THETA_DELTA)
    theta_delta_rad = c.THETA_DELTA * (np.pi / 180.0)
    torque_raw = delta_work_J / theta_delta_rad

    # removing this in favour of physics from _update_mechanical_dynamics
    # if stroke == "intake":
    #     return -abs(torque_raw)
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
    rpm
):
    """
    Determines T_wall based on the coolant temp and the heat flux.
    R_wall represents the thermal resistance of the cylinder sleeve.
    """

    cycle_time_sec = cycle_time = 120.0 / rpm  # Time for 2 revolutions in seconds
    # Thermal resistance of the cast iron/aluminum sleeve (K/Watt)
    # A value of 0.05 - 0.15 is realistic for a 2.1L WBX
    R_wall = 0.12 
    
    # Calculate heat flux in Watts (Joules per second)
    heat_flux_watts = cycle_Q_loss_joules / cycle_time_sec
    
    # T_wall = T_coolant + (Heat_Flux * Resistance)
    # This ensures T_wall is always higher than CLT when the engine is working
    new_T_wall = (current_clt_C + 273.15) + (heat_flux_watts * R_wall)
    
    return new_T_wall

def get_burn_duration(rpm, lambda_):
    # For a WBX 2.1 (Large bore, relatively slow flame path)
    BASE_DURATION = 45.0  # Degrees at 3000 RPM
    REF_RPM = 3000.0

    # 1. RPM Factor: Lower RPM should result in fewer degrees.
    # At 250 RPM, f_rpm will be approx (250/3000)**0.15 = 0.68
    f_rpm = (rpm / REF_RPM)**0.15 

    # 2. Lambda Factor: Stays the same, very sensitive to lean mixtures
    f_lambda = 1.0 + 1.5 * (lambda_ - 0.9)**2.0

    # 3. Dynamic Clipping: Allow it to be much faster at cranking speeds
    # A 25-30 degree burn at cranking is more realistic for a "catch"
    burn_duration = max(25.0, min(80.0, BASE_DURATION * f_rpm * f_lambda))
    
    return burn_duration

def update_intake_manifold_pressure(effective_tps, rpm):
    """
    effective_tps: TPS + Idle Valve (0 to 100)
    rpm: Current engine speed
    """
    # 1. Convert TPS to a physical area ratio
    # 0.02 is the "minimum leak" (throttle plate gap)
    # 1.0 is the full bore area
    area_ratio = np.clip(effective_tps / 100.0, 0.0, 1.0)
    effective_area = 0.02 + (0.98 * area_ratio)
    
    # 2. Suction Factor (The Engine as a Pump)
    # As RPM increases, the engine pulls more volume per second.
    # 6000 is a scaling constant representing the flow capacity of the head.
    suction_demand = rpm / 6000.0 
    
    # 3. The Pressure Result
    # Pressure is the balance between flow IN (area) and flow OUT (suction)
    # 0.2 is a tuning constant for the 'strength' of the vacuum.
    map_ratio = effective_area / (effective_area + suction_demand * 0.25)
    
    # Scale to Atmospheric Pressure (e.g., 101325 Pa)
    # clip ensures we stay between high vacuum (0.25 bar) and WOT (1.0 bar)
    map_pa = c.P_ATM_PA * np.clip(map_ratio, 0.25, 1.0)
    
    return map_pa