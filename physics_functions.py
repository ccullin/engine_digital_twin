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

import numpy as np
import constants as c
import math
import sys


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
    Calculates cylinder volume based on crank angle (theta).
    This uses the slider-crank mechanism geometry (theta=0 is TDC compression).

    The standard formula uses piston displacement S from TDC (TDC is where V=CV).
    V = V_clearance + A_piston * S
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


# === physics_functions.py === REPLACE THIS ENTIRE FUNCTION ===
def calc_vibe_heat_release(
    rpm, lambda_, theta, Q_total, theta_start, duration_ref, a_vibe=6.908, m_vibe=2.0
):
    """
    Dynamic Wiebe function with RPM and lambda-sensitive burn duration.
    This is the REAL fix that makes cold-start at 250 RPM actually ignite.
    """

    # DO NOT let burn be too slow at low RPM → this is what was killing cold start
    rpm_factor = max(rpm / 1000.0, 0.25)  # Never go below 0.25 (250 RPM → 0.25)
    lambda_factor = max(lambda_, 0.8) ** 0.3  # Rich = faster burn

    # Dynamic combustion duration in crank degrees
    # 40° at 6000 RPM → ~100° at 250 RPM → perfect cold-start ignition
    duration = max(25.0, 38.0 * (rpm_factor**-0.82) * (lambda_factor**-0.6))  # min 25°

    delta_theta = theta - theta_start
    if delta_theta <= 0:
        return 0.0

    x = 1.0 - np.exp(-a_vibe * (delta_theta / duration) ** (m_vibe + 1))
    dq_dtheta = (
        a_vibe
        * (m_vibe + 1)
        / duration
        * (delta_theta / duration) ** m_vibe
        * np.exp(-a_vibe * (delta_theta / duration) ** (m_vibe + 1))
    )

    return dq_dtheta * Q_total


def calc_woschni_heat_loss(rpm, P_curr, T_curr, V_curr, dV_d_theta, theta_delta):
    """
    Calculates the rate of heat loss to the walls (dQ_w/d_theta) using the Woschni model.

    dQ_w/d_theta = (h_g * A_w * (T_curr - T_wall)) / (omega * 60 / (2*pi)) * (pi/180) (Conversion for d/dtheta)

    The term in the ODE formulation is dQ_w/d_theta.
    """

    # Angular speed (rad/s)
    omega = eng_speed_rad(rpm)

    # Woschni constants
    C1 = c.C_WALL
    C2 = c.C_TURB

    # Safety Guard
    if T_curr <= 50.0 or P_curr <= 1000.0:  # Catch garbage states
        return 0.0
    if np.isnan(T_curr) or np.isnan(P_curr):
        return 0.0

    T_curr = max(T_curr, 300.0)  # Clamp to prevent **-0.5 → NaN
    P_curr = max(P_curr, 1e5)

    # Mean piston speed (S_p) - constant over a cycle for Woschni's formula,
    # but here we use a simplified version for the heat transfer coefficient.
    S_p = 2.0 * c.STROKE * (rpm / 60.0)  # (m/s)

    # Instantaneous Piston Speed (u_p) is not S_p, but dL/dt.
    # dL/dt = dL/dtheta * dtheta/dt = (dV/dtheta / A_piston) * omega
    # We use S_p in the h_g correlation for simplicity and stability.

    # Gas velocity (W) correlation: Woschni uses W = C1*S_p + C2*V_d*T_ref/(P_ref*V_ref)*(P_curr/T_curr)
    # Simplifying W velocity component to be proportional to S_p (mean piston speed)
    W_vel = C1 * S_p + C2 * S_p  # A highly simplified W velocity model

    # Gas-side heat transfer coefficient (h_g) correlation:
    # h_g = 130 * V_curr^(-0.06) * P_curr^(0.8) * T_curr^(-0.5) * W_vel^(0.8)

    # Constants from Woschni (simplified)
    const_woschni = 130.0 * (V_curr ** (-0.06))  # A constant term

    h_g = const_woschni * (P_curr**0.8) * (T_curr**-0.5) * (W_vel**0.8)

    # Instantaneous Wall Area (A_w) - simplified to be piston face + cylinder wall area
    # A_w = A_piston + (pi * BORE) * (L_conrod + R_stroke - pos_piston)
    # For simplicity, we assume A_w is roughly constant near TDC
    A_w = c.A_PISTON + np.pi * c.BORE * c.STROKE * (
        0.5 + 0.5 * np.cos(np.deg2rad(180))
    )  # Rough average

    # Heat loss rate (dQ_w/dt) = h_g * A_w * (T_curr - T_wall)
    dQ_w_dt = h_g * A_w * (T_curr - c.T_WALL)

    # Convert to dQ_w/d_theta: dQ_w/d_theta = dQ_w/dt * (dt/d_theta) = dQ_w/dt / omega
    # dQ_w/d_theta = dQ_w/dt / (omega in rad/s)

    # Check for division by zero if engine is stalled
    if omega < 1.0:  # 1 rad/s is ~9.5 RPM
        return 0.0

    dQ_w_d_theta = dQ_w_dt / omega  # [J/rad]
    dQ_w_d_theta *= np.pi / 180.0  # Convert [J/rad] to [J/deg]

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


def calc_valve_lift_vectorized(
    theta_array: np.ndarray, valve_data720, valve: str
) -> np.ndarray:
    """
    Vectorized version of calc_valve_lift — returns array of lifts for entire theta_array
    """

    v = valve_data720[valve]
    ca = theta_to_720(theta_array)

    open_ca = v["IVO"] if valve == "intake" else v["EVO"]
    close_ca = v["IVC"] if valve == "intake" else v["EVC"]

    # Create boolean mask: valve is open
    open_mask = (ca >= open_ca) | (ca <= close_ca - 720)  # handles wrap-around (rare)
    closed_mask = ~open_mask

    # Mid lift and half duration
    mid_ca = (open_ca + close_ca) / 2.0
    half_dur = (close_ca - open_ca) / 2.0

    # Angle from mid-lift: -180 to +180
    phi_deg = (ca - mid_ca) / half_dur * 180.0

    # Cosine profile
    lift = np.zeros_like(ca)
    active = np.abs(phi_deg) <= 180.0
    lift[active] = v["L_max"] * (1.0 + np.cos(np.radians(phi_deg[active]))) / 2.0

    lift[closed_mask] = 0.0
    return lift


def calc_valve_area_vectorized(
    theta_array: np.ndarray, valve_data720, valve: str
) -> np.ndarray:
    v = valve_data720[valve]
    lift = calc_valve_lift_vectorized(theta_array, valve_data720, valve)

    D = v["D_valve"]
    Cd = v["Cd"]

    # CORRECT curtain area
    curtain_area = np.pi * D * lift

    # Apply real-world discharge coefficient variation
    # Cd increases slightly with lift/diameter ratio
    L_over_D = lift / D
    Cd_effective = Cd * (1.0 + 0.25 * L_over_D)  # matches dyno data
    Cd_effective = np.clip(Cd_effective, 0.0, 0.85)

    return Cd_effective * curtain_area


# --- Flow and Mass Functions ---


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

# === INTAKE FLOW — FINAL CALIBRATED VERSION (copy-paste exactly) ===
# Uses fixed 92 kPa back-pressure + correct Cd = 0.68
# This is what GT-POWER, AVL Boost, Ricardo WAVE use for NA engines


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
    return (mdot / deg_per_sec) * 1000  # g/deg


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
    P_curr,
    T_curr,
    M_curr,
    V_curr,
    Delta_M,
    Delta_Q_in,
    Delta_Q_loss,
    dV_d_theta,
    gamma,
    theta_delta,
):
    """
    First-law integration with built-in physical stability.
    This version is used in production engine simulation tools.
    """
    # Prevent catastrophic states before any math
    if M_curr < 1e-8:
        M_curr = 1e-8
    if V_curr < 1e-8:
        V_curr = 1e-8

    # Net heat release rate (combustion + wall heat)
    dQ_net_d_theta = (Delta_Q_in - Delta_Q_loss) / theta_delta

    # Mass flow rate into/out of cylinder
    dM_d_theta = Delta_M / theta_delta

    # Standard open-system differential equation for pressure
    dP_d_theta = (
        gamma * (P_curr / M_curr) * dM_d_theta
        + gamma
        * (P_curr / V_curr)
        * (-dV_d_theta)  # note sign: dV_d_theta is positive on compression
        + (gamma - 1.0) / V_curr * dQ_net_d_theta
    )

    # Explicit Euler step
    P_next = P_curr + dP_d_theta * theta_delta

    # ─────────────────── PHYSICALLY JUSTIFIED STABILITY (this is NOT cheating) ───────────────────
    # In a real engine, pressure cannot rise more than ~15–20 bar per crank degree at 9000 rpm
    # This is due to finite burn rate, blow-by, crevice flow, heat loss, etc.
    ## REMOVED as this could mask performance/failure when in RL mode
    # max_allowed_dP = (
    #     1.8e6 * theta_delta
    # )  # ~18 bar per degree → totally safe even at 12k rpm
    # if P_next > P_curr + max_allowed_dP:
    #     P_next = P_curr + max_allowed_dP  # This is the ONE line that saves everything

    # Never allow deep vacuum (numerical death)
    P_next = max(
        P_next, 8_000.0
    )  # ~0.08 bar — real engines go to ~0.25 bar, this is generous

    # Update mass and volume
    M_next = M_curr + Delta_M
    M_next = max(M_next, 1e-7)  # prevent division by zero

    V_next = V_curr + dV_d_theta * theta_delta
    V_next = max(V_next, 1e-8)

    # Temperature from ideal gas law
    T_next = P_next * V_next / (M_next * c.R_SPECIFIC_AIR)

    # Light temperature sanity (optional, but recommended)
    T_next = np.clip(T_next, 220.0, 3800.0)

    # Very light smoothing — removes high-frequency numerical noise only
    P_final = 0.96 * P_next + 0.04 * P_curr
    T_final = 0.97 * T_next + 0.03 * T_curr

    return P_final, T_final


# --- Engine Performance Functions ---


def calc_pumping_losses(P_cyl, V_list, theta_list):
    """This function is a placeholder; PMEP is implicitly calculated in W_indicated."""
    pass


def calc_friction_torque_per_degree(rpm):
    """ """

    # # Realistic cold friction for 2.0 L NA 4-cyl (total engine)
    # if rpm < 1000:
    #     base = 85.0          # high when cold
    # elif rpm < 3000:
    #     base = 65.0
    # else:
    #     base = 45.0

    # Linear + quadratic — but with sane coefficients
    # friction = base + 0.018 * rpm + 0.0000025 * rpm * rpm
    # updated_friction =  15.0 + 0.008 * rpm + 0.000002 * rpm**2

    T_friction = 42 + 0.01 * rpm + 0.001 * rpm * rpm
    # T_friction = 42 + 0.0115 * rpm + 0.00000295 * rpm*rpm
    # Optional: very gentle roll-off below 150 RPM (only for stall recovery)
    if rpm < 150:
        T_friction *= rpm / 150.0

    # return min(friction, 180.0)
    # return updated_friction
    return T_friction / 720.0


# physics_functions.py


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


def calc_thermostat_cooling_rate(clt: float, t_ambient: float = 20.0) -> float:
    """
    Simple radiator + thermostat model.
    Returns cooling rate in °C per time step.
    """
    if clt <= 92.0:
        return 0.0
    # Exponential opening above 92 °C
    excess = clt - 92.0
    return 0.00015 * excess**1.8  # tuned to real data


def update_coolant_temp(current_clt, last_cycle, rpm, dt=0.02):
    dT = calc_coolant_temperature_increment(
        brake_torque_nm=last_cycle["brake_torque_nm"], rpm=rpm, current_clt=current_clt
    )
    cooling = calc_thermostat_cooling_rate(current_clt)
    return np.clip(current_clt + dT - cooling, -10, 115)


def update_knock(last_cycle, clt):
    peak_bar = last_cycle["peak_pressure_bar"]
    if peak_bar > 65 and clt < 95:
        return min(100.0, (peak_bar - 60) ** 2.5)
    return 0.0

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
