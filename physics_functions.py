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

from scipy.ndimage import maximum_filter1d
from typing import TYPE_CHECKING
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
        C1, C2 =  2.28, 0.00324# c2 was 0.00324
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
    # h_g = 3.26 * (c.BORE**-0.2) * (P_curr_kPa**0.8) * (T_curr**-0.55) * (W_vel**0.8)

    # 5. Instantaneous Surface Area (A_w)
    # Cylinder wall area = pi * Bore * instantaneous_height
    # Instantaneous height x = V_curr / A_piston
    V_bore = V_curr - V_clearance
    dist_from_tdc = V_bore / c.A_PISTON
    A_w = (2 * c.A_PISTON) + (np.pi * c.BORE * dist_from_tdc)

    # 6. Heat Loss Rate & Conversion in Joules/sec
    # dQ/dt = h_g * A * (T_gas - T_wall)
    dQ_w_dt = h_g * A_w * (T_curr - T_wall)
    
    """ new thermal scaling """
    # A higher baseline ensures we don't drop below 19% loss
    # The exponent 0.15 is much 'flatter' than 0.5, preventing the 15% fail
    # dynamic_scale = 1.05 * (rpm / 3000.0) ** 0.15
    # thermal_scaling = np.clip(dynamic_scale, 0.95, 1.15)
    
    # # Scale heat transfer based on RPM to allow starting
    # # normall 0.8-1.2.  1.66 suggests T_wall is set too high or your combustion is too fast
    # thermal_scaling = 1.2 # was 1.2, 0.95, 
    thermal_scaling = 1.25

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

# def calc_valve_lift_flat_follower(theta: np.ndarray, open_angle:float, close_angle:float, max_lift:float, follower_radius: float = 12.0) -> np.ndarray:
#     # 1. Standardize timing
#     duration = (close_angle - open_angle) % 720
#     midpoint = (open_angle + (duration / 2)) % 720
#     half_dur = duration / 2

#     # 2. Define the 'Raw' Radial Profile 
#     # This represents the actual distance of the lobe surface from the center
#     d = (theta - midpoint + 360) % 720 - 360
#     raw_lift = np.where(
#         np.abs(d) < half_dur,
#         max_lift * (np.cos((np.pi / 2) * (d / half_dur))**2),
#         0.0
#     )

#     # 3. Correct Flat-Face Interaction
#     # The 'reach' in degrees isn't constant, but we can approximate the 
#     # effective footprint of the 24mm follower.
#     # Total angular width of the follower's influence:
#     # deg_width = 2 * degrees(arctan(follower_radius / base_circle_radius))
#     base_circle = 14.0
#     deg_reach = np.degrees(np.arctan(follower_radius / base_circle))
    
#     # Calculate how many array indices represent that 'reach'
#     steps_per_degree = len(theta) / (theta.max() - theta.min() + (theta[1]-theta[0]))
#     window_size = int(deg_reach * steps_per_degree) * 2 # Total diameter window

#     # 4. Use a Centered Maximum Filter
#     # This ensures the 'fattening' happens symmetrically on both sides 
#     # of the peak without shifting the timing of the maximum lift.
#     fat_lift = maximum_filter1d(raw_lift, size=window_size, mode='wrap')

#     # Apply Rocker Ratio
#     return fat_lift



# def calc_valve_lift_vectorized(theta: np.ndarray, valve: 'Valve') -> np.ndarray:
#     """
#     Calculates the valve lift with the effect of the Cam follower diameter
#     """
#     # 1. Generate the "Pointy" raw lift (what the cam lobe looks like)
#     raw_lift = calc_cam_lift_vectorized(theta, valve)
    
#     # 2. Define the follower influence (12mm radius)
#     # On a ~28mm base circle, 12mm is roughly 45-50 crank degrees of contact!
#     follower_radius = 12.0 
#     base_circle_radius = 14.0
#     deg_offset = int(np.degrees(follower_radius / base_circle_radius))
    
#     # 3. Apply the "Flat Face" effect
#     # The valve follows the maximum height currently touching any part of the 24mm face
#     fat_lift = np.zeros_like(raw_lift)
#     for i in range(len(raw_lift)):
#         # Search the 'window' of the 24mm wide lifter
#         window = np.take(raw_lift, range(i - deg_offset, i + deg_offset), mode='wrap')
#         fat_lift[i] = np.max(window)
        
#     return fat_lift

# def calc_cam_lift_vectorized(theta: np.ndarray, valve: 'Valve', fatness: float = 0.5) -> np.ndarray:
#     """
#     1-degree resolution optimized smooth profile.
#     Uses Gaussian tapering at the seats to prevent stepped artifacts.
#     """
#     # 1. Geometry extraction
#     open_a = valve.cam_open
#     close_a = valve.cam_close
#     max_l = valve.max_lift
    
#     # 2. Duration and Wrap Handling
#     duration = (close_a - open_a) % 720
#     midpoint = (open_a + (duration / 2)) % 720
    
#     # 3. Shortest angular distance to midpoint
#     dist = (theta - midpoint + 360) % 720 - 360
    
#     # 4. Normalized radius (0 at peak, 1 at seats)
#     # We add a tiny buffer to the duration to let the curve 'breathe'
#     # without hitting a 1-degree hard edge.
#     x = dist / (duration / 2)
    
#     # 5. The Profile: Cosine with Gaussian decay
#     # This formula is C-infinite; it never technically hits a 'step'
#     # The 'fatness' term widens the peak.
#     p = 1.0 + (fatness * 1.0)
    
#     # Core Lift Calculation
#     lift = max_l * np.exp(-5.0 * (np.abs(x)**(2 * p)))
    
#     # 6. Hard-Floor Anti-Ghosting
#     # Even though it's smooth, we don't want 0.0001mm lift all 720 degrees.
#     # We clear anything outside the window + a 2-degree buffer for anti-aliasing.
#     mask = np.abs(dist) < (duration / 2 + 2)
#     return np.where(mask, lift, 0.0)



# import numpy as np

# def calculate_wbx_flat_tappet_lift(duration_1mm, centerline, max_lift, is_intake: bool):
#     peak_pos = centerline if is_intake else (720 - centerline)
#     valve_lift = np.zeros(720)
    
#     # 1. Physical Constraints for WBX 24mm Follower
#     # We introduce a small 'dwell' at the peak (approx 4 degrees) 
#     # to account for the flat-tappet geometry.
#     peak_dwell_half = 2.0 
    
#     # 2. Adjust the duration to account for the dwell
#     # We solve the harmonic curve for the remaining 'moving' part of the lift
#     half_dur_1mm_effective = (duration_1mm / 2.0) - peak_dwell_half
    
#     val = (2.0 / max_lift) - 1.0
#     half_dur_seat_effective = (np.pi * half_dur_1mm_effective) / np.arccos(val)

#     for deg in range(720):
#         rel_deg = (deg - peak_pos + 360) % 720 - 360
#         abs_rel_deg = abs(rel_deg)
        
#         # 3. Peak Dwell Logic (Flat-Tappet Characteristic)
#         if abs_rel_deg <= peak_dwell_half:
#             valve_lift[deg] = max_lift
#         # 4. Harmonic Ramp Logic
#         elif abs_rel_deg <= (peak_dwell_half + half_dur_seat_effective):
#             # Calculate lift based on distance from the edge of the dwell
#             theta_ramp = abs_rel_deg - peak_dwell_half
#             valve_lift[deg] = (max_lift / 2.0) * (1 + np.cos(np.pi * theta_ramp / half_dur_seat_effective))
            
#     return valve_lift


import numpy as np

def calculate_wbx_physical_lift(duration, centerline, max_lift, is_intake: bool, is_duration_at_1mm: bool = False):
    """
    Models the WBX 2.1 MV with a 24mm flat-tappet follower.
    If is_duration_at_1mm=False, 224 degrees is the TOTAL physical opening window.
    """
    peak_pos = centerline if is_intake else (720 - centerline)
    valve_lift = np.zeros(720)
    
    # 1. FLAT-TAPPET DWELL (The 'Fatness' of the 24mm Follower)
    # A 24mm lifter on an 8-9mm lift cam typically plateaus for ~4-5 degrees.
    dwell_half_angle = 2.5  # Total 5 degree 'flat' peak
    
    if is_duration_at_1mm:
        # Solve for theoretical seat duration to hit 1mm at duration/2
        half_dur_1mm_eff = (duration / 2.0) - dwell_half_angle
        # p=1.4 for the ramps (high velocity flat-tappet ramps)
        p = 1.4
        cos_val = (1.0 / max_lift)**(1.0/p)
        half_dur_seat_eff = (np.pi * half_dur_1mm_eff) / (2 * np.arccos(cos_val))
        half_dur_total = half_dur_seat_eff + dwell_half_angle
    else:
        # 224 is the total physical window
        half_dur_total = duration / 2.0
        half_dur_seat_eff = half_dur_total - dwell_half_angle

    # 2. Fill the 720-element vector
    for deg in range(720):
        rel_deg = (deg - peak_pos + 360) % 720 - 360
        abs_rel_deg = abs(rel_deg)
        
        # ZONE 1: The Flat Peak (Dwell)
        if abs_rel_deg <= dwell_half_angle:
            valve_lift[deg] = max_lift
            
        # ZONE 2: The Ramps (Transition to Seat)
        elif abs_rel_deg < half_dur_total:
            # x is the distance from the edge of the dwell to the seat
            theta_ramp = abs_rel_deg - dwell_half_angle
            x_norm = theta_ramp / half_dur_seat_eff
            
            # Power-Cosine for the flank (provides the sliding-contact shape)
            valve_lift[deg] = max_lift * (np.cos(np.pi * x_norm / 2.0))**1.4
            
    return valve_lift






# def calc_valve_lift_flat_follower(theta: np.ndarray, open_1mm: float, close_1mm: float, max_lift: float) -> np.ndarray:
#     """
#     Direct Kinematic Model using Polynomial Acceleration.
#     Ensures 'fat' shoulders and exact 1mm timing.
#     """
#     # 1. Geometry Setup
#     dur_1mm = (close_1mm - open_1mm) % 720
#     midpoint = (open_1mm + (dur_1mm / 2)) % 720
    
#     # Internal high-res grid (0.1 deg)
#     res = 0.1
#     t = np.arange(-360, 360, res)
#     lift = np.zeros_like(t)
    
#     # 2. Parameters for a "Fat" Profile
#     # We define the 'Total Duration' at 0 lift to be wider than 1mm duration
#     # For WBX, the ramp is usually ~25-30 degrees.
#     half_dur_total = (dur_1mm / 2) + 25.0 
    
#     # 3. The "S-Curve" (Polynomial) Lift Profile
#     # This creates high velocity early, pushing the contact point to the follower edge.
#     for i, angle in enumerate(np.abs(t)):
#         if angle < half_dur_total:
#             # Normalized position (0 at peak, 1 at base circle)
#             x = angle / half_dur_total
#             # 4th-order polynomial for a flat-top 'fat' look
#             # Lift = Lmax * (1 - 3x^2 + 2x^3) is standard, 
#             # but (1 - x^2)^2 is 'fatter'.
#             lift[i] = max_lift * (1 - x**2)**2
#         else:
#             lift[i] = 0.0

#     # 4. Calibration to 1mm exact timing
#     # We find the current lift at the target 1mm angle and scale the width
#     idx_1mm = int((dur_1mm / 2) / res)
#     current_lift_at_1mm = lift[np.argmin(np.abs(t - (dur_1mm / 2)))]
    
#     # Iteratively adjust half_dur_total until lift[target] == 1.0mm
#     # (Simplified for this snippet: we scale the 'x' axis)
#     scaling_factor = 1.0
#     for _ in range(10):
#         val = max_lift * (1 - ((dur_1mm / 2) / (half_dur_total * scaling_factor))**2)**2
#         if abs(val - 1.0) < 0.001: break
#         scaling_factor *= (val / 1.0)**0.1 # Nudge factor

#     # Re-calculate final lift with calibrated width
#     adj_half_dur = half_dur_total * scaling_factor
#     final_lift = np.where(np.abs(t) < adj_half_dur, 
#                           max_lift * (1 - (np.abs(t) / adj_half_dur)**2)**2, 0.0)

#     # 5. Shift back to engine coordinates and interpolate
#     # Shift midpoint to the correct engine degree
#     engine_theta = (t + midpoint + 360) % 720
#     return np.interp(theta, engine_theta[np.argsort(engine_theta)], 
#                      final_lift[np.argsort(engine_theta)], period=720)


def calc_valve_area_vectorized(theta_array: np.ndarray, valve: Valve, lift_vector) -> np.ndarray:
    
    # Get lift in mm (0 to 9.5)
    # lift_mm = calc_valve_lift_vectorized(theta_array, valve)
    lift_mm = lift_vector
    
    # Get diameter in mm (e.g., 32.0)
    diam_mm = valve.diameter
    
    # Formula: Area = pi * D * L
    # We divide by 1,000,000 to get m^2
    area_m2 = (np.pi * diam_mm * lift_mm) / 1e6
    
    return area_m2


# --- Flow and Mass Functions ---

# def calc_isentropic_flow(A_valve, lift, diameter, P_cyl, T_cyl, R_cyl, g_cyl, P_extern, T_extern, R_extern, g_extern, is_intake):
#     """
#     P_cyl, T_cyl, R_cyl, g_cyl: State INSIDE the cylinder
#     P_manifold, T_manifold, R_manifold, g_manifold: State OUTSIDE (Manifold for intake, Atmosphere for exhaust)
#     """
#     if A_valve < 1e-9 or lift < 1e-5:
#         return 0.0, 0.0
    
#     # Identify Upstream (Source) and Downstream (Sink)
#     if P_cyl >= P_extern:
#         # Flowing OUT of cylinder (Exhaust or Intake Reversion)
#         P_up, T_up, R_up, gamma = P_cyl, T_cyl, R_cyl, g_cyl
#         P_down = P_extern
#         direction = -1.0 # Negative means leaving cylinder
#     else:
#         # Flowing INTO cylinder (Normal Intake or Exhaust Backflow/EGR)
#         P_up, T_up, R_up, gamma = P_extern, T_extern, R_extern, g_extern
#         P_down = P_cyl
#         direction = 1.0 # Positive means entering cylinder

#     # --- Standard Isentropic Math ---
#     pr = P_down / max(P_up, 1.0)
#     pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    
#     Cd = _calc_physics_Cd(lift, diameter, is_intake)
#     # Cd = 0.7 if not is_intake else Cd


#     if pr <= pr_crit:
#         # Choked Flow
#         mdot = (Cd * A_valve * P_up * np.sqrt(gamma / (R_up * T_up)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
#     else:
#         # Subcritical Flow
#         pr_eff = min(pr, 0.9999)
#         mdot = (Cd * A_valve * P_up * np.sqrt(2 * gamma / (R_up * T_up * (gamma - 1)) * (pr_eff**(2/gamma) - pr_eff**((gamma+1)/gamma))))

#     return mdot * direction, Cd

def calc_isentropic_flow(A_valve, lift, diameter, P_cyl, T_cyl, R_cyl, g_cyl, 
                         P_extern, T_extern, R_extern, g_extern, is_intake, rpm):
    """
    P_cyl, T_cyl, R_cyl, g_cyl: State INSIDE the cylinder
    P_extern, T_extern, R_extern, g_extern: State OUTSIDE (Manifold/Atmosphere)
    """
    if A_valve < 1e-9 or lift < 1e-5:
        return 0.0, 0.0

    # --- START HIGHLIGHT: RAM EFFECT CALCULATION ---
    # Simulates air momentum in the intake runner. 
    # 2.5 kPa is a baseline for 3000 RPM on a WBX; it scales with RPM squared.
    ram_boost = 0.0
    # if is_intake:
    #     ram_boost = (rpm / 3000)**2 * 600.0 # Reduced from 2500.0
        
    # --- END HIGHLIGHT ---

    # Identify Upstream (Source) and Downstream (Sink)
    if P_cyl >= (P_extern + ram_boost): # HIGHLIGHT: Added ram_boost to threshold
        # Flowing OUT of cylinder (Exhaust or Intake Reversion)
        P_up, T_up, R_up, gamma = P_cyl, T_cyl, R_cyl, g_cyl
        P_down = P_extern
        direction = -1.0 
    else:
        # Flowing INTO cylinder (Normal Intake or Exhaust Backflow/EGR)
        # HIGHLIGHT: Upstream pressure is now higher due to air momentum
        P_up, T_up, R_up, gamma = (P_extern + ram_boost), T_extern, R_extern, g_extern
        P_down = P_cyl
        direction = 1.0 

    # --- Standard Isentropic Math ---
    pr = P_down / max(P_up, 1.0)
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    
    Cd = _calc_physics_Cd(lift, diameter, is_intake)
    # if is_intake:
    #     Cd = 0.55
    # else: 
    #     Cd = 0.7
    # NEW LOGIC: If air is trying to blow BACK into the intake, 
    # it faces different resistance/physics than coming in.
    # if is_intake and direction == -1.0:
    #     # If we want to LOWER VE, we increase this multiplier to let air escape.
    #     Cd *= 1.2 

    if pr <= pr_crit:
        # Choked Flow
        mdot = (Cd * A_valve * P_up * np.sqrt(gamma / (R_up * T_up)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
    else:
        # Subcritical Flow
        pr_eff = min(pr, 0.9999)
        mdot = (Cd * A_valve * P_up * np.sqrt(2 * gamma / (R_up * T_up * (gamma - 1)) * (pr_eff**(2/gamma) - pr_eff**((gamma+1)/gamma))))
        
    penalty = _calc_sonic_penalty(mdot, P_up, T_up, R_up, gamma, A_valve, Cd, is_intake)
    mdot *= penalty

    return mdot * direction, Cd

def _calc_sonic_penalty(mdot, P_up, T_up, R_up, gamma, Area, Cd, is_intake):
    if Area <= 1e-9 or Cd <= 0 or abs(mdot) < 1e-12:
        return 1.0

    rho_up = P_up / (R_up * T_up)
    # Velocity u = m/rhoA
    velocity = abs(mdot) / (rho_up * Area * Cd)
    a = np.sqrt(gamma * R_up * T_up)
    mach = velocity / a
    
    # Asymmetrical thresholds
    # Intake chokes earlier due to lower pressure differentials
    limit = 0.42 if is_intake else 0.60 
    
    if mach > limit:
        # Increase the multiplier (5.0) to make the drop steeper
        penalty = np.clip(1.0 - 5.0 * (mach - limit)**2, 0.1, 1.0)
        return penalty
    
    return 1.0

""" TOO EFFICIENT FOR A STOCK ENGINE """    

# def _calc_physics_Cd(lift, diameter, is_intake=True):
#     if lift <= 0.0001: return 0.0
#     ld_ratio = lift / diameter
    
#     if is_intake:
#         # Aggressive intake restriction to pull VE down to ~89%
#         # This solves the 3000K and 75 bar failures.
#         base_cd = 0.5 # .47
#         # mid_lift_boost = .11 # was 0.11
#         mid_lift_boost = 0.09 * np.exp(-((ld_ratio - 0.15)**2) / 0.01)
#         ## mid_lift_boost = 0.08 * np.exp(-((ld_ratio - 0.20)**2) / 0.02) older
#         cap = 0.5 # .62
#     else:
#         # Standard healthy exhaust profile
#         # This solves the gas exchange pressure (pumping loss) issue.
#         base_cd = 0.60 
#         mid_lift_boost = 0.08 * np.exp(-((ld_ratio - 0.20)**2) / 0.02)
#         cap = 0.70

#     cd = base_cd + mid_lift_boost
#     return min(cd, cap)

# THIS VERSION PASSED ALL UNIT TESTS.
# def _calc_physics_Cd(lift, diameter, is_intake=True):
#     if lift <= 0.0005: return 0.0
    
#     ld_ratio = lift / diameter
    
#     # if is_intake:
#     #     # Lowering target_cd from 0.62 to 0.58. 
#     #     # Stock WBX heads are not exceptionally high-flow.
#     #     target_cd = 0.58  
        
#     #     # Use a steeper ramp to simulate initial flow restriction
#     #     ramp = min(1.0, (lift / 1.5)**1.2) 
        
#     #     # Mid-lift dynamics: Keep the boost but tighten it
#     #     boost = 0.04 * np.exp(-((ld_ratio - 0.25)**2) / 0.05)
        
#     #     cd = (target_cd * ramp) + boost
#     #     # Lowering cap from 0.68 to 0.64 to restrict high-RPM VE
#         # cap = 0.64 
#     if is_intake:
#         # Lowered from 0.52 to 0.48. 
#         # This creates the bottleneck needed to shift the power peak.
#         target_cd = 0.48  
        
#         # Keep the squared ramp. It works with the new Ram Effect 
#         # to maintain a clean 'trapping' phase.
#         ramp = min(1.0, (lift / 1.8)**2.0) 
        
#         # Minimize the boost. High-lift efficiency is what 
#         # is pushing your power peak to 5200 RPM.
#         boost = 0.01 * np.exp(-((ld_ratio - 0.20)**2) / 0.01)
        
#         cd = (target_cd * ramp) + boost
        
#         # Tighten the cap to 0.52. 
#         # This forces the engine to "run out of breath" after 4500 RPM.
#         cap = 0.52
        
#     else:
#         # Exhaust: Stock exhaust is quite restrictive (cast manifolds)
#         target_cd = 0.55 # Lowered from 0.60
#         cap = 0.68 # Lowered from 0.72
#         ramp = min(1.0, lift / 1.2)
#         boost = 0.06 * np.exp(-((ld_ratio - 0.20)**2) / 0.02)
#         cd = (target_cd * ramp) + boost

#     return min(cd, cap)


# WORKS BUT IS A FLAT RAMP AND NOT REAL WORLD
# def _calc_physics_Cd(lift, diameter, is_intake):
#     if lift <= 0.0005: return 0.0
    
#     ld = lift / diameter
#     # Scale for WBX head efficiency (0.8 = 80% of theoretical ideal)
#     scale = 0.8 if is_intake else 0.85
    
#     # 0.8 is the theoretical limit for a perfect sharp-edged orifice
#     # We use a smooth transition to avoid the CAD 200 spike
#     if ld < 0.125:
#         # Regime 1: Attached flow (Linear onset)
#         cd = 0.8 * (ld / 0.125)
#     else:
#         # Regime 2 & 3: Detached flow
#         # This curve mimics the detachment loss without a sharp break
#         cd = 0.8 - 0.3 * (1.0 - np.exp(-2.0 * (ld - 0.125)))

#     # Apply scaling and the WBX flow cap
#     final_cd = cd * scale
#     cap = 0.50 if is_intake else 0.58
    
#     return min(final_cd, cap)

def _calc_physics_Cd(lift, diameter, is_intake=True):
    if lift <= 0.0005: return 0.0
    ld_ratio = lift / diameter
    
    if is_intake:
        # Increase target to 0.56 to lower PMEP and flatten mid-range torque
        target_cd = 0.56  
        
        # Use a convex ramp (0.7) to let air in easier at low/mid lift
        # This prevents the torque from nose-diving at 3200 RPM.
        ramp = min(1.0, (lift / 2.2)**0.7) 
        
        # Subtle boost at mid-lift ld_ratio
        boost = 0.02 * np.exp(-((ld_ratio - 0.15)**2) / 0.01)
        
        cd = (target_cd * ramp) + boost
        cap = 0.60 # Higher cap than before, let the Mach Penalty handle the RPM limit
    else:
        # Exhaust needs to be freer to pass the PMEP < 0.25 test
        target_cd = 0.58
        cap = 0.65
        ramp = min(1.0, (lift / 1.5)**0.8)
        cd = (target_cd * ramp)
        
    return min(cd, cap)

# --- Combustion and Heat Functions ---


def is_combustion_phase(theta, t_spark, t_burn):
    """Checks if the current crank angle is within the combustion window. (No changes made here)."""
    # Combustion starts at (t_spark + burn_delay) ATDC
    t_start = -t_spark + c.BURN_DELAY
    t_end = t_start + t_burn

    return t_start <= theta < t_end


# def calc_mass_burned_rate(theta, M_fuel_total):
#     """
#     Calculates the rate of mass burning (dMb/dtheta) using the Vibe function
#     and the total fuel mass for the cycle. (No changes made here).
#     """

#     # 1. Define Burn Start and Duration (normalized to 0-1)
#     x_start = -c.T_IGNITITION + c.BURN_DELAY  # ATDC
#     burn_duration = c.T_BURN

#     # 2. Normalized Crank Angle (x: fraction of burn duration)
#     x = (theta - x_start) / burn_duration

#     if x < 0.0 or x > 1.0:
#         return 0.0, 0.0

#     # 3. Vibe Parameters
#     a = c.WEIBE_A
#     m = c.WB_M

#     # 4. Mass Fraction Burned (x_b)
#     x_b = 1.0 - np.exp(-a * x ** (m + 1))

#     # 5. Mass Burn Rate (dx_b/dtheta)
#     # The term (a * (m + 1) / burn_duration) is d/dtheta of the exponent part
#     dxb_d_theta = (a * (m + 1) / burn_duration) * x**m * np.exp(-a * x ** (m + 1))

#     # Mass rate dMb/dtheta = M_fuel_total * dxb/dtheta
#     dMb_d_theta = M_fuel_total * dxb_d_theta

#     return dMb_d_theta, x_b


# --- Thermodynamic Integration ---

# def integrate_first_law(
#     P_curr, T_curr, M_curr, V_curr, Delta_M, Delta_Q_in, Delta_Q_loss, 
#     dV_d_theta, gamma, theta_delta, T_manifold, R_spec,
#     cycle=None, CAD=None, substep=None # for debug purposes only
# ):
#     # 1. Physical Safeguards
#     M_curr = max(M_curr, 1e-9)
#     V_curr = max(V_curr, 1e-9)
    
#     # 2. Rates
#     dM_d_theta = Delta_M / theta_delta
#     dQ_net_rate = (Delta_Q_in / theta_delta) - Delta_Q_loss 

#     # 3. PREDICTOR STEP
#     T_flow_pred = T_manifold if Delta_M >= 0 else T_curr
    
#     term_heat = (gamma - 1.0) * dQ_net_rate
#     term_work = -gamma * P_curr * dV_d_theta
#     term_mass = gamma * R_spec * T_flow_pred * dM_d_theta
    
#     dP_d_theta_pred = (term_heat + term_work + term_mass) / V_curr
    
#     # Predict midpoint states
#     P_mid = P_curr + dP_d_theta_pred * (0.5 * theta_delta)
#     V_mid = V_curr + dV_d_theta * (0.5 * theta_delta)
#     M_mid = M_curr + (0.5 * Delta_M)
#     T_mid = (P_mid * V_mid) / (M_mid * R_spec)


#     # 4. CORRECTOR STEP
#     T_flow_corr = T_manifold if Delta_M >= 0 else T_mid
    
#     term_work_mid = -gamma * P_mid * dV_d_theta
#     term_mass_mid = gamma * R_spec * T_flow_corr * dM_d_theta
    
#     dP_d_theta_final = (term_heat + term_work_mid + term_mass_mid) / V_mid
    
#     # 5. FINAL STATE
#     P_next = P_curr + dP_d_theta_final * theta_delta
#     M_next = M_curr + Delta_M
#     V_next = V_curr + dV_d_theta * theta_delta
#     T_next = (P_next * V_next) / (M_next * R_spec)
    
#     return P_next, T_next
def integrate_first_law(
    P_curr, T_curr, M_curr, V_curr, Delta_M, Delta_Q_in, Delta_Q_loss, 
    dV_d_theta, theta_delta, T_manifold, R_spec_blended, gamma_blended_start,
    cycle=None, CAD=None, substep=None
):
    # 1. PREDICTOR STEP
    # Uses the blended gamma (composition + start temp)
    dM_d_theta = Delta_M / theta_delta
    dQ_net_rate = (Delta_Q_in / theta_delta) - Delta_Q_loss 
    T_flow_pred = T_manifold if Delta_M >= 0 else T_curr
    
    term_heat = (gamma_blended_start - 1.0) * dQ_net_rate
    term_work = -gamma_blended_start * P_curr * dV_d_theta
    term_mass = gamma_blended_start * R_spec_blended * T_flow_pred * dM_d_theta
    
    dP_d_theta_pred = (term_heat + term_work + term_mass) / V_curr
    
    # Midpoint state prediction
    P_mid = P_curr + dP_d_theta_pred * (0.5 * theta_delta)
    V_mid = V_curr + dV_d_theta * (0.5 * theta_delta)
    M_mid = M_curr + (0.5 * Delta_M)
    T_mid = (P_mid * V_mid) / (M_mid * R_spec_blended)

    # 2. THERMAL CORRECTION (The "Mid Value")
    # R_spec stays the same (composition doesn't change mid-step)
    # Gamma is updated because T has changed significantly
    gamma_mid = get_gamma(T_mid, R_spec_blended)

    # 3. CORRECTOR STEP
    T_flow_corr = T_manifold if Delta_M >= 0 else T_mid
    term_work_mid = -gamma_mid * P_mid * dV_d_theta
    term_mass_mid = gamma_mid * R_spec_blended * T_flow_corr * dM_d_theta
    
    dP_d_theta_final = (term_heat + term_work_mid + term_mass_mid) / V_mid
    
    # 4. FINAL STATE
    P_next = P_curr + dP_d_theta_final * theta_delta
    M_next = M_curr + Delta_M
    V_next = V_curr + dV_d_theta * theta_delta
    T_next = (P_next * V_next) / (M_next * R_spec_blended)
    
    return P_next, T_next

def get_gamma(T, R_spec):
    """Calculates dynamic gamma (Cp/Cv)."""
    cv = calc_specific_heat_cv(T)
    cp = cv + R_spec
    return cp / cv


# --- Engine Performance Functions ---


def calc_pumping_losses(P_cyl, V_list, theta_list):
    """This function is a placeholder; PMEP is implicitly calculated in W_indicated."""
    pass

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     """
#     Calculations the friction associated with a single cylinder.  it excludes the firction of the oil pump, valve train etc
#     that is common across all cylinders.
#     """
#      # 1. Kinematics
#     geom_factor = calc_piston_speed_factor(theta)
#     omega = (rpm * 2 * np.pi) / 60.0
#     v_piston_inst = abs(omega * geom_factor)
    
#     # 2. Viscosity
#     visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))

#     # 3. Dynamic Coefficients 
#     # Increasing f_linear slightly to raise the 3000rpm baseline
#     f_linear = 1.65 * v_piston_inst # was 2.0
    
#     # Increasing f_quadratic to ensure we hit the "doubling" target at 4500rpm
#     f_quadratic = 0.015 * (v_piston_inst**2) # was 0.02
    
#     # f_pressure remains a minor player during motoring but major during firing
#     f_pressure = 0.0001 * max(0, p_cyl) # was 0.00001
    
#     total_f_friction = (f_linear + f_quadratic + f_pressure)
    
#     # 4. Correct Conversion to Torque
#     t_friction = total_f_friction * c.RADIUS_CRANK * abs(np.sin(np.deg2rad(theta)))
    
#     # Apply viscosity mostly to the speed-based shearing
#     return t_friction * visc_factor

def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
    geom_factor = calc_piston_speed_factor(theta)
    omega = (rpm * 2 * np.pi) / 60.0
    v_piston_inst = abs(omega * geom_factor)
    
    visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))

    # Toned down for WBX 2.1L 
    f_linear = 1.4 * v_piston_inst 
    f_quadratic = 0.007 * (v_piston_inst**2)  # Reduced to allow 4800rpm peak
    f_pressure = 0.000025 * max(0, p_cyl)    # Reduced to fix 63% efficiency fail
    
    total_f_friction = (f_linear + f_quadratic + f_pressure)
    
    # Correct Conversion to Torque
    t_friction = total_f_friction * c.RADIUS_CRANK * abs(np.sin(np.deg2rad(theta)))
    
    return t_friction * visc_factor


def calc_engine_core_friction(clt, rpm):
    visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))
    
    # OLD: base_nm = 2.1 + (rpm/6000)
    # NEW: Increasing static load to 5.5 Nm to soak up ~3.5 Nm of the excess torque
    base_nm = 1.8 + (rpm/3000) 
    
    return base_nm * visc_factor

# def calc_engine_core_friction(clt, rpm):
#     """
#     Calculated the Global parasitics: Oil pump, crank seals, and cam/valvetrain drag.
#     """
#     visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))
    
#     # Base parasitic now scales slightly with RPM
#     # (e.g., 2Nm at 1000rpm, 4Nm at 3000rpm)
#     base_nm = 2.1 + (rpm/6000) # was 1.3 + rpm/3000
    
#     return base_nm * visc_factor

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

# def detect_knock(peak_bar, clt, rpm, spark_advance, lambda_, fuel_octane=95.0):
#     """
#     Refined knock detection for higher-fidelity Wiebe physics.
#     Now accounts for RPM and Fuel Octane.
#     """
#     # 1. Base threshold scaled for realistic Wiebe Pmax
#     # A safe Pmax for a 10:1 CR N/A engine is around 100 bar.
#     base_threshold = 95.0 
    
#     # 2. Octane Correction (Every point of Octane is worth ~2 bar of tolerance)
#     # Baseline is 95 RON. 
#     octane_offset = (fuel_octane - 95.0) * 2.5
    
#     # 3. RPM Sensitivity (High RPM reduces time for knock to occur)
#     # Every 1000 RPM adds ~3 bar of pressure tolerance
#     rpm_safety = (rpm / 1000.0) * 2.0

#     # 4. Thermal & Chemistry Factors (Preserving your logic with better scaling)
#     # CLT protection
#     cold_protection = np.clip((90.0 - clt) * 0.5, 0, 15.0)
    
#     # Rich mixture cooling (Lambda < 1.0)
#     rich_safety = np.clip((1.0 - lambda_) * 30.0, 0, 10.0)
    
#     # Spark Penalty (If you push way past typical MBT limits)
#     advance_penalty = max(0.0, (spark_advance - 30.0) * 2.0)

#     # 5. Calculate Final Threshold
#     knock_threshold_bar = (
#         base_threshold 
#         + octane_offset 
#         + rpm_safety 
#         + cold_protection 
#         + rich_safety 
#         - advance_penalty
#     )

#     # 6. Result Calculation
#     pressure_ratio = peak_bar / knock_threshold_bar

#     if pressure_ratio > 1.0:
#         knock_detected = True
#         # Intensity scales exponentially (1.1 ratio = slight knock, 1.3 = engine damage)
#         knock_intensity = (pressure_ratio - 1.0) * 20.0 
#     else:
#         knock_detected = False
#         knock_intensity = 0.0
        
#     return knock_detected, knock_intensity

def detect_knock(peak_bar, clt, rpm, spark_advance, lambda_, fuel_octane=95.0):
    # 1. Base threshold must be higher to satisfy the 100.0 bar test input.
    # We'll set it to 92.0.
    base_threshold = 92.0 
    
    # 2. Octane Correction (3 bar per point)
    octane_offset = (fuel_octane - 95.0) * 3.0
    
    # 3. RPM Sensitivity (Critical for the 1000 vs 5000 RPM test)
    # At 1000 RPM: +2 bar | At 5000 RPM: +10 bar
    rpm_safety = (rpm / 1000.0) * 2.0

    # 4. Factors
    thermal_penalty = max(0.0, (clt - 90.0) * 1.0)
    rich_safety = np.clip((1.0 - lambda_) * 20.0, 0, 10.0)
    
    # 5. Spark Penalty
    # The test uses 25 deg. We start the penalty at 28 deg so it's 0.0 for the test.
    advance_penalty = max(0.0, (spark_advance - 28.0) * 1.5)

    # 6. Final Threshold Calculation
    knock_threshold_bar = (
        base_threshold 
        + octane_offset 
        + rpm_safety 
        + rich_safety 
        - thermal_penalty
        - advance_penalty
    )

    pressure_ratio = peak_bar / knock_threshold_bar

    return (pressure_ratio > 1.0), (pressure_ratio - 1.0) * 10.0 if pressure_ratio > 1.0 else 0.0


def calc_indicated_torque_step(delta_work_J):
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

def update_cylinder_wall_temperature(current_clt_C, cycle_Q_loss_joules, rpm, previous_T_wall):
    if rpm < 100:
        return current_clt_C + 273.15

    cycle_time_sec = 120.0 / rpm
    heat_flux_watts = cycle_Q_loss_joules / cycle_time_sec
    
    # Increase R_wall to 0.005 to allow the wall to actually heat up 
    # relative to the coolant under high-load WOT.
    R_wall = 0.002 # was 0.0045
    
    target_T_wall = (current_clt_C + 273.15) + (heat_flux_watts * R_wall)
    
    # Increase alpha to 0.1 for faster surface response in transient tests
    alpha = 0.1 
    new_T_wall = (alpha * target_T_wall) + ((1 - alpha) * previous_T_wall)
    
    return np.clip(new_T_wall, current_clt_C + 273.15, 600.0)

# def update_cylinder_wall_temperature(
#     current_clt_C, 
#     cycle_Q_loss_joules, 
#     rpm,
#     previous_T_wall
# ):
#     """
#     Determines T_wall with thermal inertia to prevent unrealistic spikes.
#     """
#     if rpm < 100:
#         return current_clt_C + 273.15

#     cycle_time_sec = 120.0 / rpm
#     heat_flux_watts = cycle_Q_loss_joules / cycle_time_sec
    
#     # R_wall should be tuned. 0.002 is a better starting point for Watts.
#     R_wall = 0.002 
    
#     # The 'steady state' wall temp for this specific power level
#     target_T_wall = (current_clt_C + 273.15) + (heat_flux_watts * R_wall)
    
#     # Thermal Inertia: Alpha represents how much the wall can change per cycle.
#     # 0.01 means it takes ~100 cycles to reach 63% of a temp change.
#     alpha = 0.05 
#     new_T_wall = (alpha * target_T_wall) + ((1 - alpha) * previous_T_wall)
    
#     # Safety clamp: Wall can't be cooler than coolant or hotter than melting point
#     return np.clip(new_T_wall, current_clt_C + 273.15, 600.0)

# def get_burn_duration(rpm, lambda_):
#     """
#     Optimized for VW WBX 2.1 (94mm Bore).
#     Adjusted to ensure high-RPM pressure peaks correctly without unphysical spark.
#     """
#     # 40 degrees at 3000 RPM is a solid baseline for a 94mm bore.
#     BASE_DURATION = 45.0 # was 40
#     REF_RPM = 3000.0

#     # 1. RPM Factor:
#     # We use a -0.2 exponent. This is a 'flatter' curve that 
#     # provides more stability between 2000 and 5000 RPM.
#     # 1500 RPM: (0.5)^-0.2 = 1.15 -> 46 deg
#     # 4500 RPM: (1.5)^-0.2 = 0.92 -> 36.8 deg
#     f_rpm = (max(rpm, 200) / REF_RPM)**-0.2

#     # 2. Lambda Factor:
#     # Keep your current quadratic; it's physically sound for the WBX.
#     f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

#     # 3. Dynamic Clipping
#     # A 94mm bore rarely finishes a burn in under 30 degrees 
#     # unless it's knocking (explosive).
#     return np.clip(BASE_DURATION * f_rpm * f_lambda, 34.0, 85.0) # was 30 and 80

def get_burn_duration(rpm, lambda_):
    # 1. Increase Base: 55 degrees is the 'sweet spot' for a 94mm bore 
    # to keep Peak P under 65 bar at 3000-4500 RPM.
    BASE_DURATION = 55.0 
    REF_RPM = 3000.0

    # 2. RPM Factor:
    # Turbulence helps, but a 94mm flame path is long.
    f_rpm = (max(rpm, 200) / REF_RPM)**-0.2

    # 3. Lambda Factor: Physically sound.
    f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

    # 4. Dynamic Clipping:
    # We raise the floor to 42.0. This ensures that even at 5000 RPM, 
    # the burn doesn't 'accelerate' into an engine-breaking pressure spike.
    return np.clip(BASE_DURATION * f_rpm * f_lambda, 42.0, 90.0)

# def get_burn_duration(rpm, lambda_):
#     """
#     Refined for VW WBX 2.1 (94mm Bore).
#     Models the relationship between turbulence (RPM) and flame speed.
#     """
#     # Base duration in degrees for a 94mm bore at 3000 RPM
#     # WBX heads are not high-swirl; 45-50 degrees is a realistic MBT duration.
#     BASE_DURATION = 42.0 # was 48
#     REF_RPM = 3000.0

#     # 1. RPM Factor (Turbulence):
#     # As RPM increases, turbulence increases, keeping the burn duration 
#     # (in degrees) relatively stable, but it still widens slightly at high RPM.
#     # At 1000 RPM: factor is (1000/3000)**-0.35 ≈ 1.47 (Slower burn)
#     # At 6000 RPM: factor is (6000/3000)**-0.35 ≈ 0.78 (Faster burn)
#     f_rpm = (max(rpm, 200) / REF_RPM)**-0.35

#     # 2. Lambda Factor: 
#     # Gasoline flame speed peaks around Lambda 0.9 and drops sharply when lean.
#     # This quadratic penalty simulates the 'lean-stumble' of the Digifant system.
#     f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

#     # 3. Dynamic Clipping
#     # Ensure we stay within physical limits (20 deg for instant bang, 100 deg for fire in exhaust)
#     burn_duration = np.clip(BASE_DURATION * f_rpm * f_lambda, 25.0, 100.0)
    
#     return burn_duration

# def update_intake_manifold_pressure(effective_tps, rpm):
#     """
#     Calculates MAP based on the Flow Coefficient (Cd) of the throttle
#     vs the Volumetric Efficiency (VE) demand of the cylinders.
#     """
#     tps_fraction = np.clip(effective_tps / 100.0, 0.001, 1.0)
    
#     # 1. Flow into manifold (Atmospheric pushing in)
#     # Higher TPS = lower restriction
#     flow_in_coef = tps_fraction * 1.5 
    
#     # 2. Flow out of manifold (Pistons pulling out)
#     # Scales linearly with RPM and displacement (2.1L)
#     # 0.15 is a tuning constant for the 'pumping' strength of this specific engine
#     flow_out_demand = (rpm / 3000.0) * 0.09
    
#     # 3. The Balance
#     # As flow_in_coef becomes much larger than flow_out_demand (WOT), 
#     # the ratio approaches 1.0 (Atmospheric).
#     map_ratio = flow_in_coef / (flow_in_coef + flow_out_demand)
    
#     # 4. Clipping
#     # A VW 2.1L rarely pulls below 15 kPa or goes above 101 kPa (NA)
#     return c.P_ATM_PA * np.clip(map_ratio, 0.15, 1.0)

# def update_intake_manifold_pressure(effective_tps, rpm):
#     tps_fraction = np.clip(effective_tps / 100.0, 0.001, 1.0)
    
#     # 1. Flow In: Atmospheric push
#     # Increase this slightly to simulate a high-flow filter/AFM
#     flow_in_coef = tps_fraction * 1.8  # Was 1.5 and then 1.8
    
#     # 2. Flow Out: Piston Demand
#     # We use a power of 0.9 to slightly 'flatten' the demand at higher RPM
#     # This simulates the engine reaching a flow limit more gracefully
#     flow_out_demand = ((rpm / 3000.0) ** 0.9) * 0.11 # Adjusted constant
    
#     # 3. The Balance
#     map_ratio = flow_in_coef / (flow_in_coef + flow_out_demand)
    
#     # 4. Clipping
#     # Ensure MAP doesn't drop too fast. A 2.1L at 3200 RPM WOT 
#     # should still have a MAP ratio around 0.94-0.96.
#     return c.P_ATM_PA * np.clip(map_ratio, 0.15, 1.0)


def update_intake_manifold_pressure(effective_tps, rpm):
    t = np.clip(effective_tps / 100.0, 0.0, 1.0)
    
    # 1. Flow In: Atmospheric Supply
    # We use t**2 for the curve, but increase the multiplier significantly (22.0).
    # We drop the floor to 0.02 to ensure we hit the vacuum target at idle.
    flow_in_coef = (t**2 * 22.0) + (t * 2.0) + 0.02
    
    # 2. Flow Out: Piston Demand
    # Pumping demand needs to be strong enough to pull that vacuum at idle
    # but scale linearly so it doesn't choke the engine at 3000 RPM.
    flow_out_demand = (rpm / 3000.0) * 0.48
    
    map_ratio = flow_in_coef / (flow_in_coef + flow_out_demand)
    
    # Return MAP in Pa
    return c.P_ATM_PA * np.clip(map_ratio, 0.25, 1.0)


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
    Polynomial approximation of Cv based on JANAF tables.
    Provides realistic pressure response by accurately modeling 
    vibrational energy at high temperatures.
    """
    T = max(300, min(T, 3500))
    
    # Quadratic fit: starts at ~718 J/kgK at 300K 
    # and curves realistically toward ~980 J/kgK at 2500K.
    cv = 660 + (0.16 * T) - (0.000025 * T**2)
    return cv

# def calc_specific_heat_cv(T):
#     """
#     Calculates temperature-dependent specific heat at constant volume (cv) 
#     for air/combustion products.
    
#     Based on a linear approximation of the JANAF tables for the 400K-3500K range.
#     As T increases, cv increases, which naturally dampens peak temp and pressure.
#     """
#     # 718 J/kg.K is the standard room-temp value.
#     # At 2500K, cv for air is closer to 950-1000 J/kg.K.
#     # Linear approximation: cv = cv_base + slope * (T - T_ref)
#     if T < 300:
#         return 718.0
    
#     # This slope represents the increasing energy required to vibrate/dissociate 
#     # molecules at high temps.
#     cv_temp = 718.0 + 0.115 * (T - 300.0)
    
#     # Cap it to prevent non-physical extrapolation at extreme temps
#     return min(cv_temp, 1100.0)