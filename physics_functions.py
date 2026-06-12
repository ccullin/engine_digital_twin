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

# from __future__ import annotations
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
    """ 
    standard wiebe that calculates the total burn energy during  combusion 
    A return of 0.0 is before the burn a 1.0 is after the burn.
    """
        
    delta_theta = theta - ignition_start_theta
    if delta_theta < -360:
        delta_theta += 720
    elif delta_theta > 360:
        delta_theta -= 720
    
    
    if delta_theta <= 0:
        return 0.0
    if delta_theta >= burn_duration:
        return 1.0
    
    burn_energy_accumulative_J = 1.0 - np.exp(-a_vibe * (delta_theta / burn_duration)**(m_vibe + 1))
    
    return burn_energy_accumulative_J


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
        C1, C2 =  2.28, 0.003# c2 was 0.00324
        # C1, C2 = 2.8, 0.00324
    # Phase 2: Compression (from IVC to Spark)
    elif IVC <= CAD < spark:
        C1, C2 =  2.28, 0.0 
        # C1, C2 =  2.8, 0.0 
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
    # h_g = 3.26 * (c.BORE**-0.2) * (P_curr_kPa**0.8) * (T_curr**-0.53) * (W_vel**0.8)
    h_g = 3.26 * (c.BORE**-0.2) * (P_curr_kPa**0.8) * (T_curr**-0.55) * (W_vel**0.8)

    # 5. Instantaneous Surface Area (A_w)
    # Cylinder wall area = pi * Bore * instantaneous_height
    # Instantaneous height x = V_curr / A_piston
    V_bore = V_curr - V_clearance
    dist_from_tdc = V_bore / c.A_PISTON
    # A_w = (2 * c.A_PISTON) + (np.pi * c.BORE * dist_from_tdc)
    A_w = (c.A_PISTON * 1.3) + c.A_PISTON + (np.pi * c.BORE * dist_from_tdc) # allows for wbx curved head shape

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
    thermal_scaling = 0.85 # was 1.2, 0.95, 
    # thermal_scaling =  1.5 #2.5 1.5 # 1.25

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
    dwell_half_angle = 5.0 #2.5  # Total 5 degree 'flat' peak
    
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

    # Simulates air momentum in the intake runner. 
    # # 2.5 kPa is a baseline for 3000 RPM on a WBX; it scales with RPM squared.
    # ram_boost = 0.0
    # if is_intake and rpm > 1500:
    #     # ram_boost = (rpm / 3000)**2 * 1200.0 # Reduced from 2500.0
    #     ram_boost = (rpm / 2800)**2 * 3600.0
    ram_boost = 0
    
    # # Current: (rpm / 2800)**2 * 3600.0
    # # At 3000 RPM = 4132 Pa (0.04 bar)
    # # At 4500 RPM = 9298 Pa (0.09 bar) -> This is likely too high for a stock plenum

    # # Proposed: Use a Sigmoid or Clip to simulate 'Tuning Peak'
    # # This keeps the 'hit' at 3000 but prevents it from becoming a 'Turbo' at 4500.
    # tuning_peak_rpm = 3200
    # ram_scale = np.clip(rpm / tuning_peak_rpm, 0, 1.2)**2 
    # ram_boost = ram_scale * 3200.0 # Max boost of ~0.032 bar
        


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
    
    Cd = _calc_wbx_valve_cd(lift, diameter, is_intake, rpm)
    # Cd = _calc_wbx_valve_cd(lift, diameter, is_intake)
    # Cd = 0.2 if is_intake and rpm <2000 else Cd
    Cd = 0.7

    if pr <= pr_crit:
        # Choked Flow
        mdot = (Cd * A_valve * P_up * np.sqrt(gamma / (R_up * T_up)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
    else:
        # Subcritical Flow
        pr_eff = min(pr, 0.9999)
        mdot = (Cd * A_valve * P_up * np.sqrt(2 * gamma / (R_up * T_up * (gamma - 1)) * (pr_eff**(2/gamma) - pr_eff**((gamma+1)/gamma))))
        
    # low RPM correct as flow is never resistance free even when internal and external pressures are nearly equal.
    # NEW CODE: Smooth transition using hyperbolic tangent to avoid the 'step' glitch.
    # This removes the hard 'if < 5000' gate.
    pressure_delta = abs(P_up - P_down)
    # Scales smoothly from ~0.8 at low delta to 1.0 at high delta
    resistance_factor = 0.8 + 0.2 * np.tanh(pressure_delta / 2500.0)
    resistance_factor = 0.2 + 0.8 * np.tanh(pressure_delta / 5000.0)
    mdot *= resistance_factor
    
    # gas velocity penalty    
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
    # Raise intake limit slightly to allow 4800 RPM flow
    limit = 0.40 if is_intake else 0.60 #was 0.55 0.45 0.41
    
    if mach > limit:
        # We need a curve that is steeper than 5.0 but softer than 18.0
        # 12.0 is the mathematical 'mid-point' for a 4800 RPM roll-off gives peak at 4600
        penalty = np.clip(1.0 - 20.0 * (mach - limit)**2, 0.1, 1.0)
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

# def _calc_physics_Cd(lift, diameter, is_intake=True):
#     if lift <= 0.0005: return 0.0
#     ld_ratio = lift / diameter
    
#     if is_intake:
#         # Increase target to 0.56 to lower PMEP and flatten mid-range torque
#         target_cd = 0.56  
        
#         # Use a convex ramp (0.7) to let air in easier at low/mid lift
#         # This prevents the torque from nose-diving at 3200 RPM.
#         # ramp = min(1.0, (lift / 2.2)**0.7) 
#         # Change from 0.7 to 1.5 to simulate valve seat restriction
#         ramp = min(1.0, (lift / 2.2)**1.5)
        
#         # Subtle boost at mid-lift ld_ratio
#         boost = 0.02 * np.exp(-((ld_ratio - 0.15)**2) / 0.01)
        
#         cd = (target_cd * ramp) + boost
#         cap = 0.60 # Higher cap than before, let the Mach Penalty handle the RPM limit
#     else:
#         # Exhaust needs to be freer to pass the PMEP < 0.25 test
#         target_cd = 0.58
#         cap = 0.65
#         ramp = min(1.0, (lift / 1.5)**0.8)
#         cd = (target_cd * ramp)
        
#     return min(cd, cap)

# def _calc_physics_Cd(lift, diameter, is_intake=True):
#     if lift <= 0.0005: return 0.0
    
#     # 1. THE RAMP (Valve Curtain Restriction)
#     # Using a slightly higher exponent (1.8-2.0) keeps it restrictive at low lift.
#     # We normalize by an expected "Full Lift" (e.g., 9.0mm)
#     # norm_lift = lift / 9.0 
#     # ramp = norm_lift**2.0
    
#     # Normalize by your actual max lift (e.g., 9.5mm)
#     # This spreads the 'opening' effect across the whole cam lobe.
#     norm_lift = np.clip(lift / 6, 0.0, 1.0) # was 9.5
#     ramp = norm_lift**1.5 # 1.5 is a natural curve; 2.0 is more restrictive early
    
#     # 2. THE SATURATION (Port Restriction)
#     # Using tanh or a soft-clamping function makes the top "flat" 
#     # and prevents the sinusoidal look.
#     max_efficiency = 0.55 if is_intake else 0.58
#     cd = max_efficiency * np.tanh(ramp * 3.0) 
    
#     # 3. THE "CAP" (Hard Physical Limit)
#     return min(cd, max_efficiency)

# def _calc_physics_Cd(lift, diameter, is_intake=True):
#     """
#     Calculates the Discharge Coefficient (Cd) based on L/D ratio.
#     Reference: Heywood, Internal Combustion Engine Fundamentals.
#     """
#     if lift <= 0:
#         return 0.0
    
#     # Calculate Lift-to-Diameter ratio
#     l_d = lift / diameter
    
#     # Define Peak Cd (Standard 2-valve head values)
#     # A WBX head isn't a high-flow racing head; 0.65-0.70 is realistic.
#     peak_cd = 0.68 if is_intake else 0.63
    
#     # Empirical Cd curve fit
#     # Standard profile: starts low, rises steeply, plateaus around L/D = 0.25-0.3
#     if l_d < 0.25:
#         # Quadratic/Polynomial ramp up
#         # This creates a smooth 'S' shape rather than a linear snap
#         cd = peak_cd * (1.0 - (1.0 - (l_d / 0.25))**2)
#     else:
#         # Plateau with very slight increase for high lift
#         cd = peak_cd + (l_d - 0.25) * 0.05
    
#     # Physical cap: No production 2V head exceeds 0.75-0.80 
#     return min(cd, 0.75)

# import numpy as np

# def _calc_wbx_valve_cd(lift, diameter, is_intake=True):
#     if lift <= 0: return 0.0
#     ld = lift / diameter
    
#     if is_intake:
#         # For 8/40 ratio (max ld = 0.20)
#         peak_cd = 0.68
#         optimal_ld = 0.22 # Intake stays curtain-limited throughout
#         base_cd = 0.25
#         if ld <= optimal_ld:
#             cd = base_cd + (peak_cd - base_cd) * np.sin((ld / optimal_ld) * (np.pi / 2))
#         else:
#             cd = peak_cd * np.exp(-1.5 * (ld - optimal_ld))
#     else:
#         # For 9/34 ratio (max ld = 0.26)
#         peak_cd = 0.63
#         optimal_ld = 0.22 # Exhaust transitions to port-limited at the end of lift
#         base_cd = 0.45 #0.30    # Higher floor helps bleed the 720-deg spike
#         if ld <= optimal_ld:
#             cd = base_cd + (peak_cd - base_cd) * np.sin((ld / optimal_ld) * (np.pi / 2))
#         else:
#             cd = peak_cd * np.exp(-1.2 * (ld - optimal_ld))
            
#     return max(cd, 0.15) # .35

def _calc_wbx_valve_cd(lift, diameter, is_intake, rpm):
    if lift <= 0: return 0.0
    ld = lift / diameter
    
    if is_intake:
        peak_cd = 0.68
        optimal_ld = 0.22 
        # Lower base_cd to 0.15 forces air to 'wait' for higher lift
        base_cd = 0.15 
        if ld <= optimal_ld:
            cd = base_cd + (peak_cd - base_cd) * np.sin((ld / optimal_ld) * (np.pi / 2))
        else:
            cd = peak_cd * np.exp(-1.5 * (ld - optimal_ld))
    else:
        peak_cd = 0.63
        optimal_ld = 0.22 
        # Lowering exhaust floor from 0.45 to 0.20 reduces low-RPM over-scavenging
        base_cd = 0.20 
        if ld <= optimal_ld:
            cd = base_cd + (peak_cd - base_cd) * np.sin((ld / optimal_ld) * (np.pi / 2))
        else:
            cd = peak_cd * np.exp(-1.2 * (ld - optimal_ld))
            
    return cd # No hard floor needed if base_cd is correct

# def _calc_wbx_valve_cd(lift, diameter, is_intake, rpm):
#     # 1. Base Cd from Lift/Diameter ratio (Existing Logic)
#     # L/D = lift / diameter
#     # base_cd = ... (your existing lookup or formula)
#     base_cd = 0.65 # Placeholder for your actual base logic

#     # 2. Mean Piston Speed (MPS) scaling for WBX (76mm stroke)
#     # MPS = (2 * Stroke * RPM) / 60
#     mps = (2 * 0.076 * rpm) / 60.0
    
#     # 3. Velocity Factor: 
#     # Starts at 0.70 (lazy) at low RPM and reaches 1.0 (full efficiency) 
#     # when MPS hits ~7.1 m/s (approx 2800 RPM).
#     velocity_factor = np.clip(0.60 + (0.40 * (mps / 8.5)), 0.60, 1.0)
    
#     return base_cd * velocity_factor

# def _calc_wbx_valve_cd(lift, diameter, is_intake, rpm):
#     base_cd = 0.62 # Lowered base efficiency
#     mps = (2 * 0.076 * rpm) / 60.0
#     # Sharpen the penalty to hollow out 1000-2000 RPM
#     velocity_factor = np.clip(0.45 + (0.55 * (mps / 8.5)), 0.45, 1.0)
#     return base_cd * velocity_factor

# def _calc_wbx_valve_cd(lift, diameter, is_intake, rpm):
#     if lift <= 1e-6: return 0.0
#     ld = lift / diameter
    
#     # 1. Geometry (Physical restriction of the valve head)
#     # Peak Cd for a 2.1L WBX is roughly 0.65-0.68 for Intake
#     peak_cd = 0.65 if is_intake else 0.63
    
#     # Sine-curve for the 'Curtain Area' efficiency
#     # Efficiency peaks at ld=0.25 (Standard for poppet valves)
#     if ld <= 0.25:
#         # Starts at 0.15 to prevent 'instant gulping' at low lift
#         cd = 0.15 + (peak_cd - 0.15) * np.sin((ld / 0.25) * (np.pi / 2))
#     else:
#         # Shrouding: Slight drop as the valve moves too far into the chamber
#         cd = peak_cd * np.exp(-0.6 * (ld - 0.25))
        
#     # 2. Dynamic Velocity Penalty (The 'Wall' at high RPM)
#     # Mean Piston Speed (MPS) proxy
#     mps = (2 * 0.076 * rpm) / 60.0
#     # At 4500 RPM (mps=11.4), factor is ~0.94. At 6000, it's ~0.89.
#     velocity_penalty = np.clip(1.0 - (mps / 45.0)**2, 0.8, 1.0)
    
#     return cd * velocity_penalty


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
#     dM_d_theta = Delta_M #/ theta_delta
#     # dQ_net_rate = (Delta_Q_in / theta_delta) - Delta_Q_loss 
#     dQ_net_rate = Delta_Q_in - Delta_Q_loss 

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

# def integrate_first_law_old(
#     P_curr, T_curr, M_curr, V_curr, Delta_M, Delta_Q_in, Delta_Q_loss, 
#     dV_d_theta, theta_delta, T_manifold, R_spec_blended, gamma_blended_start,
#     cycle=None, CAD=None, substep=None
# ):
    

    
#     # 1. PREDICTOR STEP
#     # Uses the blended gamma (composition + start temp)
#     dM_d_theta = Delta_M #/ theta_delta
#     dQ_net_rate = Delta_Q_in - Delta_Q_loss 
#     T_flow_pred = T_manifold if Delta_M >= 0 else T_curr
    
#     term_heat = (gamma_blended_start - 1.0) * dQ_net_rate
#     term_work = -gamma_blended_start * P_curr * dV_d_theta
#     term_mass = gamma_blended_start * R_spec_blended * T_flow_pred * dM_d_theta
    
#     dP_d_theta_pred = (term_heat + term_work + term_mass) / V_curr
    
#     # Midpoint state prediction
#     P_mid = P_curr + dP_d_theta_pred * (0.5 * theta_delta)
#     V_mid = V_curr + dV_d_theta * (0.5 * theta_delta)
#     M_mid = M_curr + (0.5 * Delta_M)
#     T_mid = (P_mid * V_mid) / (M_mid * R_spec_blended)

#     # 2. THERMAL CORRECTION (The "Mid Value")
#     # R_spec stays the same (composition doesn't change mid-step)
#     # Gamma is updated because T has changed significantly
#     gamma_mid = get_gamma(T_mid, R_spec_blended)

#     # 3. CORRECTOR STEP
#     T_flow_corr = T_manifold if Delta_M >= 0 else T_mid
#     term_work_mid = -gamma_mid * P_mid * dV_d_theta
#     term_mass_mid = gamma_mid * R_spec_blended * T_flow_corr * dM_d_theta
    
#     dP_d_theta_final = (term_heat + term_work_mid + term_mass_mid) / V_mid
    
#     # 4. FINAL STATE
#     P_next = P_curr + dP_d_theta_final * theta_delta
#     M_next = M_curr + Delta_M
#     V_next = V_curr + dV_d_theta * theta_delta
#     T_next = (P_next * V_next) / (M_next * R_spec_blended)
    
#     return P_next, T_next

# def integrate_first_law(CAD, P_curr, T_curr, M_curr, V_curr, 
#                         Delta_Q_in,    # Joules (absolute)
#                         Delta_Q_loss,  # Joules (absolute)
#                         dV,            # m3 (absolute)
#                         Delta_M,       # kg (absolute)
#                         R_spec, T_manifold):
    
#     cv = calc_specific_heat_cv(T_curr)

#     # 1. Total Net Heat for this substep
#     dQ_substep = Delta_Q_in - Delta_Q_loss

#     # 2. Work done (P * dV)
#     dW_substep = P_curr * dV

#     # 3. Mass energy exchange (Enthalpy)
#     # If mass is entering, it brings energy; if leaving, it takes it.
#     gamma = (cv + R_spec) / cv
#     h_flow = T_manifold * (cv + R_spec) if Delta_M > 0 else T_curr * (cv + R_spec)
#     dEnthalpy = Delta_M * h_flow

#     # 4. First Law: Change in Internal Energy (dU)
#     dU = dQ_substep - dW_substep + dEnthalpy
   
#         # 1. Calculate current internal energy
#     U_curr = M_curr * cv * T_curr

#     # 2. Add the change (Net energy from Work, Heat, and Enthalpy flow)
#     U_next = U_curr + dU

#     # 3. Update Mass and Volume
#     M_next = M_curr + Delta_M
#     V_next = V_curr + dV

#     # 4. Calculate T_next based on NEW mass
#     # Note: ideally cv should be updated for T_next, but T_curr is an okay approximation
#     T_next = U_next / (M_next * cv)

#     # 5. Then calculate Pressure
#     P_next = (M_next * R_spec * T_next) / V_next

#     return P_next, T_next

def integrate_first_law(CAD, P_curr, T_curr, M_curr, V_curr, 
                        Delta_Q_in, Delta_Q_loss, dV, Delta_M, 
                        R_spec, T_manifold, lambda_):
    # 1. Get dynamic properties
    cv = calc_specific_heat_cv(T_curr, lambda_)
    cp = cv + R_spec
    
    # 2. Net Energy for this substep (Joules)
    dQ_net = Delta_Q_in - Delta_Q_loss
    dW = P_curr * dV
    
    # 3. Mass energy exchange
    # If Delta_M is positive, mass enters with T_manifold enthalpy
    # If negative, mass leaves with T_curr enthalpy
    h_flow = cp * (T_manifold if Delta_M > 0 else T_curr)
    
    # 4. THE TEMPERATURE SOLVER (Differential Form)
    # This replaces the U_next / (M * Cv) logic which is prone to error
    # Equation: m*cv*dT = dQ - P*dV + h*dm - u*dm
    u_curr = cv * T_curr
    m_next = M_curr + Delta_M
    
    # solving for dT
    dT = (dQ_net - dW + (h_flow * Delta_M) - (u_curr * Delta_M)) / (m_next * cv)
    
    T_next = T_curr + dT
    P_next = (m_next * R_spec * T_next) / (V_curr + dV)
    
    return P_next, T_next


# def get_gamma(T, R_spec):
#     """Calculates dynamic gamma (Cp/Cv)."""
#     cv = calc_specific_heat_cv(T)
#     cp = cv + R_spec
    
#     return cp / cv

def calc_specific_heat_cv(T, lambda_):
    T_clamped = max(300, min(T, 3500))
    # Revised coefficients to hit ~1350 J/kgK at 3000K
    cv = 715 + (0.22 * T_clamped) 
    
    # Keep your rich-mixture logic—it's excellent physics.
    if lambda_ < 1.0:
        f_extra = (1.0 - lambda_) * 0.068 
        cv = (cv * (1 - f_extra)) + (2500 * f_extra)
    return cv

# def calc_specific_heat_cv(T, lambda_):
#     T_clamped = max(300, min(T, 3500))
    
#     # Tuned for WBX Combustion Products (Lambda 0.85 - 1.0)
#     # Starts at ~720 at 300K, hits ~1100 at 2500K
#     cv = 680 + (0.25 * T_clamped) - (0.00003 * T_clamped**2)
    
#     # If rich, the unburned fuel acts as a heat sink
#     if lambda_ < 1.0:
#         f_extra = (1.0 - lambda_) * 0.068 # mass fraction of excess fuel
#         # Fuel vapor has much higher Cv (~2500) than exhaust (~1100)
#         cv = (cv * (1 - f_extra)) + (2500 * f_extra)
    
#     return cv

# def calc_specific_heat_cv(T):
#     """
#     Polynomial approximation of Cv based on JANAF tables.
#     Provides realistic pressure response by accurately modeling 
#     vibrational energy at high temperatures.
#     """
#     T = max(300, min(T, 3500))
    
#     # Quadratic fit: starts at ~718 J/kgK at 300K 
#     # and curves realistically toward ~980 J/kgK at 2500K.
#     cv = 660 + (0.16 * T) - (0.000025 * T**2)
#     return cv


# --- Engine Performance Functions ---


def calc_pumping_losses_nm(current_map):
    """ calculated the toruqe pumping losses """
    # Pumping Torque Calculation (Nm)
    # Formula: (Delta_Pressure * Displacement) / (4 * pi)
    # For a 2.1L (0.0021 m^3) 4-stroke engine:

    p_delta = c.P_ATM_PA - current_map  # Pressure difference in Pascals
    V_d = c.V_DISPLACED * c.NUM_CYL # Displacement in m^3

    # This is the 'Brake' applied by the vacuum
    pumping_torque_nm = (p_delta * V_d) / (4 * np.pi)

    return pumping_torque_nm

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     geom_factor = calc_piston_speed_factor(theta)
#     omega = (rpm * 2 * np.pi) / 60.0
#     v_piston_inst = abs(omega * geom_factor)
    
#     visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))

#     # Toned down for WBX 2.1L 
#     f_linear = 1.4 * v_piston_inst 
#     f_quadratic = 0.007 * (v_piston_inst**2)  # Reduced to allow 4800rpm peak
#     f_pressure = 0.000025 * max(0, p_cyl)    # Reduced to fix 63% efficiency fail
    
#     total_f_friction = (f_linear + f_quadratic + f_pressure)
    
#     # Correct Conversion to Torque
#     t_friction = total_f_friction * c.RADIUS_CRANK * abs(np.sin(np.deg2rad(theta)))
    
#     return t_friction * visc_factor


# def calc_engine_core_friction(clt, rpm):
#     visc_factor = max(1.0, 1.0 + 2.0 * np.exp(-0.05 * (clt - 40)))
    
#     # OLD: base_nm = 2.1 + (rpm/6000)
#     # NEW: Increasing static load to 5.5 Nm to soak up ~3.5 Nm of the excess torque
#     base_nm = 1.8 + (rpm/3000) 
    
#     return base_nm * visc_factor

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     # 1. Ring Tension (The Floor)
#     # This is constant. 0.8 to 1.0 Nm is realistic for a 94mm bore.
#     t_ring_tension = 0.8 # was 0.9
    
#     # 2. Viscous Drag (Oil Film)
#     # Reduce the sensitivity. Use a smaller exponent or a "clamped" scalar.
#     visc_scalar = max(1.0, (100.0 / (clt + 10.0))**0.5) 
#     t_viscous = (rpm / 3000.0) * 0.3 * visc_scalar
    
#     # 3. Pressure Loading (Rings pushed against wall by gas)
#     t_gas_load = abs(p_cyl - c.P_ATM_PA) * 2e-7 # 1.5e-7 to 3.0e-7
    
#     return t_ring_tension + t_viscous + t_gas_load

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     t_ring_tension = 0.8
    
#     visc_scalar = max(1.0, (100.0 / (clt + 10.0))**0.5) 
#     t_viscous = (rpm / 3000.0) * 0.3 * visc_scalar
    
#     # INCREASE THIS: Move from 2e-7 to 3.5e-6
#     # This represents the combined drag of all rings + piston side-loading
#     t_gas_load = abs(p_cyl - c.P_ATM_PA) * 8.5e-6 
    
#     return t_ring_tension + t_viscous + t_gas_load

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     # 1. CONSTANT: Boundary friction (Rings vs Walls)
#     t_ring_tension = 0.5 # 0.8
    
#     # 2. VISCOUS SCALAR: Oil thinning with temp
#     visc_scalar = max(1.0, (100.0 / (clt + 10.0))**0.5) 
    
#     # 3. LINEAR TERM: Oil Film Shearing (Hydrodynamic)
#     # Scales with RPM
#     # t_hydro = (rpm / 3000.0) * 0.4 * visc_scalar
#     t_hydro = (rpm / 3000.0) * 0.3 * visc_scalar
    
#     # 4. QUADRATIC TERM: Windage and Turbulence (Pumping/Churning)
#     # Scales with RPM^2. This is the 'Sound Physics' key for high RPM growth.
#     # t_windage = (rpm / 3000.0)**2 * 0.25
#     t_windage = (rpm / 3000.0)**2 * 0.4
    
#     # 5. GAS LOAD: Piston Side-Loading
#     t_gas_load = abs(p_cyl - c.P_ATM_PA) * 8.5e-6 
    
#     return t_ring_tension + t_hydro + t_windage + t_gas_load

# def calc_engine_core_friction(rpm, clt):
#     t_static = 0.9 #1.3 
    
#     # Hydrodynamic drag (Bearings)
#     visc_scalar = (100.0 / (clt + 10.0))**0.8 
#     # t_hydro = (rpm / 3000.0) * 1.0 * visc_scalar
#     t_hydro = (rpm / 3000.0) * 0.8 * visc_scalar
    
#     # Windage & Oil Pump (Parasitic) - Quadratic
#     # Even in a core test, the oil pump work increases with the square of speed.
#     t_parasitic = (rpm / 3000.0)**2 * 0.35 # 0.4 
    
#     return t_static + t_hydro + t_parasitic

# def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
#     # 1. CONSTANT
#     t_ring_tension = 0.68 # 0.51  # Shaved from 0.52
    
#     # 2. VISCOUS SCALAR
#     # Lowering exponent to 1.15 to guarantee cyl_ratio < 2.0
#     visc_scalar = max(1.0, (90.0 / (clt + 5.0))**1.15) 
    
#     # 3. LINEAR TERM
#     t_hydro = (rpm / 3000.0) * 0.35 * visc_scalar 
    
#     # 4. QUADRATIC TERM (The 'Ceiling' Killer)
#     # Shaving this term slightly to get under the 7.0Nm total
#     t_windage = (rpm / 3000.0)**2 * 0.33 # Shaved from 0.35
    
#     # 5. GAS LOAD
#     t_gas_load = abs(p_cyl - c.P_ATM_PA) * 1.0e-5 
    
#     return t_ring_tension + t_hydro + t_windage + t_gas_load

def calc_single_cylinder_friction(theta, rpm, p_cyl, clt):
    # Increase ring tension to contribute to the 40Nm drop
    t_ring_tension = 1.1 
    
    # Quadratic scaling for piston speed (MPS^2 proxy)
    t_viscous = ((rpm / 4000.0)**2) * 1.0
    
    # Keep Gas Load as a polynomial 'B' term
    t_gas_load = abs(p_cyl - c.P_ATM_PA) * 0.8e-5 
    
    return t_ring_tension + t_viscous + t_gas_load

def calc_engine_core_friction(rpm, clt):
    # 1. Increase Static Floor to 5.5 to drop base torque by ~30-40Nm
    t_static = 2.5 #5.5 
    
    visc_scalar = (70.0 / (clt + 5.0))**1.2 
    
    # 2. Hydrodynamic: Use a 0.8 exponent. 
    # This provides a steady but non-explosive growth in the mid-range.
    # t_hydro = ((rpm / 3000.0)**0.8) * 2.5 * visc_scalar 
    t_hydro = ((rpm / 3200.0)**0.7) * 2.0 * visc_scalar 
    
    # 3. Parasitic (Quadratic): Increase this to 'kill' the torque after 3000 RPM.
    # Increasing from 0.5 to 1.8 creates the aggressive high-end ramp you requested.
    t_parasitic = (rpm / 5500.0)**2 * 1.8
    # t_parasitic = ((rpm / 4500.0)**2) * 2.5
    
    return t_static + t_hydro + t_parasitic

# def calc_engine_core_friction(rpm, clt):
#     # 1. Increase Static Floor to 5.5 to drop base torque by ~30-40Nm
#     t_static = 2.5 #5.5 
    
#     visc_scalar = (70.0 / (clt + 5.0))**1.2 
    
#     # 2. Hydrodynamic: Use a 0.8 exponent. 
#     # This provides a steady but non-explosive growth in the mid-range.
#     # t_hydro = ((rpm / 3000.0)**0.8) * 2.5 * visc_scalar 
#     t_hydro = ((rpm / 3000.0)**0.8) * 2.5 * visc_scalar 
    
#     # 3. Parasitic (Quadratic): Increase this to 'kill' the torque after 3000 RPM.
#     # Increasing from 0.5 to 1.8 creates the aggressive high-end ramp you requested.
#     # t_parasitic = (rpm / 3000.0)**2 * 1.8
#     t_parasitic = ((rpm / 3000.0)**2) * 2.5
    
#     return t_static + t_hydro + t_parasitic


# def calc_engine_core_friction(rpm, clt):
#     # 1. Static/Boundary Friction (Seals and initial shear)
#     # Increase from 1.35 to 2.8. This anchors your FMEP floor.
#     t_static = 3.10 #2.80 
    
#     # 2. Hydrodynamic Friction (Bearings and Oil Film)
#     # 90.0 / (clt + 5.0) is a good thermal curve.
#     # Increase the multiplier from 0.8 to 2.2.
#     visc_scalar = (90.0 / (clt + 5.0))**1.4 
#     # t_hydro = (rpm / 3000.0) * 2.2 * visc_scalar 
#     t_hydro = ((rpm / 3000.0)**0.7) * 2.8 * visc_scalar
    
#     # 3. Parasitic Friction (Oil pump, Water pump, Windage)
#     # Increase from 0.4 to 1.2. The WBX oil pump is a massive gear-driven unit.
#     # t_parasitic = (rpm / 3000.0)**2 * 1.20 
#     t_parasitic = (rpm / 3000.0)**2 * 0.4
    
#     return t_static + t_hydro + t_parasitic

# def calc_engine_core_friction(rpm, clt):
#     t_static = 1.35 # 1.0 
    
#     # Keep core sensitivity slightly higher (1.4) as bearings are very temp dependent
#     visc_scalar = (90.0 / (clt + 5.0))**1.4 
#     t_hydro = (rpm / 3000.0) * 0.8 * visc_scalar 
#     t_parasitic = (rpm / 3000.0)**2 * 0.40 
    
#     return t_static + t_hydro + t_parasitic

# def calc_engine_core_friction(rpm, clt):
#     # Base drag for valvetrain/seals (Constant)
#     t_static = 1.3 # was 1.5 
    
#     # Hydrodynamic drag for bearings (Temp Sensitive)
#     # This is where the 3.0x ratio should live.
#     visc_scalar = (100.0 / (clt + 10.0))**0.8 
#     t_hydro = (rpm / 3000.0) * 1.0 * visc_scalar
    
#     return t_static + t_hydro

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

# def calc_wiebe_heat_rate(theta, theta_start, duration, total_heat_J):
#     """
#     Calculates the instantaneous heat release (Joules/degree) using the Wiebe function.
#     """
#     # a=5.0 and m=2.0 are standard for spark-ignition engines
#     a = 5.0
#     m = 2.0
    
#     # Normalized progress through the burn (0.0 to 1.0)
#     delta_theta = theta - theta_start
    
#     if delta_theta < 0 or delta_theta > duration:
#         # # Add this inside the active burn window logic
#         # if total_heat_J > 0:
#         #     print(f"DEBUG_BURN | θ:{theta:05.1f} | "
#         #         f"Burn_Progress:{y:4.2f} | "
#         #         f"dQ_this_deg:{delta_theta * c.THETA_DELTA:6.2f}J | "
#         #         f"Total_Expected:{total_heat_J:6.2f}J")
#         return 0.0
    
#     # Normalized position
#     y = delta_theta / duration
    
#     # Wiebe Mass Fraction Burned (MFB) derivative: dXb/dtheta
#     # This gives us the fraction of total heat released during THIS degree
#     term1 = a * (m + 1) / duration
#     term2 = y**m
#     term3 = np.exp(-a * y**(m + 1))
    
#     dxb_dtheta = term1 * term2 * term3
    
#     return dxb_dtheta * total_heat_J

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
    T_wall_raw = (alpha * target_T_wall) + ((1 - alpha) * previous_T_wall)
    
    new_t_wall = np.clip(T_wall_raw, current_clt_C + 273.15, 600.0)
    
    return new_t_wall


def get_burn_duration(rpm, lambda_):
    BASE_DURATION = 42.0  # Dropped from 55.0 to 42.0
    REF_RPM = 3000.0

    # Softer exponent (-0.08) prevents the 'Idle Stretch'
    f_rpm = (max(rpm, 200) / REF_RPM)**-0.08 

    # Keep your existing Lambda scaling
    f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

    # Clip to ensure high-rpm safety
    return np.clip(BASE_DURATION * f_rpm * f_lambda, 35.0, 80.0)


# def get_burn_duration(rpm, lambda_):
#     # 1. Increase Base: 55 degrees is the 'sweet spot' for a 94mm bore 
#     # to keep Peak P under 65 bar at 3000-4500 RPM.
#     BASE_DURATION = 55.0 
#     REF_RPM = 3000.0

#     # 2. RPM Factor:
#     # Turbulence helps, but a 94mm flame path is long.
#     # f_rpm = (max(rpm, 200) / REF_RPM)**-0.2
#     f_rpm = (max(rpm, 200) / REF_RPM)**-0.15 # Softer scaling for idle

#     # 3. Lambda Factor: Physically sound.
#     f_lambda = 1.0 + 2.2 * (lambda_ - 0.9)**2.0

#     # 4. Dynamic Clipping:
#     # We raise the floor to 42.0. This ensures that even at 5000 RPM, 
#     # the burn doesn't 'accelerate' into an engine-breaking pressure spike.
#     return np.clip(BASE_DURATION * f_rpm * f_lambda, 42.0, 90.0)

# def update_intake_manifold_pressure(current_map, effective_tps, rpm, crank_angle, iat_k, dV_list):
#     """
#     Dynamic Filling and Emptying Model.
    
#     Args:
#         current_map: Current P_manifold_Pa from previous step
#         effective_tps: Throttle position (0-100) including idle air
#         rpm: Engine speed
#         crank_angle: Current CAD (0-719)
#         iat_k: Intake Air Temperature in Kelvin
#         dt: Time step in seconds (1/6*rpm)
#     """
#     # 1. CONSTANTS
#     V_manifold = c.V_DISPLACED # 2.1 Liters plenum volume (typical for WBX)
#     R_air = c.R_SPECIFIC_AIR
#     P_amb = c.P_ATM_PA # Ambient pressure
    
#     dt = 1.0 / (6.0 * max(0.1, rpm))
    
#     # 2. FLOW IN (Through Throttle)
#     # Area-based flow. 0 TPS = 0 Flow (RL-Proof Stall)
#     throttle_area = effective_tps / 100.0
#     # Pressure delta drives flow into the manifold
#     dp_in = max(0, P_amb - current_map)
#     # Constant tuned so WOT recovery matches real-world response times
#     mass_in = throttle_area * np.sqrt(dp_in) * 0.0015 * dt 

#     # 3. FLOW OUT (Into Cylinders)
#     # We sum the dV of any cylinder currently on an intake stroke
#     total_dv = 0.0
#     # A 4-cylinder Boxer has intake pulses every 180 degrees
#     for offset in [0, 180, 360, 540]:
#         cyl_cad = (crank_angle + offset) % 720
#         # Intake stroke is roughly 0 to 180 degrees
#         if 0 <= cyl_cad < 180:
#             # We need the dV for this specific 1-degree movement
#             # In your model, this comes from the slider-crank v_array
#             # For simplicity here: total_dv = cylinder_dv_at_this_angle
#             total_dv += dV_list[cyl_cad]

#     # Mass = Density * Volume
#     rho_manifold = current_map / (R_air * iat_k)
#     mass_out = rho_manifold * total_dv

#     # 4. STATE UPDATE (Ideal Gas Law Derivative)
#     # dP = (dm * R * T) / V
#     dm = mass_in - mass_out
#     dp = (dm * R_air * iat_k) / V_manifold
    
#     new_map = current_map + dp
    
#     # Return new pressure, no clipping. If it hits 0, the engine stalls.
#     return max(0.0, new_map)

# def update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, total_dv):
#     R_air = 287.05
#     V_manifold = 0.0021 
#     dt = 1.0 / (6.0 * max(0.1, rpm))
    
#     # 1. EMPTYING (The Geometric "Suction")
#     rho_manifold = current_map / (R_air * iat_k)
#     mass_out = rho_manifold * total_dv 

#     # 2. FILLING (The Temporal "Leak")
#     # To fix the "Zero Delta" problem in a single-step test, 
#     # we use the P_amb vs current_map. 
#     # But since current_map == P_amb at start, let's use a 
#     # small epsilon or the pressure AFTER suction to see the delta.
    
#     # PHYSICAL TRUTH: Flow is driven by the delta. 
#     # Let's assume the throttle sees the vacuum created by the piston movement.
#     estimated_vac = max(10, c.P_ATM_PA - (current_map - 882)) # Use a representative delta
    
#     # Increase this constant to 0.8 for the WBX throttle body scale
#     m_dot_in = (effective_tps / 100.0) * np.sqrt(estimated_vac) * 0.8
#     mass_in = m_dot_in * dt 

#     # 3. INTEGRATION
#     dm = mass_in - mass_out
#     dp = (dm * R_air * iat_k) / V_manifold
    
#     return current_map + dp

# def update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, total_dv):
#     """
#     Standard Filling-and-Emptying Model.
    
#     Category: 
#     - c. constants: Design specs (Volumes, Areas, Gas Constants)
#     - dynamic args: State variables (RPM, IAT, MAP)
#     """
#     THROTTLE_FLOW_COEFF = 0.012
    
#     # 1. TIME STEP CALCULATIONS
#     # Time for 1 degree of crank rotation
#     dt = 1.0 / (6.0 * max(0.1, rpm)) 

#     # 2. FILLING (Mass In from Throttle)
#     # Flow is driven by the pressure delta between Atmosphere and Manifold
#     delta_p = c.P_ATM_PA - current_map
    
#     # Standard Orifice Equation: Area * Velocity * Density
#     # c.THROTTLE_FLOW_COEFF should be tuned to the physical 50mm plate flow capacity
#     throttle_area_ratio = effective_tps / 100.0
#     m_dot_in = throttle_area_ratio * np.sign(delta_p) * np.sqrt(abs(delta_p)) * THROTTLE_FLOW_COEFF
#     mass_in = m_dot_in * dt

#     # 3. EMPTYING (Mass Out to Cylinders)
#     # Gas Density in the manifold (kg/m^3)
#     rho_manifold = current_map / (c.R_SPECIFIC_AIR * iat_k)
#     # Mass = Density * Volume Displaced
#     mass_out = rho_manifold * total_dv

#     # 4. PRESSURE DERIVATIVE (Ideal Gas Law)
#     # dP = (dm * R * T) / V
#     dm_net = mass_in - mass_out
#     dp = (dm_net * c.R_SPECIFIC_AIR * iat_k) / c.V_INTAKE_MANIFOLD
    
#     # Return updated state (capped to prevent vacuum lower than absolute zero)
#     return max(100.0, current_map + dp)

# def update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, dm_in_engine):
#     """
#     Revised Filling-and-Emptying Model using Direct Mass.
    
#     Args:
#         current_map: Manifold pressure (Pa) from previous step
#         effective_tps: Throttle opening (%)
#         rpm: Engine speed
#         iat_k: Intake Air Temp (K)
#         dm_in_engine: Actual mass (kg) sucked out by all cylinders in the last degree
#     """
#     THROTTLE_FLOW_COEFF = 0.0020 # 0.012 # Tune this so MAP hits ~100kPa at WOT/high RPM
    
#     # 1. TIME STEP
#     dt = 1.0 / (6.0 * max(0.1, rpm)) 

#     # 2. FILLING (Mass In from Atmosphere)
#     # Pushing air into the manifold
#     delta_p = c.P_ATM_PA - current_map
    
#     # Simple orifice flow for throttle
#     throttle_area_ratio = effective_tps / 100.0
#     m_dot_in = throttle_area_ratio * np.sign(delta_p) * np.sqrt(abs(delta_p)) * THROTTLE_FLOW_COEFF
#     mass_in = m_dot_in * dt

#     # 3. EMPTYING (Mass Out to Cylinders)
#     # We no longer calculate mass_out via density * dV. 
#     # We use the actual kg provided by the cylinder flow physics.
#     mass_out = dm_in_engine

#     # 4. PRESSURE DERIVATIVE (Ideal Gas Law)
#     # PV = mRT -> P = (m * R * T) / V
#     # dP = (dm_net * R * T) / V_manifold
#     dm_net = mass_in - mass_out
    
#     # Critical: Use the VOLUME of the manifold (Plenum + Runners), not displacement!
#     dp = (dm_net * c.R_SPECIFIC_AIR * iat_k) / c.V_INTAKE_MANIFOLD
    
#     # 5. INTEGRATION
#     # 100 Pa floor prevents math from breaking (absolute vacuum)
#     return max(100.0, current_map + dp)


def update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, dm_in_engine):
    # Constants for Air
    GAMMA = c.GAMMA_AIR
    R = c.R_SPECIFIC_AIR
    P_AMB = c.P_ATM_PA
    T_AMB = iat_k 
    
    # 1. CALCULATE EFFECTIVE AREA (A_eff)
    # Physically: Bore area * sin(angle) or similar. 
    # For now, let's use the actual geometric area of a 50mm throttle
    throttle_dia = 0.050 # 50mm
    max_area = (np.pi * (throttle_dia**2)) / 4.0
    A_geom = max_area * (effective_tps / 100.0)
    
    CD = _get_dynamic_throttle_cd(effective_tps)
    A_eff = A_geom * CD

    # 2. ISENTROPIC FLOW (Filling)
    # Use absolute pressure ratio to determine flow direction
    pr = current_map / P_AMB 

    # Handle backflow (Manifold > Ambient)
    if pr > 1.0:
        # Technically, you should swap the pressures and calc flow OUT of the manifold
        # For a simple manifold model, we can just treat it as zero or reverse subsonic
        m_dot_in = 0.0 # Simplest guard
        # Or implement reverse flow logic here if needed
    else:
        pr_crit = (2 / (GAMMA + 1))**(GAMMA / (GAMMA - 1))

        if pr <= pr_crit:
            # --- CHOKED FLOW ---
            term_val = GAMMA * (2/(GAMMA+1))**((GAMMA+1)/(GAMMA-1))
            m_dot_in = (A_eff * P_AMB / np.sqrt(R * T_AMB)) * np.sqrt(term_val)
        else:
            # --- SUBSONIC FLOW ---
            # Added np.maximum(0, ...) to guard against tiny negative floating point errors
            inner_val = (pr**(2/GAMMA) - pr**((GAMMA+1)/GAMMA))
            term = np.sqrt((2*GAMMA / (GAMMA-1)) * np.maximum(0, inner_val))
            m_dot_in = (A_eff * P_AMB / np.sqrt(R * T_AMB)) * term

    # 3. INTEGRATION (Per Degree)
    # dt for 1 degree at current RPM
    dt = 1.0 / (6.0 * max(0.1, rpm)) 
    
    mass_in = m_dot_in * dt
    mass_out = dm_in_engine # Mass sucked out by cylinder logic this degree
    
    dm_net = mass_in - mass_out
    
    # Ideal Gas Law for Pressure Change
    dp = (dm_net * R * T_AMB) / c.V_INTAKE_MANIFOLD
    
    return max(100.0, current_map + dp)


def _get_dynamic_throttle_cd(tps):
    # tps is 0-100
    if tps < 6.0:
        # IACV dominant range: high restriction, low efficiency
        # Linear interpolation from 0.2 to 0.45
        return 0.28 + (0.45 - 0.2) * (tps / 6.0)
    else:
        # Main throttle dominant: efficiency increases to a peak
        # Linear interpolation from 0.45 at 6% to 0.65-0.70 at WOT
        return 0.45 + (0.68 - 0.45) * ((tps - 6.0) / 94.0)

def update_exhaust_pressure(rpm, ambient):
    # Refined WBX Exhaust Backpressure Model
    # 101325 is base atmospheric. 
    # 1.8e-5 is a tuned constant for a stock WBX-2.1L muffler.
    # At 3000 RPM, this adds ~1.6 kPa of backpressure.
    # p_exhaust = ambient + (1.8e-5 * (rpm**2))
    p_exhaust = 101325.0 + (4.0e-5 * (rpm**2))
    return p_exhaust


# def calculate_combustion_dq(cad, substep_idx, substep_size, cyl_state, lambda_):
#     """
#     Calculates the heat release (dQ) for a specific degree/substep.
#     """
#     # 1. Determine Wiebe Slice
#     theta_start = cad + (substep_idx * substep_size)
#     theta_next = theta_start + substep_size
    
#     f1 = calc_wiebe_fraction(theta_start, cyl_state.ignition_start_theta, 
#                              cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
#     f2 = calc_wiebe_fraction(theta_next, cyl_state.ignition_start_theta, 
#                              cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
#     step_fraction = max(0.0, f2 - f1)

#     # 2. Dynamic Efficiency
#     # Increased to 0.97 for the lambda power window to help hit 65 bar target
#     eff = 0.97 if 0.85 <= lambda_ <= 1.05 else 0.85
    
#     # 3. Energy release limited by the limiting reactant
#     # Proportional burn of the mass present at the time of spark
#     dm_fuel_potential = cyl_state.fuel_mass_at_spark * step_fraction
#     dm_air_potential  = cyl_state.air_mass_at_spark  * step_fraction

#     # Clamp by remaining cylinder contents
#     actual_fuel_burned = min(cyl_state.fuel_mass_kg, dm_fuel_potential)
#     actual_air_burned  = min(cyl_state.air_mass_kg, dm_air_potential)

#     # 4. Stoichiometric Heat Calculation (Rich vs Lean)
#     if lambda_ < 1.0:
#         # Rich: Limited by available Oxygen (Air)
#         dq = (actual_air_burned / 14.7) * c.LHV_FUEL_GASOLINE * eff
#     else:
#         # Lean: Limited by available Fuel
#         dq = actual_fuel_burned * c.LHV_FUEL_GASOLINE * eff

#     return dq, actual_fuel_burned, actual_air_burned


def calculate_combustion_dq(cad, substep_idx, substep_size, cyl_state, lambda_):
    # 1. Determine Wiebe Slice (Keep your stable f2 - f1 logic)
    theta_start = cad + (substep_idx * substep_size)
    theta_next = theta_start + substep_size
    
    f1 = calc_wiebe_fraction(theta_start, cyl_state.ignition_start_theta, 
                             cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
    f2 = calc_wiebe_fraction(theta_next, cyl_state.ignition_start_theta, 
                             cyl_state.burn_duration, a_vibe=5.0, m_vibe=cyl_state.m_vibe)
    step_fraction = max(0.0, f2 - f1)

    # 2. Stoichiometry & Mass Limitation
    dm_fuel_potential = cyl_state.fuel_mass_at_spark * step_fraction
    dm_air_potential  = cyl_state.air_mass_at_spark  * step_fraction

    actual_fuel_burned = min(cyl_state.fuel_mass_kg, dm_fuel_potential)
    actual_air_burned  = min(cyl_state.air_mass_kg, dm_air_potential)

    # 3. Gross Chemical Energy Release
    # Use a standard efficiency; dissociation is handled separately below
    base_eff = 0.97 if 0.85 <= lambda_ <= 1.05 else 0.88
    
    if lambda_ < 1.0:
        dq_gross = (actual_air_burned / 14.7) * c.LHV_FUEL_GASOLINE * base_eff
    else:
        dq_gross = actual_fuel_burned * c.LHV_FUEL_GASOLINE * base_eff

    # 4. Thermal Brakes (The "Sound Physics" Additions)
    
    # A) Dissociation (Move it HERE, not in EngineModel)
    # As T rises, the effective energy release drops because molecules break apart
    t_factor = 1.0
    # if cyl_state.T_curr > 2300: # Lowered threshold to catch the rise earlier
    #     # t_factor = max(0.75, 1.0 - (cyl_state.T_curr - 2300) / 2000.0)
    #     T_norm = (cyl_state.T_curr - 2300) / 700.0  # reaches 1.0 at 3000K
    #     t_factor = max(0.82, 1.0 - (0.18 * T_norm**2)) # Up to 18% energy absorption
    # if cyl_state.T_curr > 2400:
    #     # A 12% reduction at 2900K is physically standard
    #     t_factor = 1.0 - (min(0.12, (cyl_state.T_curr - 2400) / 4000.0))
    #     dq_gross *= t_factor
    # Inside calculate_combustion_dq step 4A
    # if cyl_state.T_curr > 2200:
    #     # Use a steeper curve for dissociation energy absorption
    #     T_excess = max(0, cyl_state.T_curr - 2200)
    #     # At 2800K (600 excess), factor is ~0.70 (30% energy absorption)
    #     t_factor = 1.0 - (T_excess / 2000.0)**1.5 
    #     t_factor = max(0.65, t_factor)
    if cyl_state.T_curr > 2450:
        # We only start 'braking' when we are actually near the limit.
        # This keeps the early P-V work high (cooling the gas via expansion).
        T_excess = max(0, cyl_state.T_curr - 2450)
        
        # Use a very aggressive power (3.0) to 'Wall' the temperature.
        # At 2750K (300 excess), factor is 1.0 - (300/500)^3 = 0.784 (21% drop)
        t_factor = 1.0 - (T_excess / 500.0)**3.0
        t_factor = max(0.65, t_factor)

    
    dq_after_dissociation = dq_gross * t_factor

    # B) Vaporization Cooling (Latent Heat)
    # Every bit of fuel entering the 'burn' must vaporize
    q_vaporization = dm_fuel_potential * 350000.0

    # C) Rich Mixture Heating (Sensible Heat)
    # If rich, calculate the heating cost of the excess fuel in THIS slice
    excess_fuel_in_slice = max(0.0, dm_fuel_potential - actual_fuel_burned)
    # Fuel vapor Cp is ~2500 J/kgK. Heating from port temp (~350K) to T_curr
    q_rich_heating = excess_fuel_in_slice * 2500.0 * (cyl_state.T_curr - 350.0)

    # 5. Final Net dQ
    dq_net = dq_after_dissociation - q_vaporization - q_rich_heating

    return dq_net, actual_fuel_burned, actual_air_burned


# def calc_physical_m_vibe(rpm, lambda_val, residual_fraction=0.10):
#     """
#     Calculates m_vibe based on mechanical turbulence (RPM), 
#     chemical mixture (Lambda), and dilution (Residuals).
#     """
#     # 1. Base Mechanical Shape (Proxy for Turbulence/Piston Speed)
#     #    Keep the burn 'fast' (2.1) through the peak, then 
#     # ramp up sharply only AFTER 4800 RPM.
#     rpm_points = [1000, 2000, 3500, 4800, 5500]
#     m_base_pts = [3.8,  3.1,  2.1,  2.1,  3.8]
#     m_base = np.interp(rpm, rpm_points, m_base_pts)

#     # 2. Lambda Penalty (Chemical Speed)
#     # Flame speed peaks at ~0.85-0.90. It slows down significantly as you go lean.
#     # We use a quadratic penalty centered at 0.88.
#     lambda_ref = 0.88
#     # If lambda is 1.0 (Stoich), penalty is ~1.1x. If 1.2 (Lean), penalty is ~1.5x.
#     k_lambda = 1.0 + 3.0 * (max(0, lambda_val - lambda_ref)**2)

#     # 3. Residual Penalty (Dilution/EGR)
#     # High residuals at low RPM/load 'dampen' the flame front.
#     # Normal residuals are ~10%. If they spike to 20%, m_vibe increases.
#     k_res = 1.0 + (residual_fraction * 2.0)

#     # Final Combined m_vibe
#     m_final = m_base * k_lambda * k_res
    
#     # Physical safety clamp for the Wiebe function stability
#     return np.clip(m_final, 1.2, 5.0)

def calc_physical_m_vibe(rpm, lambda_val, residual_fraction=0.10):
    # 1. Base Mechanical Shape
    # Standardize around 2.0 for the bulk of the power band.
    # Higher m at low RPM (slow piston speed = lazy flame start)
    rpm_points = [1000, 2000, 3000, 5000]
    m_base_pts = [2.0, 1.9, 1.8, 2.2] # Linear/Crisper start at idle
    # m_base_pts = [2.5,  2.1,  1.8,  2.2] 
    m_base = np.interp(rpm, rpm_points, m_base_pts)

    # 2. Lambda Penalty (Chemical Speed)
    # Use a softer scaling. Flame speed doesn't change the SHAPE 
    # as much as it changes the DURATION.
    lambda_ref = 0.88
    # We only want to bump m_vibe significantly if we go very lean.
    m_lambda_offset = 1.5 * (max(0, lambda_val - lambda_ref)**2)

    # 3. Residual Penalty
    # Offset rather than multiplier
    m_res_offset = residual_fraction * 1.5

    # Final m_vibe calculation (Base + Offsets)
    m_final = m_base + m_lambda_offset + m_res_offset
    
    return np.clip(m_final, 1.1, 4.0)


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