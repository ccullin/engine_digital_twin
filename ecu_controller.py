# ecu_controller.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin


import sys
from engine_model import FixedKeyDictionary
import constants as c
import numpy as np

# REQUIRED: Import necessary for 2D map interpolation (Spark/AFR tables)
from scipy.interpolate import RegularGridInterpolator


class ECUController:
    """
    Simulates the Engine Control Unit (ECU).
    It determines control outputs (Spark, AFR, Idle Valve Position) based on sensor inputs.
    """

    def __init__(self):

        # State for PID Idle Control
        self.idle_integral = 0.0
        self.last_error = 0.0
        self.idle_target_rpm = c.IDLE_RPM

        # --- ECU Standard Outputs ---
        self.spark_active = False
        self.spark_advance_btdc = 0.0
        self.injector_is_active = False  # NEW: Boolean flag for twin
        self.afr_target = 0.0
        self.idle_valve_position = 0.0
        self.fuel_cut_active = False  # Flag for Deceleration Fuel Cut-Off (DFCO)
        self.trapped_air_mass_kg = 0.0
        self.ve_fraction = 0.0

        # --- Fuel System Outputs ---
        # self.fuel_mass_mg = 0.0                 # Mass of fuel required (mg)
        # self.injector_pw_msec = 0.0      # Final command to injector (ms)
        self.injector_end_timing_degree = 360.0
        self.injector_start_timing_degree = 0.0  # Store SOI angle

        # other
        # self._remaining_pw_ms = 0.0      # NEW: Internal counter for duration tracking
        # self.injector_start_deg = 0.0    # NEW: Store SOI angle

        # --- Crank and Cam timing ---
        self.crank_tooth = 0
        self.cam_sync = False
        self.crank_sync = False
        self.calculated_theta = (
            0.0  # will be crank * cam 0-719 in 10 degree increments.
        )
        self.last_calculated_theta = 0
        self.first_crank_rotation = True
        self.cycle = 1

        # --- for internal debugging
        self._required_fuel_g = 0

        self.output_dict = FixedKeyDictionary(
            {
                "spark": self.spark_active,
                "afr_target": self.afr_target,
                "idle_valve_position": self.idle_valve_position,
                "trapped_air_mass_kg": self.trapped_air_mass_kg,
                "ve_fraction": self.ve_fraction,
                "injector_on": self.injector_is_active,
                # 'injector_pulse_width_ms'   : self.injector_pulse_width_ms,
                # 'injector_timing_deg'       : self.injector_timing_deg,
                "fuel_cut_active": self.fuel_cut_active,
            }
        )

        # --- ECU Look-Up Table Definitions (6x6) -------------------------------------------------------------------------
        RPMS = [100, 600, 2000, 4000, 6000, 8000]
        MAPS = [30, 50, 75, 95, 105, 150]

        # --- AFR (AIR-FUEL RATIO) ------------------------------------------------------------------------------------------
        # CRITICAL FIX: The 100 RPM row is set to 8.0:1 (very rich) to ensure the engine gets enough fuel to start.
        afr_data = np.array(
            [
                [
                    11.5,
                    11.5,
                    11.5,
                    11.5,
                    11.5,
                    11.5,
                ],  # NEW: 100 rpm (CRANKING FIX: Very rich)
                [11.5, 14.7, 14.7, 14.7, 14.0, 14.0],  # 600 rpm (Idle/Low Load)
                [14.7, 14.7, 14.7, 13.5, 13.0, 13.0],  # 2000 rpm
                [14.7, 14.7, 14.0, 12.8, 12.5, 12.5],  # 4000 rpm
                [14.7, 14.0, 13.0, 12.5, 12.2, 12.2],  # 6000 rpm
                [14.0, 13.5, 12.5, 12.0, 12.0, 12.0],  # 8000 rpm
            ]
        )
        self.afr_interp = RegularGridInterpolator(
            (RPMS, MAPS), afr_data, bounds_error=False, fill_value=None
        )

        # --- VE (VOLUMETRIC EFFICIENCY)-------------------------------------------------------------------------------------
        # Extrapolated the 600 RPM data for the new 150 kPa column.
        ve_data = np.array(
            [
                [100, 100, 100, 100, 100, 100],  # NEW: 100 rpm (Low Cranking VE)
                [70, 50, 60, 70, 75, 75],  # 600 rpm
                [50, 70, 85, 95, 100, 100],  # 2000 rpm
                [55, 80, 95, 102, 105, 105],  # 4000 rpm
                [60, 75, 90, 98, 102, 102],  # 6000 rpm
                [50, 65, 80, 90, 95, 95],  # 8000 rpm
            ]
        )
        self.ve_interp = RegularGridInterpolator(
            (RPMS, MAPS), ve_data, bounds_error=False, fill_value=None
        )

        # --- SPARK (BTDC)---------------------------------------------------------------------------------------------------
        # CRITICAL FIX: The 100 RPM row is set to 5 degrees BTDC to prevent the starter from fighting combustion pressure.
        spark_data = np.array(
            [
                [8, 8, 8, 8, 8, 8],  # NEW: 100 rpm (CRANKING FIX: Safe Retarded Spark)
                [
                    8,
                    30,
                    25,
                    20,
                    18,
                    15,
                ],  # 600 rpm (Extrapolated the 150 kPa column as retarded for load)
                [38, 33, 28, 23, 21, 18],  # 2000 rpm
                [42, 37, 32, 27, 25, 22],  # 4000 rpm
                [38, 33, 28, 25, 23, 20],  # 6000 rpm
                [35, 30, 25, 23, 21, 18],  # 8000 rpm
            ]
        )
        self.spark_interp = RegularGridInterpolator(
            (RPMS, MAPS), spark_data, bounds_error=False, fill_value=None
        )

        # --- INJECTION (degrees) -------------------------------------------------------------------------------------------
        injector_timing_data = np.array(
            [
                [150, 140, 120, 100, 90, 80],  #  100 RPM
                [155, 140, 120, 100, 90, 80],  #  600 RPM
                [160, 150, 130, 110, 100, 90],  # 2000 RPM
                [170, 160, 150, 130, 120, 110],  # 4000 RPM
                [180, 170, 160, 150, 140, 130],  # 6000 RPM
                [190, 180, 170, 160, 150, 140],  # 8000 RPM
            ]
        )
        self.inj_timing_interp = RegularGridInterpolator(
            (RPMS, MAPS), injector_timing_data, bounds_error=False, fill_value=None
        )

    # -----------------------------------------------------------------------------------------

    def get_outputs(self):
        self.output_dict.update(
            {
                "spark": self.spark_active,
                "afr_target": self.afr_target,
                "idle_valve_position": self.idle_valve_position,
                "trapped_air_mass_kg": self.trapped_air_mass_kg,
                "ve_fraction": self.ve_fraction,
                "injector_on": self.injector_is_active,
                # 'injector_pulse_width_ms'   : self.injector_pulse_width_ms,
                # 'injector_timing_deg'       : self.injector_timing_deg,
                "fuel_cut_active": self.fuel_cut_active,
            }
        )

        return self.output_dict

    # -----------------------------------------------------------------------------------------
    def update(self, sensors):
        """
        ECU update — called every crank degree.
        This is the single source of truth for all control outputs.
        """
        # =================================================================
        # 1. Sensor inputs
        # =================================================================
        RPM = sensors["RPM"]
        MAP_kPa = sensors["MAP_kPa"]
        TPS = sensors["TPS_percent"]
        CLT_C = sensors["CLT_C"]
        crank_pos = sensors["crank_pos"]
        cam_pulse = sensors["cam_sync"]

        current_rpm_safe = max(RPM, 10.0)

        # =================================================================
        # Engine Timing Sync
        # =================================================================

        cam_rising_edge = False
        cam_rising_edge = not cam_rising_edge and cam_pulse

        if cam_rising_edge:
            self.cam_sync = True

        if self.cam_sync and crank_pos == 0:
            self.crank_sync = True

        # if self.crank_sync and crank_pos == 0 and self.first_crank_rotation:  # then first time though and crank at theta720=0
        #     self.last_calculated_theta = 0    # only included for clarity, it is already 0
        #     self.calculated_theta = 0         # redundant statement but added for clarity
        #     self.first_crank_rotation = False
        # elif self.crank_sync:
        if self.crank_sync:
            self.last_calculated_theta = self.calculated_theta
            self.calculated_theta = (self.calculated_theta + 1) % 720.0
            if self.calculated_theta % 720 == 0:
                self.cycle += 1

        # =================================================================
        # 2. Idle valve control (ECU owns this)
        # =================================================================
        idle_pos = self._idle_pid(TPS, RPM, CLT_C)
        self.idle_valve_position = idle_pos

        # =================================================================
        # 3. Effective throttle = driver pedal + idle valve
        # =================================================================
        effective_tps = np.clip(TPS + idle_pos, 0.0, 100.0)

        # =================================================================
        # 4. Spark advance (2D table)
        # =================================================================

        self.spark_advance_btdc = float(self.spark_interp([RPM, MAP_kPa]))
        # Convert to 720° domain for engine
        spark_fire_angle_720 = 360.0 - self.spark_advance_btdc

        # Only fire spark at the exact degree
        spark_this_degree = (
            self.last_calculated_theta < spark_fire_angle_720 <= self.calculated_theta
        )

        # print(f"{self.last_calculated_theta:3.3f} <= {spark_fire_angle_720:3.3f} <= {self.calculated_theta:3.3f}  sparke_now={spark_this_degree:3.3f}")

        if spark_this_degree and not self.fuel_cut_active:
            self.spark_active = True
        else:
            self.spark_active = False

        # =================================================================
        # 5. Target AFR (2D table + WOT enrichment)
        # =================================================================
        afr_target = float(self.afr_interp([RPM, MAP_kPa]))
        # should the WOT enrichment be retained.  This may complicate RL
        # if effective_tps > 90.0:
        #     afr_target = np.clip(afr_target, 11.8, 12.8)
        self.afr_target = afr_target

        # =================================================================
        # 6. Volumetric Efficiency (2D table)
        # =================================================================
        self.ve_fraction = float(self.ve_interp([RPM, MAP_kPa])) / 100

        # =================================================================
        # 7. Trapped air mass — speed-density (per cylinder)
        # =================================================================
        self.trapped_air_mass_kg = self._calculate_trapped_air_mass(
            MAP_kPa, c.T_INTAKE_K, self.ve_fraction, RPM
        )

        # =================================================================
        # 8. Required fuel mass
        # =================================================================
        _required_fuel_kg = self.trapped_air_mass_kg / afr_target
        # self._required_fuel_g = required_fuel_g
        _required_fuel_vol_cc = _required_fuel_kg / c.FUEL_DENSITY_KG_CC

        # if self.calculated_theta == 180: # end of intake
        #     print(f"Mair = {self.trapped_air_mass_kg:6.4f}kg | "
        #         f"reqd fuel (kg): {_required_fuel_kg:4.2e} | "
        #         f"reqd fuel (cc) = {_required_fuel_vol_cc:6.4f} | "
        #         f"MAP = {MAP_kPa:6.4f} | "

        #         )

        # =================================================================
        # 9. Injector pulse width
        # =================================================================
        # injector_flow_g_per_ms = (c.INJECTOR_FLOW_CC_PER_MIN / 60_000.0) * c.FUEL_DENSITY_G_CC
        # calculated_pw_ms = required_fuel_g / injector_flow_g_per_ms
        # calculated_pw_ms += c.INJECTOR_DEAD_TIME_MS

        _injector_pw_msec = (
            _required_fuel_vol_cc / c.INJECTOR_FLOW_CC_PER_MIN
        ) * 60_000
        _injector_pw_degree = _injector_pw_msec * 360 * RPM / 60_000

        # print(
        #     f"trapped air: {self.trapped_air_mass_kg} | "
        #     f"taregt AFR: {afr_target} | "
        #     f"required fuel g {_required_fuel_kg} | "
        #     f"required fuel cc: {_required_fuel_vol_cc} | "
        #     f"injector pw ms: {_injector_pw_msec} | "
        #     )
        # calculated_pw_ms = np.clip(calculated_pw_ms, 0.5, 15.0)

        # =================================================================
        # 10. Injection timing (2D table lookup)
        # =================================================================
        _injector_end_timing_degree = float(self.inj_timing_interp([RPM, MAP_kPa]))

        # print(self.injector_timing_deg)
        # print(calculated_pw_ms)

        # =================================================================
        # 11. Injector Logic
        # =================================================================
        # --- Time conversion for the current step ---

        # time_per_degree_ms = 1000.0 / (current_rpm_safe * 6.0)

        # 1. Calculate Start of Injection (SOI)
        # We need to convert the pulse width (ms) to degrees at the current RPM
        # pw_in_degrees = calculated_pw_ms / time_per_degree_ms

        # SOI is EOI minus PW, then wrap-around if necessary
        self.injector_start_timing_degree = (
            _injector_end_timing_degree - _injector_pw_degree
        ) % 720.0
        self.injector_end_timing_degree = _injector_end_timing_degree % 720

        # print(
        #     f"injector start: {self.injector_start_timing_degree} | "
        #     f"injector end: {_injector_end_timing_degree} | "
        #     f"injector timing: {_injector_pw_degree} | "
        # )

        self.injector_is_active = False
        # if int(self.injector_start_timing_degree) <= self.calculated_theta < self.injector_end_timing_degree:
        #     self.injector_is_active = True if not self.fuel_cut_active else False

        if not self.fuel_cut_active:
            # 1. Determine if the injection event wraps around the 720/0 degree boundary
            is_wrapping = (
                self.injector_start_timing_degree > self.injector_end_timing_degree
            )

            # Get the current crank angle (theta) safely
            current_theta = self.calculated_theta % 720.0

            # 2. Check Activation
            if is_wrapping:
                # Case A: Injection crosses the 720/0 boundary (Start > End)
                # We are active if (theta >= Start) OR (theta < End)
                if (current_theta >= self.injector_start_timing_degree) or (
                    current_theta < self.injector_end_timing_degree
                ):
                    self.injector_is_active = True
            else:
                # Case B: Standard injection (Start <= End)
                # We are active if (theta >= Start) AND (theta < End)
                if (current_theta >= self.injector_start_timing_degree) and (
                    current_theta < self.injector_end_timing_degree
                ):
                    self.injector_is_active = True

        # =================================================================
        # 12. Return outputs
        # =================================================================

        # print(
        #     f"cycle = {self.cycle:2.0f} | "
        #     f"theta = {self.calculated_theta:5.0f} | "
        #     f"RPM = {RPM:3.0f} | "
        #     f"MAP = {MAP_kPa:3.0f} | "
        #     f"spark = {spark_fire_angle_720:5.0f} | "
        #     f"afr_target = {self.afr_target:4.1f} | "
        #     f"ve_fraction = {self.ve_fraction:5.1f} | "
        #     f"reqd_fuel = {self._required_fuel_g * 1000:4.1f}mg | "
        #     f"pulse_width = {pw_in_degrees:4.1f} | "
        #     f"str={self.injector_start_deg:4.0f} | "
        #     f"end={self.injector_end_deg:4.0f} | "
        # )

        return self.get_outputs()

    # -----------------------------------------------------------------------------------------
    def _idle_pid(self, TPS_percent, RPM, CLT_C):
        idle_pos = 0.0
        err = 0.0
        self.fuel_cut_active = False

        # Deceleration Fuel Cut-Off (DFCO) Thresholds
        DFCO_ENGAGE_RPM = 2000  # RPM must be above this to engage DFCO (coasting)
        # DFCO_DISENGAGE_RPM = self.idle_target_rpm + 50 # Fuel injection resumes below this to prevent stall
        DFCO_DISENGAGE_RPM = 1350
        # DFCO Logic: High RPM, foot off pedal
        if TPS_percent < 1.0 and RPM > DFCO_ENGAGE_RPM:
            self.fuel_cut_active = True
            # self.idle_integral = 0.0 # Reset integral term when fuel is cut
            idle_pos = 0.0  # Close idle valve completely (0% WOT equivalent flow)

        # PID Idle Control Logic
        # Only engage PID if not in DFCO and the throttle is mostly closed
        elif TPS_percent < 5.0 and RPM < DFCO_DISENGAGE_RPM:
            # Error is positive when RPM is too slow
            err = self.idle_target_rpm - RPM

            # PID gains — Re-tuned for the 0.0 to 5.0 WOT equivalent output range
            # Note: Gains are significantly smaller than the old ones (which targeted the 20-68 actuator range)
            # kp, ki, kd = 0.009, 0.0003, 0.0012
            kp = 0.007  # was 0.009
            ki = 0.00018  # was 0.0003 → slower integral, no windup
            kd = 0.0009  # was 0.0012 → slightly less D overshoot

            # Integral with anti-windup (limits remain the same as they operate on the error in RPM)
            self.idle_integral += err
            self.idle_integral = np.clip(self.idle_integral, -1200, 1200)

            # Derivative
            deriv = err - self.last_error

            # Base idle valve position (WOT equivalent flow percentage) - Normalized
            BASE_IDLE_FLOW = (
                1.9  # 1.8% WOT equivalent flow for warm engine (Replaces old 32.0)
            )
            COLD_SENSITIVITY = (
                0.065  # 0.065% flow per degree C under 70C (Replaces old 0.65)
            )

            # Base position includes cold enrichment
            cold_bonus = max(0.0, (70.0 - CLT_C) * COLD_SENSITIVITY)
            base_pos = BASE_IDLE_FLOW + cold_bonus

            # Raw PID output
            pid_output = kp * err + ki * self.idle_integral + kd * deriv

            # Final position (Base + PID)
            idle_pos = base_pos + pid_output

            # The clip is now a safety guard against extreme, incorrect values
            idle_pos = np.clip(idle_pos, 0.0, 20.0)

        else:
            # Driver on throttle (> 5.0%) or DFCO has been disabled
            # self.idle_integral = 0.0
            idle_pos = 0.0

        self.last_error = err if TPS_percent < 5.0 else 0.0

        return idle_pos

    # -----------------------------------------------------------------------------------------
    def _calculate_trapped_air_mass(self, MAP_kPa, T_intake_K, VE, rpm):
        air_density_kg_m3 = (
            MAP_kPa * 1000.0 / (c.R_SPECIFIC_AIR * T_intake_K)
        )  # Ideal Gas Law
        theoretical_air_kg = air_density_kg_m3 * c.V_DISPLACED * VE

        # model is theoretically perfect so lets make it more representative of real road cars
        if rpm < 3500:
            # Ideal or peak VE is allowed up to this point (e.g., 90% potential)
            ve_limit = 0.90
        elif rpm < 4500:
            # Beginning of flow restriction and valve inertia issues (VE starts dropping)
            # Interpolate from 90% down to 80%
            ve_limit = 0.90 - (rpm - 3500) / 1000 * 0.10
        elif rpm < 5500:
            # Severe restriction, reaching the physical redline limit (valve float)
            # Interpolate from 80% down to 50%
            ve_limit = 0.80 - (rpm - 4500) / 1000 * 0.30
        else:
            # Above the redline, VE crashes due to severe valve float and lack of flow
            ve_limit = 0.50 - (rpm - 5500) / 2000 * 0.20  # Drops further at high RPM
            ve_limit = max(ve_limit, 0.30)  # Maintain a floor

        theoretical_air_kg = theoretical_air_kg * ve_limit

        # print(
        #     f"MAP={MAP_kPa*1000} | "
        #     f"T_intake_K={T_intake_K} | "
        #     f"air density={air_density_kg_m3}m3 | "
        #     f"m air={theoretical_air_kg}kg | "
        #     f"VE={VE} | "
        #     )

        return theoretical_air_kg

    # -----------------------------------------------------------------------------------------
    def _calculate_fuel_mass(self, trapped_air_mass_g, afr_target):
        """Calculates the required mass of fuel in milligrams (mg)."""
        # M_fuel (g) = M_air (g) / AFR
        required_fuel_mass_g = trapped_air_mass_g / afr_target
        # Convert to milligrams for standard ECU resolution
        return required_fuel_mass_g * 1000.0  # mg

    # -----------------------------------------------------------------------------------------
    # def _calculate_pulse_width(self, fuel_mass_mg):
    #     """
    #     Converts required fuel mass (mg) into an Injector Pulse Width (ms).
    #     This is a core injector model calculation.
    #     """

    #     # 1. Convert Mass Flow Rate to Volume Flow Rate (V_fuel)
    #     # Fuel density is in g/cc. Mass is in mg. Convert mg to g:
    #     fuel_mass_g = fuel_mass_mg / 1000.0

    #     # V_fuel (cc) = M_fuel (g) / Fuel_Density (g/cc)
    #     fuel_volume_cc = fuel_mass_g / c.FUEL_DENSITY_G_CC

    #     # 2. Convert Volume Flow Rate to Time (Pulse Width)
    #     # Time (min) = V_fuel (cc) / I_FLOW_CC_MIN (cc/min)
    #     required_time_min = fuel_volume_cc / c.INJECTOR_FLOW_CC_PER_MIN

    #     # 3. Convert Time to milliseconds (ms) and apply dead time
    #     # Time (ms) = Time (min) * 60,000 ms/min
    #     required_time_ms = required_time_min * 60000.0

    #     # Total Pulse Width = Required Open Time + Injector Dead Time/Offset
    #     #
    #     pulse_width_ms = required_time_ms + c.V_OFFSET_MS

    #     # Must be positive; clamp to a small minimum if the calculated time is near zero
    #     return max(0.0, pulse_width_ms)
