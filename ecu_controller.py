# ecu_controller.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin


import sys
from engine_model import FixedKeyDictionary
import constants as c
import numpy as np
from efi_tables import EFITables

# REQUIRED: Import necessary for 2D map interpolation (Spark/AFR tables)
from scipy.interpolate import RegularGridInterpolator


class ECUController:
    """
    Simulates the Engine Control Unit (ECU).
    It determines control outputs (Spark, AFR, Idle Valve Position) based on sensor inputs.
    """
    
    def __init__(self, rl_idle_mode=False, rl_wot_spark_mode=False):
        
        # RL bypass variables
        """
        rl_idle_mode:       If True, bypasses internal PID and uses external_idle_command
        rl_wot_spark_mode:  If True, bypasses spark table lookup and uses external_spark_advance
        """
        self.rl_idle_mode = rl_idle_mode
        self.rl_wot_spark_mode = rl_wot_spark_mode

        self.external_idle_command = 0.0      # used when rl_idle_mode=True
        self.external_spark_advance = 0.0     # used when rl_wot_spark_mode=True
        
        # EFI Tables and lookup values
        self.tables = EFITables()
        self.ve_fraction = 0.0
        self.spark_advance_btdc = 0.0
        self.afr_target = 0.0
        self.injector_end_timing_degree = 360.0
        

        # State for PID Idle Control
        self.idle_integral = 0.0
        self.last_error_proportional = 0.0
        self.idle_target_rpm = c.IDLE_RPM
        self._idle_pid_base_flow = 1.9
        self._cold_bonus_mult = 1.0
        self._kp_mult = 1.0

        # --- ECU Standard Outputs ---
        self.spark_active = False

        self.injector_is_active = False  # NEW: Boolean flag for twin

        self.idle_valve_position = 0.0
        self.fuel_cut_active = False  # Flag for Deceleration Fuel Cut-Off (DFCO)
        self.trapped_air_mass_kg = 0.0


        # --- Fuel System Outputs ---
        # self.fuel_mass_mg = 0.0                 # Mass of fuel required (mg)
        # self.injector_pw_msec = 0.0      # Final command to injector (ms)

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

        # --- for internal debugging ---
        self._required_fuel_g = 0
        
        # --- idle pid variables ---
        self.pid_error_integral = 0.0
        self._filtered_drpm = 0.0

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
  
        # 1. extract sensor inputs
        RPM = sensors["RPM"]
        rpm_history = sensors["rpm_720_history"]
        MAP_kPa = sensors["MAP_kPa"]
        TPS = sensors["TPS_percent"]
        CLT_C = sensors["CLT_C"]
        crank_pos = sensors["crank_pos"]
        cam_pulse = sensors["cam_sync"]

        current_rpm_safe = max(RPM, 10.0)
        
        # 2. Subsystems
        self._sync_timing(crank_pos, cam_pulse)
        self._lookup_tables(RPM, MAP_kPa)
        # RL spark override
        if self.rl_wot_spark_mode:
            self.spark_advance_btdc = self.external_spark_advance  # RL override
            
        self._calculate_spark_timing()
        self._calculate_fuel_delivery(MAP_kPa, c.T_INTAKE_K, RPM)
        self._calculate_idle_valve(TPS, RPM, rpm_history, CLT_C)

        # 3. Build outputs
        return self.get_outputs()
    
        

    # ---------------------------------------------------------------------
    def _sync_timing(self, crank_pos, cam_pulse):
        """Handle crank/cam sync and theta calculation."""

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
                
    # ---------------------------------------------------------------------
    def _calculate_idle_valve(self, TPS, RPM, rpm_history, CLT_C):
        """Idle valve control — uses PID or external command."""

        if self.rl_idle_mode:
            # RL mode: do NOTHING here — idle valve will be set externally
            idle_valve_position = self.external_idle_command  # set by RL env
        else:
            # Normal mode: use PID
            idle_valve_position = self._idle_pid(TPS, RPM, rpm_history, CLT_C)
        
        self.idle_valve_position = idle_valve_position
        
        effective_tps = np.clip(TPS + idle_valve_position, 0.0, 100.0)
        
    
    # ---------------------------------------------------------------------
    def _lookup_tables(self, RPM, MAP_kPa):
        """EFI table lookups (VE, spark, AFR, injector timing)."""
        
        lookup = self.tables.lookup(RPM, MAP_kPa)
        self.ve_fraction = lookup["ve"] / 100  # convert from deg to fraction
        self.spark_advance_btdc = lookup["spark"]
        self.afr_target = lookup["afr"]
        self.injector_end_timing_degree = lookup["injector"] 

    # ---------------------------------------------------------------------
    def _calculate_spark_timing(self):
        """Determine if spark should fire this degree."""

        # self.spark_advance_btdc = float(self.spark_interp([RPM, MAP_kPa]))
        # Convert to 720° domain for engine
        spark_fire_angle_720 = 360.0 - self.spark_advance_btdc

        # Only fire spark at the exact degree
        spark_this_degree = (
            self.last_calculated_theta < spark_fire_angle_720 <= self.calculated_theta
        )
        
        if spark_this_degree and not self.fuel_cut_active:
            self.spark_active = True
        else:
            self.spark_active = False
    
    # ---------------------------------------------------------------------
    def _calculate_fuel_delivery(self, MAP_kPa, T_intake_K, RPM):
        """Calculate trapped air, required fuel, and injector timing."""

        # Trapped air
        self.trapped_air_mass_kg = self._calculate_trapped_air_mass(MAP_kPa, c.T_INTAKE_K, self.ve_fraction, RPM)

        # Required fuel
        _required_fuel_kg = self.trapped_air_mass_kg / self.afr_target
        _required_fuel_vol_cc = _required_fuel_kg / c.FUEL_DENSITY_KG_CC

        # Injector pulse width in degrees
        _injector_pw_msec = (_required_fuel_vol_cc / c.INJECTOR_FLOW_CC_PER_MIN) * 60_000
        _injector_pw_degree = _injector_pw_msec * 360 * RPM / 60_000


        # SOI is EOI minus PW, then wrap-around if necessary
        self.injector_start_timing_degree = (self.injector_end_timing_degree - _injector_pw_degree) % 720.0
        self.injector_is_active = False

        if not self.fuel_cut_active:
            # Injector active this degree?
            current_theta = self.calculated_theta % 720.0
            is_wrapping = (self.injector_start_timing_degree > self.injector_end_timing_degree)

            if is_wrapping:
                active = (current_theta >= self.injector_start_timing_degree) or (current_theta < self.injector_end_timing_degree)
            else:
                active = (current_theta >= self.injector_start_timing_degree) and (current_theta < self.injector_end_timing_degree)

            self.injector_is_active = active and not self.fuel_cut_active


    # -----------------------------------------------------------------------------------------
    def _idle_pid(self, TPS_percent, RPM, rpm_history, CLT_C):
        current_theta = int(self.calculated_theta % 720)
        idle_pos = 0.0
        rpm_avg = np.mean(rpm_history)
        prev_rpm = rpm_history[(current_theta - 2) % 720]
        
        self.fuel_cut_active = False

        # Deceleration Fuel Cut-Off (DFCO) Thresholds
        DFCO_ENGAGE_RPM = 3000  # RPM must be above this to engage DFCO (coasting)
        # DFCO_DISENGAGE_RPM = self.idle_target_rpm + 50 # Fuel injection resumes below this to prevent stall
        DFCO_DISENGAGE_RPM = 2000
        # DFCO Logic: High RPM, foot off pedal
        if TPS_percent < 1.0 and rpm_avg > DFCO_ENGAGE_RPM:
            self.fuel_cut_active = True
            # self.idle_integral = 0.0 # Reset integral term when fuel is cut
            idle_pos = 0.0  # Close idle valve completely (0% WOT equivalent flow)
            
        # PID Idle Control Logic
        # Only engage PID if not in DFCO and the throttle is mostly closed
        elif TPS_percent < 5.0 and rpm_avg < DFCO_DISENGAGE_RPM:
                
            # ------------------------------------------------------------------
            # 2. Errors
            # ------------------------------------------------------------------
            error_instant = self.idle_target_rpm - RPM
            error_avg = self.idle_target_rpm - rpm_avg
            
            # PID gains — Re-tuned for the 0.0 to 5.0 WOT equivalent output range
            # Note: Gains are significantly smaller than the old ones (which targeted the 20-68 actuator range)
            # kp, ki, kd = 0.009, 0.0003, 0.0012
            # kp = 0.007  # was 0.009
            # ki = 0.00018  # was 0.0003 → slower integral, no windup
            # kd = 0.0009  # was 0.0012 → slightly less D overshoot
            kp = 0.005  # was 0.009
            ki = 0.0004 # was 0.0003 → slower integral, no windup
            kd = 0.0 
            # PID_BACK_CALC_GAIN = 2000
            
            # ------------------------------------------------------------------
            # 3. Proportional – on lightly filtered RPM
            # ------------------------------------------------------------------
            P = (kp * self._kp_mult) * error_instant 
                  
            # ------------------------------------------------------------------
            # 4. Integral – on average rpm (aka filtered)
            # ------------------------------------------------------------------
            self.idle_integral += ki * error_avg
            self.idle_integral = np.clip(self.idle_integral, -5.0, 5.0)  # small limits in % equiv
            I = self.idle_integral
            
            # ------------------------------------------------------------------
            # 5. Derivative – on instantaneous rate (proper sign & filtered)
            # ------------------------------------------------------------------
            rpm_rate = RPM - prev_rpm # rate of change
            D = kd * rpm_rate # opposes acceleration
       
            # ------------------------------------------------------------------
            # 6. Cold Base 
            # ------------------------------------------------------------------
            self._idle_pid_base_flow = 2.8

            # Base + cold bonus
            cold_bonus = max(0.0, (70.0 - CLT_C) * 0.065)
            base_pos = (self._idle_pid_base_flow + cold_bonus) * self._cold_bonus_mult
            
            # ------------------------------------------------------------------
            # 7. output
            # ------------------------------------------------------------------
            pid_output = P + I + D

            idle_pos = base_pos + pid_output
            # Soft safety clip (wider than before if needed, but not required)
            idle_pos = np.clip(idle_pos, 0.0, 30.0)


        else:
            idle_pos = 0.0

        # self.last_error_propotional = err_proportional if TPS_percent < 5.0 else 0.0

        # if current_theta % 20 == 0:
        # print(
        #     f"theta: {current_theta:3d} | "
        #     f"RPM: {int(RPM):3d} | "
        #     f"prev_rpm: {int(prev_rpm):3d} | "
        #     f"P: {P:6.0f} | "
        #     f"I: {self.pid_error_integral:6.0f} | "
        #     f"D: {D:4.2e} | "
        #     f"PID output: {pid_clamped:10.2f} | "
        #     f"Idle Pos: {idle_pos:10.2f}"
        # )
        
        # if current_theta % 90 == 0:
        #     print(
        #         f"theta: {current_theta:3d} | "
        #         f"IDLE | RPM:{int(RPM):4d} avg:{int(rpm_avg):4d} | "
        #         f"P:{P:+6.3f} I:{I:+6.3f} | base:{base_pos:5.2f} → pos:{idle_pos:5.2f}")


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
