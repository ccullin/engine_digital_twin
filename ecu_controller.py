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
    
    def __init__(self, rl_idle_mode=False, rl_ecu_spark_mode=False):
        
        # RL bypass variables
        """
        rl_idle_mode:       If True, bypasses internal PID and uses external_idle_command
        rl_ecu_spark_mode:  If True, bypasses spark table lookup and uses external_spark_advance
        """
        self.rl_idle_mode = rl_idle_mode
        self.rl_ecu_spark_mode = rl_ecu_spark_mode

        # Flags set externally, typically by Driver Strategy
        self.external_idle_command = 0.0      # used when rl_idle_mode=True
        self.external_spark_advance = 0.0     # used when rl_ecu_spark_mode=True
        self.is_motoring = False              # set when engine running in motoring mode
        self.fuel_enabled = True              # typically used by MotoringStrategy to toggle fuel
        self.spark_enabled = True             # typically used by MotoringStrategy to P.V plot a running engine
        
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
        self.iacv_wot_equiv = 0.0
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
        self.calculated_theta = 0
        self.last_calculated_theta = 0
        self.first_crank_rotation = True
        self.cycle = 1
        self.rpm_history = np.zeros(720)

        # --- for internal debugging ---
        self._required_fuel_g = 0
        
        # --- idle pid variables ---
        self.pid_error_integral = 0.0
        self._filtered_drpm = 0.0
        self.pid_P = 0.0
        self.pid_I = 0.0
        self.pid_D = 0.0
        
        # --- idle overrides
        self.idle_afr = 13.8
        self.filtered_map = 35.0
        

        self.output_dict = FixedKeyDictionary(
            {
                "spark": self.spark_active,
                "spark_timing": int(self.spark_advance_btdc),
                "afr_target": self.afr_target,
                "target_rpm": self.idle_target_rpm,
                "iacv_pos": self.idle_valve_position,
                "iacv_wot_equiv": self.iacv_wot_equiv,
                "pid_P": self.pid_P,
                "pid_I" : self.pid_I,
                "pid_D": self.pid_D,
                "trapped_air_mass_kg": self.trapped_air_mass_kg,
                "ve_fraction": self.ve_fraction,
                "injector_on": self.injector_is_active,
                'injector_start_deg'   : int(self.injector_start_timing_degree),
                'injector_end_deg'     : int(self.injector_end_timing_degree),
                "fuel_cut_active": self.fuel_cut_active,
            }
        )
        


    # -----------------------------------------------------------------------------------------
    def get_outputs(self):
        self.output_dict.update(
            {
                "spark": self.spark_active,
                "spark_timing": int(self.spark_advance_btdc),
                "afr_target": self.afr_target,
                "target_rpm": self.idle_target_rpm,
                "iacv_pos": self.idle_valve_position,
                "iacv_wot_equiv": self.iacv_wot_equiv,
                "pid_P": self.pid_P,
                "pid_I" : self.pid_I,
                "pid_D": self.pid_D,
                "trapped_air_mass_kg": self.trapped_air_mass_kg,
                "ve_fraction": self.ve_fraction,
                "injector_on": self.injector_is_active,
                'injector_start_deg'   : int(self.injector_start_timing_degree),
                'injector_end_deg'     : int(self.injector_end_timing_degree),
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
        
        def _get_smoothed_map(current_map, filtered_map):
            """
            returns a smoothed map
            
            :param self: Description
            :param current_map: current map from engine
            :param filtered_map: initial guess for idle map
            """
            # Alpha of 0.1 to 0.2 is common. 
            # Lower = smoother/slower, Higher = noisier/faster
            # alpha = 0.15 
            # Use a very heavy filter (Alpha = 0.05) to ignore valve pulses
            # and only follow the actual air-mass trend.
            alpha = 0.05
            self.filtered_map = (alpha * current_map) + ((1 - alpha) * filtered_map)
            
            return self.filtered_map
  
        CAD = int(self.calculated_theta) % 720
        # 1. extract sensor inputs
        RPM = sensors.rpm
        # rpm_history = sensors.rpm_720_history
        MAP_kPa = sensors.MAP_kPa
        TPS = sensors.TPS_percent
        CLT_C = sensors.CLT_C
        crank_pos = sensors.crank_pos
        cam_pulse = sensors.cam_sync

        current_rpm_safe = max(RPM, 10.0)
        self.rpm_history[CAD] = RPM
        
        is_idle = sensors.TPS_percent < 5.0
        
        # 2. Subsystems with idle overrides
        self._sync_timing(crank_pos, cam_pulse)
        
        if is_idle: 
            tables_dict = self._set_idle_overrides(RPM, MAP_kPa)      
            map = _get_smoothed_map(MAP_kPa, self.filtered_map) # smooth MAP for smoother fuel delivery
        else:
            tables_dict = self._lookup_tables(RPM, MAP_kPa, TPS)
            map = MAP_kPa
               
        # store the EFI table / idle parameters
        self.afr_target = tables_dict['afr']
        self.spark_advance_btdc = tables_dict['spark']
        self.ve_fraction = tables_dict['ve']
        self.injector_end_timing_degree = tables_dict['injector']
        
        # calculate fuel and timing 
        self._calculate_fuel_delivery(map, c.T_INTAKE_K, RPM)
        self._calculate_spark_timing()
        self._calculate_idle_valve(TPS, RPM, CLT_C)

        return self.get_outputs()


    def _set_idle_overrides(self, RPM, MAP_kPa):
        """
        Sets AFR and Spark timing when in idle mode
        """  
        
        lookup = self.tables.lookup(RPM, MAP_kPa) # uses actual MAP
        injector_end_timing_degree = lookup["injector"] 
        
            
        # 1. OVERRIDE AFR.  Locks AFR to a stable 'torque-rich' target. 
        # 13.5 - 13.8 is common for smooth idling in older engines like the 2.1L WBX.
        afr_target = self.idle_afr
        
        if self.rl_ecu_spark_mode:
            spark_advance_btdc = self.external_spark_advance # RL spark override
        else: 
            # 2. OVERRIDE SPARK advance: Use Spark Reserve Strategy
            # We use a fixed base (e.g., 10 deg) plus a dynamic correction
            idle_base_spark = 10.0 
            error = RPM - self.idle_target_rpm
            # Correction factor: 1 degree per 25 RPM error (Aggressive but stable)
            # Limit the swing to +/- 10 degrees to prevent stalls or knock
            correction = np.clip(error / 25.0, -10.0, 10.0)   
            spark_advance = idle_base_spark - correction # Negative because error > 0 (high RPM) needs retard (-)
            spark_advance_btdc = np.clip(spark_advance, -5, 45)
        
        # 3. VE OVERRIDE. Use a static VE value for the 'Idle Zone' 
        # to prevent the table from 'stepping' between cells
        # lookup = self.tables.lookup(rpm=self.idle_target_rpm, map_kpa = 35.0) # uses static RPM and MAP
        # ve_fraction = lookup["ve"] / 100
        ve_fraction = 0.45
   


        return {
            "ve": ve_fraction,
            "spark": spark_advance_btdc,
            "afr": afr_target,
            "injector": injector_end_timing_degree,
        }
   
   
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
    def _calculate_idle_valve(self, TPS, RPM, CLT_C):
        """Idle valve control — uses PID or external command."""

        if self.rl_idle_mode:
            # RL mode: do NOTHING here — idle valve will be set externally
            idle_valve_position = self.external_idle_command  # set by RL env
        else:
            # Normal mode: use PID
            idle_valve_position = self._idle_pid(TPS, RPM, CLT_C)
        self.idle_valve_position = idle_valve_position
        
        # Map idle valve position to WOT
        # A 12mm IACV bore is ~6% of a 50mm Throttle Body area
        IACV_MAX_WOT_EQUIV = 0.06
        self.iacv_wot_equiv = idle_valve_position * IACV_MAX_WOT_EQUIV
        
        
    
    # ---------------------------------------------------------------------
    def _lookup_tables(self, RPM, MAP_kPa, TPS):
        """EFI table lookups (VE, spark, AFR, injector timing)."""
        
        lookup = self.tables.lookup(RPM, MAP_kPa)
        ve_fraction = lookup["ve"] / 100  # converted to fraction
        
        if self.rl_ecu_spark_mode:
            spark_advance_btdc = self.external_spark_advance # RL spark override
        else: 
            spark_advance_btdc = lookup["spark"]
            
        afr_target = lookup["afr"] 
        injector_end_timing_degree = lookup["injector"] 
        
        return {
            "ve": ve_fraction,
            "spark": spark_advance_btdc,
            "afr": afr_target,
            "injector": injector_end_timing_degree,
        }

    # ---------------------------------------------------------------------
    def _calculate_spark_timing(self):
        """Determine if spark should fire this degree."""

        # self.spark_advance_btdc = float(self.spark_interp([RPM, MAP_kPa]))
        # Convert to 720° domain for engine
        spark_fire_angle_720 = 360.0 - self.spark_advance_btdc
        rl_spark = 360.0 - self.external_spark_advance
        # Spark Gate
        can_spark = (not self.is_motoring) or self.spark_enabled
        
        # Only fire spark at the exact degree
        spark_this_degree = (
            self.last_calculated_theta < spark_fire_angle_720 <= self.calculated_theta
        )
        
        if can_spark and spark_this_degree and not self.fuel_cut_active:
            self.spark_active = True
        else:
            self.spark_active = False

    # ---------------------------------------------------------------------
    def _calculate_fuel_delivery(self, MAP_kPa, T_intake_K, RPM):
        """Calculate trapped air, required fuel, and injector timing."""
        
        # Fuel Gate
        can_inject = (not self.is_motoring) or self.fuel_enabled

        # Trapped air
        self.trapped_air_mass_kg = self._calculate_trapped_air_mass(MAP_kPa, c.T_INTAKE_K, self.ve_fraction, RPM)

        # Required fuel
        _required_fuel_kg = self.trapped_air_mass_kg / self.afr_target
        _required_fuel_kg = _required_fuel_kg if RPM > (c.IDLE_RPM) else _required_fuel_kg * 1.7
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

            if can_inject:
                self.injector_is_active = active and not self.fuel_cut_active
                
        # if int(self.calculated_theta) == int(self.injector_start_timing_degree):
        #     print(f"ECU injector start:{self.injector_start_timing_degree:3.0f} end_deg:{self.injector_end_timing_degree:3.0f} MAP:{MAP_kPa} rpm:{RPM:4.0f} "
        #       f"estimated_air:{self.trapped_air_mass_kg:7.5f}kg reqd_fuel:{_required_fuel_vol_cc:7.5f}cc {_required_fuel_kg:7.5f}kg "
        #       f"pw_deg:{_injector_pw_degree:3.0f}")

    # -----------------------------------------------------------------------------------------
    def _idle_pid(self, TPS_percent, RPM, CLT_C):
        """
        Calculates the Idle Air Control Valve (IACV) position.
        Returns: float (0.0 to 0.06) representing the effective WOT equivalent area.
        """
        current_theta = int(self.calculated_theta % 720)
        rpm_avg = np.mean(self.rpm_history)
        prev_rpm = self.rpm_history[(current_theta - 2) % 720]
        
        self.fuel_cut_active = False
        

        
        # 2. DFCO Logic (Deceleration Fuel Cut-Off)
        DFCO_ENGAGE_RPM = 3000
        DFCO_DISENGAGE_RPM = 1500 # Slightly lower to allow PID to catch
        
        """ DEBUG """
        # if TPS_percent < 1.0 and rpm_avg > DFCO_ENGAGE_RPM:
        #     self.fuel_cut_active = True
        #     return 0.0  # Close valve completely during coasting
            
        # 3. PID Engagement Check
        # Only run PID if throttle is closed and we aren't at high RPM
        """ DEBUG """
        if TPS_percent < 5.0: #and rpm_avg < DFCO_DISENGAGE_RPM:
            
            # --- Errors ---
            error_instant = self.idle_target_rpm - RPM
            error_avg = self.idle_target_rpm - rpm_avg
            
            if abs(error_instant) < 20:
                iacv_pos = self.idle_valve_position
            else:
                # --- Gains (Calibrated for 0-100 internal scale) ---
                kp = 0.005   # 0.015 Responsive Proportional
                ki = 0.0004  # 0.0005 Slow Integral to prevent hunting
                kd = 0.000   # 0.001 Damping to prevent overshoot during flare
                
                # --- Proportional ---
                P = kp * error_instant 
                    
                # --- Integral (with Anti-Windup) ---
                self.idle_integral += ki * error_avg
                # Clamp integral to +/- 20% of valve range
                self.idle_integral = np.clip(self.idle_integral, -20.0, 20.0)
                I = self.idle_integral
                
                # --- Derivative ---
                rpm_rate = RPM - prev_rpm
                D = kd * rpm_rate
        
                # --- Base Flow Logic (0-100 scale) ---
                # 35.0 is a typical warm idle DC for a VW 2.1L
                base_valve_pos = 10.0 

                # Cold bonus: Add ~25% more air when stone cold (0C)
                # Tapers to 0 at 70C
                cold_bonus = max(0.0, (70.0 - CLT_C) * 0.4)
                
                effective_base = base_valve_pos + cold_bonus

                # --- CRANKING / STARTUP FLARE ---
                # If engine is below target or cranking, force valve to max (100%)
                if RPM < (self.idle_target_rpm - 100):
                    effective_base = 100.0 

                # --- Total Raw Output (0-100 scale) ---
                iacv_pos_raw = effective_base + P + I - D # D opposes direction of travel
                
                # Clamp to physical hardware limits
                iacv_pos = np.clip(iacv_pos_raw, 0.0, 100.0)
                
                self.pid_P = P
                self.pid_I = I
                self.pid_D = D

                # if current_theta % 20 == 0:
                #     print(
                #         f"    DEBUG IDLE PID. "
                #         f"theta: {current_theta:3d} | "
                #         f"RPM: {int(RPM):3d} | "
                #         f"prev_rpm: {int(prev_rpm):3d} | "
                #         f"P: {P:6.0f} | "
                #         f"I: {I:6.0f} | "
                #         f"D: {D:4.2e} | "
                #         f"raw output: {raw_output:10.2f} | "
                #         f"Idle Pos: {iacv_wot_equiv:10.2f}"
                #     )
        else:
            # If driver is on the throttle, the IACV usually holds a 
            # "dashpot" position to prevent stalling when they lift off.
            iacv_pos = 25 # Fixed small bypass (1.5% WOT)
            """ DEBUG """
            iacv_pos = 0.0
            
  
        return iacv_pos  

    # -----------------------------------------------------------------------------------------
    """Pre converstion to WOT scaler updates.  this worked for the IACV=TPS 1:1 mapping, but that is not real world."""
    # def _idle_pid(self, TPS_percent, RPM, CLT_C):
    #     current_theta = int(self.calculated_theta % 720)
    #     idle_pos = 0.0
    #     rpm_avg = np.mean(self.rpm_history)
    #     prev_rpm = self.rpm_history[(current_theta - 2) % 720]
        
    #     self.fuel_cut_active = False

    #     # Deceleration Fuel Cut-Off (DFCO) Thresholds
    #     DFCO_ENGAGE_RPM = 3000  # RPM must be above this to engage DFCO (coasting)
    #     # DFCO_DISENGAGE_RPM = self.idle_target_rpm + 50 # Fuel injection resumes below this to prevent stall
    #     DFCO_DISENGAGE_RPM = 2000
    #     # DFCO Logic: High RPM, foot off pedal
    #     if TPS_percent < 1.0 and rpm_avg > DFCO_ENGAGE_RPM:
    #         self.fuel_cut_active = True
    #         # self.idle_integral = 0.0 # Reset integral term when fuel is cut
    #         idle_pos = 0.0  # Close idle valve completely (0% WOT equivalent flow)
            
    #     # PID Idle Control Logic
    #     # Only engage PID if not in DFCO and the throttle is mostly closed
    #     elif TPS_percent < 5.0 and rpm_avg < DFCO_DISENGAGE_RPM:
                
    #         # ------------------------------------------------------------------
    #         # 2. Errors
    #         # ------------------------------------------------------------------
    #         error_instant = self.idle_target_rpm - RPM
    #         error_avg = self.idle_target_rpm - rpm_avg
            
    #         # PID gains — Re-tuned for the 0.0 to 5.0 WOT equivalent output range
    #         # Note: Gains are significantly smaller than the old ones (which targeted the 20-68 actuator range)
    #         # kp, ki, kd = 0.009, 0.0003, 0.0012
    #         # kp = 0.007  # was 0.009
    #         # ki = 0.00018  # was 0.0003 → slower integral, no windup
    #         # kd = 0.0009  # was 0.0012 → slightly less D overshoot
    #         kp = 0.005  # was 0.005
    #         ki = 0.0004 # was 0.0003 → slower integral, no windup
    #         kd = 0.0 
    #         # PID_BACK_CALC_GAIN = 2000
            
    #         # ------------------------------------------------------------------
    #         # 3. Proportional – on lightly filtered RPM
    #         # ------------------------------------------------------------------
    #         P = (kp * self._kp_mult) * error_instant 
                  
    #         # ------------------------------------------------------------------
    #         # 4. Integral – on average rpm (aka filtered)
    #         # ------------------------------------------------------------------
    #         self.idle_integral += ki * error_avg
    #         self.idle_integral = np.clip(self.idle_integral, -5.0, 5.0)  # small limits in % equiv
    #         I = self.idle_integral
            
    #         # ------------------------------------------------------------------
    #         # 5. Derivative – on instantaneous rate (proper sign & filtered)
    #         # ------------------------------------------------------------------
    #         rpm_rate = RPM - prev_rpm # rate of change
    #         D = kd * rpm_rate # opposes acceleration
       
    #         # ------------------------------------------------------------------
    #         # 6. Cold Base + Cranking Flare
    #         # ------------------------------------------------------------------
    #         self._idle_pid_base_flow = 2.8

    #         # Base + cold bonus
    #         cold_bonus = max(0.0, (70.0 - CLT_C) * 0.065)
    #         base_pos = (self._idle_pid_base_flow + cold_bonus) * self._cold_bonus_mult

    #         # --- CRANKING OVERRIDE ---
    #         if RPM < 900:
    #             # Force the valve open to 25-30% to provide enough air mass to "catch"
    #             # This acts as an automatic 'high idle' during the start attempt
    #             base_pos = max(base_pos, 100.0) 

    #         # ------------------------------------------------------------------
    #         # 7. output
    #         # ------------------------------------------------------------------
    #         pid_output = P + I + D

    #         idle_pos = base_pos + pid_output
    #         # Soft safety clip (wider than before if needed, but not required)
    #         # idle_pos = np.clip(idle_pos, 0.0, 30.0)


    #     else:
    #         idle_pos = 0.0

    #     # self.last_error_propotional = err_proportional if TPS_percent < 5.0 else 0.0

    #     # if current_theta % 20 == 0:
    #     #     print(
    #     #         f"theta: {current_theta:3d} | "
    #     #         f"RPM: {int(RPM):3d} | "
    #     #         f"prev_rpm: {int(prev_rpm):3d} | "
    #     #         f"P: {P:6.0f} | "
    #     #         f"I: {self.pid_error_integral:6.0f} | "
    #     #         f"D: {D:4.2e} | "
    #     #         # f"PID output: {pid_clamped:10.2f} | "
    #     #         f"Idle Pos: {idle_pos:10.2f}"
    #     #     )
        
    #     # if current_theta % 90 == 0:
    #     #     print(
    #     #         f"theta: {current_theta:3d} | "
    #     #         f"IDLE | RPM:{int(RPM):4d} avg:{int(rpm_avg):4d} | "
    #     #         f"P:{P:+6.3f} I:{I:+6.3f} | base:{base_pos:5.2f} → pos:{idle_pos:5.2f}")


    #     return idle_pos

    # -----------------------------------------------------------------------------------------
    def _calculate_trapped_air_mass(self, MAP_kPa, T_intake_K, VE, rpm):
        air_density_kg_m3 = (MAP_kPa * 1000.0) / (c.R_SPECIFIC_AIR * T_intake_K)  # Ideal Gas Law
        theoretical_air = air_density_kg_m3 * c.V_DISPLACED
        theoretical_air_kg = theoretical_air * VE

        # # model is theoretically perfect so lets make it more representative of real road cars
        # if rpm < 3500:
        #     # Ideal or peak VE is allowed up to this point (e.g., 90% potential)
        #     ve_limit = 0.90
        # elif rpm < 4500:
        #     # Beginning of flow restriction and valve inertia issues (VE starts dropping)
        #     # Interpolate from 90% down to 80%
        #     ve_limit = 0.90 - (rpm - 3500) / 1000 * 0.10
        # elif rpm < 5500:
        #     # Severe restriction, reaching the physical redline limit (valve float)
        #     # Interpolate from 80% down to 50%
        #     ve_limit = 0.80 - (rpm - 4500) / 1000 * 0.30
        # else:
        #     # Above the redline, VE crashes due to severe valve float and lack of flow
        #     ve_limit = 0.50 - (rpm - 5500) / 2000 * 0.20  # Drops further at high RPM
        #     ve_limit = max(ve_limit, 0.30)  # Maintain a floor

        # theoretical_air_kg = theoretical_air_kg * ve_limit

        # print(
        #     f"MAP={MAP_kPa*1000} | "
        #     f"T_intake_K={T_intake_K} | "
        #     f"air density={air_density_kg_m3}m3 | "
        #     f"m air={theoretical_air_kg}kg | "
        #     f"VE={VE} | "
        #     )

        return theoretical_air_kg
