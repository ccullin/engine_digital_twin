# engine_model.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin


import numpy as np
import physics_functions as pf
import constants as c
import sys

import collections.abc


class FixedKeyDictionary(dict):
    """A dictionary that only allows assignments or updates to predefined keys."""

    def __init__(self, *args, **kwargs):
        # We allow keys to be set ONLY during the initial super().__init__ call
        super().__init__(*args, **kwargs)
        # Store the set of valid keys defined at creation
        self._valid_keys = set(self.keys())
        self._is_initialized = True

    def __setitem__(self, key, value):
        # 1. Enforce key restriction for single assignment (d[key] = value)
        if hasattr(self, "_is_initialized") and key not in self._valid_keys:
            raise KeyError(
                f"Attempted to assign a new key '{key}'. "
                f"Only existing keys ({list(self._valid_keys)}) are allowed."
            )
        super().__setitem__(key, value)

    def update(self, other=None, **kwargs):
        """Overrides dict.update() to enforce key restriction."""
        if hasattr(self, "_is_initialized"):

            # Check keys from the 'other' argument (e.g., d.update(new_dict))
            if other:
                if isinstance(other, collections.abc.Mapping):
                    # If 'other' is a dictionary or mapping
                    incoming_keys = other.keys()
                else:
                    # If 'other' is a sequence of (key, value) pairs
                    incoming_keys = [k for k, v in other]

                for key in incoming_keys:
                    if key not in self._valid_keys:
                        raise KeyError(
                            f"Attempted to update with a new key '{key}' from other. "
                            f"Only existing keys are allowed."
                        )

            # Check keys from keyword arguments (e.g., d.update(key1=value1))
            for key in kwargs:
                if key not in self._valid_keys:
                    raise KeyError(
                        f"Attempted to update with a new key '{key}' from kwargs. "
                        f"Only existing keys are allowed."
                    )

        # If all checks pass OR if we are still initializing, call the original update
        super().update(other, **kwargs)

    def setdefault(self, key, default=None):
        """Overrides dict.setdefault() to enforce key restriction."""
        # setdefault only adds a key if it's MISSING.
        # If the key is missing AND initialization is complete, we must raise an error.
        if hasattr(self, "_is_initialized") and key not in self._valid_keys:
            raise KeyError(
                f"Attempted to set a new key '{key}' using setdefault. "
                f"Only existing keys are allowed."
            )
        return super().setdefault(key, default)



class EngineModel:
    def __init__(self, rpm):
        # =================================================================
        # 1. Geometry
        # =================================================================
        self.A_piston = c.A_PISTON
        self.V_displaced = c.V_DISPLACED
        self.V_clearance = self.V_displaced / (c.COMP_RATIO - 1.0)

        self.theta_list = np.arange(
            c.THETA_MIN, c.THETA_MAX + c.THETA_DELTA, c.THETA_DELTA
            )
        self.V_list = [
            pf.v_cyl(th, self.A_piston, self.V_clearance) for th in self.theta_list
            ]

        # =================================================================
        # 2. Crank & Cam Sync — REAL HARDWARE STANDARD
        # =================================================================
        self.crank_teeth_total = 36
        self.missing_tooth_index = 35
        self.deg_per_tooth = 360.0 / self.crank_teeth_total

        self.current_theta = 0.0  # 0–720° (true position)
        self.next_theta = 0.0  # looks ahead to end of cycle, used for logging.
        self.crank_tooth = 0  # 0–35
        self.cam_sync = False  # differentiations the 1st 360 and Cyl 1 and 3.

        # Sync status
        self.crank_synced = False

        # =================================================================
        # 3. State
        # =================================================================
        self.rpm = rpm
        self.P_manifold = c.P_ATM_PA
        self.P_cyl = c.P_ATM_PA
        self.T_cyl = c.T_AMBIENT
        self.M_gas = 5.8e-4  # kg.  Mass of a cylinder of air
        self.clt = c.COOLANT_START
        self.tps_sensor = 0.0
        self.effective_tps = 0.0
        self.P_ambient_sensor = c.P_ATM_PA
        self.afr_sensor = 0
        self.knock = False
        self.knock_intensity = 0.0
        self.wheel_load = 0.0
        self._cycle_fuel_injected_cc = 0
        self._cylinder_total_air_mass_kg = 0
        self._exhaust_g_per_degree = 0
        # self._cycle_work_J = 0.0
        self._burn_heat_per_deg = 0.0
        self._burn_duration_remaining = 0.0
        self._cycle_count = 0
        self.total_cycle_heat_J = 0
        self.combustion_active = False

        # self.cranking = True if self.rpm <= c.CRANK_RPM else False

        # =================================================================
        # 4. per cylinder INSTANTANEOUS INDICATED TORQUE
        # =================================================================
        self.torque_indicated_cyl_1 = np.zeros(720)
        self.torque_indicated_cyl_2 = np.zeros(720)
        self.torque_indicated_cyl_3 = np.zeros(720)
        self.torque_indicated_cyl_4 = np.zeros(720)
        self.torque_indicated_engine = np.zeros(720)
        self.torque_brake_720 = np.zeros(720)
        self.torque_friction = 0.0
        self.torque_brake = 0.0
        self.T_net_engine = np.zeros(720)
        self.power_history = np.zeros(720)

        # =================================================================
        # 4. Sensors & last_cycle
        # =================================================================
        self.rpm_history = np.zeros(720)
        self.map_history = np.zeros(720)
        
        self.sensors_dict = FixedKeyDictionary({
                "RPM": self.rpm,
                "rpm_720_history": self.rpm_history,
                "MAP_kPa": self.P_manifold / 1000.0,
                "Ambient_pressure": self.P_ambient_sensor,
                "actual_AFR": 99,
                "lambda_actual": 99,
                "Knock": self.knock,
                "knock_intensity": self.knock_intensity,
                "CLT_C": self.clt,
                "TPS_percent": self.tps_sensor,
                "crank_pos": self.crank_tooth,
                "cam_sync": self.cam_sync,
            })
        
        self.spark_advance_btdc = 0
        self.peak_pressure_bar = 0
        
        self.engine_data_dict = FixedKeyDictionary({
            "theta": 0,

            # Primary performance outputs
            "torque_history": self.torque_brake_720,
            "rpm_history": self.rpm_history,
            "power_history": self.power_history,
            # "brake_power_avg_kw": np.mean(self.torque_brake_720) * np.mean(self.rpm_history) / 9549.0,  # approximate, or compute properly

            # Air / Fuel / Combustion (very useful for idle learning)
            # "air_mass_cycle_kg": self._cylinder_total_air_mass_kg,          # if you track it
            "air_mass_per_cyl_kg": self._cylinder_total_air_mass_kg,          # per cylinder average
            # "injected_fuel_per_cyl_cc": total_fuel_injected_cc,     # total over 720°
            # "actual_AFR_cycle": total_air_kg / (total_fuel_kg) if total_fuel_kg > 0 else 14.7,

            # Pressure and efficiency
            "peak_pressure_bar": self.peak_pressure_bar,
            # "mean_piston_speed_m_s": calculate from RPM and stroke,
            # "ve_percent": (actual_air_mass / theoretical_air_mass) * 100,  # if you can compute

            # Smoothness and stability metrics (gold for idle control)
            # "rpm_avg_720": np.mean(self.rpm_history),
            # "rpm_std_720": np.std(self.rpm_history),
            # "rpm_min_720": np.min(self.rpm_history),
            # "rpm_max_720": np.max(self.rpm_history),

            # Load / disturbance indicators
            "map_avg_kPa": np.mean(self.map_history),
            "torque_std_nm": np.std(self.torque_brake_720),

            # Optional advanced
            # "bsfc_g_kwh": fuel_consumption / power if power > 0 else 0,
            # "friction_torque_nm": estimated friction loss,
        })

        self.log = {"P": [], "V": [], "T": []}

    # ----------------------------------------------------------------------
    def get_sensors(self):
        self.sensors_dict.update({
                "RPM": self.rpm,
                "rpm_720_history": self.rpm_history,
                "MAP_kPa": self.P_manifold / 1000.0,
                "Ambient_pressure": self.P_ambient_sensor,
                "TPS_percent": self.tps_sensor,
                "actual_AFR": self.afr_sensor,
                "lambda_actual": self.afr_sensor / 14.7,
                "Knock": self.knock,
                "knock_intensity": self.knock_intensity,
                "CLT_C": self.clt,
                "crank_pos": self.crank_tooth,
                "cam_sync": self.cam_sync,
            })

        return self.sensors_dict

    # ----------------------------------------------------------------------
    def get_engine_data(self):
        self.engine_data_dict.update({
            "theta": self.current_theta,

            # Primary performance outputs
            "torque_history": self.torque_brake_720,
            # "brake_power_avg_kw": np.mean(self.torque_brake_720) * np.mean(self.rpm_history) / 9549.0,  # approximate, or compute properly
            "rpm_history": self.rpm_history,
            "power_history": self.power_history,

            # Air / Fuel / Combustion (very useful for idle learning)
            # "air_mass_cycle_kg": self._cylinder_total_air_mass_kg,          # if you track it
            "air_mass_per_cyl_kg": self._cylinder_total_air_mass_kg,          # per cylinder average
            # "injected_fuel_per_cyl_cc": total_fuel_injected_cc,     # total over 720°
            # "actual_AFR_cycle": total_air_kg / (total_fuel_kg) if total_fuel_kg > 0 else 14.7,

            # Pressure and efficiency
            "peak_pressure_bar": self.peak_pressure_bar,
            # "mean_piston_speed_m_s": calculate from RPM and stroke,
            # "ve_percent": (actual_air_mass / theoretical_air_mass) * 100,  # if you can compute

            # Smoothness and stability metrics (gold for idle control)
            # "rpm_avg_720": np.mean(self.rpm_history),
            # "rpm_std_720": np.std(self.rpm_history),
            # "rpm_min_720": np.min(self.rpm_history),
            # "rpm_max_720": np.max(self.rpm_history),

            # Load / disturbance indicators
            "map_avg_kPa": np.mean(self.map_history),
            "torque_std_nm": np.std(self.torque_brake_720),

            # Optional advanced
            # "bsfc_g_kwh": fuel_consumption / power if power > 0 else 0,
            # "friction_torque_nm": estimated friction loss,
        })
        return self.engine_data_dict

    # =================================================================
    # REAL-TIME STEP — CALLED EVERY DEGREE
    # =================================================================
    def step(self, ecu_outputs):
        """
        Called every crank degree (or every 6°/10° if you want).
        This is how real ECUs work.
        """
        self.current_theta = (
            self.next_theta
        )  # both are 0 after __init__ so cycle 0 works.

        # Crank: 36-1 wheel
        tooth_float = (self.current_theta % 360.0) / self.deg_per_tooth
        new_tooth = int(tooth_float) % self.crank_teeth_total

        if new_tooth != self.crank_tooth:
            self.crank_tooth = new_tooth
            if new_tooth == 0 and not self.crank_synced:
                # self.current_theta = 0.0  # resync on missing tooth
                self.crank_synced = True

        # Cam: pulse at 630° (90° BTDC)
        self.cam_sync = 620 <= self.current_theta <= 640

        # update throttle position for MAP calculations
        # self.tps sensor is updated in the main loop from the Driver inputs.
        effective_tps = np.clip(
            self.tps_sensor + ecu_outputs["idle_valve_position"], 0, 100
        )  # clipo to ensure there is no idle_valve open at WOT
        self.effective_tps = effective_tps

        # Realistic throttle → MAP curve (non-linear, matches real engines)
        # 0%   → ~28 kPa (deep vacuum)
        # 2.5% → ~38 kPa (typical idle)
        # 100% → 99+ kPa (WOT)
        tps_frac = np.clip(effective_tps / 100.0, 0.0, 1.0)
        tps_frac_low = effective_tps / 20.0 if effective_tps < 20.0 else 1.0
        
        if effective_tps < 20.0:
            # Deep vacuum at closed + small idle bypass
            map_kPa = c.P_ATM_PA * (0.30 + 0.20 * tps_frac_low**2)  # 30 kPa at 0%, ~50 kPa at 20%
        else:
            # Transition to linear/full above idle range
            map_kPa = c.P_ATM_PA * (0.50 + 0.50 * tps_frac**1.3)        
        
        engine_running = True if self.rpm > 600 else False
        self.P_manifold = map_kPa
        self.map_history[int(self.current_theta)] = map_kPa

        # Run physics for this ONE degree
        self._step_one_degree(ecu_outputs, engine_running)

        # Increment theta and check for end of cycle
        self.next_theta = (self.current_theta + 1.0) % 720.0
        if (
            self.current_theta >= 719.0
        ):  # this is the last cycle has completed and the next cycle is 0
            self._cycle_count += 1

        # return self.sensors_dict, self.engine_data_dict
        return self.get_sensors(), self.get_engine_data()
    
    # ----------------------------------------------------------------------
    def _step_one_degree(self, ecu_outputs, engine_running):
        """
        Called every 1° of crank rotation.
        """
        theta = self.current_theta
        i = int(np.round(theta)) % 720
        stroke, _ = self._get_stroke()
        
        # 1. Stroke Transitions & Reset Logic
        if self._is_dead_center(i):
            self._handle_stroke_transitions(i, stroke)

        # 2. Mass Flow (Air & Fuel)
        self._update_mass_flow(ecu_outputs, stroke)

        # 3. Combustion Heat Release (Current Linear Model)
        # Q_in = self._calculate_linear_heat_release(ecu_outputs["spark"])
        Q_in = self._calculate_combustion_heat(ecu_outputs["spark"]
            
        )

        # 4. Thermodynamics (First Law Integration)
        V_curr = self.V_list[i]
        dV = self.V_list[(i + 1) % 720] - V_curr
        
        P_next, T_next = pf.integrate_first_law(
            P_curr=self.P_cyl, T_curr=self.T_cyl, M_curr=self.M_gas,
            V_curr=V_curr, Delta_M=0.0, Delta_Q_in=Q_in, Delta_Q_loss=0.0,
            dV_d_theta=dV, gamma=c.GAMMA_AIR, theta_delta=c.THETA_DELTA
        )

        # 5. Mechanical Dynamics & RPM Update
        self._update_mechanical_dynamics(i, stroke, P_next, dV)

        # 6. State Updates & Logging
        self.P_cyl = np.clip(P_next, 5_000, 18_000_000)
        self.T_cyl = np.clip(T_next, 220, 3500)
        
        self.log["P"].append(self.P_cyl)
        self.log["V"].append(V_curr)
        self.log["T"].append(self.T_cyl)

        if self.current_theta >= 719.0:
            self._end_of_cycle_update()
            
    # --- Helper Methods to maintain functionality ---

    def _handle_stroke_transitions(self, i, stroke):
        """Preserves the debug/reset logic at dead centers."""
        if stroke == "exhaust":
            self.P_cyl = self.P_manifold
            self._cycle_fuel_injected_cc = 0.0
        # (Note: T_ind_720/power_kw calculations from original are preserved here if needed for logging)

    def _update_mass_flow(self, ecu_outputs, stroke):
        """Preserves injector and air intake logic."""
        current_rpm_safe = max(self.rpm, 10.0)
        fuel_per_deg = c.INJECTOR_FLOW_CC_PER_MIN / (360 * current_rpm_safe)
        
        if ecu_outputs["injector_on"]:
            self._cycle_fuel_injected_cc += fuel_per_deg

        trapped_air_per_deg = ecu_outputs["trapped_air_mass_kg"] / 180
        if stroke == "intake":
            self._cylinder_total_air_mass_kg += trapped_air_per_deg
        elif stroke == "exhaust":
            if abs(self.current_theta - 540) % 540 <= 0.01:
                self._exhaust_kg_per_degree = self._cylinder_total_air_mass_kg / 180
            self._cylinder_total_air_mass_kg -= self._exhaust_kg_per_degree

    def _calculate_combustion_heat(self, spark_command):
        """Calculates heat release using the Wiebe S-curve with energy conservation."""
        
        # 1. Trigger Spark Event
        if spark_command:
            self.spark_event_theta = self.current_theta
            self.combustion_active = True
            self.cumulative_heat_released = 0.0 # Track total energy released
            
            # Setup fuel/AFR/Efficiency
            fuel_kg = self._cycle_fuel_injected_cc * c.FUEL_DENSITY_KG_CC
            afr = (self._cylinder_total_air_mass_kg / fuel_kg) if fuel_kg > 0 else 99.0
            self.afr_sensor = afr
            eff = 0.94 if 11.5 <= afr <= 16.0 else 0.80
            
            self.total_cycle_heat_J = fuel_kg * c.LHV_FUEL_GASOLINE * eff
            self._cycle_fuel_injected_cc = 0.0

        # 2. Calculate Heat Release
        if self.combustion_active:
            q_step = pf.calc_wiebe_heat_rate(
                theta=self.current_theta,
                theta_start=self.spark_event_theta,
                duration=c.BURN_DURATION_DEG,
                total_heat_J=self.total_cycle_heat_J
            )
            
            # Energy Conservation Guard: Ensure we don't exceed total_heat_J
            # or leave small remainders behind.
            if self.cumulative_heat_released + q_step > self.total_cycle_heat_J:
                q_step = self.total_cycle_heat_J - self.cumulative_heat_released
            
            self.cumulative_heat_released += q_step
            
            # Check if burn is finished
            if self.current_theta >= (self.spark_event_theta + c.BURN_DURATION_DEG):
                # Final cleanup: If there's a tiny remainder due to float math, dump it here
                remainder = self.total_cycle_heat_J - self.cumulative_heat_released
                q_step += max(0.0, remainder) 
                
                self.combustion_active = False
                
            return q_step
            
        return 0.0

    # def _calculate_linear_heat_release(self, spark_command):
    #     """Preserves your existing linear combustion logic exactly."""
    #     if spark_command:
    #         self.spark_advance_btdc = 360.0 - (self.current_theta % 360.0)
    #         fuel_kg = self._cycle_fuel_injected_cc * c.FUEL_DENSITY_KG_CC
    #         afr = (self._cylinder_total_air_mass_kg / fuel_kg) if fuel_kg > 0 else 99.0
    #         self.afr_sensor = afr
    #         eff = 0.94 if 11.5 <= afr <= 16.0 else 0.80
            
    #         self.total_cycle_heat_J = fuel_kg * c.LHV_FUEL_GASOLINE * eff
    #         self._burn_heat_per_deg = self.total_cycle_heat_J / c.BURN_DURATION_DEG
    #         self._burn_duration_remaining = c.BURN_DURATION_DEG
    #         self._cycle_fuel_injected_cc = 0.0

    #     if self._burn_duration_remaining > 0.0:
    #         self._burn_duration_remaining -= 1.0
    #         return self._burn_heat_per_deg
    #     return 0.0

    def _update_mechanical_dynamics(self, i, stroke, P_next, dV):
        """Preserves torque integration and RPM physics."""
        P_avg = (self.P_cyl + P_next) / 2.0
        delta_work_J = P_avg * dV
        
        t_ind_cyl1 = pf.calc_indicated_torque_step(delta_work_J, stroke)
        
        # Distribute to all 4 cylinders
        self.torque_indicated_cyl_1[i] = t_ind_cyl1
        self.torque_indicated_cyl_4[(i + 180) % 720] = t_ind_cyl1
        self.torque_indicated_cyl_3[(i + 360) % 720] = t_ind_cyl1
        self.torque_indicated_cyl_2[(i + 540) % 720] = t_ind_cyl1

        self.torque_indicated_engine[i] = (
            self.torque_indicated_cyl_1[i] + self.torque_indicated_cyl_2[i] +
            self.torque_indicated_cyl_3[i] + self.torque_indicated_cyl_4[i]
        )
        
        self.torque_friction = pf.calc_friction_torque_per_degree(self.rpm)
        self.torque_brake = self.torque_indicated_engine[i] - self.torque_friction
        self.torque_brake_720[i] = self.torque_brake
        self.T_net_engine[i] = self.torque_brake - self.wheel_load

        # RPM Integration
        omega = pf.eng_speed_rad(self.rpm)
        dt = np.deg2rad(1.0) / omega
        alpha = self.T_net_engine[i] / c.MOMENT_OF_INERTIA
        self.rpm = max(c.CRANK_RPM, self.rpm + alpha * dt * 30.0 / np.pi)
        
        self.rpm_history[i] = self.rpm
        self.power_history[i] = self.torque_brake * self.rpm / 9549.3
        
        
        
        
        
        # # =================================================================
        # # 1. Extract ECU commands (fresh every degree)
        # # =================================================================
        # ecu_spark_command = ecu_outputs["spark"]
        # ecu_injector_on = ecu_outputs["injector_on"]
        # trapped_air_kg = ecu_outputs["trapped_air_mass_kg"]

        # trapped_air_kg_per_deg = trapped_air_kg / 180
        # current_rpm_safe = max(self.rpm, 10.0)
        # fuel_cc_per_degree = c.INJECTOR_FLOW_CC_PER_MIN / (360 * current_rpm_safe)

        # # =================================================================
        # # 2. Current crank angle (0–720°)
        # # =================================================================
        # theta = self.current_theta
        # i = int(np.round(theta)) % 720
        # i_next = (i + 1) % 720
        # stroke, previous_stroke = self._get_stroke()

        # # =================================================================
        # # 2. Current crank angle (0–720°)
        # # =================================================================
        # if self._is_dead_center(i):
        #     #  DEBUG print for INJECTION and SPARK

        #     stroke_start = (i - 180) % 720
        #     if stroke_start < i:
        #         T_ind_stroke = self.torque_indicated_engine[stroke_start:i]
        #         T_net_stroke = self.T_net_engine[stroke_start:i]
        #     else:
        #         T_ind_stroke = self.torque_indicated_engine[i:stroke_start]
        #         T_net_stroke = self.T_net_engine[i:stroke_start]
        #     T_ind_stroke_mean = sum(T_ind_stroke) / 180
        #     T_net_stroke_mean = sum(T_net_stroke) / 180

        #     if stroke == "exhaust":
        #         self.P_cyl = self.P_manifold
        #         self._cycle_fuel_injected_cc = (
        #             0.0  # vent any fuel that did not get ignited due to DFCO
        #         )

        #     if stroke == "intake":  # start of a new cycle, previous cycle just finished
        #         # Full 720° averages (this is what dynos, papers, and ECUs use)
        #         T_ind_720 = np.mean(self.torque_indicated_engine)  # Nm
        #         # T_fric_720 = np.mean([pf.calc_friction_torque_mean(r) for r in np.linspace(800, self.rpm, 10)])  # or just your mean function
        #         # T_brake_720 = T_ind_720 - T_fric_720
        #         # T_net_720   = T_brake_720 - self.wheel_load
        #         T_net_720 = np.mean(self.T_net_engine)
        #         power_kw = T_net_720 * self.rpm * np.pi / (30.0 * 1000.0)  # kW


        # # =================================================================
        # # 3. Volume & dV this step
        # # =================================================================
        # # Find nearest index in pre-computed theta_list
        # V_current = self.V_list[i]
        # V_next = self.V_list[i_next]
        # dV = V_next - V_current

        # # =================================================================
        # # 4. FUEL INJECTION — real, per-degree delivery
        # # =================================================================
        # if ecu_injector_on:
        #     # Injector is commanded ON for the full 1° duration
        #     self._cycle_fuel_injected_cc = (
        #         self._cycle_fuel_injected_cc + fuel_cc_per_degree
        #     )

        # # =================================================================
        # # 4. AIR INTAKE per-degree linear intake and exhaust
        # # =================================================================
        # if stroke == "intake":
        #     self._cylinder_total_air_mass_kg = (
        #         self._cylinder_total_air_mass_kg + trapped_air_kg_per_deg
        #     )
        #     self._exhaust_g_per_degree = 0
        # elif stroke == "exhaust":
        #     if abs(theta - 540) % 540 <= 0.01:
        #         self._exhaust_kg_per_degree = self._cylinder_total_air_mass_kg / 180
        #     self._cylinder_total_air_mass_kg = (
        #         self._cylinder_total_air_mass_kg - self._exhaust_kg_per_degree
        #     )

        # # =================================================================
        # # 5. COMBUSTION — triggered by ECU spark command
        # # =================================================================
        # Q_in_per_degree = 0.0
        # total_heat_J = 0.0

        # # Check if spark is commanded (START OF BURN)
        # if ecu_spark_command:
        #     # === RECORD SPARK ADVANCE FOR KNOCK DETECTION ===
        #     spark_theta = self.current_theta
        #     self.spark_advance_btdc = 360.0 - (spark_theta % 360.0)
            
        #     # === calculate afr ===
        #     fuel_available_kg = self._cycle_fuel_injected_cc * c.FUEL_DENSITY_KG_CC
        #     actual_afr = (
        #         (self._cylinder_total_air_mass_kg / fuel_available_kg)
        #         if fuel_available_kg > 0
        #         else 99.0
        #     )
        #     self.afr_sensor = actual_afr
        #     combustion_efficiency = 0.94 if 11.5 <= actual_afr <= 16.0 else 0.80

        #     # Calculate and set the heat rate state
        #     total_heat_J = (
        #         fuel_available_kg * c.LHV_FUEL_GASOLINE * combustion_efficiency
        #     )
        #     self.total_cycle_heat_J = total_heat_J
        #     self._burn_heat_per_deg = total_heat_J / c.BURN_DURATION_DEG
        #     self._burn_duration_remaining = c.BURN_DURATION_DEG

        #     self._cycle_fuel_injected_cc = 0.0

        # # Apply heat if burn is in progress
        # if self._burn_duration_remaining > 0.0:
        #     Q_in_per_degree = self._burn_heat_per_deg
        #     self._burn_duration_remaining -= 1.0  # Decrement burn counter by 1 degree

        #     # Log total heat for sanity check (only logs once per cycle)
        #     # total_heat_J = self._burn_heat_per_deg * c.BURN_DURATION_DEG
        # else:
        #     # total_heat_J = 0.0
        #     # self.total_cycle_heat_J = 0.0
        #     self._burn_heat_per_deg = 0.0

        # # =================================================================
        # # 6. First law — 1° integration
        # # =================================================================
        # P_next, T_next = pf.integrate_first_law(
        #     P_curr=self.P_cyl,
        #     T_curr=self.T_cyl,
        #     M_curr=self.M_gas,
        #     V_curr=V_current,
        #     Delta_M=0.0,
        #     Delta_Q_in=Q_in_per_degree,
        #     Delta_Q_loss=0.0,
        #     dV_d_theta=dV,
        #     gamma=c.GAMMA_AIR,
        #     theta_delta=c.THETA_DELTA,
        # )

        # # =================================================================
        # # 5. Update engine performance per degree
        # # =================================================================
        # # delta_work_J = self.P_cyl * dV
        # P_avg = (self.P_cyl + P_next) / 2.0
        # delta_work_J = P_avg * dV

        # # 1. CONSTANTS FOR INSTANTANEOUS CALCULATION
        # # Convert 1.0 degree to radians (assuming c.THETA_DELTA is 1.0)
        # THETA_DELTA_RAD = c.THETA_DELTA * (np.pi / 180.0)

        # # 2. INSTANTANEOUS INDICATED TORQUE (Replaces the cycle-averaged calculation)
        # torque_indicated_cyl1_raw = delta_work_J / THETA_DELTA_RAD

        # if stroke == "intake":
        #     # Pumping stroke: Must consume work (Negative Torque)
        #     # T_ind_raw is currently POSITIVE during intake, so we must negate it.
        #     torque_indicated_cyl1 = -abs(torque_indicated_cyl1_raw)

        # else:
        #     # Power (positive) and Compression (negative) signs are correct.
        #     torque_indicated_cyl1 = torque_indicated_cyl1_raw

        # # # firing order is not important but using 1432 because I love early air-cooled
        # self.torque_indicated_cyl_1[i] = torque_indicated_cyl1
        # self.torque_indicated_cyl_4[(i + 180) % 720] = torque_indicated_cyl1
        # self.torque_indicated_cyl_3[(i + 360) % 720] = torque_indicated_cyl1
        # self.torque_indicated_cyl_2[(i + 540) % 720] = torque_indicated_cyl1

        # torque_indicated = (
        #     self.torque_indicated_cyl_1[i]
        #     + self.torque_indicated_cyl_2[i]
        #     + self.torque_indicated_cyl_3[i]
        #     + self.torque_indicated_cyl_4[i]
        # )
        # self.torque_indicated_engine[i] = torque_indicated
        
        # # 3. FRICTION, BRAKE, AND NET TORQUE
        # torque_friction = pf.calc_friction_torque_per_degree(self.rpm)
        # torque_brake = torque_indicated - torque_friction
        # self.torque_brake_720[i] = torque_brake        
        
        # T_net_engine = torque_brake - self.wheel_load
        # self.torque_friction = torque_friction
        # self.torque_brake = torque_brake
        # self.T_net_engine[i] = T_net_engine

        # # 4. DYNAMIC RPM UPDATE (Requires time step correction)
        # omega = pf.eng_speed_rad(self.rpm)
        # dt = np.deg2rad(1.0) / omega

        # alpha = T_net_engine / c.MOMENT_OF_INERTIA  # rad/s²
        # self.rpm = self.rpm + alpha * dt * 30.0 / np.pi  # convert back to RPM

        # if self.rpm < c.CRANK_RPM:
        #     self.rpm = c.CRANK_RPM
        
        # self.rpm_history[i] = self.rpm
        # self.power_history[i] = torque_brake * self.rpm / 9549.3
            

        # # Stability — keeps model alive forever
        # P_next = np.clip(P_next, 5_000, 18_000_000)
        # # P_next = 0.70 * P_next + 0.30 * self.P_cyl
        # T_next = np.clip(T_next, 220, 3500)
        # # T_next = 0.80 * T_next + 0.20 * self.T_cyl


        # # Update state
        # self.P_cyl = P_next
        # self.T_cyl = T_next

        # # Log
        # self.log["P"].append(P_next)
        # self.log["V"].append(V_current)
        # self.log["T"].append(T_next)


        # if self.current_theta >= 719.0:  # this is the last cycle has completed and the next cycle is 0
        #     self._end_of_cycle_update()

        # return None

    # ----------------------------------------------------------------------
    def _end_of_cycle_update(self):
        """Called exactly once per 720° — torque, RPM, sensors"""
        
        # end of cycle checks
        peak_bar = max(self.log["P"]) / 100000.0 # Ensure P is in bar
        self.peak_pressure_bar = peak_bar

        self.knock_detected, self.knock_intensity = pf.detect_knock(
            peak_bar = peak_bar,
            clt = self.clt, 
            rpm = self.rpm,   
            spark_advance = self.spark_advance_btdc,
            lambda_ = self.afr_sensor / 14.7,
            fuel_octane = c.FUEL_OCTANE
        )

                
        # prep for next cycle
        self.log["P"].clear()
        self.log["V"].clear()
        self.log["T"].clear()
        
    # ----------------------------------------------------------------------
    def _is_dead_center(self, theta, tolerance=1e-3):
        """Checks if the crank angle (theta) is at a Dead Center."""

        # Use standard Python modulo operator (%)
        remainder = theta % 180.0

        # Use NumPy's robust float comparison
        is_close_to_zero = np.isclose(remainder, 0.0, atol=tolerance)
        is_close_to_180 = np.isclose(remainder, 180.0, atol=tolerance)

        return is_close_to_zero | is_close_to_180

    # ----------------------------------------------------------------------
    def _get_stroke(self):
        stroke = None
        previous_stroke = None
        phase = int(self.current_theta / 180) + 1
        if phase == 1:
            stroke = "intake"
            previous_stroke = "exhaust"
        elif phase == 2:
            stroke = "compression"
            previous_stroke = "intake"
        elif phase == 3:
            stroke = "power"
            previous_stroke = "compression"
        elif phase == 4:
            stroke = "exhaust"
            previous_stroke = "power"
        return stroke, previous_stroke

 