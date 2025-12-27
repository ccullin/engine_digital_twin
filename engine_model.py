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

    # Note: methods like pop() and clear() are safe as they only remove items.


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
        self.knock_intensity = 0
        self.wheel_load = 0.0
        self._cycle_fuel_injected_cc = 0
        self._cylinder_total_air_mass_kg = 0
        self._exhaust_g_per_degree = 0
        # self._cycle_work_J = 0.0
        self._burn_heat_per_deg = 0.0
        self._burn_duration_remaining = 0.0
        self._cycle_count = 0
        self.total_cycle_heat_J = 0

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
                "Knock": 0.0,
                "CLT_C": self.clt,
                "TPS_percent": self.tps_sensor,
                "crank_pos": self.crank_tooth,
                "cam_sync": self.cam_sync,
            })

        # self.engine_data_dict = FixedKeyDictionary({  # ONLY UPDATED AT THE END OF EACH 720 CYCLE.
        #         "theta": 0,
        #         # --- Primary Outputs ---
        #         "brake_torque_nm": 0,  # can be either torque_ind or torque_net
        #         "brake_power_kw": 0,  # kW (Standard)
        #         # 'brake_power_hp': 0,                # hp (Dashboard
        #         # --- Fuel/Air/Combustion ---
        #         # 'injected_fuel_cc': 0,
        #         # 'total_air_kg': 0,
        #         # 'air_mass_per_cyl_kg': 0,
        #         "peak_pressure_bar": 0,
        #         # --- Efficiency and Load Metrics ---
        #         # 'bsfc_g_kwh': 0,                    # Brake Specific Fuel Consumption
        #         # 've_actual': 0,                     # Requires V_displaced_g_per_cyl calculation in ICE model
        #         # 'mean_effective_pressure_bar': 0    # Assumes Indicated MEP is calculated elsewhere
        #     })
        
        
        self.engine_data_dict = FixedKeyDictionary({
            "theta": 0,

            # Primary performance outputs
            "torque_history": self.torque_brake_720,
            "rpm_history": self.rpm_history,
            # "brake_power_avg_kw": np.mean(self.torque_brake_720) * np.mean(self.rpm_history) / 9549.0,  # approximate, or compute properly

            # Air / Fuel / Combustion (very useful for idle learning)
            # "air_mass_cycle_kg": self._cylinder_total_air_mass_kg,          # if you track it
            "air_mass_per_cyl_kg": self._cylinder_total_air_mass_kg,          # per cylinder average
            # "injected_fuel_per_cyl_cc": total_fuel_injected_cc,     # total over 720°
            # "actual_AFR_cycle": total_air_kg / (total_fuel_kg) if total_fuel_kg > 0 else 14.7,

            # Pressure and efficiency
            # "peak_pressure_bar": np.max(self.cylinder_pressure_history),
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
                "Knock": self.knock_intensity,
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

            # Air / Fuel / Combustion (very useful for idle learning)
            # "air_mass_cycle_kg": self._cylinder_total_air_mass_kg,          # if you track it
            "air_mass_per_cyl_kg": self._cylinder_total_air_mass_kg,          # per cylinder average
            # "injected_fuel_per_cyl_cc": total_fuel_injected_cc,     # total over 720°
            # "actual_AFR_cycle": total_air_kg / (total_fuel_kg) if total_fuel_kg > 0 else 14.7,

            # Pressure and efficiency
            # "peak_pressure_bar": np.max(self.cylinder_pressure_history),
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

        # if effective_tps < 20.0:
        #     # More sensitive in 0–10%, flatter above — better resolution for PID/RL
        #     map_kPa = c.P_ATM_PA * (0.29 + 0.21 * tps_frac_low**1.8)  # ~29 kPa at 0%, ~45 kPa at 20%
        # else:
        #     # Smooth transition above idle range
        #     tps_frac = effective_tps / 100.0
        #     map_kPa = c.P_ATM_PA * (0.45 + 0.55 * tps_frac**1.4)
        
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

        # Update sensors
        # self.sensors_dict.update(
        #     {
        #         "RPM": self.rpm,
        #         "MAP_kPa": self.P_manifold / 1000.0,
        #         "crank_pos": self.crank_tooth,
        #         "cam_sync": self.cam_sync,
        #     }
        # )

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
        # =================================================================
        # 1. Extract ECU commands (fresh every degree)
        # =================================================================
        ecu_spark_command = ecu_outputs["spark"]
        ecu_injector_on = ecu_outputs["injector_on"]
        trapped_air_kg = ecu_outputs["trapped_air_mass_kg"]

        trapped_air_kg_per_deg = trapped_air_kg / 180

        # injector_timing_deg     = ecu_outputs['injector_timing_deg']
        # injector_pulse_width_ms = ecu_outputs['injector_pulse_width_ms']

        # --- Pre-calculate Constants for this Step ---
        # Time duration of the 1 degree step (in milliseconds)
        # Use a minimum RPM to prevent division by zero during startup
        current_rpm_safe = max(self.rpm, 10.0)

        # time_per_degree_ms = 1000.0 / (current_rpm_safe * 6.0)

        # # Injector flow rate (grams per millisecond)
        # fuel_flow_g_per_ms = (c.INJECTOR_FLOW_CC_PER_MIN / 60_000.0) * c.FUEL_DENSITY_G_CC

        # Fuel mass delivered in exactly one 1° step (assuming constant RPM over the step)
        # fuel_g_per_degree_step = fuel_flow_g_per_ms * time_per_degree_ms

        fuel_cc_per_degree = c.INJECTOR_FLOW_CC_PER_MIN / (360 * current_rpm_safe)
        # fuel_g_per_degree = fuel_cc_per_degree * c.FUEL_DENSITY_G_CC

        # =================================================================
        # 2. Current crank angle (0–720°)
        # =================================================================
        theta = self.current_theta
        i = int(np.round(theta)) % 720
        i_next = (i + 1) % 720
        stroke, previous_stroke = self._get_stroke()

        # =================================================================
        # 2. Current crank angle (0–720°)
        # =================================================================

        if self._is_dead_center(i):
            #  DEBUG print for INJECTION and SPARK

            stroke_start = (i - 180) % 720
            if stroke_start < i:
                T_ind_stroke = self.torque_indicated_engine[stroke_start:i]
                T_net_stroke = self.T_net_engine[stroke_start:i]
            else:
                T_ind_stroke = self.torque_indicated_engine[i:stroke_start]
                T_net_stroke = self.T_net_engine[i:stroke_start]
            T_ind_stroke_mean = sum(T_ind_stroke) / 180
            T_net_stroke_mean = sum(T_net_stroke) / 180

            # print(
            #     f"cyc{self._cycle_count:3d} | "
            #     f"{previous_stroke:6.6s} | "
            #     f"rpm={self.rpm:4.0f} | "
            #     f"MAP={self.P_manifold/1000:6.2f} | "
            #     f"eff_TPS={self.effective_tps:4.1F} | "
            #     f"<<CYL>> "
            #     f"fuel={self._cycle_fuel_injected_cc:6.4f}cc | "
            #     f"heat={(self.total_cycle_heat_J if previous_stroke == "power" else 0.0):4.0f}J | "
            #     # f"T_ind_cyl={self.torque_indicated_cyl1:7.1f} | "
            #     # f"d_work={delta_work_J:5.1f}J | "
            #     f"<<ENGINE>> "
            #     f"T_ind={T_ind_stroke_mean:7.1f} | "
            #     # f"T_fric={self.torque_friction:7.1f} | "
            #     # f"T_brk={self.torque_brake:7.1f} | "
            #     f"T_net={T_net_stroke_mean:7.1f}"
            # )
            # print("")

            if stroke == "exhaust":
                self.P_cyl = self.P_manifold
                self._cycle_fuel_injected_cc = (
                    0.0  # vent any fuel that did not get ignited due to DFCO
                )

            if stroke == "intake":  # every full cycle
                # Full 720° averages (this is what dynos, papers, and ECUs use)
                T_ind_720 = np.mean(self.torque_indicated_engine)  # Nm
                # T_fric_720 = np.mean([pf.calc_friction_torque_mean(r) for r in np.linspace(800, self.rpm, 10)])  # or just your mean function
                # T_brake_720 = T_ind_720 - T_fric_720
                # T_net_720   = T_brake_720 - self.wheel_load
                T_net_720 = np.mean(self.T_net_engine)
                power_kw = T_net_720 * self.rpm * np.pi / (30.0 * 1000.0)  # kW

                # update data for logging and dashboard
                # self.engine_data_dict.update({
                #         "theta": self.current_theta,
                #         "brake_torque_nm": self.torque_brake,
                #         # "brake_torque_nm": T_net_720,
                #         # 'torque_brake_nm': T_brake_720,
                #         # 'torque_net': T_net_720,
                #         "brake_power_kw": power_kw,
                #         # 'air_mass_per_cyl_kg': self._cylinder_total_air_mass_kg,
                #         # 'injected_fuel_cc': self._cycle_fuel_injected_cc,
                #         "peak_pressure_bar": (max(self.log["P"]) / 1e5 if self.log["P"] else 1.0),
                #     })

                # ======== PRINT END OF CYCLE SUMMARY =========
                # print(
                #     f"  720° CYCLE SUMMARY | Cycle {self._cycle_count:2d} | "
                #     f"  RPM {self.rpm:4.0f} | "
                #     f"  MAP {self.P_manifold/1000:5.1f} kPa | "
                #     f"  Torque Ind: {T_ind_720:+6.1f} Nm | "
                #     f"  Wheel load: {self.wheel_load:+6.1f} Nm | "
                #     f"  Torque Eng : {T_net_720:+6.1f} Nm | "
                #     f"  Power : {power_kw:+6.2f} kW"
                # )

        # =================================================================
        # 3. Volume & dV this step
        # =================================================================
        # Find nearest index in pre-computed theta_list
        V_current = self.V_list[i]
        V_next = self.V_list[i_next]
        dV = V_next - V_current

        # =================================================================
        # 4. FUEL INJECTION — real, per-degree delivery
        # =================================================================
        if ecu_injector_on:
            # Injector is commanded ON for the full 1° duration
            self._cycle_fuel_injected_cc = (
                self._cycle_fuel_injected_cc + fuel_cc_per_degree
            )

        # =================================================================
        # 4. AIR INTAKE per-degree linear intake and exhaust
        # =================================================================
        if stroke == "intake":
            self._cylinder_total_air_mass_kg = (
                self._cylinder_total_air_mass_kg + trapped_air_kg_per_deg
            )
            self._exhaust_g_per_degree = 0
        elif stroke == "exhaust":
            if abs(theta - 540) % 540 <= 0.01:
                self._exhaust_kg_per_degree = self._cylinder_total_air_mass_kg / 180
            self._cylinder_total_air_mass_kg = (
                self._cylinder_total_air_mass_kg - self._exhaust_kg_per_degree
            )

        # =================================================================
        # 5. COMBUSTION — triggered by ECU spark command
        # =================================================================
        Q_in_per_degree = 0.0
        total_heat_J = 0.0

        # Check if spark is commanded (START OF BURN)
        if ecu_spark_command:
            fuel_available_kg = self._cycle_fuel_injected_cc * c.FUEL_DENSITY_KG_CC
            actual_afr = (
                (self._cylinder_total_air_mass_kg / fuel_available_kg)
                if fuel_available_kg > 0
                else 99.0
            )
            self.afr_sensor = actual_afr
            combustion_efficiency = 0.94 if 11.5 <= actual_afr <= 16.0 else 0.80

            # Calculate and set the heat rate state
            total_heat_J = (
                fuel_available_kg * c.LHV_FUEL_GASOLINE * combustion_efficiency
            )
            self.total_cycle_heat_J = total_heat_J
            self._burn_heat_per_deg = total_heat_J / c.BURN_DURATION_DEG
            self._burn_duration_remaining = c.BURN_DURATION_DEG

            self._cycle_fuel_injected_cc = 0.0

        # Apply heat if burn is in progress
        if self._burn_duration_remaining > 0.0:
            Q_in_per_degree = self._burn_heat_per_deg
            self._burn_duration_remaining -= 1.0  # Decrement burn counter by 1 degree

            # Log total heat for sanity check (only logs once per cycle)
            # total_heat_J = self._burn_heat_per_deg * c.BURN_DURATION_DEG
        else:
            # total_heat_J = 0.0
            # self.total_cycle_heat_J = 0.0
            self._burn_heat_per_deg = 0.0

        # =================================================================
        # 6. First law — 1° integration
        # =================================================================
        P_next, T_next = pf.integrate_first_law(
            P_curr=self.P_cyl,
            T_curr=self.T_cyl,
            M_curr=self.M_gas,
            V_curr=V_current,
            Delta_M=0.0,
            Delta_Q_in=Q_in_per_degree,
            Delta_Q_loss=0.0,
            dV_d_theta=dV,
            gamma=c.GAMMA_AIR,
            theta_delta=c.THETA_DELTA,
        )

        # =================================================================
        # 5. Update engine performance per degree
        # =================================================================
        # delta_work_J = self.P_cyl * dV
        P_avg = (self.P_cyl + P_next) / 2.0
        delta_work_J = P_avg * dV
        # --- ASSUMED PRIOR CALCULATIONS ---
        # 1. delta_work_J = self.current_P * self.dV_last_step  # instantaneous work (P*dV)
        # 2. self.dV_last_step is the change in volume from last step to current step.
        # --------------------------------------------------------------------------------

        # 1. CONSTANTS FOR INSTANTANEOUS CALCULATION
        # Convert 1.0 degree to radians (assuming c.THETA_DELTA is 1.0)
        THETA_DELTA_RAD = c.THETA_DELTA * (np.pi / 180.0)

        # 2. INSTANTANEOUS INDICATED TORQUE (Replaces the cycle-averaged calculation)
        # T_ind_cyl3 = Work / Radians
        # T_ind, total = T_ind, cyl3 *
        torque_indicated_cyl1_raw = delta_work_J / THETA_DELTA_RAD

        if stroke == "intake":
            # Pumping stroke: Must consume work (Negative Torque)
            # T_ind_raw is currently POSITIVE during intake, so we must negate it.
            torque_indicated_cyl1 = -abs(torque_indicated_cyl1_raw)

        else:
            # Power (positive) and Compression (negative) signs are correct.
            torque_indicated_cyl1 = torque_indicated_cyl1_raw

        # # firing order is not important but using 1432 because I love early air-cooled
        self.torque_indicated_cyl_1[i] = torque_indicated_cyl1
        self.torque_indicated_cyl_4[(i + 180) % 720] = torque_indicated_cyl1
        self.torque_indicated_cyl_3[(i + 360) % 720] = torque_indicated_cyl1
        self.torque_indicated_cyl_2[(i + 540) % 720] = torque_indicated_cyl1

        torque_indicated = (
            self.torque_indicated_cyl_1[i]
            + self.torque_indicated_cyl_2[i]
            + self.torque_indicated_cyl_3[i]
            + self.torque_indicated_cyl_4[i]
        )
        self.torque_indicated_engine[i] = torque_indicated
        


        # 3. FRICTION, BRAKE, AND NET TORQUE
        torque_friction = pf.calc_friction_torque_per_degree(self.rpm)
        torque_brake = torque_indicated - torque_friction
        self.torque_brake_720[i] = torque_brake
        
        # --- DEBUG CODE ----
        if self.wheel_load is None:
            print(f"ERROR ENGINE: wheel_load is None at theta={self.current_theta}")
            print(f"Last driver load: {getattr(self, '_last_driver_load', 'unknown')}")
            import traceback
            traceback.print_stack()
            sys.exit()
        # ---
        
        
        T_net_engine = torque_brake - self.wheel_load
        
        # print(
        #     "EngineModel  "
        #     f"{self._cycle_count}/{i} | "
        #     f"brake torque= {torque_brake:6.0f} | "
        #     f"wheel laod= {self.wheel_load:6.0f} | "
        #     f"T_net= {T_net_engine:6.0f} | "
        # )

        self.torque_friction = torque_friction
        self.torque_brake = torque_brake
        self.T_net_engine[i] = T_net_engine

        # 4. DYNAMIC RPM UPDATE (Requires time step correction)

        # if not engine_running:
        # if self.rpm <900:
        #     damping = 0.2
        #     alpha = alpha * damping

        # CORRECTED TIME STEP (dt) - Time taken to rotate 1.0 degree at current RPM
        # dt (sec/deg) = 1.0 / (RPM * 6.0)
        # if self.rpm > 5.0:
        #     dt = c.THETA_DELTA / (self.rpm * 6.0)
        # else:
        #     # Use a small, safe time step near stall
        #     dt = 0.001
        # dt = 0.001
        # alpha = T_net_engine / c.MOMENT_OF_INERTIA # Assuming c.MOMENT_OF_INERTIA is flywheel inertia (I)

        omega = pf.eng_speed_rad(self.rpm)
        dt = np.deg2rad(1.0) / omega

        alpha = T_net_engine / c.MOMENT_OF_INERTIA  # rad/s²
        self.rpm = self.rpm + alpha * dt * 30.0 / np.pi  # convert back to RPM
        # rpm_next = self.rpm + alpha * dt * 30.0 / np.pi       # convert back to RPM

        # # === REV LIMITER LOGIC ===
        # RPM_LIMIT_SOFT = c.RPM_LIMIT - 300
        # if rpm_next > c.RPM_LIMIT:
        #     # Hard wall – engine cannot physically spin faster
        #     self.rpm = c.RPM_LIMIT
        # elif rpm_next > RPM_LIMIT_SOFT:
        #     # Soft limiter zone – progressive cut (realistic and RL-friendly)
        #     # This gives a nice "buzz" feel and prevents hard bang-bang
        #     excess = rpm_next - RPM_LIMIT_SOFT
        #     limit_factor = 1.0 - (excess / (c.RPM_LIMIT - RPM_LIMIT_SOFT))
        #     limit_factor = max(0.0, limit_factor)          # 1.0 → 0.0 as RPM climbs
        #     self.rpm = RPM_LIMIT_SOFT + excess * limit_factor**2   # quadratic drop-off, feels natural
        # else:
        #     # Normal operation
        #     self.rpm = rpm_next

        # # Final safety clamp + idle floor
        # self.rpm = max(0.0, min(self.rpm, c.RPM_LIMIT))

        if self.rpm < c.CRANK_RPM:
            self.rpm = c.CRANK_RPM
        
        self.rpm_history[i] = self.rpm
            
        # if i % 120 == 0:
        #     print(
        #         f"theta: {self._cycle_count:3d}/{i:3d} | "
        #         f"Cycle end instataneous RPM: {self.rpm:4.0f} | "
        #         f" 720 average rpm: {np.mean(self.rpm_history):4.0f} | "
        #         f" Effective TPS: {self.effective_tps:4.1f}"
        #         )
        
        # if i % 20 == 0:
        #     print(f"theta: {i:3d}  T_ind:  "
        #         f"1: {self.torque_indicated_cyl_1[i]:4.0f}    "
        #         f"2: {self.torque_indicated_cyl_2[i]:4.0f}    "
        #         f"3: {self.torque_indicated_cyl_3[i]:4.0f}    "
        #         f"4: {self.torque_indicated_cyl_4[i]:4.0f}    "
        #         f"engine:  {torque_indicated:4.0f}    "
        #         f"avg: {np.mean(self.torque_indicated_engine):4.0f}   "
        #         f"rpm: {self.rpm:3.0f}"
        #         )

        # --------------------------------------------------------------------------------
        # *IMPORTANT: Keep self._cycle_work_J accumulation, but for logging only.*
        # self._cycle_work_J += delta_work_J
        # --------------------------------------------------------------------------------

        # DEBUG FOR ANALYSIS OF CYLINDER TORQUE
        # print(
        #     f"cyc{self._cycle_count:3d} | "
        #     f"θ={i:3.0f} | "
        # #     f"{stroke:8s} | "
        # #     # f"rpm={self.rpm:4.0f} | "
        # #     f"spark={'1' if ecu_spark_command else '':1.1s} | "
        # #     # f"inj={'1' if ecu_injector_on else '':1.1s} | "
        # #     # f"fuel={self._cycle_fuel_injected_g*1000:4.1f}mg | "
        # #     # f"heat={self._burn_heat_per_deg:6.1f}J | "
        # #     # f"Q_in={Q_in_per_degree:4.1f}/deg | "
        # #     f"d_work={delta_work_J:5.1f}J | "
        # #     f"dV={dV:9.2e}m3 | "
        # #     f"P_cyl={self.P_cyl / 1e5:6.1f}bar | "
        # #     f"P_next={P_next / 1e5:6.1f}bar | "
        # #     f"V_curr={V_current:9.2e}m3 | "

        # #     f"V_next={V_next:9.2e}m3 | "

        #     # f"T_ind={torque_indicated:7.1f} | "
        #     # f"totals ||"
        #     # f"T_ind={torque_indicated:7.1f} | "
        #     # f"T_fric={torque_friction:7.1f} | "
        #     f"T_brk= {torque_brake:7.1f} | "
        #     f"wheel load= {self.wheel_load:5.1f} | "
        #     f"T_net= {T_net_engine:7.1f} | "
        # )

        #  DEBUG print for INJECTION and SPARK
        # print(
        #     f"cyc{self._cycle_count:3d} | "
        #     f"θ={i:3.0f} | "
        #     f"{stroke:0.5s} | "
        #     f"rpm={self.rpm:4.0f} | "
        #     # f"eTPS={self.effective_tps:4.0f} | "
        #     f"inj={'1' if ecu_injector_on else '':1.1s} | "
        #     f"fuel={self._cycle_fuel_injected_cc:6.4f}cc | "
        #     f"spark={'1' if ecu_spark_command else '':1.1s} | "
        #     f"heat={self._burn_heat_per_deg:6.4f}J | "
        #     f"P_next={P_next/1000:6.1f}kPa | "
        #     # f"Q_in={Q_in_per_degree:4.1f}/deg | "
        #     # f"dV={dV:9.2e}m3 | "
        #     # f"V_next={V_next:1.2e} | "
        #     # f"V_current={V_current:1.2e} | "
        #     f"d_work={delta_work_J:5.1f}J | "
        #     f"T_ind={torque_indicated:7.1f} | "
        #     # f"T_fric={torque_friction:7.1f} | "
        #     # f"T_brk={torque_brake:7.1f} | "
        #     # f"T_net={T_net:7.1f} | "
        # )

        # Stability — keeps model alive forever
        P_next = np.clip(P_next, 5_000, 18_000_000)
        # P_next = 0.70 * P_next + 0.30 * self.P_cyl
        T_next = np.clip(T_next, 220, 3500)
        # T_next = 0.80 * T_next + 0.20 * self.T_cyl

        # Indicated work this degree
        # self._cycle_work_J = self._cycle_work_J + P_next * dV

        # print(
        #     f"cycle_count{self._cycle_count:3d} | "
        #     f"theta={self.current_theta:4.0f} | "
        #     f"rpm={self.rpm:4.0f} | "
        #     f"spark={'1' if ecu_spark_command else '':1.1s} |"
        #     f"heat={total_heat_J:6.1f}J | "
        #     f"Q_in={Q_in_per_degree:4.1f}/deg | "
        #     f"P-next={P_next:8.0f}J | "
        #     f"dV={dV:1.4e}m3 | "
        #     f"total_heat={total_heat_J:6.1f} |"
        # )

        # print(
        #     "#=================================="
        #     "# OLD END OF CYCLE DEBUG LOG"
        #     "#=================================="
        # )
        # print(
        #     f"theta={self.current_theta:4.0f} | "
        #     f"old_rpm={old_rpm:4.0f} | "
        #     f"new_rpm={self.rpm:4.0f} |"
        #     f"afr={self.afr_sensor:4.1f} | "
        #     f"torque_ind={torque_indicated:5.3f} | "
        #     f"friction={torque_friction:4.1f} | "
        #     f"crank={starter_motor:4.1f} | "
        #     f"extern_load={self.wheel_load} | "
        # )
        # print(
        #     f"T_net={T_net:3.2f} | "
        #     f"alpha={alpha:3.2f} | "
        #     f"dt={dt:4.2f} | "
        # )
        # print(
        #     "#==================================")

        # Update state
        self.P_cyl = P_next
        self.T_cyl = T_next

        # Log
        self.log["P"].append(P_next)
        self.log["V"].append(V_current)
        self.log["T"].append(T_next)

        # # update data for logging and dashboard
        # self.engine_data_dict.update({
        #     'theta': self.current_theta,  # start of next cycle
        #     'brake_torque_nm': torque_brake,
        #     'brake_power_kw': (torque_brake * self.rpm) / 9549.0,
        #     'air_mass_per_cyl_kg': self._cylinder_total_air_mass_kg,
        #     'injected_fuel_cc': self._cycle_fuel_injected_cc,
        #     'peak_pressure_bar': max(self.log['P']) / 1e5 if self.log['P'] else 1.0,
        # })

        # if self.current_theta >= 719.0:  # this is the last cycle has completed and the next cycle is 0
        #     self._end_of_cycle_update()

        return None

    # ----------------------------------------------------------------------
    def _end_of_cycle_update(self):
        """Called exactly once per 720° — torque, RPM, sensors"""
        # # Torque
        # torque_indicated = (self._cycle_work_J * c.NUM_CYL) / (4 * np.pi)
        # torque_friction = pf.calc_friction_torque(self.rpm)
        # torque_brake = torque_indicated - torque_friction

        # starter_gain = max(1, 1.001 * (c.CRANK_RPM - self.rpm))
        # # cranking = min(30, Gain  * (c.CRANK_RPM - self.rpm))
        # starter_motor = torque_friction if self.rpm <= c.CRANK_RPM else 0.0
        # T_net = torque_brake + starter_motor - self.wheel_load

        # # RPM update
        # alpha = T_net / c.MOMENT_OF_INERTIA
        # dt = 120.0 / max(self.rpm, 5.0)
        # omega_new = pf.eng_speed_rad(self.rpm) + alpha * dt
        # old_rpm = self.rpm
        # self.rpm = np.clip(omega_new * 30.0 / np.pi, 0.1, 9500.0)

        # print(
        #     "#=================================="
        #     "# END OF CYCLE DEBUG LOG"
        #     "#=================================="
        # )
        # print(
        #     f"theta={self.current_theta:4.0f} | "
        #     # f"old_rpm={old_rpm:4.0f} | "
        #     f"new_rpm={self.rpm:4.0f} |"
        #     f"afr={self.afr_sensor:4.1f} | "
        # f"torque_ind={torque_indicated:5.3f} | "
        # f"friction={torque_friction:4.1f} | "
        # f"crank={starter_motor:4.1f} | "
        #     f"extern_load={self.wheel_load} | "
        # )
        # print(
        #     f"T_net={T_net:3.2f} | "
        #     f"alpha={alpha:3.2f} | "
        #     f"dt={dt:4.2f} | "
        #     f"omega_new={omega_new:6.2f} |"
        # )
        # print(
        #     "#==================================")

        # # Fuel & AFR
        # fuel_injected_g = self._cycle_fuel_injected_g
        # actual_afr = (trapped_air_g / max(fuel_injected_g, 1e-9)
        #               if fuel_injected_g > 0 else 99.0)
        # self.afr_sensor = actual_afr

        # print(self._cycle_fuel_injected_g, trapped_air_g, fuel_injected_g)

        # Update engine_data_dict — only once per cycle
        # self.engine_data_dict.update({
        #     'theta': self.current_theta,  # start of next cycle
        #     'brake_torque_nm': torque_brake,
        #     'brake_power_kw': (torque_brake * self.rpm) / 9549.0,
        #     'total_air_mg': trapped_air_g * 1000.0 * c.NUM_CYL,
        #     'air_mg_per_cyl': trapped_air_g * 1000.0,
        #     'injected_fuel_mg': self._cycle_fuel_injected_g * 1000.0,
        #     'peak_pressure_bar': max(self.log['P']) / 1e5 if self.log['P'] else 1.0,
        # })

        # Reset cycle accumulators
        # self._cycle_work_J = 0.0
        # self._cycle_fuel_injected_g = 0.0
        # self._cylinder_total_air_mass_g = 0.0
        # self.log = {'P': [], 'V': [], 'T': []}
        # self._cycle_count += 1

    def _is_dead_center(self, theta, tolerance=1e-3):
        """Checks if the crank angle (theta) is at a Dead Center."""

        # Use standard Python modulo operator (%)
        remainder = theta % 180.0

        # Use NumPy's robust float comparison
        is_close_to_zero = np.isclose(remainder, 0.0, atol=tolerance)
        is_close_to_180 = np.isclose(remainder, 180.0, atol=tolerance)

        return is_close_to_zero | is_close_to_180

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

    # ----------------------------------------------------------------------
    # def _update_crank_cam_sync(self):
    #     """
    #     Updates crank: 36-1 wheel → 10° per tooth, missing tooth = reference
    #     cam: single pulse at 630° (90° BTDC cyl #1)
    #     """
    #     # Advance true position
    #     self.current_theta_720 = (self.current_theta_720 + c.THETA_DELTA) % 720.0
    #     self.current_theta_360 = self.current_theta_720 % 360.0

    #     # === CRANK SIGNAL (36-1) ===
    #     tooth_float = self.current_theta_360 / self.crank_deg_per_tooth
    #     tooth_int = int(tooth_float) % self.crank_teeth_total

    #     # Detect tooth edge (rising)
    #     if tooth_int != self.current_crank_tooth:
    #         self.current_crank_tooth = tooth_int

    #         # Missing tooth = sync point (should be at TDC #1)
    #         if tooth_int == 0:  # first tooth after missing = reference
    #             if self.crank_sync_lost:
    #                 # Resync crank to known position
    #                 self.current_theta_720 = 0.0
    #                 self.current_theta_360 = 0.0
    #                 self.crank_sync_lost = False

    #     # === CAM SIGNAL — single pulse at 630° (90° BTDC cyl #1)
    #     cam_pulse_width = 30.0  # degrees wide
    #     if (self.cam_sync_pulse_deg_720 - cam_pulse_width/2 <=
    #         self.current_theta_720 <=
    #         self.cam_sync_pulse_deg_720 + cam_pulse_width/2):
    #         self.cam_sync_active = True
    #     else:
    #         self.cam_sync_active = False

    #     # Rising edge = sync
    #     if self.cam_sync_active and self.cam_sync_lost:
    #         self.cam_sync_lost = False
    #         self.engine_phase_known = True  # now we know it's cylinder 1
