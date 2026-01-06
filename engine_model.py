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
from dataclasses import dataclass, field

@dataclass
class EngineSensors:
    """The interface between the engine physics and the ECU"""
    rpm: float
    P_manifold_Pa: float = c.P_ATM_PA
    MAP_kPa: float = P_manifold_Pa / 1000.0
    TPS_percent: float = 0.0
    CLT_C: float = c.COOLANT_START
    AFR: float = 14.7
    lambda_: float = AFR / 14.7
    knock: bool = False
    knock_intensity: float = 0.0
    crank_pos: int = 0
    cam_sync: bool = False
    ambient_pressure: float = c.P_ATM_PA
    rpm_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    


@ dataclass
class CylinderState:
    """Master Cylinder (Cyl 1) Attributes and Instantaneous State"""
    
    # Geometry Attributes
    A_piston: float = c.A_PISTON
    V_displaced: float = c.V_DISPLACED
    V_clearance: float = c.V_DISPLACED / (c.COMP_RATIO - 1.0)

    # Thermodynamic State (Uppercase for Physics Formulas)
    P: float = c.P_ATM_PA   # Pa
    T: float = c.T_AMBIENT  # Kelvin
    M_gas: float = 5.8e-4   # kg.  Mass of a cylinder of air
    log_P: np.ndarray = field(default_factory=lambda: np.zeros(720))
    log_V: np.ndarray = field(default_factory=lambda: np.zeros(720))
    log_T: np.ndarray = field(default_factory=lambda: np.zeros(720))
    P_peak_bar: float = 0.0
    P_peak_angle: float = 0.0
    
    # Flow Accumulators
    air_mass_kg: float = 0.0
    fuel_mass_cc: float = 0.0
    fuel_mass_kg: float = 0.0
    exhaust_kg_per_deg: float = 0.0
    
    
    # Combustion State
    spark_advance_btdc: float = 0.0
    combustion_active: bool = False
    spark_event_theta: float = 0.0
    total_cycle_heat_J: float = 0.0
    cumulative_heat_released: float = 0.0
    _burn_heat_per_deg: float = 0.0
    _burn_duration_remaining:float = 0.0
    _exhaust_g_per_degree: float = 0.0
    knock: bool = False
    knock_intensity: float = 0.0
   
    # History for Phasing
    torque_indicated_history: np.ndarray = field(default_factory=lambda: np.zeros(720)) # indicated
    
    def __post_init__(self):
        self.valves = ValveGeom()
        
        # Precomputed cylinder  vectors    
        theta_array = np.arange(c.THETA_MIN, c.THETA_MAX, c.THETA_DELTA)
        self.V_list = pf.v_cyl(theta_array, self.A_piston, self.V_clearance)

    
@dataclass
class ValveGeom:
    """stores valve geometry"""
    
    # Intake valve geometry
    IVO: float = 720 - c.VALVE_TIMING['intake']['open_btdc']
    IVC: float = 180 + c.VALVE_TIMING['intake']['close_abdc']
    intake_lift_mm: float = c.VALVE_TIMING['intake']['max_lift']
    intake_diam_mm: float = c.VALVE_TIMING['intake']['diameter']
          
    # Exhaust valve geometry  
    EVO: float = 540 - c.VALVE_TIMING['exhaust']['open_bbdc']
    EVC: float = 0 + c.VALVE_TIMING['exhaust']['close_atdc']
    exhaust_lift_mm: float = c.VALVE_TIMING['exhaust']['max_lift']
    exhaust_diam_mm: float = c.VALVE_TIMING['exhaust']['diameter']
    
 
        

@dataclass
class EngineState:
    """Mechanical and Operational State of the Engine"""
    
    # geometry
    crank_teeth_total: int = 36
    missing_tooth_index: int = 35
    deg_per_tooth: int = 360.0 / crank_teeth_total
    
    # timing
    current_theta: float = 0.0
    next_theta: float = 0.0
    crank_synced: bool = False
    cam_sync: bool = False
 
    
    # state
    # P_manifold: float = c.P_ATM_PA
    # rpm: float = 0.0
    # map_Pa: float = c.P_ATM_PA
    effective_tps: float = 0.0
    # crank_tooth: int = 0  # 0–35
    # afr: float = 14.7
    # lambda_: float = afr / 14.7
    
       
    # Global History Buffers
    # rpm_history: np.ndarray          = field(default_factory=lambda: np.zeros(720))
    map_history: np.ndarray          = field(default_factory=lambda: np.zeros(720))
    torque_brake_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    torque_net_history: np.ndarray   = field(default_factory=lambda: np.zeros(720))
    power_history: np.ndarray        = field(default_factory=lambda: np.zeros(720))
    
    # engine output
    wheel_load: float = 0.0
    torque_indicated: float = 0.0
    torque_friction: float = 0.0
    torque_brake: float = 0.0
    
    

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
        # Initialize Blocks
        self.cyl = CylinderState()
        self.state = EngineState()
        self.sensors = EngineSensors(rpm=rpm)
        
        # counters
        self._cycle_count = 0


        self.engine_data_dict = FixedKeyDictionary({
            "theta": 0,
            "torque_history": self.state.torque_brake_history,
            # "rpm_history": self.sensors.rpm_history,
            "power_history": self.state.power_history,
            "air_mass_per_cyl_kg": self.cyl.air_mass_kg,          # per cylinder average
            "peak_pressure_bar": self.cyl.P_peak_bar,
            "map_avg_kPa": np.mean(self.state.map_history),
            "torque_std_nm": np.std(self.state.torque_brake_history),
        })



    # ----------------------------------------------------------------------
    def get_sensors(self):
        s = self.sensors
               
        # s.rpm = self.sensors.rpm
        # # s.rpm_720_history = self.sensors.rpm_history
        # s.MAP_kPa = self.sensors.P_manifold_Pa / 1000.0
        # s.ambient_pressure = s.ambient_pressure # set externally by ECU
        # s.TPS_percent = s.TPS_percent # set externally by ECU
        # s.AFR = self.sensors.afr
        # s.lambda_ = self.sensors.afr / 14.7
        # s.knock = self.sensors.knock
        # s.knock_intensity = self.sensors.knock_intensity
        # s.clt_C = self.state.clt
        # s.crank_pos = self.state.crank_tooth
        # s.cam_sync = self.state.cam_sync
        return self.sensors
        

    # ----------------------------------------------------------------------
    def get_engine_data(self):
        self.engine_data_dict.update({
            "theta": self.state.current_theta,
            "torque_history": self.state.torque_brake_history,
            # "rpm_history": self.sensors.rpm_history,
            "power_history": self.state.power_history,
            "air_mass_per_cyl_kg": self.cyl.air_mass_kg,          # per cylinder average
            "peak_pressure_bar": self.cyl.P_peak_bar,
            "map_avg_kPa": np.mean(self.state.map_history),
            "torque_std_nm": np.std(self.state.torque_brake_history),
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
        
        print(f"theta: {self.state.current_theta} rpm: {self.sensors.rpm}  MAP_kPa: {self.sensors.MAP_kPa} air:{self.cyl.air_mass_kg}")
        print(ecu_outputs)
        print("\n")
        
        self.state.current_theta = self.state.next_theta
        theta = self.state.current_theta  # both are 0 at __init so cycle 0 works.
        CAD = int(theta) # Crank Angle Deegree
        
        # Crank: 36-1 wheel
        tooth_float = (theta % 360.0) / self.state.deg_per_tooth
        new_tooth = int(tooth_float) % self.state.crank_teeth_total

        if new_tooth != self.sensors.crank_pos:
            self.sensors.crank_pos = new_tooth
            if new_tooth == 0 and not self.state.crank_synced:
                # self.state.current_theta = 0.0  # resync on missing tooth
                self.state.crank_synced = True

        # Cam: pulse at 630° (90° BTDC)
        self.state.cam_sync = 620 <= theta <= 640

        # Manifold Pressure (MAP) Physics
        # Combine driver TPS and ECU idle control
        effective_tps = np.clip(
            self.sensors.TPS_percent + ecu_outputs["idle_valve_position"], 0, 100
        )  # clipo to ensure there is no idle_valve open at WOT
        self.state.effective_tps = effective_tps

        # Realistic throttle → non-linear MAP curve
        tps_frac = np.clip(effective_tps / 100.0, 0.0, 1.0)
        
        if effective_tps < 20.0:
            tps_frac_low = effective_tps / 20.0
            # Deep vacuum at closed + small idle bypass
            map_Pa = c.P_ATM_PA * (0.30 + 0.20 * tps_frac_low**2)  # 30 kPa at 0%, ~50 kPa at 20%
        else:
            # Transition to linear/full above idle range
            map_Pa = c.P_ATM_PA * (0.50 + 0.50 * tps_frac**1.3)   
            
        # store truth     
        self.sensors.P_manifold_Pa = map_Pa
        self.state.map_history[CAD] = map_Pa
        
        # engine_running state determined by physical RPM truth
        engine_running = self.sensors.rpm > 600

        # Run physics for this ONE degree
        self._step_one_degree(ecu_outputs, engine_running)

        # Increment and cycle management
        self.state.next_theta = (theta + 1.0) % 720.0
        if theta >= 719.0:  # this is the last cycle has completed and the next cycle is 0
            self._cycle_count += 1
            # self._handle_end_of_cycle()
            
        # print(f"ENGINE at theta {theta}: "
        #     f"next_theta: {self.state.next_theta} | "
        #     f"current_theta: {self.state.current_theta} | "
        #     )

        return self.get_sensors(), self.get_engine_data()
    
    # ----------------------------------------------------------------------
    def _step_one_degree(self, ecu_outputs, engine_running):
        """
        Called every 1° of crank rotation.
        """
        theta = self.state.current_theta
        CAD = int(np.round(theta)) % 720
        stroke, _ = self._get_stroke()
        
        # 1. Stroke Transitions & Reset Logic
        self._handle_stroke_transitions(CAD, stroke)

        # 2. Mass Flow (Air & Fuel)
        self._update_mass_flow(ecu_outputs, stroke)

        # 3. Combustion Heat Release (Current Linear Model)
        # Q_in = self._calculate_linear_heat_release(ecu_outputs["spark"])
        Q_in = self._calculate_combustion_heat(ecu_outputs["spark"]
            
        )

        # 4. Thermodynamics (First Law Integration)
        V_curr = self.cyl.V_list[CAD]
        dV = self.cyl.V_list[(CAD + 1) % 720] - V_curr
        
        P_next, T_next = pf.integrate_first_law(
            P_curr=self.cyl.P, T_curr=self.cyl.T, M_curr=self.cyl.M_gas,
            V_curr=V_curr, Delta_M=0.0, Delta_Q_in=Q_in, Delta_Q_loss=0.0,
            dV_d_theta=dV, gamma=c.GAMMA_AIR, theta_delta=c.THETA_DELTA
        )

        # 5. Mechanical Dynamics & RPM Update
        self._update_mechanical_dynamics(CAD, stroke, P_next, dV)

        # 6. State Updates & Logging
        self.cyl.P = np.clip(P_next, 5_000, 18_000_000)
        self.cyl.T = np.clip(T_next, 220, 3500)
        
        self.cyl.log_P[CAD] = P_next
        self.cyl.log_V[CAD] = V_curr
        self.cyl.log_T[CAD] = T_next
        
        
        # print(f"ENGINE at theta {self.state.current_theta}: "
        #     f"Actual_Air: {self.cyl.air_mass_kg:.6f} Kg | "
        #     f"Actual_Fuel: {self.cyl.fuel_mass_cc} cc | "
        #     f"Lambda: { self.sensors.lambda_ :.3f} | "
        #     f"P_cyl: {self.cyl.P/1000:.1f} kPa"
        #     )

        if CAD >= 719.0:
            self._handle_end_of_cycle()

            
    # --- Helper Methods to maintain functionality ---

    def _handle_stroke_transitions(self, CAD, stroke):
        """
        Resets cylinder pressure based on valve opening events 
        to maintain physical consistency.
        """
        
        # if CAD == self.cyl.valves.IVO: # start of gas intake
        if CAD == 0: # start of gas intake
            self.cyl.P = self.sensors.P_manifold_Pa
            self.cyl.air_mass_kg = 0.0
            
        if CAD == 540: # start of exhaust stroke
            self.cyl.P = c.P_ATM_PA # Pressure equalizes with the exhaust manifold (Atmosphere)
            self.cyl.fuel_mass_kg = 0.0
            self.cyl.fuel_mass_cc = 0.0

    # reverting back to the physics based determination of mass flow.
    def _update_mass_flow(self, ecu_outputs, stroke):
        """Calculates physical mass flow using valve geometry and pressure delta."""
        current_rpm_safe = max(self.sensors.rpm, 10.0)
        
        # 1. FIXED FUEL LOGIC (Restored from your 'Old Working Version')
        fuel_per_deg = c.INJECTOR_FLOW_CC_PER_MIN / (current_rpm_safe * 360) 
        if ecu_outputs["injector_on"]:
            print("INJECTOR ON")
            self.cyl.fuel_mass_cc += fuel_per_deg
            self.cyl.fuel_mass_kg = self.cyl.fuel_mass_cc * c.FUEL_DENSITY_KG_CC

        # 3. PHYSICS-BASED AIR CALCULATION (Intake)
        theta_arr = np.array([self.state.current_theta])
                
        A_intake_mm2 = pf.calc_valve_curtain_vectorized(theta_arr, 
                                                        self.cyl.valves,
                                                        "intake",
                                                        self.sensors.rpm)[0]
        A_intake_m2 = A_intake_mm2 * 1e-6
        
        dm_air_kg = pf.intake_mass_flow(
            A_valve=A_intake_m2, 
            P_man=self.sensors.P_manifold_Pa, 
            T_man=c.T_AMBIENT, 
            rpm=self.sensors.rpm
        )
        



        # Update physical state for Intake
        self.cyl.air_mass_kg += dm_air_kg
        self.cyl.M_gas += dm_air_kg
        self.cyl._current_step_dm = dm_air_kg # Critical for integrate_first_law
        
        # print("DEBUG: dm_air  -> "
        #     f"Theta: {self.state.current_theta:3.0f} | "
        #     f"RPM: {self.sensors.rpm:<4.0f} | "
        #     f"P_man: {self.sensors.P_manifold_Pa:>6.0f}Pa | "
        #     f"A_valve: {A_intake_m2:>8.2e}m2 | "
        #     f"dm_air: {dm_air_kg:>10.8f}kg | "
        #     f"total_air: {self.cyl.air_mass_kg:>10.8f}kg | "
        #     )

        # 4. EXHAUST LOGIC (Removing mass from the cylinder)
        if stroke == "exhaust":
            # Determine how much to remove per degree to reach 0 by TDC
            if abs(self.state.current_theta - 540) <= 0.5:
                # Calculate the 'slug' of mass to remove over the 180 deg stroke
                self.cyl.exhaust_kg_per_deg = self.cyl.M_gas / 180.0
            
            # Subtract mass from both the tracker and the physical gas state
            self.cyl.air_mass_kg -= self.cyl.exhaust_kg_per_deg
            self.cyl.M_gas -= self.cyl.exhaust_kg_per_deg
            # Pass the negative mass flow to the integrator to allow pressure to drop
            self.cyl._current_step_dm = -self.cyl.exhaust_kg_per_deg

    def _calculate_combustion_heat(self, spark_command):
        """Calculates heat release using the Wiebe S-curve with energy conservation."""
        
        # 1. Trigger Spark Event
        if spark_command:
            self.cyl.spark_event_theta = self.state.current_theta
            self.cyl.combustion_active = True
            self.cyl.cumulative_heat_released = 0.0 # Track total energy released
            self.cyl.spark_advance_btdc = (360.0 - (self.state.current_theta % 360.0)) % 360
            
            # Setup fuel/AFR/Efficiency
            fuel_kg = self.cyl.fuel_mass_cc * c.FUEL_DENSITY_KG_CC
            afr = (self.cyl.air_mass_kg / fuel_kg) if fuel_kg > 0 else 99.0
            self.sensors.afr = afr
            eff = 0.94 if 11.5 <= afr <= 16.0 else 0.80
            
            self.cyl.total_cycle_heat_J = fuel_kg * c.LHV_FUEL_GASOLINE * eff
            self.cyl.fuel_mass_cc = 0.0
            self.cyl.fuel_mass_kg = 0.0

        # 2. Calculate Heat Release
        if self.cyl.combustion_active:
            q_step = pf.calc_wiebe_heat_rate(
                theta=self.state.current_theta,
                theta_start=self.cyl.spark_event_theta,
                duration=c.BURN_DURATION_DEG,
                total_heat_J=self.cyl.total_cycle_heat_J
            )
            
            # Energy Conservation Guard: Ensure we don't exceed total_heat_J
            # or leave small remainders behind.
            if self.cyl.cumulative_heat_released + q_step > self.cyl.total_cycle_heat_J:
                q_step = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
            
            self.cyl.cumulative_heat_released += q_step
            
            # Check if burn is finished
            if self.state.current_theta >= (self.cyl.spark_event_theta + c.BURN_DURATION_DEG):
                # Final cleanup: If there's a tiny remainder due to float math, dump it here
                remainder = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
                q_step += max(0.0, remainder) 
                
                self.cyl.combustion_active = False
                
            return q_step
            
        return 0.0

    def _update_mechanical_dynamics(self, i, stroke, P_next, dV):
        """Preserves torque integration and RPM physics."""
        P_avg = (self.cyl.P + P_next) / 2.0
        # delta_work_J = P_avg * dV
        # P_ambient is the pressure pushing on the "back" of the piston
        delta_work_J = (P_avg - c.P_ATM_PA) * dV
        
        t_ind_cyl = pf.calc_indicated_torque_step(delta_work_J, stroke)
        
        # after 720 the cylinder buffer is full and simulates 4 cylinders
        self.cyl.torque_indicated_history[i] = t_ind_cyl
        t_total_indicated = (
            self.cyl.torque_indicated_history[i] +
            self.cyl.torque_indicated_history[(i + 180) % 720] +
            self.cyl.torque_indicated_history[(i + 360) % 720] +
            self.cyl.torque_indicated_history[(i + 540) % 720]
        )

        self.state.torque_indicated = t_total_indicated
        
        self.state.torque_friction = pf.calc_friction_torque_per_degree(self.sensors.rpm)
        self.state.torque_brake = self.state.torque_indicated - self.state.torque_friction
        self.state.torque_brake_history[i] = self.state.torque_brake
        self.state.torque_net_history[i] = self.state.torque_brake - self.state.wheel_load

        # RPM Integration
        omega = pf.eng_speed_rad(self.sensors.rpm)
        dt = np.deg2rad(1.0) / omega
        alpha = self.state.torque_net_history[i] / c.MOMENT_OF_INERTIA
        self.sensors.rpm = max(c.CRANK_RPM, self.sensors.rpm + alpha * dt * 30.0 / np.pi)      
        
        self.sensors.rpm_history[i] = self.sensors.rpm
        self.state.power_history[i] = self.state.torque_brake * self.sensors.rpm / 9549.3
        

    # ----------------------------------------------------------------------
    def _handle_end_of_cycle(self):
        """Called exactly once per 720° — torque, RPM, sensors"""
        
        # end of cycle checks
        peak_bar = max(self.cyl.log_P) / 100000.0 # Ensure P is in bar
        self.cyl.P_peak_bar = peak_bar
        # Find the index (crank angle) where that max pressure occurred
        self.cyl.P_peak_angle = np.argmax(self.cyl.log_P)

        self.sensors.knock, self.sensors.knock_intensity = pf.detect_knock(
            peak_bar = peak_bar,
            clt = self.sensors.CLT_C, 
            rpm = self.sensors.rpm,   
            spark_advance = self.cyl.spark_advance_btdc,
            lambda_ = self.sensors.AFR / 14.7,
            fuel_octane = c.FUEL_OCTANE
        )

                
        # prep for next cycle
        self.cyl.log_P.fill(0.0)
        self.cyl.log_V.fill(0.0)
        self.cyl.log_T.fill(0.0)
        
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
        phase = int(self.state.current_theta / 180) + 1
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

 