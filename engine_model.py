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
    TPS_percent: float = 0.0
    CLT_C: float = c.COOLANT_START
    afr: float = 0.0
    knock: bool = False
    knock_intensity: float = 0.0
    crank_pos: int = 0
    cam_sync: bool = False
    ambient_pressure: float = c.P_ATM_PA
    rpm_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    #IAT ??
    
    @property
    def MAP_kPa(self) -> float:
        return self.P_manifold_Pa / 1000.0

    @property
    def lambda_(self) -> float:
        return self.afr / 14.7 if self.afr > 0 else 0.0
    


@ dataclass
class CylinderState:
    """Master Cylinder (Cyl 1) Attributes and Instantaneous State"""
    
    # Geometry Attributes
    A_piston: float = c.A_PISTON
    V_displaced: float = c.V_DISPLACED
    V_clearance: float = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
    V_list: np.ndarray = field(init=False, repr=False)
    dV_list: np.ndarray = field(init=False, repr=False)

    # Thermodynamic State
    P: float = c.P_ATM_PA   # Pa
    T: float = c.T_AMBIENT  # Kelvin
    # M_gas: float = 5.8e-4  
    # M_gas: float = field(init = False) # kg.  Mass of a cylinder of gas (fresh and exhaust)
    log_P: np.ndarray = field(default_factory=lambda: np.full(720, c.P_ATM_PA))
    log_V: np.ndarray = field(default_factory=lambda: np.zeros(720))
    log_T: np.ndarray = field(default_factory=lambda: np.full(720, c.T_AMBIENT))
    P_peak_bar: float = 0.0
    P_peak_angle: float = 0.0
    T_wall: float = c.T_AMBIENT + 20   # adding buffer for start stability and speed
    
    # Thermodynamic tracking
    Q_loss_total: float = 0.0 # heat loss from ctl to cyl_wall
    
    # Flow State
    dm_total: float = 0.0 # Change in cylinder mass air_in - air_out + fuel
    R_specific_blend: float = c.R_SPECIFIC_AIR
    gamma_blend: float = c.GAMMA_AIR
   
    # Flow Accumulators
    air_mass_kg: float = 0.0
    fuel_mass_cc: float = 0.0
    fuel_mass_kg: float = 0.0
    total_mass_kg: float = field(init=False)
    
    # flow debug 
    dm_in_history:  np.ndarray = field(default_factory=lambda: np.zeros(720))
    dm_ex_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    Cd_in_history:  np.ndarray = field(default_factory=lambda: np.zeros(720))
    Cd_ex_history:  np.ndarray = field(default_factory=lambda: np.zeros(720))
    
    
    # Combustion State
    spark_advance_btdc: float = 0.0
    combustion_active: bool = False
    spark_event_theta: float = 0.0
    total_cycle_heat_J: float = 0.0
    cumulative_heat_released: float = 0.0
    burn_duration: float = 0.0
    _burn_heat_per_deg: float = 0.0
    _burn_duration_remaining:float = 0.0
    _exhaust_g_per_degree: float = 0.0
    knock: bool = False
    knock_intensity: float = 0.0
    air_mass_at_spark: float = 0.0
    fuel_mass_at_spark: float = 0.0
    ignition_start_theta: float = 0.0
    m_vibe: float = 0.0
   
    # History for Phasing
    torque_indicated_history: np.ndarray = field(default_factory=lambda: np.zeros(720)) # indicated
    

      
    def __post_init__(self):
        # Precomputed cylinder  vectors    
        theta_array = np.arange(720)
        self.V_list = pf.v_cyl(theta_array, self.A_piston, self.V_clearance)
        V_next = np.roll(self.V_list, -1)  # done this way rather than np.diff to handle index wrapping
        self.dV_list = V_next - self.V_list
        self.total_mass_kg = (self.P * self.V_clearance) / (c.R_SPECIFIC_EXHAUST * self.T)
        

    
@dataclass
class Valve:
    """stores valve geometry"""
    open_angle: float
    close_angle: float
    max_lift: float
    diameter: float
        
@dataclass    
class Valves:
    intake: Valve = field(default_factory=lambda: Valve(
        c.VALVE_TIMING['intake']['open'] % 720,
        c.VALVE_TIMING['intake']['close'] % 720,
        c.VALVE_TIMING['intake']['max_lift'],
        c.VALVE_TIMING['intake']['diameter']
    ))
    
    exhaust: Valve = field(default_factory=lambda: Valve(
        c.VALVE_TIMING['exhaust']['open'] % 720,
        c.VALVE_TIMING['exhaust']['close'] % 720,
        c.VALVE_TIMING['exhaust']['max_lift'],
        c.VALVE_TIMING['exhaust']['diameter']
    ))
    
    # pre computed vectors
    intake_lift_table: np.ndarray = field(init=False, repr=False)
    intake_area_table: np.ndarray = field(init=False, repr=False)
    exhaust_lift_table: np.ndarray = field(init=False, repr=False)
    exhaust_area_table: np.ndarray = field(init=False, repr=False)
    
    
    def __post_init__(self):
        theta_range = np.arange(720)
        self.intake_lift_table = pf.calc_valve_lift_vectorized(theta_range, self.intake) # mm
        self.intake_area_table = pf.calc_valve_area_vectorized(theta_range, self.intake) # m2        
        self.exhaust_lift_table = pf.calc_valve_lift_vectorized(theta_range, self.exhaust) # mm
        self.exhaust_area_table = pf.calc_valve_area_vectorized(theta_range, self.exhaust) # m2
        
 
        

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
    effective_tps: float = 0.0
    T_exhaust_manifold: float = c.T_AMBIENT
 
    # Global History Buffers
    map_history: np.ndarray              = field(default_factory=lambda: np.zeros(720))
    torque_indicated_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    torque_friction_history: np.ndarray  = field(default_factory=lambda: np.zeros(720))
    torque_brake_history: np.ndarray     = field(default_factory=lambda: np.zeros(720))
    torque_net_history: np.ndarray       = field(default_factory=lambda: np.zeros(720))
    power_history: np.ndarray            = field(default_factory=lambda: np.zeros(720))
    
    # engine output
    wheel_load: float = 0.0
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
        self.valves = Valves()
        
        # counters
        # self.state.next_theta = self.valves.intake.open_angle # start the engine at IVO, useful for debugging
        # self.state.current_theta = self.valves.intake.open_angle # start the engine at IVO, useful for debugging
        self._cycle_count = 0
        self.temp_total_dm_f = 0.0
        
        # update cyl gas composition based on initialisation air, fuel and exhaust mix
        self._update_gas_properties()
        
        self.crank_rpm = c.CRANK_RPM  # in motoring mode the driver strategy will set this for engine analysis.


        self.engine_data_dict = FixedKeyDictionary({
            "theta": 0,
            "torque_history": self.state.torque_brake_history,
            "torque_I_history" : self.state.torque_indicated_history,
            "torque_F_history" : self.state.torque_friction_history,
            # "rpm_history": self.sensors.rpm_history,
            "power_history": self.state.power_history,
            "air_mass_per_cyl_kg": self.cyl.air_mass_kg,          # per cylinder average
            "peak_pressure_bar": self.cyl.P_peak_bar,
            "map_avg_kPa": np.mean(self.state.map_history),
            "torque_std_nm": np.std(self.state.torque_brake_history),
            "trapped_air_mass_kg": self.cyl.total_mass_kg,  ##  NOTE THE NAME IS NOT CORRECT
            "peak_pressure_pa": np.max(self.cyl.log_P),
            "peak_temperature_k": np.max(self.cyl.log_T),
            "P_peak_angle": self.cyl.P_peak_angle,
            
            
            # The Energy Audit
            "work_pumping_j": 0.0,
            "work_compression_j": 0.0,
            "work_expansion_j": 0.0,
            "net_work_j": 0.0,
            "friction_work_j": np.sum(self.state.torque_friction_history * (np.pi / 180.0)),
            "log_P": self.cyl.log_P, # The 720-degree pressure history
            "log_V": self.cyl.V_list, # The 720-degree volume history
            "dm_in": self.cyl.dm_in_history,  # Add this to debug oscillations
            "dm_out": self.cyl.dm_ex_history, # Add this
            "work_deg": 0.0,
            "theta_list": np.arange(720).astype(int),
            
            # Valve and Pressure analysis in Motoring strategy
            "intake_area_vec": self.valves.intake_area_table,
            "exhaust_area_vec": self.valves.exhaust_area_table,
            "intake_lift_vec": self.valves.intake_lift_table,
            "exhaust_lift_vec": self.valves.exhaust_lift_table,
        })
        
        self.print_geom()


    def print_geom(self):
        print("")
        print(f"//" *80)
        print(f"IVO:{self.valves.intake.open_angle} IVC:{self.valves.intake.close_angle} EVO:{self.valves.exhaust.open_angle} EVC:{self.valves.exhaust.close_angle}")

    # ----------------------------------------------------------------------
    def get_sensors(self):
        return self.sensors
        

    # ----------------------------------------------------------------------
    def get_engine_data(self):
        work_stats = self._calculate_cycle_work()
        rad_per_degree = np.pi / 180.0
        
        self.engine_data_dict.update({
            "theta": self.state.current_theta,
            "torque_history": self.state.torque_brake_history,
            "torque_I_history" : self.state.torque_indicated_history,
            "torque_F_history" : self.state.torque_friction_history,
            # "rpm_history": self.sensors.rpm_history,
            "power_history": self.state.power_history,
            "air_mass_per_cyl_kg": self.cyl.air_mass_kg,          # per cylinder average
            "peak_pressure_bar": self.cyl.P_peak_bar,
            "map_avg_kPa": np.mean(self.state.map_history),
            "torque_std_nm": np.std(self.state.torque_brake_history),

            # required for motoring dtrategy plot
            "trapped_air_mass_kg": self.cyl.total_mass_kg,
            "peak_pressure_pa": np.max(self.cyl.log_P),
            "peak_temperature_k": np.max(self.cyl.log_T),
            "P_peak_angle": self.cyl.P_peak_angle,
            
            # The Energy Audit
            "work_pumping_j": work_stats["work_pumping_j"],
            "work_compression_j": work_stats["work_compression_j"],
            "work_expansion_j": work_stats["work_expansion_j"],
            "friction_work_j": np.sum(self.state.torque_friction_history) * rad_per_degree,
            "net_work_j": np.sum(self.state.torque_brake_history) * rad_per_degree,
            "log_P": self.cyl.log_P, # The 720-degree pressure history
            "log_V": self.cyl.V_list, # The 720-degree volume history
            "dm_in": self.cyl.dm_in_history,  
            "dm_out": self.cyl.dm_ex_history, 
            # "ve_fraction": self.state.ve_fraction,
            # "friction_torque_nm": self.mechanics.friction_torque
            "work_deg": work_stats["work_deg"],
            "theta_list": np.arange(720).astype(int),
            
            # Valve and Pressure analysis in Motoring strategy
            "intake_area_vec": self.valves.intake_area_table,
            "exhaust_area_vec": self.valves.exhaust_area_table,
            "intake_lift_vec": self.valves.intake_lift_table,
            "exhaust_lift_vec": self.valves.exhaust_lift_table,
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
        
        self.state.current_theta = self.state.next_theta
        CAD = int(self.state.current_theta) # Crank Angle Deegree
        self.state.next_theta = (CAD + 1.0) % 720.0
        
        # prep start of step
        self._handle_step_init(CAD, ecu_outputs)
        
        # Crank: 36-1 wheel
        tooth_float = (CAD % 360.0) / self.state.deg_per_tooth
        new_tooth = int(tooth_float) % self.state.crank_teeth_total
        self.sensors.crank_pos = new_tooth

        # Cam: pulse at 630° (90° BTDC)
        self.sensors.cam_sync = 620 <= CAD <= 640

        # # Manifold Pressure (MAP) Physics
        # # Combine driver TPS and ECU idle control
        # effective_tps = np.clip(
        #     self.sensors.TPS_percent + ecu_outputs["idle_valve_position"], 0, 100
        # )  # clipo to ensure there is no idle_valve open at WOT
        # self.state.effective_tps = effective_tps
        
        # # Realistic throttle → non-linear MAP curve
        # tps_frac = np.clip(effective_tps / 100.0, 0.0, 1.0)
        
        # if effective_tps < 20.0:
        #     tps_frac_low = effective_tps / 20.0
        #     # Deep vacuum at closed + small idle bypass
        #     map_Pa = c.P_ATM_PA * (0.30 + 0.20 * tps_frac_low**2)  # 30 kPa at 0%, ~50 kPa at 20%
        # else:
        #     # Transition to linear/full above idle range
        #     map_Pa = c.P_ATM_PA * (0.50 + 0.50 * tps_frac**1.3)   
            
        # # store truth     
        # self.sensors.P_manifold_Pa = map_Pa
        # self.state.map_history[CAD] = map_Pa
        
        # Run physics for this ONE degree
        self._step_one_degree(ecu_outputs)

        self._handle_step_end(CAD, ecu_outputs)

        return self.get_sensors(), self.get_engine_data()
    
    # ----------------------------------------------------------------------
    def _step_one_degree(self, ecu_outputs):
        """
        Called every 1° of crank rotation.
        """        
        
        CAD = int(self.state.current_theta)
        stroke, _ = self._get_stroke()
        
        self.cyl.log_P[CAD] = self.cyl.P
        self.cyl.log_T[CAD] = self.cyl.T
        self.cyl.log_V[CAD] = self.cyl.V_list[CAD]

        # 2. Mass Flow (Air & Fuel)
        # adding sub-degree increments for flow calculations
        n_sub = 5
        substep_size = 1.0 / n_sub
        
        for substep in range(n_sub):
            # 1. Calculate Deltas (Pure)
            deltas = self._calc_flow_deltas(ecu_outputs, substep_size)
            self.cyl.dm_in_history[CAD] = deltas['dm_i']
            self.cyl.dm_ex_history[CAD] = deltas['dm_e']
            self.cyl.Cd_in_history[CAD] = deltas['Cd_i']
            self.cyl.Cd_ex_history[CAD] = deltas['Cd_e']
            
            # 2. Volume and Heat Loss
            dV_d_theta = self.cyl.dV_list[CAD]
            V_curr = self.cyl.V_list[CAD] + (dV_d_theta * substep_size * substep)
            
            Q_in_sub = self._calculate_combustion_heat_substep(CAD, substep, substep_size, ecu_outputs["spark"])
            Q_loss_sub = pf.calc_woschni_heat_loss(
                theta=CAD,
                rpm=self.sensors.rpm,
                P_curr=self.cyl.P,
                T_curr=self.cyl.T,
                V_curr=V_curr,
                T_wall=self.cyl.T_wall,
                V_clearance = self.cyl.V_clearance,
                theta_delta=substep_size
            )
            # self.cyl.Q_loss_total += Q_loss_sub # track total per cyl for cyl wall T update at end of cycle.
            """ DEBUG TEST """
            # Q_loss_sub = 0

            # 3. Integrate First Law
            # Note: We pass the CURRENT mass here. The integrator handles dM/dtheta.
            P_next, T_next = pf.integrate_first_law(
                P_curr=self.cyl.P, T_curr=self.cyl.T, M_curr=self.cyl.total_mass_kg,
                V_curr=V_curr, Delta_M=deltas["dm_tot"], 
                Delta_Q_in=Q_in_sub, Delta_Q_loss=Q_loss_sub,
                dV_d_theta=dV_d_theta, gamma=self.cyl.gamma_blend, 
                theta_delta=substep_size, T_manifold=deltas["T_inflow"], 
                R_spec=self.cyl.R_specific_blend,
                cycle=self._cycle_count, CAD=CAD, substep=substep
            )

            # 4. ATOMIC STATE UPDATE
            m_old = max(self.cyl.total_mass_kg, 1e-9)
            f_air = self.cyl.air_mass_kg / m_old
            f_fuel = self.cyl.fuel_mass_kg / m_old
            # f_exh is implicitly (1 - f_air - f_fuel)
            
            if 210 < CAD < 460:
                if deltas["dm_i"] > 0: print(f"WANRING: dm_in:{deltas["dm_i"]:.2e} >0 between IVC and EVO at CAD:{CAD}")
                if deltas["dm_e"] < 0: print(f"WANRING: dm_out:{deltas["dm_e"]:.2e} <0 between IVC and EVO at CAD:{CAD}")

            # --- INTAKE PORT FLOW ---
            if deltas["dm_i"] > 0:
                self.cyl.air_mass_kg += deltas["dm_i"]  # Pure air enters from manifold
            else:
                # Mixture leaves (Reversion)
                self.cyl.air_mass_kg  += deltas["dm_i"] * f_air
                self.cyl.fuel_mass_kg += deltas["dm_i"] * f_fuel

            # --- EXHAUST PORT FLOW ---
            if deltas["dm_e"] < 0:
                # Normal Exhaust: Mixture leaves
                self.cyl.air_mass_kg  += deltas["dm_e"] * f_air
                self.cyl.fuel_mass_kg += deltas["dm_e"] * f_fuel
            else:
                # Exhaust Backflow (EGR): Only affects total_mass (it's pure exhaust)
                pass 
            
            # Check for "Phantom Mass"
            sum_species = self.cyl.air_mass_kg + self.cyl.fuel_mass_kg
            if sum_species > self.cyl.total_mass_kg * 1.01:
                print(f"!!! SPECIES OVERFLOW at CAD {CAD}: Sum({sum_species}) > Total({self.cyl.total_mass_kg})")

            # --- FUEL INJECTION ---
            self.cyl.fuel_mass_kg += deltas["dm_f"]

            # --- FINAL TOTAL ---
            self.cyl.total_mass_kg += deltas["dm_tot"]
            self.cyl.P = P_next
            self.cyl.T = T_next                    

            # 5. Re-calculate properties for the NEXT substep
            self._update_gas_properties() # Helper to refresh R_blend and gamma_blend
            
            # --- COMBUSTION DEBUG ---
            # Only print during the expansion stroke where combustion happens
            if 335 <= CAD <= 540 and substep == 4:
                if Q_in_sub > 0:
                # Calculate the 'Energy Density' to see if the heat addition is sane
                    specific_energy_j_kg = Q_in_sub / max(self.cyl.total_mass_kg, 1e-9)
                    
                    print(f"    DEBUG COMB | θ:{self._cycle_count}/{CAD}/{substep} | P:{P_next/1e5:4.1f}bar | T:{T_next:4.0f}K "
                          f"T_wall:{self.cyl.T_wall:4.0f}K "
                          f"dQ_in:{Q_in_sub:6.2f}J | dQ/m:{specific_energy_j_kg:6.0f}J/kg | gamma:{self.cyl.gamma_blend:.3f} "
                          f"dQ_loss:{Q_loss_sub:6.2f}J "
                        )
                    
                # Check for "Thermal Runaway"
                if T_next > 3500 or T_next < 200:
                    print(f"  !!! INSTABILITY DETECTED at CAD {CAD}/{substep}. 3500K < Temp:{T_next:4.0f} < 200K.   Physics breaking.")

        
        # 4. Mechanical Dynamics (Calculated once per degree using final state)
        # We use the full degree dV here because work is summed across the step
        self._update_mechanical_dynamics(CAD, stroke, self.cyl.P, self.cyl.dV_list[CAD])
         
        # if CAD % 10 == 0:
        #     print(
        #     f"DEBUG_cycle "
        #         f"θ: {self._cycle_count}/{CAD} {stroke}  "
        #         f"P:{self.cyl.P:6.0f} T:{self.cyl.T:3.0f} V_curr:{self.cyl.V_list[CAD]:8.2e} " 
        #     )

            
        
        # # We will accumulate mass flow for the 1-degree history log
        # dm_in_acc = 0.0
        # dm_out_acc = 0.0
        # dm_total_acc = 0.0
        # dV_acc = 0.0
        # Q_in_acc = 0.0
        # Q_loss_acc = 0.0
        
        # for substep in range(n_sub):
            
        #     # 1. MASS FLOW UPDATES (the ingredients)
        #     dm_in, dm_out, dm_fuel, dm_total, T_inflow, Cd_i, Cd_e, R_blend, gamma_blend = self._update_mass_flow(ecu_outputs, substep_size)
        #     dm_in_acc += dm_in
        #     dm_out_acc += dm_out
        #     dm_total_acc += dm_total

        #     # 2. COMBUSTION (the energy release)
        #     Q_in_sub = self._calculate_combustion_heat_substep(CAD, substep_size, ecu_outputs["spark"])
        #     Q_in_acc += Q_in_sub

        #     # 3. HEAT LOSS 
        #     # --- Negative Q_loss means heat is flow from cyl wall to the trapped air.
        #     dV_d_theta = self.cyl.dV_list[CAD]
        #     dV_sub = self.cyl.dV_list[CAD] * substep_size
        #     dV_acc += dV_sub
        #     V_curr = self.cyl.V_list[CAD] + dV_sub * substep
                
        #     Q_loss_sub = pf.calc_woschni_heat_loss(
        #         theta=CAD,
        #         rpm=self.sensors.rpm,
        #         P_curr=self.cyl.P,
        #         T_curr=self.cyl.T,
        #         V_curr=V_curr,
        #         T_wall=self.cyl.T_wall,
        #         # dV_d_theta=dV_d_theta,
        #         theta_delta=substep_size
        #     )
        #     Q_loss_acc += Q_loss_sub
        #     self.cyl.Q_loss_total += Q_loss_sub
            
        #     # if CAD < 4:
        #     #     print(
        #     #         f"DEBUG_WOSHCI  "
        #     #         f"θ: {self._cycle_count}/{CAD}/{substep}/{stroke} "
        #     #         f"rpm:{self.sensors.rpm:4.0f} "
        #     #         f"P_curr:{self.cyl.P:6.0f} T_curr:{self.cyl.T:3.0f} V_curr:{V_curr:8.2e} T_wall:{self.cyl.T_wall:3.0f} "  
        #     #         f"dV_d_theta:{dV_d_theta:9.2e}  "
        #     #         f"Q_loss:{Q_loss_sub:6.3f}"
        #     #     )

            
        #     # DEBUG PRINTS
        #     # P = self.cyl.P
        #     # air = self.cyl.air_mass_kg
        #     # fuel = self.cyl.fuel_mass_kg
        #     # mass = self.cyl.total_mass_kg
        #     # exh = max(0, mass - air - fuel)
        #     # i_open = self.valves.intake.open_angle
        #     # e_open = self.valves.exhaust.open_angle
        #     # if CAD >= i_open or CAD <= self.valves.intake.close_angle:
        #     #     if CAD == i_open or CAD == 719 or CAD % 10 == 0 or CAD == self.valves.intake.close_angle:
        #     #         print(f"DEBUG_AIR: CAD:{self._cycle_count}/{CAD:3.0f}/{substep}-{stroke:6s} ")
        #     #         print(f"    Air:{air:9.2e}kg total_m:{mass:9.2e}kg dm_in:{dm_in:8.2e} dm_out:{dm_out:8.2e} dm_total:{dm_total:8.2e} Q_in:{Q_in_sub:8.2e} Q_loss:{Q_loss_sub:8.2e} ")
        #     #         print(f"    P:{P:6.0f}Pa T:{self.cyl.T:3.0f}K P.dV:{P * dV_sub:5.3f} P_i_map:{self.sensors.P_manifold_Pa:6.0f}Pa P_e_map:{c.P_ATM_PA:6.0f}Pa ")
        #     #         print(f"    A_v:{self.valves.intake_area_table[CAD]:8.6f}m2 v_lift:{self.valves.intake_lift_table[CAD]:4.2f}mm Cd_i:{Cd_i:4.3f} ")
        #     #         print(f"    A_v_Ex:{self.valves.exhaust_area_table[CAD]:8.6f}m2 lift_Ex:{self.valves.exhaust_lift_table[CAD]:4.2f}mm Cd_e:{Cd_e:4.3f} ")                   
                    
                
        #     # if CAD >= e_open or CAD <= self.valves.exhaust.close_angle:       
        #     #     if CAD == e_open or CAD % 10 == 0 or CAD == self.valves.exhaust.close_angle or self.valves.exhaust_lift_table[CAD] < 1.2:
        #     #         print(f"DEBUG_AIR: CAD:{self._cycle_count}/{CAD:3.0f}/{substep}-{stroke:6s} ")
        #     #         print(f"    Air:{air:9.2e}kg total_m:{mass:9.2e}kg dm_in:{dm_in:8.2e} dm_out:{dm_out:8.2e} dm_total:{dm_total:8.2e} Q_in:{Q_in_sub:8.2e} Q_loss:{Q_loss_sub:8.2e} ")
        #     #         print(f"    P:{P:6.0f}Pa T:{self.cyl.T:3.0f}K P.dV:{P * dV_sub:5.3f} P_i_map:{self.sensors.P_manifold_Pa:6.0f}Pa P_e_map:{c.P_ATM_PA:6.0f}Pa ")
        #     #         print(f"    A_v:{self.valves.intake_area_table[CAD]:8.6f}m2 v_lift:{self.valves.intake_lift_table[CAD]:4.2f}mm Cd_i:{Cd_i:4.3f} ")
        #     #         print(f"    A_v_Ex:{self.valves.exhaust_area_table[CAD]:8.6f}m2 lift_Ex:{self.valves.exhaust_lift_table[CAD]:4.2f}mm Cd_e:{Cd_e:4.3f} ")      
            

            
        #     # 4. INTERGRATION (state transition)
        #     P_next, T_next = pf.integrate_first_law(
        #         P_curr=self.cyl.P, T_curr=self.cyl.T, M_curr=self.cyl.total_mass_kg,
        #         V_curr=V_curr, Delta_M=dm_total, Delta_Q_in=Q_in_sub, Delta_Q_loss=Q_loss_sub,
        #         dV_d_theta=dV_d_theta, gamma=gamma_blend, theta_delta=substep_size,
        #         T_manifold=T_inflow, R_spec=R_blend, 
        #         cycle=self._cycle_count, CAD=CAD, substep=substep
        #     )
            


        #     # 5. STATE UPDATE
        #     P_pre = self.cyl.P
        #     T_pre = self.cyl.T
        #     self.cyl.P = P_next
        #     self.cyl.T = T_next
            
      
        # # if CAD % 10 == 0:
        # #     print(
        # #     f"DEBUG_cycle "
        # #         f"θ: {self._cycle_count}/{CAD} {stroke}  "
        # #         f"P_curr:{P_pre:6.0f}->{self.cyl.P:6.0f} T_curr:{T_pre:3.0f}->{self.cyl.T:3.0f} V_curr:{V_curr:8.2e} " 
        # #         f"dMass:{dm_total:9.2e} T_inflow:{T_inflow:3.0f} " 
        # #         f"dQ_in:{Q_in_sub:5.2e} dQ_loss:{Q_loss_sub:9.2e}  "
        # #         f"dV_d_theta:{dV_sub:9.2e}  "
        # #     )
                            

        # # 4. Mechanical Dynamics (Calculated once per degree using final state)
        # # We use the full degree dV here because work is summed across the step
        # self._update_mechanical_dynamics(CAD, stroke, self.cyl.P, self.cyl.dV_list[CAD])

        # # 5. Logging & Debugging
        # self.cyl.log_P[CAD] = self.cyl.P
        # self.cyl.log_V[CAD] = self.cyl.V_list[CAD]
        # self.cyl.log_T[CAD] = self.cyl.T
        # self.cyl.dm_in_history[CAD] = dm_in_acc
        # self.cyl.dm_ex_history[CAD] = dm_out_acc
        


            
    # --- Helper Methods to maintain functionality ---
    
    def _calc_flow_deltas(self, ecu_outputs, step_size):
        """
        PURE FUNCTION: Calculates mass and property deltas for a substep.
        Does NOT modify self.cyl state.
        """
        current_rpm_safe = max(self.sensors.rpm, 10.0)
        CAD = int(self.state.current_theta % 720)
        dt = step_size / (6.0 * current_rpm_safe)

        # 1. Valve Geometry
        L_i = self.valves.intake_lift_table[CAD]
        A_i = self.valves.intake_area_table[CAD] 
        L_e = self.valves.exhaust_lift_table[CAD] 
        A_e = self.valves.exhaust_area_table[CAD] 

        # 2. Isentropic Flow Calculations
        # Intake
        dm_i_raw, Cd_i = pf.calc_isentropic_flow(
            A_i, L_i, self.valves.intake.diameter,
            P_cyl=self.cyl.P, T_cyl=self.cyl.T, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
            P_extern=self.sensors.P_manifold_Pa, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
            is_intake=True
        )
        dm_i = dm_i_raw * dt
        
        # Exhaust
        dm_e_raw, Cd_e = pf.calc_isentropic_flow(
            A_e, L_e, self.valves.exhaust.diameter,
            P_cyl=self.cyl.P, T_cyl=self.cyl.T, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
            P_extern=c.P_ATM_PA, T_extern=self.state.T_exhaust_manifold, R_extern=c.R_SPECIFIC_EXHAUST, g_extern=c.GAMMA_EXHAUST,
            is_intake=False
        )
        dm_e = dm_e_raw * dt

        # 3. Fuel Calculation (Scale by step_size)
        dm_f = 0.0
        if ecu_outputs["injector_on"]:
            fuel_cc_per_subdeg = c.INJECTOR_FLOW_CC_PER_MIN * step_size/ (current_rpm_safe * 360.0)
            dm_f = fuel_cc_per_subdeg * c.FUEL_DENSITY_KG_CC 
            self.temp_total_dm_f += dm_f
            # print(f"DEBUG FUEL TOTAL: CAD:{CAD} rpm:{int(current_rpm_safe)} dm_f:{fuel_cc_per_subdeg:6.4f}cc = {dm_f:9.6f}kg total_dm_f:{self.temp_total_dm_f:9.6f}kg ")
            # print(f"                  {c.INJECTOR_FLOW_CC_PER_MIN} {step_size}")

        dm_tot = dm_i + dm_e + dm_f

        # 4. Enthalpy/Inflow Temperature Logic
        mass_in = 0.0
        weighted_T = 0.0
        if dm_i > 0:
            mass_in += dm_i
            weighted_T += dm_i * c.T_AMBIENT
        if dm_e > 0:
            mass_in += dm_e
            weighted_T += dm_e * self.state.T_exhaust_manifold
        if dm_f > 0:
            mass_in += dm_f
            weighted_T += dm_f * c.T_FUEL_K

        T_inflow = weighted_T / mass_in if mass_in > 1e-18 else self.cyl.T

        # 5. Return everything needed for the sub-step integration
        return {
            "dm_i": dm_i, "dm_e": dm_e, "dm_f": dm_f, "dm_tot": dm_tot,
            "T_inflow": T_inflow, "Cd_i": Cd_i, "Cd_e": Cd_e
        }
        
    def _update_gas_properties(self):

        # --- Calculate gas mixture
        m_total_pre = max(self.cyl.total_mass_kg, 1e-9)
        m_air = self.cyl.air_mass_kg
        m_fuel = self.cyl.fuel_mass_kg
        m_exh = max(0, self.cyl.total_mass_kg - (m_air + m_fuel)) 
        
        f_air = m_air / m_total_pre
        f_fuel = m_fuel / m_total_pre
        f_exh = m_exh / m_total_pre

        # Calculate the air/fuel mixture R_specific and Gamma
        # --- R_specific
        R_blend = (f_air * c.R_SPECIFIC_AIR) + (f_fuel * c.R_SPECIFIC_FUEL) + (f_exh * c.R_SPECIFIC_EXHAUST)
        self.cyl.R_specific_blend = R_blend
        
        # --- Gamma
        cv_air =  c.R_SPECIFIC_AIR / (c.GAMMA_AIR - 1.0)
        cv_fuel = c.R_SPECIFIC_FUEL / (c.GAMMA_FUEL - 1.0)
        cv_exh =  c.R_SPECIFIC_EXHAUST / (c.GAMMA_EXHAUST - 1.0)
        cv_blend = (f_air * cv_air) + (f_fuel * cv_fuel) + (f_exh * cv_exh)
        gamma_blend = (R_blend / cv_blend) + 1
        self.cyl.gamma_blend = gamma_blend
        

    # def _update_mass_flow(self, ecu_outputs, step_size=1.0):
    #     """
    #     Calculates physical mass flow. 
    #     step_size: fraction of a degree (e.g., 0.2 for 5 sub-steps).
    #     """
    #     current_rpm_safe = max(self.sensors.rpm, 10.0)
    #     CAD = int(self.state.current_theta % 720)
             
    #     # 1. FUEL MASS DELTA
    #     # --- Fuel flow is rate-based, so we scale it by the step_size
    #     dm_fuel_kg = 0.0    
    #     if ecu_outputs["injector_on"]:
    #         fuel_cc_per_deg = c.INJECTOR_FLOW_CC_PER_MIN / (current_rpm_safe * 6.0) # 6.0 is (360/60)
    #         dm_fuel_kg = fuel_cc_per_deg * c.FUEL_DENSITY_KG_CC * step_size
        
    #     # 2. GAS FLOW
    #     L_i = self.valves.intake_lift_table[CAD]
    #     A_i = self.valves.intake_area_table[CAD] 
    #     L_e = self.valves.exhaust_lift_table[CAD] 
    #     A_e = self.valves.exhaust_area_table[CAD] 
    #     dt = step_size / (6.0 * self.sensors.rpm)

    #     # -- Intake Valve Flow
    #     dm_intake_port, Cd_i = pf.calc_isentropic_flow(
    #         A_i, L_i, self.valves.intake.diameter,
    #         P_cyl=self.cyl.P, T_cyl=self.cyl.T, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
    #         P_extern=self.sensors.P_manifold_Pa, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
    #         is_intake = True
    #     )
    #     dm_intake_port *= dt
        
    #     # if CAD < 30:
    #     #     print(f"DEBUG dm_intake:CAD:{self._cycle_count}/{CAD} "
    #     #         f"P_cyl:{self.cyl.P:6.0f}Pa, P_ext:{self.sensors.P_manifold_Pa:6.0f}Pa "
    #     #         f"dm_in:{dm_intake_port:9.2e}"
    #     #         )

    #     # -- Exhaust Valve Flow (-ve for flow out, +ve for flow into cyl)
    #     dm_exhaust_port, Cd_e = pf.calc_isentropic_flow(
    #         A_e, L_e, self.valves.exhaust.diameter,
    #         P_cyl=self.cyl.P, T_cyl=self.cyl.T, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
    #         P_extern=c.P_ATM_PA, T_extern=self.state.T_exhaust_manifold, R_extern=c.R_SPECIFIC_EXHAUST, g_extern=c.GAMMA_EXHAUST,
    #         is_intake=False
    #     )
    #     dm_exhaust_port *= dt

    #     # --- Update Intake Flow ---
    #     # --- Calculate Current Fractions/mixture BEFORE applying deltas ---
    #     m_total_pre = max(self.cyl.total_mass_kg, 1e-9)
    #     m_air = self.cyl.air_mass_kg
    #     m_fuel = self.cyl.fuel_mass_kg
    #     m_exh = max(0, self.cyl.total_mass_kg - (m_air + m_fuel)) 
        
    #     f_air = m_air / m_total_pre
    #     f_fuel = m_fuel / m_total_pre
    #     f_exh = m_exh / m_total_pre
        
    #     if dm_intake_port >= 0:
    #         # Forward flow: Pure Air entering from Port
    #         self.cyl.air_mass_kg += dm_intake_port
    #     else:
    #         # Backflow: Mixture leaving the cylinder
    #         # We remove air and fuel according to their current presence
    #         self.cyl.air_mass_kg += dm_intake_port * f_air
    #         self.cyl.fuel_mass_kg += dm_intake_port * f_fuel
    #         # The remainder of dm_intake_port (the exhaust part) is handled by the total_mass update

    #     # --- Update Exhaust Flow ---
    #     # if dm_exhaust_port < 0:
    #     #     # Forward flow: Mixture leaving to exhaust
    #     #     self.cyl.air_mass_kg += dm_exhaust_port  * f_air
    #     #     self.cyl.fuel_mass_kg += dm_exhaust_port * f_fuel
    #     # else:
    #     #     # Backflow: Exhaust gas re-entering (Recompression/EGR)
    #     #     # This adds to total_mass but DOES NOT add to air_mass or fuel_mass
    #     #     pass 

    #     # --- Mass Balance ---
    #     dm_total_kg = dm_intake_port + dm_exhaust_port + dm_fuel_kg
    #     self.cyl.dm_total = dm_total_kg
    #     self.cyl.total_mass_kg += dm_total_kg

    #     # --- Calculate Current Fractions/mixture BEFORE applying deltas ---
    #     m_total_pre = max(self.cyl.total_mass_kg, 1e-9)
    #     m_air = self.cyl.air_mass_kg
    #     m_fuel = self.cyl.fuel_mass_kg
    #     m_exh = max(0, self.cyl.total_mass_kg - (m_air + m_fuel)) 
        
    #     f_air = m_air / m_total_pre
    #     f_fuel = m_fuel / m_total_pre
    #     f_exh = m_exh / m_total_pre



    #     # 4. Determine Temperature of flows (Enthalpy Source/Sink)
    #     mass_entering = 0.0
    #     weighted_enthalpy_temp = 0.0

    #     # -- Case A: Fresh Charge entering (Normal Intake)
    #     if dm_intake_port >= 0:
    #         mass_entering += dm_intake_port
    #         weighted_enthalpy_temp += dm_intake_port * c.T_AMBIENT

    #     # --- Case B: Exhaust Backflow (Re-entry during Overlap)
    #     if dm_exhaust_port > 0:
    #         mass_entering += dm_exhaust_port
    #         weighted_enthalpy_temp += dm_exhaust_port * self.state.T_exhaust_manifold

    #     # --- Case C: Fuel Vapor (Port Injection)
    #     if dm_fuel_kg > 0:
    #         mass_entering += dm_fuel_kg
    #         weighted_enthalpy_temp += dm_fuel_kg * c.T_FUEL_K

    #     # --- Determine the blended temperature of the INFLOW
    #     if mass_entering > 1e-18:
    #         T_manifold_source = weighted_enthalpy_temp / mass_entering
    #     else:
    #         # If no mass is entering, this value won't be used by the 
    #         # new integrator (it will use T_curr), but we set a safe default.
    #         T_manifold_source = c.T_AMBIENT
            
    #     # print(f"DEBUG_VALVE | θ:{CAD} | Cd_i:{Cd_i:.3f} | Cd_e:{Cd_e:.3f} | RPM:{self.sensors.rpm:.0f} | A_valve_in:{self.valves.intake_area_table[CAD]:.8f} |  A_valve_ex:{self.valves.exhaust_area_table[CAD]:.8f}")        
    #     # print(f"DEBUG_DM | θ:{CAD} in:{dm_intake_port:4.2e} out:{dm_exhaust_port:4.2e} fuel:{dm_fuel_kg:4.2e} total:{dm_total:4.2e}" )

    #     # 5. Calculate the air/fuel mixture R_specific and Gamma
    #     # --- R_specific
    #     R_blend = (f_air * c.R_SPECIFIC_AIR) + (f_fuel * c.R_SPECIFIC_FUEL) + (f_exh * c.R_SPECIFIC_EXHAUST)
    #     self.cyl.R_specific_blend = R_blend
        
    #     # --- Gamma
    #     cv_air =  c.R_SPECIFIC_AIR / (c.GAMMA_AIR - 1.0)
    #     cv_fuel = c.R_SPECIFIC_FUEL / (c.GAMMA_FUEL - 1.0)
    #     cv_exh =  c.R_SPECIFIC_EXHAUST / (c.GAMMA_EXHAUST - 1.0)
    #     cv_blend = (f_air * cv_air) + (f_fuel * cv_fuel) + (f_exh * cv_exh)
    #     gamma_blend = (R_blend / cv_blend) + 1
    #     self.cyl.gamma_blend = gamma_blend

    #     return dm_intake_port, dm_exhaust_port, dm_fuel_kg, dm_total_kg, T_manifold_source, Cd_i, Cd_e, R_blend, gamma_blend


    # def _update_mass_flow(self, ecu_outputs):
    #     """Calculates physical mass flow using valve geometry and pressure delta."""
    #     current_rpm_safe = max(self.sensors.rpm, 10.0)
        
    #     # 1. Calculate Fuel Mass Delta (kg)
    #     dm_fuel_kg = 0.0    
    #     if ecu_outputs["injector_on"]:
    #         fuel_cc_per_deg = c.INJECTOR_FLOW_CC_PER_MIN / (current_rpm_safe * 360)
    #         # self._cycle_fuel_injected_cc += fuel_cc_per_deg
    #         dm_fuel_kg = fuel_cc_per_deg * c.FUEL_DENSITY_KG_CC
    #         self.cyl.fuel_mass_kg += dm_fuel_kg

    #     CAD = int(self.state.current_theta % 720)
    #     # get valve area and lift (in metres)
    #     L_i = self.valves.intake_lift_table[CAD]
    #     A_i = self.valves.intake_area_table[CAD] 
        
    #     L_e = self.valves.exhaust_lift_table[CAD] 
    #     A_e = self.valves.exhaust_area_table[CAD] 
        
    #     # Calculate flows using the single unified physics function
    #     dm_intake_port = pf.calc_isentropic_flow(
    #         A_i, L_i, self.valves.intake.diameter,
    #         P_up=self.sensors.P_manifold_Pa, T_up=c.T_AMBIENT, 
    #         P_down=self.cyl.P, T_down=self.cyl.T, rpm=self.sensors.rpm, is_intake=True
    #     )
        
    #     # print(f"DEBUG_FLOW  | θ:{self.state.current_theta:05.1f} | dm_g:{dm_intake_port} | Av:{A_i:8.2e} | P_m:{self.sensors.P_manifold_Pa:7.0f}")

    #     # Check for non-physical mass changes when valves are closed
    #     # stroke = self._get_stroke()[0]
    #     # if stroke in ["compression", "power"] and abs(dm_intake_port) > 1e-12:
    #     #     print(f"!!! MASS LEAK ALERT !!! | θ:{self.state.current_theta:05.1f} | "
    #     #         f"Stroke:{stroke} | Leak_dm:{dm_intake_port:e} kg")



    #     dm_exhaust_port = pf.calc_isentropic_flow(
    #         A_e, L_e, self.valves.exhaust.diameter,
    #         P_up=self.cyl.P, T_up=self.cyl.T, 
    #         P_down=c.P_ATM_PA, T_down=c.T_AMBIENT, rpm=self.sensors.rpm, is_intake=False
    #     )


    #     # 3. Store the total mass change for this degree
    #     # integrate_first_law NEEDS this
    #     # self.dm_total = (dm_intake_port - dm_exhaust_port) + dm_fuel_kg
    #     dm_total_raw = (dm_intake_port - dm_exhaust_port) + dm_fuel_kg
    #     # dm_total = np.clip(dm_total, -0.01 * self.M_gas, 0.01 * self.M_gas)  # symmetric damp; tune 0.01 to 0.05
    #     dm_total = np.clip(dm_total_raw, -0.1 * self.cyl.M_gas, 0.1 * self.cyl.M_gas)
    #     self.cyl.dm_total = dm_total
        
    #     if dm_total_raw != dm_total:
    #         print(f"!!! FLOW CLIPPED at CAD {CAD}: {dm_total_raw:e} -> {dm_total:e}")
        
        
    #     # 2. Fresh air Accumulator for AFR calculation
    #     # We only count positive flow through the intake valve
    #     # self.cylinder_air_kg += max(dm_intake_port, 0.0)
    #     self.cyl.air_mass_kg += dm_intake_port
    #     # self.cyl.air_mass_kg += dm_intake_port if dm_intake_port > 0 \
    #     #         else dm_intake_port * (self.cyl.air_mass_kg / self.cyl.M_gas if self.cyl.M_gas > 0 \
    #     #         else 0)
   
    #     # if L_i > 0: # Only print when the valve is actually open
    #     #     print(f"DEBUG ISEN| CAD: {self.current_theta:3.1f} | RPM: {self.rpm:.0f}")
    #     #     print(f"      | Inputs: Area={A_i:.6f} m2, Lift={L_i:.3f} mm, D={self.v_data_720['intake']['D_valve']:.1f} mm")
    #     #     print(f"      | Pressures: P_up (Manifold)={self.P_manifold:.1f} Pa, P_down (Cyl)={self.P_cyl:.1f} Pa")
    #     #     print(f"      | Pr Ratio: {self.P_cyl / self.P_manifold:.4f} (Critical is ~0.528)")
    #     #     print(f"      | Result: dm_intake_port={dm_intake_port:.8f} kg/deg"
    #     #           f"dm_out: {dm_exhaust_port} | dm_total: {dm_total}")
    #     #     print("-" * 50)
    #     #     print("DEBUG AIR  "
    #     #         f"total air: {self.cylinder_air_kg:10.6f} kg | "
    #     #         f"ECU air: {ecu_outputs['trapped_air_mass_kg']:10.6f} kg"
    #     #         )
    #     #     print("-" * 50)
        
    #     return dm_intake_port, dm_exhaust_port

    """ 
    CHANGE FROM STEP TO RATE OF HEAT RELEASE  dQ/theta. 
    integrate_first_law can then multiply this rate by substep size (theta_delta).
    """
    def _calculate_combustion_heat_substep(self, CAD, substep, substep_size, spark_command):
        """
        Substep-friendly Wiebe heat release.
        Returns dQ (Joules) for the specific delta_theta provided.
        """
        
        # 1. Trigger Spark Event (Logic remains similar, but usually handled 
        # outside the substep loop to avoid multiple resets)
        if spark_command and substep == 0 and not self.cyl.combustion_active:
            self.cyl.spark_event_theta = CAD
            ign_delay = 6.0 # CHECK THIS
            self.cyl.ignition_start_theta = CAD + ign_delay
            self.cyl.combustion_active = True
            self.cyl.cumulative_heat_released = 0.0
            self.cyl.spark_advance_btdc = (360.0 - (CAD % 360.0)) % 360
                        
            # Setup fuel/AFR/Efficiency
            fuel_kg = self.cyl.fuel_mass_kg
            air_kg = self.cyl.air_mass_kg 
            
            # record the air and fuel levels at time of spark
            self.cyl.air_mass_at_spark = air_kg
            self.cyl.fuel_mass_at_spark = fuel_kg
            
            # updates AFR sensors
            afr = (air_kg / fuel_kg) if fuel_kg > 0 else 99.0
            # lambda_ = afr/14.7 if fuel_kg > 0 else 99 # changed to a sensors attribute
            self.sensors.afr = afr
            # self.sensors.lambda_ = lambda_ # changed to a sensors attribute
            
            # Determing heat to be released
            eff = 0.94 if 0.8 <= self.sensors.lambda_ <= 1.1 else 0.80
            self.cyl.total_cycle_heat_J = fuel_kg * c.LHV_FUEL_GASOLINE * eff

            # self.cyl.burn_duration = max(25.0, 38.0 * (self.sensors.rpm**-0.82) * (lambda_**-0.6))  # min 25°
            # self.cyl.burn_duration = 55.0
            self.cyl.burn_duration = pf.get_burn_duration(self.sensors.rpm, self.sensors.lambda_)
            self.cyl.m_vibe = max(2.0, 3.0 - (self.sensors.rpm / 4000.0))
            
            # Calculate stoichiometric fuel requirement
            # required_fuel_for_air = air_kg / 14.7

            # # Heat release is limited by whichever runs out first: fuel or oxygen
            # effective_fuel_burned = min(fuel_kg, required_fuel_for_air)
            # self.cyl.total_cycle_heat_J = effective_fuel_burned * c.LHV_FUEL_GASOLINE * eff

            # # update accumulators for the next cycle/step
            # # Instead of zeroing them, they should be converted to 'Exhaust' 
            # # (which is implicitly total_mass - air - fuel)
            # self.cyl.air_mass_kg -= (effective_fuel_burned * 14.7) 
            # self.cyl.fuel_mass_kg -= effective_fuel_burned
            # # Ensure they don't go negative due to rounding
            # self.cyl.air_mass_kg = max(0, self.cyl.air_mass_kg)
            # self.cyl.fuel_mass_kg = max(0, self.cyl.fuel_mass_kg)
            
            
            
            print(
                f"    SPARK for cycle:{self._cycle_count} at CAD:{CAD} "
                f"fuel:{fuel_kg:.2e}kg  air:{air_kg:.2e}kg total_mass:{self.cyl.total_mass_kg:.2e} "
                f"afr:{afr:.1f}  heat_to_release:{self.cyl.total_cycle_heat_J:.1f}J "
                f"rpm:{self.sensors.rpm:.0f} "
                f"m_vibe: {self.cyl.m_vibe:.2f} "
                f"burn_dur: {int(self.cyl.burn_duration)} "
                f"MAP:{self.sensors.MAP_kPa:.1f}kPa"
            )
            
        combustion_start = CAD >= self.cyl.ignition_start_theta

        # if combustion_start and self.cyl.combustion_active:
        #     # 1. Determine Wiebe Slice
        #     theta_start_substep = CAD + (substep * substep_size)
        #     theta_next_substep = theta_start_substep + substep_size
        #     theta_previous_substep = theta_start_substep - substep_size
            
        #     f1 = pf.calc_wiebe_fraction(theta_start_substep, self.cyl.ignition_start_theta, self.cyl.burn_duration, a_vibe=5.0, m_vibe=self.cyl.m_vibe)
        #     f2 = pf.calc_wiebe_fraction(theta_next_substep, self.cyl.ignition_start_theta, self.cyl.burn_duration, a_vibe=5.0, m_vibe=self.cyl.m_vibe)
        #     step_fraction = max(0.0, f2 - f1)

        #     # 2. Check Energy Budget
        #     # Determine how much fuel we are 'allowed' to burn before hitting the total_cycle_heat_J cap
        #     eff = 0.94 if 0.8 <= self.sensors.lambda_ <= 1.1 else 0.80
        #     remaining_joules = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
        #     max_fuel_allowed_by_energy = remaining_joules / (c.LHV_FUEL_GASOLINE * eff)

        #     # 3. Calculate Potential Mass to burn (Proportional to snapshot at Spark)
        #     dm_fuel_potential = self.cyl.fuel_mass_at_spark * step_fraction
        #     dm_air_potential  = self.cyl.air_mass_at_spark  * step_fraction

        #     # 4. Determine Actual Burned (Clamped by Cylinder contents AND Energy budget)
        #     # This prevents the 'Gap' where mass is deleted but energy is capped
        #     actual_fuel_burned = min(self.cyl.fuel_mass_kg, dm_fuel_potential, max_fuel_allowed_by_energy)
        #     actual_air_burned  = min(self.cyl.air_mass_kg, dm_air_potential)

        #     # 5. Chemical Heat Limit (Stoichiometric Check)
        #     # Even if we 'burn' 2mg of fuel, we only get heat for what the air can support
        #     # This is critical for your current 'Firehose' / Rich condition
        #     fuel_that_actually_reacts = min(actual_fuel_burned, actual_air_burned / 14.7)
        #     dQ_combustion = fuel_that_actually_reacts * c.LHV_FUEL_GASOLINE * eff

        #     # 6. Update Physics & Tracker
        #     self.cyl.fuel_mass_kg -= actual_fuel_burned
        #     self.cyl.air_mass_kg  -= actual_air_burned
        #     self.cyl.cumulative_heat_released += dQ_combustion

        #     # 7. Clean Exit (Physical Species Preservation)
        #     completion_ratio = self.cyl.cumulative_heat_released / self.cyl.total_cycle_heat_J
            
        #     if completion_ratio >= 0.995 or self.cyl.fuel_mass_kg < 1e-12 or self.cyl.air_mass_kg < 1e-12:
        #         # Catch last bit of Energy
        #         final_snap = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
                
        #         # Calculate mass needed for final snap
        #         final_fuel_req = final_snap / (c.LHV_FUEL_GASOLINE * eff)
        #         final_air_req  = final_fuel_req * self.sensors.afr
                
        #         # Final decrement (clamped at zero)
        #         self.cyl.fuel_mass_kg = max(0.0, self.cyl.fuel_mass_kg - final_fuel_req)
        #         self.cyl.air_mass_kg  = max(0.0, self.cyl.air_mass_kg - final_air_req)
                
        #         # Synchronize and Close
        #         dQ_combustion += final_snap
        #         self.cyl.cumulative_heat_released = self.cyl.total_cycle_heat_J
        #         self.cyl.combustion_active = False
                
        #         print(f"    PHYSICAL EXIT: CAD={CAD} ratio:{completion_ratio:.4f} "
        #             f"Remaining Fuel: {self.cyl.fuel_mass_kg:7.3e}kg, Air: {self.cyl.air_mass_kg:7.3e}kg")
                
            # if CAD % 10 == 0:
            #     ign_delay = int(self.cyl.ignition_start_theta - self.cyl.spark_event_theta)
            #     print(
            #         f"    DEBUG COMBUSTION: {self._cycle_count}/{CAD}/{substep}  "
            #         f"rpm:{self.sensors.rpm:4.0f} "
            #         # f"f1: {f1:8.2e} f2: {f2:8.2e} "
            #         f"dQ:{dQ_combustion:6.4f} heat_released:{self.cyl.cumulative_heat_released:7.1f} "
            #         f"air_remaining:{self.cyl.air_mass_kg:9.2e} fuel_remaining:{self.cyl.fuel_mass_kg:9.2e} "
            #         f"ign_delay:{ign_delay} duration: {CAD - int(self.cyl.ignition_start_theta)} "
            #         f"ratio:{completion_ratio:4.2f} P:{self.cyl.P/1e5:4.1f}bar "
            #     )

        #     return dQ_combustion
        
        if combustion_start and self.cyl.combustion_active:
            # 1. Determine Wiebe Slice
            theta_start_substep = CAD + (substep * substep_size)
            theta_next_substep = theta_start_substep + substep_size
            
            f1 = pf.calc_wiebe_fraction(theta_start_substep, self.cyl.ignition_start_theta, self.cyl.burn_duration, a_vibe=5.0, m_vibe=self.cyl.m_vibe)
            f2 = pf.calc_wiebe_fraction(theta_next_substep, self.cyl.ignition_start_theta, self.cyl.burn_duration, a_vibe=5.0, m_vibe=self.cyl.m_vibe)
            step_fraction = max(0.0, f2 - f1)

            # 2. Check Energy Budget
            # Adjust efficiency based on Lambda window
            eff = 0.94 if 0.8 <= self.sensors.lambda_ <= 1.1 else 0.80
            remaining_joules = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
            max_fuel_allowed_by_energy = remaining_joules / (c.LHV_FUEL_GASOLINE * eff)

            # 3. Calculate Potential Mass to burn (Proportional to snapshot at Spark)
            dm_fuel_potential = self.cyl.fuel_mass_at_spark * step_fraction
            dm_air_potential  = self.cyl.air_mass_at_spark  * step_fraction

            # 4. Determine Actual Burned (Clamped by Cylinder contents AND Energy budget)
            actual_fuel_burned = min(self.cyl.fuel_mass_kg, dm_fuel_potential, max_fuel_allowed_by_energy)
            actual_air_burned  = min(self.cyl.air_mass_kg, dm_air_potential)

            # 5. Chemical Heat Limit (Stoichiometric Check)
            # Identify the limiting reactant for this specific slice
            max_fuel_by_air = actual_air_burned / 14.7
            
            # Heat is only generated by the fuel that actually has oxygen to react with
            fuel_that_actually_reacts = min(actual_fuel_burned, max_fuel_by_air)
            dQ_combustion = fuel_that_actually_reacts * c.LHV_FUEL_GASOLINE * eff

            # 6. Update Physics & Tracker
            self.cyl.fuel_mass_kg -= actual_fuel_burned
            self.cyl.air_mass_kg  -= actual_air_burned
            self.cyl.cumulative_heat_released += dQ_combustion

            # 7. Physical Exit (Species Preservation)
            completion_ratio = self.cyl.cumulative_heat_released / self.cyl.total_cycle_heat_J
            
            # Identify if we are actually out of a reactant
            out_of_fuel = self.cyl.fuel_mass_kg < 1e-12
            out_of_air = self.cyl.air_mass_kg < 1e-12
            
            if completion_ratio >= 0.995 or out_of_fuel or out_of_air:
                # 1. Calculate remaining energy gap
                final_snap_joules = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
                
                if final_snap_joules > 0:
                    # 2. Stoichiometric limit: How much can we POSSIBLY burn with what's left?
                    # We can only burn as much fuel as we have air for (and vice versa)
                    potential_fuel_by_air = self.cyl.air_mass_kg / 14.7
                    max_physically_burnable_fuel = min(self.cyl.fuel_mass_kg, potential_fuel_by_air)
                    
                    # 3. How much fuel do we NEED to reach the energy cap?
                    fuel_needed_for_cap = final_snap_joules / (c.LHV_FUEL_GASOLINE * eff)
                    
                    # 4. Actual snap mass (limited by physics AND the cap)
                    snap_fuel = min(max_physically_burnable_fuel, fuel_needed_for_cap)
                    snap_air = snap_fuel * 14.7
                    
                    actual_snap_heat = snap_fuel * c.LHV_FUEL_GASOLINE * eff
                    
                    # 5. Apply the final mass change
                    self.cyl.fuel_mass_kg = max(0.0, self.cyl.fuel_mass_kg - snap_fuel)
                    self.cyl.air_mass_kg = max(0.0, self.cyl.air_mass_kg - snap_air)
                    
                    dQ_combustion += actual_snap_heat
                    self.cyl.cumulative_heat_released += actual_snap_heat

                self.cyl.combustion_active = False
                
                print(f"    PHYSICAL EXIT: CAD={CAD} ratio:{self.cyl.cumulative_heat_released/self.cyl.total_cycle_heat_J:.4f} "
                      f"EXCESS RECOVERED - Fuel: {self.cyl.fuel_mass_kg:7.3e}kg, Air: {self.cyl.air_mass_kg:7.3e}kg")
                
            # --- Debugging Output ---
            # if CAD % 10 == 0:
            #     ign_delay = int(self.cyl.ignition_start_theta - self.cyl.spark_event_theta)
            #     print(
            #         f"    DEBUG COMBUSTION: {self._cycle_count}/{CAD}/{substep}  "
            #         f"rpm:{self.sensors.rpm:4.0f} "
            #         f"dQ:{dQ_combustion:6.4f} heat_released:{self.cyl.cumulative_heat_released:7.1f} "
            #         f"air_remaining:{self.cyl.air_mass_kg:9.2e} fuel_remaining:{self.cyl.fuel_mass_kg:9.2e} "
            #         f"ign_delay:{ign_delay} duration: {CAD - int(self.cyl.ignition_start_theta)} "
            #         f"ratio:{completion_ratio:4.2f} P:{self.cyl.P/1e5:4.1f}bar "
            #     )

            return dQ_combustion
        
        return 0.0

        # 2. Calculate Heat Release Rate
        # if self.cyl.combustion_active:
        #     # Calculate how much of the "S-Curve" we covered in this tiny slice
        #     # Current angle at the START of this substep
        #     theta_start_substep = CAD + (substep * substep_size)
        #     # Current angle at the END of this substep
        #     theta_end_substep = theta_start_substep + substep_size
        #     f1 = pf.calc_wiebe_fraction(theta_start_substep, self.cyl.spark_event_theta, self.cyl.burn_duration)
        #     f2 = pf.calc_wiebe_fraction(theta_end_substep, self.cyl.spark_event_theta, self.cyl.burn_duration)
        #     print(
        #         f"    DEBUG WIEBE: {self._cycle_count}/{CAD}/{substep} ")

        #     # # 1. Calculate the slice
        #     # dQ_combustion = (f2 - f1) * self.cyl.total_cycle_heat_J
            
        #     #--------------
        #     # 1. Determine the slice fraction from Wiebe
        #     step_fraction = max(0.0, f2 - f1)

        #     # 2. Calculate the 'Potential' mass to burn based on the Spark Snapshot
        #     # This respects the ECU's chosen AFR, whether it was 12:1 or 18:1
        #     dm_fuel_potential = self.cyl.fuel_mass_at_spark * step_fraction
        #     dm_air_potential  = self.cyl.air_mass_at_spark  * step_fraction

        #     # 3. Determine actual burned (Clamped by what's physically left in the cylinder)
        #     # This prevents math errors if the Exhaust Valve opens during the burn
        #     actual_fuel_burned = min(self.cyl.fuel_mass_kg, dm_fuel_potential)
        #     actual_air_burned  = min(self.cyl.air_mass_kg, dm_air_potential)

        #     # 4. Update Physics
        #     self.cyl.fuel_mass_kg -= actual_fuel_burned
        #     self.cyl.air_mass_kg  -= actual_air_burned

        #     # 5. Output Heat Release
        #     # Heat is only released by the fuel that actually found air to burn with.
        #     # (Note: In a very rich "Firehose" case, this correctly limits energy 
        #     # to the stoichiometric potential of the trapped air).
        #     dQ_combustion = actual_fuel_burned * c.LHV_FUEL_GASOLINE * eff
        #     # ----------------

        #     # 2. The ONLY guard you need (Safety first)
        #     remaining_energy = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
        #     if dQ_combustion > remaining_energy:
        #         dQ_combustion = max(0.0, remaining_energy)

        #     # 3. Update tracker
        #     self.cyl.cumulative_heat_released += dQ_combustion

        #     # # 4. Clean exit
        #     # if self.cyl.cumulative_heat_released >= (self.cyl.total_cycle_heat_J * 0.999):
        #     #     # Release the final tiny bit of energy if any remains
        #     #     final_snap = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
        #     #     # Add final_snap to dQ_combustion here if you want 100% energy closure
        #     #     final_snap = final_snap if final_snap > 0 else 0.0
        #     #     self.cyl.cumulative_heat_released += final_snap
        #     #     self.cyl.combustion_active = False
                
        #     # 4. Clean Exit
        #     # --- Standard energy check
        #     # completion_ratio = self.cyl.cumulative_heat_released / self.cyl.total_cycle_heat_J

        #     # # --- Exit condition (Energy OR Reactant depletion)
        #     # if completion_ratio >= 0.995 or self.cyl.fuel_mass_kg < 1e-9 or self.cyl.air_mass_kg < 1e-9: 
        #     #     print(f"    COMBUSTION CLEAN EXIT: CAD={CAD}/{substep} ratio:{completion_ratio} fuel:{self.cyl.fuel_mass_kg:7.1f} air:{self.cyl.air_mass_kg:7.1f} ")
        #     #     # -- Catch last bit of Energy for the Return
        #     #     final_snap = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
        #     #     dQ_combustion += max(0.0, final_snap)
                        
        #     #     # --- Hard Reset to synchronize state
        #     #     self.cyl.cumulative_heat_released = self.cyl.total_cycle_heat_J
        #     #     self.cyl.fuel_mass_kg = 0.0
        #     #     self.cyl.air_mass_kg = 0.0
        #     #     self.cyl.combustion_active = False
            
        #     # --- Completion Logic inside the if self.cyl.combustion_active block ---

        #     # 1. Standard energy check
        #     completion_ratio = self.cyl.cumulative_heat_released / self.cyl.total_cycle_heat_J

        #     # 2. Exit condition (Energy OR one reactant is exhausted)
        #     if completion_ratio >= 0.995 or self.cyl.fuel_mass_kg < 1e-9 or self.cyl.air_mass_kg < 1e-9: 
                
        #         # Calculate final energy "snap"
        #         final_snap = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
        #         dQ_combustion += final_snap
                
        #         # Calculate exactly how much mass is needed for this final energy snap
        #         final_fuel_to_burn = final_snap / (c.LHV_FUEL_GASOLINE * eff)
        #         final_air_to_burn = final_fuel_to_burn * 14.7
                
        #         # Decrement only what is actually needed for the final snap
        #         self.cyl.fuel_mass_kg = max(0.0, self.cyl.fuel_mass_kg - final_fuel_to_burn)
        #         self.cyl.air_mass_kg = max(0.0, self.cyl.air_mass_kg - final_air_to_burn)

        #         # --- Synchronize State ---
        #         self.cyl.cumulative_heat_released = self.cyl.total_cycle_heat_J
        #         self.cyl.combustion_active = False
                
        #         print(f"    PHYSICAL EXIT: CAD={CAD} Remaining Fuel: {self.cyl.fuel_mass_kg:7.3e}kg, Air: {self.cyl.air_mass_kg:7.3e}kg")
            
                
        #     print(
        #         f"    DEBUG COMBUSTION: {self._cycle_count}/{CAD}/{substep}  "
        #         f"rpm:{self.sensors.rpm:4.0f} "
        #         f"f1: {f1:8.2e} f2: {f2:8.2e} dQ_combust:{dQ_combustion:6.4f} heat_released:{self.cyl.cumulative_heat_released:7.1f} <> max:{self.cyl.total_cycle_heat_J:7.1f} "
        #         f"remaining: {remaining_energy:6.4f} "
        #         f"active:{self.cyl.combustion_active}"
        #     )
            

            
        #     return dQ_combustion
                

    """
    THIS FUNCTION IS STEP BASED AND NOT SUB_STEP FRIENDLY
    """
    # def _calculate_combustion_heat(self, spark_command):
    #     """Calculates heat release using the Wiebe S-curve with energy conservation."""
        
    #     # 1. Trigger Spark Event
    #     if spark_command:
    #         self.cyl.spark_event_theta = self.state.current_theta
    #         self.cyl.combustion_active = True
    #         self.cyl.cumulative_heat_released = 0.0 # Track total energy released
    #         self.cyl.spark_advance_btdc = (360.0 - (self.state.current_theta % 360.0)) % 360
            
    #         # Setup fuel/AFR/Efficiency
    #         fuel_kg = self.cyl.fuel_mass_kg
    #         air_kg = self.cyl.air_mass_kg 
    #         afr = (air_kg / fuel_kg) if fuel_kg > 0 else 99.0
    #         lambda_ = afr/14.7 if fuel_kg > 0 else 99
    #         self.sensors.afr = afr
    #         self.sensors.lambda_ = lambda_
    #         eff = 0.94 if 0.8 <= lambda_ <= 1.1 else 0.80
            
    #         self.cyl.total_cycle_heat_J = fuel_kg * c.LHV_FUEL_GASOLINE * eff
    #         self.cyl.fuel_mass_kg = 0.0
            
    #         if self._cycle_count > 1: 
    #             print(f"SPARK at CAD:{self.state.current_theta}")
    #             print(
    #                 f"fuel:{fuel_kg:7.5f}kg  air:{air_kg:7.5f}kg  total heat to be released:{self.cyl.total_cycle_heat_J}J "
    #             )
            

    #     # 2. Calculate Heat Release
    #     if self.cyl.combustion_active:
    #         q_step = pf.calc_vibe_heat_release(
    #             rpm=self.sensors.rpm,
    #             lambda_=self.sensors.lambda_,
    #             theta=self.state.current_theta,
    #             Q_total=self.cyl.total_cycle_heat_J,
    #             theta_start=self.cyl.spark_event_theta,
    #             duration_ref=c.BURN_DURATION_DEG # Used as base in the function
    #         )
            
    #         # Energy Conservation Guard: Ensure we don't exceed total_heat_J
    #         # or leave small remainders behind.
    #         if self.cyl.cumulative_heat_released + q_step > self.cyl.total_cycle_heat_J:
    #             q_step = self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released
            
    #         self.cyl.cumulative_heat_released += q_step
            
    #         # End combustion when energy is depleted (or max degrees passed)
    #         if self.cyl.cumulative_heat_released >= (self.cyl.total_cycle_heat_J * 0.999):
    #             self.combustion_active = False
                
    #         return q_step
            
    #     return 0.0

    def _update_mechanical_dynamics(self, CAD, stroke, P_next, dV):
        """Preserves torque integration and RPM physics."""
        P_avg = (self.cyl.P + P_next) / 2.0
        # delta_work_J = P_avg * dV
        # P_ambient is the pressure pushing on the "back" of the piston
        delta_work_J = (P_avg - c.P_ATM_PA) * dV
        
        t_ind_cyl = pf.calc_indicated_torque_step(delta_work_J, stroke)
        
        # after 720 the cylinder buffer is full and simulates 4 cylinders
        self.cyl.torque_indicated_history[CAD] = t_ind_cyl
        t_total_indicated = (
            self.cyl.torque_indicated_history[CAD] +
            self.cyl.torque_indicated_history[(CAD + 180) % 720] +
            self.cyl.torque_indicated_history[(CAD + 360) % 720] +
            self.cyl.torque_indicated_history[(CAD + 540) % 720]
        )

        
        # self.state.torque_friction = pf.calc_friction_torque_per_degree(self.sensors.rpm, self.sensors.CLT_C)
        self.state.torque_friction = self._calc_friction_torque_per_degree(CAD, self.sensors.rpm, self.sensors.CLT_C, self.cyl.log_P)
        
        self.state.torque_brake = t_total_indicated - self.state.torque_friction
        # store values in history arrays
        self.state.torque_indicated_history[CAD] = t_total_indicated 
        self.state.torque_friction_history[CAD] = self.state.torque_friction
        self.state.torque_brake_history[CAD] = self.state.torque_brake
        self.state.torque_net_history[CAD] = self.state.torque_brake - self.state.wheel_load

        # RPM Integration
        omega = pf.eng_speed_rad(self.sensors.rpm)
        dt = np.deg2rad(1.0) / omega
        alpha = self.state.torque_net_history[CAD] / c.MOMENT_OF_INERTIA
        self.sensors.rpm = max(self.crank_rpm, self.sensors.rpm + alpha * dt * 30.0 / np.pi)       
        
        self.sensors.rpm_history[CAD] = self.sensors.rpm
        self.state.power_history[CAD] = self.state.torque_brake * self.sensors.rpm / 9549.3
        
        rpm_delta = self.sensors.rpm_history[CAD - 1] - self.sensors.rpm
        # print(f"DEBUG_MECH | θ:{self.state.current_theta:05.1f} | "
        #     f"T_Indicated:{t_total_indicated:6.2f}Nm | "
        #     f"T_Friction:{self.state.torque_friction:6.2f}Nm | "
        #     f"T_Net:{self.state.torque_brake:6.2f}Nm | "
        #     f"RPM_Delta:{rpm_delta:8.4f}")

    # ----------------------------------------------------------------------
    def _handle_step_init(self, CAD, ecu):  
        if CAD == 0: 
            self._handle_cycle_start()
            print("-"*80,
                "\n"
                f"INTAKE CAD:{CAD} cycle:{self._cycle_count} rpm:{self.sensors.rpm:.0f} "
                f"M_air:{self.cyl.air_mass_kg:.2e}kg M_fuel:{self.cyl.fuel_mass_kg:.2e}kg M_total:{self.cyl.total_mass_kg:.2e}kg "
                f"P:{self.cyl.P:.0f}Pa T:{self.cyl.T:.0f}K "
            )
        elif CAD == 180: print(f"COMPRESSION CAD:{CAD}")
        elif CAD == 360: print(f"POWER CAD:{CAD}")
        elif CAD == 540: print(f"EXHAUST CAD:{CAD}")
            
        if CAD == self.valves.intake.open_angle: 
            print(f"  IVO θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
                  f"T_cyl:{self.cyl.T:.0f}K ")
            self.temp_total_dm_f = 0.0
            
        elif CAD == self.valves.exhaust.open_angle: 
            print(f"  EVO θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
                  f"T_cyl:{self.cyl.T:.0f}K ")
    
    def _handle_step_end(self, CAD, ecu):
        if CAD >= 719.0:
            self._handle_cycle_end(ecu)
            self._cycle_count += 1

        elif CAD == self.valves.intake.close_angle: 
            print(f"  IVC θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
                  f"afr_calculated:{self.cyl.air_mass_kg/self.cyl.fuel_mass_kg:4.1f} "
                  f"ECU_air_estimate:{ecu['trapped_air_mass_kg']:.2e} ECU_target_afr:{ecu['afr_target']:.1f} "
                  f"T_cyl:{self.cyl.T:.0f}K "
                #   f" AFR:{self.sensors.afr:4.1f} {self.cyl.air_mass_at_spark:7.5f} {self.cyl.fuel_mass_at_spark:7.5f}"
                #   f"Spark:{ecu['spark_timing']} Injector_start:{ecu['injector_start_deg']} injector_end:{ecu['injector_end_deg']} "
                  )
        
        elif CAD == self.valves.exhaust.close_angle:
            print(f"  EVC θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
                #   f"Spark:{ecu['spark_timing']} Injector_start:{ecu['injector_start_deg']} injector_end:{ecu['injector_end_deg']} "                  
                  )
            
    
    def _handle_cycle_start(self):
        # prep for next cycle
        self.cyl.log_P.fill(c.P_ATM_PA)
        self.cyl.log_V.fill(0.0)
        self.cyl.log_T.fill(c.T_AMBIENT)

    def _handle_cycle_end(self, ecu_outputs):
        CAD = int(self.state.current_theta)
        
        # determine peak pressure and peak pressure angle for dashboard reporting
        peak_bar = max(self.cyl.log_P) / 100000.0 # Ensure P is in bar
        self.cyl.P_peak_bar = peak_bar
        self.cyl.P_peak_angle = np.argmax(self.cyl.log_P)

        # Update CLT sensor
        self.sensors.CLT_C = pf.update_coolant_temp(self.sensors.CLT_C, self.state.torque_brake, self.sensors.rpm)
        
        # Update knock Sensor
        if self.cyl.cumulative_heat_released > 0.0:
            self.sensors.knock, self.sensors.knock_intensity = pf.detect_knock(
                peak_bar = peak_bar,
                clt = self.sensors.CLT_C, 
                rpm = self.sensors.rpm,   
                spark_advance = self.cyl.spark_advance_btdc,
                lambda_ = self.sensors.lambda_,
                fuel_octane = c.FUEL_OCTANE
            )
        else: 
            self.sensors.knock = False
            self.sensors.knock_intensity = 0.0
        
        # Update Cylinder Wall Temp
        self.cyl.T_wall = pf.update_cylinder_wall_temperature(
            self.sensors.CLT_C, 
            self.cyl.Q_loss_total, 
            self.sensors.rpm
        )
        self.cyl.Q_loss_total = 0.0
        
        # update Intake Manifold Pressure (MAP) sensor
        # -- Combine driver TPS and ECU idle control
        
        # -- scale idle valve % upto WOT
        IDLE_AIR_CAPACITY = 0.10  # 10% of total WOT airflow
        # idle_contribution = (ecu_outputs["idle_valve_position"] / 100.0) * IDLE_AIR_CAPACITY * 100.0
        idle_contribution = ecu_outputs["idle_valve_position"] * 100

        # calc total throttle fraction
        effective_tps = np.clip(self.sensors.TPS_percent + idle_contribution, 0, 100)
        self.state.effective_tps = effective_tps
        tps_frac = np.clip(effective_tps / 100.0, 0.0, 1.0)
        
        # --- set MAP
        map_Pa = pf.update_intake_manifold_pressure(effective_tps, self.sensors.rpm)  
        self.sensors.P_manifold_Pa = map_Pa
        self.state.map_history[CAD] = map_Pa
        
        print(
            f"    DEBUG step end: CAD={self._cycle_count}/{CAD} "
            f"rpm:{self.sensors.rpm:.0f} P-peak:{peak_bar:.0f}bar P_peak_angle:{self.cyl.P_peak_angle:.0f} T_peak:{max(self.cyl.log_T):.0f}K "
            f"T_wall:{self.cyl.T_wall:.0f}K MAP:{map_Pa:.1f}:Pa  "
        )
        
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
    
    # ----------------------------------------------------------------------
    def _calc_friction_torque_per_degree(self, CAD, rpm, clt, pressure_history):
        """
        Calculates friction for all 4 cylinders based on their specific 
        position in the 720 degree cycle.
        """
        total_fric_torque = 0
        
        # Offsets for a 4-cylinder engine
        offsets = [0, 180, 360, 540]
        
        for offset in offsets:
            # Determine the CAD for this specific cylinder
            cyl_cad = (CAD + offset) % 720
            cyl_p = pressure_history[cyl_cad]
            
            # Calculate friction for THIS cylinder at THIS specific position
            t_fric_cyl = pf.calc_single_cylinder_friction(cyl_cad, rpm, cyl_p, clt)
            total_fric_torque += t_fric_cyl
            
        return total_fric_torque
        
    
    # ----------------------------------------------------------------------
    def _calculate_cycle_work(self):
        """
        Integrates P*dV across the four strokes to find the work done in Joules.
        Positive = Engine producing work. Negative = Engine consuming energy.
        """
        # Instantaneous work for every degree (Joules)
        # Note: dV must be in m^3, P in Pa
        work_deg = self.cyl.log_P * self.cyl.dV_list
        
        # Slice by standard 4-stroke boundaries
        # 0-180: Intake | 180-360: Comp | 360-540: Power | 540-720: Exhaust
        pumping_in   = np.sum(work_deg[0:180])
        compression  = np.sum(work_deg[180:360])
        expansion    = np.sum(work_deg[360:540])
        pumping_ex   = np.sum(work_deg[540:720])
        
        return {
            "work_pumping_j": pumping_in + pumping_ex,
            "work_compression_j": compression,
            "work_expansion_j": expansion,
            "net_indicated_work_j": np.sum(work_deg),
            "work_deg": work_deg
        }

 