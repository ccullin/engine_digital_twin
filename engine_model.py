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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecu_controller import EcuOutput


@dataclass(slots=True, frozen=False)
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
    


@dataclass(slots=True, frozen=False)
class CylinderState:
    """Master Cylinder (Cyl 1) Attributes and Instantaneous State"""
    
    # Geometry Attributes
    A_piston: float = c.A_PISTON
    V_displaced: float = c.V_DISPLACED
    V_clearance: float = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
    V_list: np.ndarray = field(init=False, repr=False)
    dV_list: np.ndarray = field(init=False, repr=False)

    # Thermodynamic State
    P_curr: float = c.P_ATM_PA   # Pa
    T_curr: float = c.T_AMBIENT  # Kelvin
    V_curr: float = field(init=False, repr=False)
    P_next: float = c.P_ATM_PA   # Pa
    T_next: float = c.T_AMBIENT  # Kelvin
    # M_gas: float = 5.8e-4  
    # M_gas: float = field(init = False) # kg.  Mass of a cylinder of gas (fresh and exhaust)
    log_P: np.ndarray = field(default_factory=lambda: np.full(720, c.P_ATM_PA))
    # log_V: np.ndarray = field(default_factory=lambda: np.zeros(720))
    log_T: np.ndarray = field(default_factory=lambda: np.full(720, c.T_AMBIENT))
    P_peak_bar: float = 0.0
    P_peak_angle: float = 0.0
    T_wall: float = c.T_AMBIENT + 20   # adding buffer for start stability and speed
    
    # Thermodynamic tracking
    Q_loss_total: float = 0.0 # heat loss from ctl to cyl_wall
    Q_loss_step_sum: float = 0.0 # tracks sub-step total for engine data reporting
    Q_in_step_sum: float = 0.0 # tracks sub-step total for engine data reporting
    
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
    total_mass_at_spark: float = 0.0
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
        self.V_curr = self.V_list[0]
        self.total_mass_kg = (self.P_curr * self.V_clearance) / (c.R_SPECIFIC_EXHAUST * self.T_curr)
        

    
@dataclass(slots=True, frozen=False)
class Valve:
    """stores valve geometry"""
    open_angle: float
    close_angle: float
    max_lift: float
    diameter: float
        
@dataclass(slots=True, frozen=False)
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
        
 
        

@dataclass(slots=True, frozen=False)
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
    torque_indicated: float = 0.0
    torque_friction: float = 0.0
    torque_brake: float = 0.0
    torque_governor_history: np.ndarray = field(default_factory=lambda: np.zeros(720)) # when motoring measure the torque required to hold rpm
    
    # friction breakdown
    torque_friction_piston: float = 0.0
    torque_friction_global: float = 0.0 # captures valve train, oil pump, bearing friction
    torque_friction_piston_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
    torque_friction_global_history: np.ndarray = field(default_factory=lambda: np.zeros(720))
  
    # work audit
    work_deg: np.ndarray = field(default_factory=lambda: np.zeros(720))
    work_pumping_j: float = 0.0
    work_compression_j: float = 0.0
    work_expansion_j: float = 0.0
    work_net_indicated_j: float = 0.0
    work_gross_indicated_j: float = 0.0
     
    @property
    def map_avg_kPa(self) -> float:
        return np.mean(self.map_history)
    
    @property
    def friction_work_j(self) -> float:
        return np.sum(self.torque_friction_history) * np.pi / 180.0
    
    @property
    def net_work_j(self) -> float:
        return np.sum(self.torque_brake_history) * np.pi / 180.0
    
    @property
    def indicated_work_j(self) -> float:
        return np.sum(self.torque_indicated_history) * np.pi / 180.0
    
    
@dataclass(slots=True, frozen=True)
class EngineTelemetry:
    """includes all status information """
    cyl: CylinderState
    valves: Valves
    state: EngineState
    
    theta_list: np.ndarray = field(default_factory=lambda: np.arange(720).astype(int))
    


class EngineModel:
    def __init__(self, rpm):
        # Initialize Blocks
        self.cyl = CylinderState()
        self.state = EngineState()
        self.sensors = EngineSensors(rpm=rpm)
        self.valves = Valves()
        self.telemtry = EngineTelemetry(cyl=self.cyl, valves=self.valves, state=self.state)
        
    
        
        # counters
        # self.state.next_theta = self.valves.intake.open_angle # start the engine at IVO, useful for debugging
        # self.state.current_theta = self.valves.intake.open_angle # start the engine at IVO, useful for debugging
        self._cycle_count = 0
        self.temp_total_dm_f = 0.0
        
        # update cyl gas composition based on initialisation air, fuel and exhaust mix
        self._update_gas_properties()
        
        self.motoring_rpm = 0.0  # in motoring mode the driver strategy will set this for engine analysis.
        
        # self.print_geom()


    def print_geom(self):
        print("")
        print(f"//" *80)
        print(f"IVO:{self.valves.intake.open_angle} IVC:{self.valves.intake.close_angle} EVO:{self.valves.exhaust.open_angle} EVC:{self.valves.exhaust.close_angle}")

    # ----------------------------------------------------------------------
    def get_sensors(self):
        return self.sensors
        

    # =================================================================
    # REAL-TIME STEP — CALLED EVERY DEGREE
    # =================================================================
    def step(self, ecu_outputs: EcuOutput):
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
    
        # Run physics for this ONE degree
        self._step_one_degree(ecu_outputs)
        self._handle_step_end(CAD, ecu_outputs)

        return self.sensors, self.telemtry
    
    # ----------------------------------------------------------------------
    def _step_one_degree(self, ecu_outputs:EcuOutput):
        """
        Called every 1° of crank rotation.
        """        
        
        CAD = int(self.state.current_theta)
        stroke, _ = self._get_stroke()
        P_next = self.cyl.log_P[CAD]
        T_next = self.cyl.log_T[CAD]


        # 2. Mass Flow (Air & Fuel)
        # adding sub-degree increments for flow calculations
        n_sub = 10
        substep_size = 1.0 / n_sub
        
        Q_in_step_sum = 0.0
        Q_loss_step_sum = 0.0
        self.cyl.dm_in_history[CAD] = 0.0
        self.cyl.dm_ex_history[CAD] = 0.0
        self.cyl.Cd_in_history[CAD] = 0.0
        self.cyl.Cd_ex_history[CAD] = 0.0
        
        for substep in range(n_sub):
            
            # update P_curr & T_curr 
            P_curr = P_next
            T_curr = T_next
            self.cyl.P_curr = P_curr
            self.cyl.T_curr = T_curr
            
            # set V_curr
            dV_d_theta = self.cyl.dV_list[CAD]
            V_curr = self.cyl.V_list[CAD] + (dV_d_theta * substep_size * substep)
            self.cyl.V_curr = V_curr
            
            # 1. Calculate Deltas (Pure)
            deltas = self._calc_flow_deltas(ecu_outputs, substep_size)
            # store these by CAD for engine analysis. Sum the mass and average the Cd's
            self.cyl.dm_in_history[CAD] += deltas['dm_i']
            self.cyl.dm_ex_history[CAD] += deltas['dm_e']
            self.cyl.Cd_in_history[CAD] += deltas['Cd_i'] / n_sub
            self.cyl.Cd_ex_history[CAD] += deltas['Cd_e'] / n_sub
            
            # 2. Volume and Heat Loss
            Q_in_sub = self._calculate_combustion_heat_substep(CAD, substep, substep_size, ecu_outputs.spark)
            Q_loss_sub = pf.calc_woschni_heat_loss(CAD=CAD + (substep * substep_size), rpm=self.sensors.rpm, cyl=self.cyl, valves=self.valves)
            Q_loss_sub_joules = Q_loss_sub * substep_size
            
            Q_in_step_sum += Q_in_sub
            Q_loss_step_sum += Q_loss_sub_joules
            self.cyl.Q_loss_total += Q_loss_sub_joules # track total per cyl for cyl wall T update at end of cycle.

            # 3. Integrate First Law
            # Note: We pass the CURRENT mass here. The integrator handles dM/dtheta.
            P_next, T_next = pf.integrate_first_law(
                P_curr=P_curr, T_curr=T_curr, M_curr=self.cyl.total_mass_kg,
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
            # if sum_species > self.cyl.total_mass_kg * 1.01:
            #     print(f"!!! SPECIES OVERFLOW at CAD {CAD}: Sum({sum_species}) > Total({self.cyl.total_mass_kg})")

            # --- FUEL INJECTION ---
            self.cyl.fuel_mass_kg += deltas["dm_f"]

            # --- FINAL TOTAL ---
            self.cyl.total_mass_kg += deltas["dm_tot"]

                    
            # 5. Re-calculate properties for the NEXT substep
            self._update_gas_properties() # Helper to refresh R_blend and gamma_blend  
            
            # --- COMBUSTION DEBUG ---
            # Only print during the expansion stroke where combustion happens
            # if 335 <= CAD <= 540 and substep == 4:
            if Q_in_sub > 0:
            # Calculate the 'Energy Density' to see if the heat addition is sane
                specific_energy_j_kg = Q_in_sub / max(self.cyl.total_mass_kg, 1e-9)
                
                # print(f"    DEBUG COMB | θ:{self._cycle_count}/{CAD}/{substep} | P:{P_next/1e5:4.1f}bar | T:{T_next:4.0f}K "
                #         f"T_wall:{self.cyl.T_wall:4.0f}K "
                #         f"dQ_in:{Q_in_sub:6.2f}J | dQ/m:{specific_energy_j_kg:6.0f}J/kg | gamma:{self.cyl.gamma_blend:.3f} "
                #         f"dQ_loss:{Q_loss_sub:6.2f}J "
                #     )
                    
                # Check for "Thermal Runaway"
                if T_next > 3500 or T_next < 200:
                    print(f"  !!! INSTABILITY DETECTED at CAD {CAD}/{substep}. 3500K < Temp:{T_next:4.0f} < 200K.   Physics breaking.")

        # update logs with the results of the final substep
        self.cyl.log_P[(CAD+1) %720] = P_next
        self.cyl.log_T[(CAD+1) %720] = T_next
        # self.cyl.log_V[CAD] = self.cyl.V_list[CAD]   

        # UPDATE INTAKE MANIFOLD PRESSURE (MAP SENSOR) 
        # calc total throttle fraction
        idle_valve = ecu_outputs.iacv_wot_equiv
        effective_tps = np.clip(self.sensors.TPS_percent + idle_valve, 0, 100)
        self.state.effective_tps = effective_tps
        tps_frac = np.clip(effective_tps / 100.0, 0.0, 1.0)
        

        
        # update heat loss trackers for Engine Data reporting
        self.cyl.Q_in_step_sum = Q_in_step_sum
        self.cyl.Q_loss_step_sum = Q_loss_step_sum

        
        #  MECHANICAL DYNAMICS (Calculated once per degree using final state)
        # We use the full degree dV here because work is summed across the step
        self._update_mechanical_dynamics(CAD, stroke, self.cyl.log_P[(CAD - 1) % 720], P_curr, self.cyl.dV_list[CAD])
        
        # --- set MAP using latest rpm
        map_Pa = pf.update_intake_manifold_pressure(effective_tps, self.sensors.rpm)    
        self.sensors.P_manifold_Pa = map_Pa
        self.state.map_history[CAD] = map_Pa

            
    # --- Helper Methods to maintain functionality ---
    
    def _calc_flow_deltas(self, ecu_outputs:EcuOutput, step_size):
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
            P_cyl=self.cyl.P_curr, T_cyl=self.cyl.T_curr, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
            P_extern=self.sensors.P_manifold_Pa, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
            is_intake=True
        )
        dm_i = dm_i_raw * dt
        
        # Exhaust
        dm_e_raw, Cd_e = pf.calc_isentropic_flow(
            A_e, L_e, self.valves.exhaust.diameter,
            P_cyl=self.cyl.P_curr, T_cyl=self.cyl.T_curr, R_cyl=self.cyl.R_specific_blend, g_cyl=self.cyl.gamma_blend,
            P_extern=c.P_ATM_PA, T_extern=self.state.T_exhaust_manifold, R_extern=c.R_SPECIFIC_EXHAUST, g_extern=c.GAMMA_EXHAUST,
            is_intake=False
        )
        dm_e = dm_e_raw * dt

        # 3. Fuel Calculation (Scale by step_size)
        dm_f = 0.0
        if ecu_outputs.injector_on:
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

        T_inflow = weighted_T / mass_in if mass_in > 1e-18 else self.cyl.T_curr

        # 5. Return everything needed for the sub-step integration
        return {
            "dm_i": dm_i, "dm_e": dm_e, "dm_f": dm_f, "dm_tot": dm_tot,
            "T_inflow": T_inflow, "Cd_i": Cd_i, "Cd_e": Cd_e
        }
        
        
        
    def _update_gas_properties(self):
        m_total_pre = max(self.cyl.total_mass_kg, 1e-9)
        f_air = self.cyl.air_mass_kg / m_total_pre
        f_fuel = self.cyl.fuel_mass_kg / m_total_pre
        f_exh = max(0, self.cyl.total_mass_kg - (self.cyl.air_mass_kg + self.cyl.fuel_mass_kg)) / m_total_pre

        # NEW: Get temperature-dependent cv for the dominant species
        # This replaces the static c.R / (c.GAMMA - 1) calculation
        cv_air_dyn = pf.calc_specific_heat_cv(self.cyl.T_curr) # Dynamic based on T
        cv_fuel = c.R_SPECIFIC_FUEL / (c.GAMMA_FUEL - 1.0) # Fuel remains constant
        cv_exh_dyn = cv_air_dyn * 1.05 # Exhaust products typically have ~5% higher cv

        # Blend using the dynamic values
        cv_blend = (f_air * cv_air_dyn) + (f_fuel * cv_fuel) + (f_exh * cv_exh_dyn)
        
        # Calculate R_blend (R is physically constant even if cv changes)
        R_blend = (f_air * c.R_SPECIFIC_AIR) + (f_fuel * c.R_SPECIFIC_FUEL) + (f_exh * c.R_SPECIFIC_EXHAUST)
        
        # Update state
        self.cyl.R_specific_blend = R_blend
        self.cyl.gamma_blend = (R_blend / cv_blend) + 1
        
        
    def _calculate_combustion_heat_substep(self, CAD, substep, substep_size, spark_command):
        """
        Substep-friendly Wiebe heat release with corrected energy return.
        """
        # 1. Trigger Spark Event (ECU-synced at 1-degree increments)
        if spark_command and substep == 0 and not self.cyl.combustion_active:
            self.cyl.spark_event_theta = CAD
            
            # Physical Delay: ~1.0ms is a standard SI kernel formation time
            ign_delay = (self.sensors.rpm / 60.0) * 360.0 * 0.001
            self.cyl.ignition_start_theta = CAD + ign_delay
            
            self.cyl.combustion_active = True
            self.cyl.cumulative_heat_released = 0.0
            
            # Snapshot ingredients and set AFR for this burn
            self.cyl.air_mass_at_spark = self.cyl.air_mass_kg
            self.cyl.fuel_mass_at_spark = self.cyl.fuel_mass_kg
            self.cyl.total_mass_at_spark = self.cyl.total_mass_kg
            self.sensors.afr = (self.cyl.air_mass_kg / self.cyl.fuel_mass_kg) if self.cyl.fuel_mass_kg > 0 else 99.0
            
            # Determine total energy 'bucket' with 0.97 efficiency for peak power
            # eff = 0.97 if 0.85 <= self.sensors.lambda_ <= 1.05 else 0.85
            # eff = 0.94 if 0.8 <= self.sensors.lambda_ <= 1.1 else 0.80
            # if self.sensors.lambda_ < 1.0:
            #     self.cyl.total_cycle_heat_J = (self.cyl.air_mass_kg / 14.7) * c.LHV_FUEL_GASOLINE * eff
            # else:
            #     self.cyl.total_cycle_heat_J = self.cyl.fuel_mass_kg * c.LHV_FUEL_GASOLINE * eff
            
            # First principles: Energy is limited by whichever reactant is depleted first (Stoichiometry)
            stoich_fuel_req = self.cyl.air_mass_kg / 14.7

            # Theoretical max energy available if 100% efficient
            theoretical_max_energy = min(self.cyl.fuel_mass_kg, stoich_fuel_req) * c.LHV_FUEL_GASOLINE

            # Combustion Efficiency should be a function of physics (e.g., turbulence, temp)
            # For now, keep it as a constant, but separate it from the energy math
            combustion_efficiency = 0.95 
            self.cyl.total_cycle_heat_J = theoretical_max_energy * combustion_efficiency

            self.cyl.burn_duration = pf.get_burn_duration(self.sensors.rpm, self.sensors.lambda_)
            self.cyl.m_vibe = max(1.5, 2.5 - (self.sensors.rpm / 4000.0))

        # 2. Execution Logic
        # Use precise theta to ensure we don't 'miss' the start due to substep alignment
        precise_theta = CAD + (substep * substep_size)
        
        if precise_theta >= self.cyl.ignition_start_theta and self.cyl.combustion_active:
            # Physics Calculation (Slice of the Wiebe curve)
            dQ_combustion, dm_fuel, dm_air = pf.calculate_combustion_dq(
                CAD, substep, substep_size, self.cyl, self.sensors.lambda_
            )
            
            # Update state
            self.cyl.fuel_mass_kg -= dm_fuel
            self.cyl.air_mass_kg  -= dm_air
            self.cyl.cumulative_heat_released += dQ_combustion

            # 3. Completion and 'The Snap'
            completion_ratio = 0.0
            if self.cyl.total_cycle_heat_J > 0:
                completion_ratio = self.cyl.cumulative_heat_released / self.cyl.total_cycle_heat_J
            
            out_of_fuel = self.cyl.fuel_mass_kg < 1e-12
            out_of_air = self.cyl.air_mass_kg < 1e-12
            
            if completion_ratio >= 0.995 or out_of_fuel or out_of_air:
                final_snap_joules = max(0.0, self.cyl.total_cycle_heat_J - self.cyl.cumulative_heat_released)
                
                # IMPORTANT: Add the snap to the dQ we are returning for THIS substep
                dQ_combustion += final_snap_joules
                self.cyl.cumulative_heat_released += final_snap_joules
                
                # Mass cleanup
                if self.sensors.lambda_ < 1.0:
                    self.cyl.air_mass_kg = 0.0
                else:
                    self.cyl.fuel_mass_kg = 0.0

                self.cyl.combustion_active = False

            # Return the total energy released in this substep (Wiebe slice + any Snap)
            return dQ_combustion
        
        return 0.0

    def _update_mechanical_dynamics(self, CAD, stroke, cyl_P_previous, cyl_P, dV):
        """Preserves torque integration and RPM physics."""
        P_avg = (cyl_P_previous + cyl_P) / 2.0
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
        self.state.torque_indicated = t_total_indicated
        self.state.torque_friction = self._calc_friction_torque_per_degree(CAD, self.sensors.rpm, self.sensors.CLT_C, self.cyl.log_P)
        self.state.torque_brake = self.state.torque_indicated - self.state.torque_friction
        
        # store values in history arrays
        self.state.torque_indicated_history[CAD] = t_total_indicated 
        self.state.torque_friction_history[CAD] = self.state.torque_friction
        self.state.torque_brake_history[CAD] = self.state.torque_brake
        self.state.torque_net_history[CAD] = self.state.torque_brake - self.state.wheel_load

        # RPM Integration
        omega = pf.eng_speed_rad(self.sensors.rpm)
        dt = np.deg2rad(1.0) / omega
        alpha = self.state.torque_net_history[CAD] / c.MOMENT_OF_INERTIA
        next_rpm = self.sensors.rpm + alpha * dt * 30.0 / np.pi
            
        if self.motoring_rpm > 0.0:  # engine is in motoring mode for testing
            rpm_delta = self.motoring_rpm - next_rpm

            # Calculate exactly how many Nm are needed to fix that error in ONE dt
            # Formula: T = J * alpha -> T = J * (delta_omega / dt)
            delta_omega = (rpm_delta * np.pi) / 30.0
            required_governor_torque = (c.MOMENT_OF_INERTIA * delta_omega) / dt

            # Apply it unconditionally (Motor or Brake)
            self.state.torque_net_history[CAD] += required_governor_torque
            self.state.torque_governor_history[CAD] = required_governor_torque
            self.sensors.rpm = self.motoring_rpm # Force steady state for the next step
        else:   
            self.sensors.rpm = max(c.CRANK_RPM, next_rpm)       
        
        self.sensors.rpm_history[CAD] = self.sensors.rpm
        self.state.power_history[CAD] = self.state.torque_brake * self.sensors.rpm / 9549.3
        
        # rpm_delta = self.sensors.rpm_history[CAD - 1] - self.sensors.rpm
        # if CAD == 719:
        #     print(f"DEBUG_MECH | θ:{self._cycle_count}/{self.state.current_theta:05.1f} | "
        #         f"T_Indicated:{np.average(self.state.torque_indicated_history):6.2f}Nm | "
        #         f"T_Friction:{np.average(self.state.torque_friction_history):6.2f}Nm | "
        #         f"T_Net:{np.average(self.state.torque_brake_history):6.2f}Nm | "
        #         f"RPM_Delta:{rpm_delta:8.4f}")

    # ----------------------------------------------------------------------
    def _handle_step_init(self, CAD, ecu):  
        if CAD == 0: 
            self._handle_cycle_start()
            # print("-"*80,
            #     "\n"
            #     f"INTAKE CAD:{CAD} cycle:{self._cycle_count} rpm:{self.sensors.rpm:.0f} "
            #     f"M_air:{self.cyl.air_mass_kg:.2e}kg M_fuel:{self.cyl.fuel_mass_kg:.2e}kg M_total:{self.cyl.total_mass_kg:.2e}kg "
            #     f"P:{self.cyl.P:.0f}Pa T:{self.cyl.T:.0f}K "
            # )
        # elif CAD == 180: print(f"COMPRESSION CAD:{CAD}")
        # elif CAD == 360: print(f"POWER CAD:{CAD}")
        # elif CAD == 540: print(f"EXHAUST CAD:{CAD}")
            
        if CAD == self.valves.intake.open_angle: 
            # print(f"  IVO θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
            #       f"T_cyl:{self.cyl.T:.0f}K "
            #       f"eff_TPS:{self.state.effective_tps:.0f} IACV:{ecu['iacv_wot_equiv']}"
            #       )
            self.temp_total_dm_f = 0.0
            
        # elif CAD == self.valves.exhaust.open_angle: 
        #     print(f"  EVO θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
        #           f"T_cyl:{self.cyl.T:.0f}K ")
    
    def _handle_step_end(self, CAD, ecu):
        if CAD >= 719.0:
            self._handle_cycle_end(ecu)
            self._cycle_count += 1

        # elif CAD == self.valves.intake.close_angle: 
        #     print(f"  IVC θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
        #           f"afr_calculated:{self.cyl.air_mass_kg/self.cyl.fuel_mass_kg:4.1f} "
        #           f"ECU_air_estimate:{ecu['trapped_air_mass_kg']:.2e} ECU_target_afr:{ecu['afr_target']:.1f} "
        #           f"T_cyl:{self.cyl.T:.0f}K "
        #         #   f" AFR:{self.sensors.afr:4.1f} {self.cyl.air_mass_at_spark:7.5f} {self.cyl.fuel_mass_at_spark:7.5f}"
        #         #   f"Spark:{ecu['spark_timing']} Injector_start:{ecu['injector_start_deg']} injector_end:{ecu['injector_end_deg']} "
        #           )
        
        # elif CAD == self.valves.exhaust.close_angle:
        #     print(f"  EVC θ:{CAD:3d} air_mass:{self.cyl.air_mass_kg:.2e} fuel_mass:{self.cyl.fuel_mass_kg:.2e} total_mass:{self.cyl.total_mass_kg:.2e} "
        #         #   f"Spark:{ecu['spark_timing']} Injector_start:{ecu['injector_start_deg']} injector_end:{ecu['injector_end_deg']} "                  
        #           )
            
    
    def _handle_cycle_start(self):
        # prep for next cycle
        # self.cyl.log_P.fill(c.P_ATM_PA)
        # self.cyl.log_V.fill(0.0)
        # self.cyl.log_T.fill(c.T_AMBIENT)
        
        self.cyl.Q_loss_total = 0.0 # reset at start so it can be used by analysis at end

    def _handle_cycle_end(self, ecu_outputs:EcuOutput):
        CAD = int(self.state.current_theta)
        
        # determine peak pressure and peak pressure angle for dashboard reporting
        peak_bar = max(self.cyl.log_P) / 1e5 # Ensure P is in bar
        self.cyl.P_peak_bar = peak_bar
        self.cyl.P_peak_angle = np.argmax(self.cyl.log_P)

        # update thermal state
        self._update_thermal_state()
         
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
        
        # print(
        #     f"    DEBUG step end: CAD={self._cycle_count}/{CAD} "
        #     f"rpm:{self.sensors.rpm:.0f} P-peak:{peak_bar:.0f}bar P_peak_angle:{self.cyl.P_peak_angle:.0f} T_peak:{max(self.cyl.log_T):.0f}K "
        #     f"T_wall:{self.cyl.T_wall:.0f}K MAP:{self.sensors.MAP_kPa:.1f}:Pa  "
        # )
    
    # ----------------------------------------------------------------------
    def _update_thermal_state(self):
        avg_torque_brake = np.average(self.state.torque_brake_history)
        avg_rpm = np.average(self.sensors.rpm_history)
        self.sensors.CLT_C = pf.update_coolant_temp(self.sensors.CLT_C, avg_torque_brake, avg_rpm)
        
        self.cyl.T_wall = pf.update_cylinder_wall_temperature(
            self.sensors.CLT_C, 
            self.cyl.Q_loss_total, 
            avg_rpm,
            self.cyl.T_wall
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
        total_friction_torque = 0
        
        # Offsets for a 4-cylinder engine
        offsets = [0, 180, 360, 540]
        
        for offset in offsets:
            # Determine the CAD for this specific cylinder
            cyl_cad = (CAD + offset) % 720
            cyl_p = pressure_history[cyl_cad]
            
            # Calculate friction for THIS cylinder at THIS specific position
            t_fric_cyl = pf.calc_single_cylinder_friction(cyl_cad, rpm, cyl_p, clt)
            total_friction_torque += t_fric_cyl
        
        global_parasitic = pf.calc_engine_core_friction(rpm=rpm, clt=clt)
                
        self.state.torque_friction_piston = total_friction_torque
        self.state.torque_friction_piston_history[CAD] = total_friction_torque
        
        self.state.torque_friction_global = global_parasitic
        self.state.torque_friction_global_history[CAD] = global_parasitic
        

         
        return total_friction_torque + global_parasitic
        
    
    # ----------------------------------------------------------------------
    def _calculate_cycle_work(self):
        """
        Calculates Engine Work Done 
        Integrates P*dV across the four strokes to find the work done in Joules.
        Positive = Engine producing work. Negative = Engine consuming energy.
        """
        # 1. Calculate base work for the master cylinder
        # work = P * dV (Joules per degree)
        w_cyl = self.cyl.log_P * self.cyl.dV_list
        
        # 2. Create the Engine-Level work_deg array
        # We sum 4 versions of the array, each shifted by the firing interval (180 deg)
        engine_work_deg = (
            w_cyl + 
            np.roll(w_cyl, 180) + 
            np.roll(w_cyl, 360) + 
            np.roll(w_cyl, 540)
        )
        
        # 3. Calculate Stroke Totals for the whole engine
        # Note: Using the single cylinder sum * NUM_CYL is equivalent and faster
        pumping_work = (np.sum(w_cyl[0:180]) + np.sum(w_cyl[540:720])) * c.NUM_CYL
        compression  = np.sum(w_cyl[180:360]) * c.NUM_CYL
        expansion    = np.sum(w_cyl[360:540]) * c.NUM_CYL
    
        # Gross and Net Indictaed Work
        work_net_indicated   = np.sum(w_cyl) * c.NUM_CYL
        work_gross_indicated = (np.sum(w_cyl[180:360]) + np.sum(w_cyl[360:540])) * c.NUM_CYL
        
        # if self.state.current_theta >= 719:
        #     print(
        #         f"DEBUG WORK: cyc:{self._cycle_count:.0f} "
        #         f"rpm_avg:{np.mean(self.sensors.rpm_history):.0f} "
        #         f"work_pumping_j:{pumping_in + pumping_ex:.2f} "
        #         f"work_compression_j:{compression:.2f} "
        #         f"work_expansion_j:{expansion:.2f} "
        #         f"net_indicated_work_j:{np.sum(work_deg):.2f} "
        #     )
        
        self.state.work_pumping_j = pumping_work
        self.state.work_compression_j = compression
        self.state.work_expansion_j = expansion
        self.state.work_net_indicated_j = work_net_indicated
        self.state.work_gross_indicated_j = work_gross_indicated
        self.state.work_deg = engine_work_deg
        

 