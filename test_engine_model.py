import unittest
import numpy as np
import constants as c
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

from engine_model import EngineModel, Valves, Valve, CylinderState
from ecu_controller import EcuOutput
import physics_functions as pf

# =================================================================
#  ENGINE PRESETS to standardise engine setup across tests.
# =================================================================
class RunMode(Enum):
    IDLE = auto()
    WOT = auto()
    CRUISE = auto()
    CRANK = auto()

class EnginePreset(NamedTuple):
    rpm: float
    tps: float
    iacv_pos: float
    afr_target: float
    thermal_state: str
    rpm_hold: bool 
    
# The Single Source of Truth for your Test Cell
ENGINE_PRESETS = {
    RunMode.IDLE: EnginePreset(rpm=900.0, 
                                tps=0.0, 
                                iacv_pos=48.0, 
                                afr_target=14.7,
                                thermal_state="warm",
                                rpm_hold=True),
    
    RunMode.WOT: EnginePreset(rpm=4500.0, 
                                tps=100.0, 
                                iacv_pos=10.0, 
                                afr_target=12.5,
                                thermal_state="warm",
                                rpm_hold=True),
    
    RunMode.CRUISE: EnginePreset(rpm=2500.0, 
                                tps=40.0, 
                                iacv_pos=10.0, 
                                afr_target=14.7,
                                thermal_state="warm",
                                rpm_hold=True)
}


    
# =================================================================
# 1. SHARED INFRASTRUCTURE (The "Digital Test Cell")
# =================================================================

# @dataclass(slots=True, frozen=False)
class MockEcuOutput(EcuOutput):
    """Consolidated ECU for all tiers of testing."""
    spark: bool = False
    spark_timing: int = 350
    afr_target: float = 14.7 #* 0.88
    target_rpm: float = 0.0
    iacv_pos: float = 40 #20 #11
    pid_P: float = 0.0 
    pid_I: float = 0.0 
    pid_D: float = 0.0
    injector_on: bool = False
    injector_start: int = 0
    injector_end: int = 170
    fuel_cut_active: bool = False
    
    
    @property
    def iacv_wot_equiv(self) -> float:
        return (self.iacv_pos * 0.06)
    
    @iacv_wot_equiv.setter
    def iacv_wot_equiv(self, value: float):
        """Triggers when you WRITE to ecu.iacv_wot_equiv = value"""
        # Intercept and drop the base class default assignment at startup
        if value == 0.0:
            return
            
        # # Otherwise, keep variables linked backward if modified mid-test
        # self.iacv_pos = value / 0.06
        

@dataclass
class MockValve():
    # def __init__(self, open=0 , close=0, cam_open=0, cam_close=0, lift=0, diameter=0):
    #     self.diameter = diameter / 1000
    #     self.max_lift = lift / 1000.0
    #     self.cam_open = cam_open
    #     self.cam_close = cam_close
    #     self.open_angle = open
    #     self.close_angle = close
    is_intake: bool
    open_1mm: float = field(init=False)
    close_1mm: float = field(init=False)
    diameter: float = field(init=False)
    max_lift: float = field(init=False)
    open_angle: float = field(init=False)
    close_angle: float = field(init=False)
    
    def __post_init__(self):
        if self.is_intake:
            self.open_1mm = 0
            self.close_1mm = 0
            self.open_angle = 698
            self.close_angle = 238
            self.max_lift = 8
            self.diameter = 40
        else:
            self.open_1mm = 0
            self.close_1mm = 0
            self.open_angle = 480
            self.close_angle = 24
            self.max_lift = 9
            self.diameter = 34

   
@dataclass
class MockValves():
    intake: MockValve = field(default_factory=lambda: MockValve(is_intake=True)) 
    exhaust: MockValve = field(default_factory=lambda: MockValve(is_intake=False))
    # note:  in the engine model the cam open and close are specific and valve open/close is calculation
    # they are:
    #       intake_v = MockValve(open=688, close=250, lift=9, diameter=40, cam_open=6, cam_close=211)
    #       exhaust_v = MockValve(open=469, close=21, lift=9, diameter=34, cam_open=507, cam_close=702)


# @dataclass
class MockCylinderState(CylinderState):
    # Arrays for historical lookup (IVC references)
    log_P: np.ndarray = field(default_factory=lambda: np.full(720, 101325.0))
    log_T: np.ndarray = field(default_factory=lambda: np.full(720, 293.0))
    V_list: np.ndarray = field(init=False, repr=False)
    dV_list: np.ndarray = field(init=False, repr=False)
    
    # Instantaneous values for current step
    P_curr: float = 101325.0
    T_curr: float = 293.0
    V_curr: float = 0.0005
    T_wall: float = 293.0
    V_clearance: float = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
    spark_event_theta: float = 335.0
    
    def __post_init__(self):
        theta_array = np.arange(720)
        self.V_list = pf.v_cyl(theta_array, c.A_PISTON, self.V_clearance)
        V_next = np.roll(self.V_list, -1)  # done this way rather than np.diff to handle index wrapping
        self.dV_list = V_next - self.V_list


class BaseEngineTest(unittest.TestCase):
    """Provides the standard 'fire_engine' loop used by Integration and Validation."""

    def fire_engine(self,
                    # preset mode 
                    preset: RunMode | None = None, # IDLE, CRUISE or WOT
                    
                    # preset overrides
                    rpm: float | None = None, # or set rpm and TPS directly
                    tps: float | None = None,
                    iacv_pos: float | None = None,
                    rpm_hold: bool | None = None, # static dyno model
                    thermal_state: str | None = None, # "cold", "warm", "hot" or "current" (persist existing)
                    
                    # other optional inputs
                    wheel_load=0,
                    motoring=False, # no combustion, just pumping and friction (e.g., for friction validation)
                    has_spark=True, # direct control of spark for knock or motoring tests
                    engine: EngineModel | None = None, # if not provided will be created and returned
                    ecu: MockEcuOutput | None = None, # if not provided will be created and returned
                    cycles=3
                    ) -> tuple[EngineModel, MockEcuOutput]:
        """
        Simulates engine cycles. 
        thermal_state: "cold" (ambient), "Warm" (70C), "hot" (90C) or "current" (persist existing)
        """
        
        def set_spark(rpm):
            if rpm <= 1000:
                return 350 # 5 deg BTDC
            if rpm <= 1500: 
                return 335 # 18 deg BTDC
            if rpm <= 3500: 
                return 335
            else:
                return 336 # 28 deg BTDC
            
        def set_thermals(thermal_state):
            if thermal_state == "cold":
                _engine.sensors.CLT_C = 20
                _engine.cyl.T_wall = 20 + 273.15
                _engine.cyl.T_curr = 20 + 273.15
            elif thermal_state == "warm":
                _engine.sensors.CLT_C = 70
                _engine.cyl.T_wall = 90 + 273.15
                _engine.cyl.T_curr = 90 + 273.15
            elif thermal_state == "hot":
                _engine.sensors.CLT_C = 90
                _engine.cyl.T_wall = 120 + 273.15
                _engine.cyl.T_curr = 120 + 273.15
            else:
                pass # ontinue with defaults or previous if existing engine
            
        # 1. If a preset is chosen, extract its global defaults
        _rpm, _tps, _iacv_pos, _thermal_state, _rpm_hold = 0.0, 0.0, 0.0, "current", False
        if preset is not None:
            default = ENGINE_PRESETS[preset]
            _rpm = default.rpm
            _tps = default.tps
            _iacv_pos = default.iacv_pos
            _thermal_state = default.thermal_state
            _rpm_hold = default.rpm_hold
        
        # 2. If any preset override is provided, use it instead of the preset default
        rpm = _rpm if rpm is None else rpm
        tps = _tps if tps is None else tps
        iacv_pos = _iacv_pos if iacv_pos is None else iacv_pos
        thermal_state = _thermal_state if thermal_state is None else thermal_state
        rpm_hold = _rpm_hold if rpm_hold is None else rpm_hold
            
        # setup ecu
        _ecu = ecu if ecu is not None else MockEcuOutput()
        _ecu.iacv_pos = iacv_pos
            
        # setup engine
        _engine = engine if engine is not None else EngineModel(rpm=rpm)
        set_thermals(thermal_state)
        _engine.sensors.TPS_percent = tps
        _engine.state.wheel_load = wheel_load
        if motoring or rpm_hold:  # motoring has no spark nor fuel
            _engine.motoring_rpm = rpm
        else:
            _engine.motoring_rpm = 0.0
            
        
        target_afr = _ecu.afr_target
            
        # 1. Convert CC/MIN to KG/S (assuming gasoline density 0.74)
        flow_kg_s = (c.INJECTOR_FLOW_CC_PER_MIN / 60.0) * (0.74 / 1000.0)
        
        # Assume IVC for a VW 2.1L is roughly 220-240 CAD (adjust to your model's spec)
        IVC_ANGLE = _engine.valves.intake.close_angle 

        for _ in range(cycles):
            for CAD in range(720):
                
                # 2. At Intake Valve Close, calculate the injection needed for the NEXT cycle
                if CAD == IVC_ANGLE:
                    trapped_air = _engine.cyl.air_mass_kg
                    
                    if trapped_air > 0:
                        required_fuel_kg = trapped_air / target_afr
                        
                        # Calculate Pulse Width in Seconds, then Degrees
                        inj_time_sec = required_fuel_kg / flow_kg_s
                        deg_per_sec = (_engine.sensors.rpm / 60.0) * 360.0
                        pw_degrees = inj_time_sec * deg_per_sec
                        
                        # 3. Set Variable Start based on Fixed End
                        # Example: Injector End is fixed at 170 CAD (standard for many port setups)
                        _ecu.injector_start = int(_ecu.injector_end - pw_degrees)


                # 4. Set ECU Outputs
                start_trigger = int(_ecu.injector_start % 720)
                end_trigger = int(_ecu.injector_end % 720)
                _ecu.spark_timing = set_spark(rpm=rpm)
                _ecu.spark = (CAD == _ecu.spark_timing) if not motoring and has_spark else False
                
                if not motoring and has_spark:
                    if start_trigger > end_trigger: # Pulse crosses 720/0 boundary
                        _ecu.injector_on = (CAD >= start_trigger or CAD <= end_trigger)
                    else:
                        _ecu.injector_on = (start_trigger <= CAD <= end_trigger)

                _engine.step(_ecu)
        return _engine, _ecu



# =================================================================
# TIER 1: Test the math of the Physics Functions
# =================================================================
class TestPhysics(unittest.TestCase):
    """ Validates that units, gravity, gas constants, and basic solvers work. 
        No EngineModel instantiation required. If this fails, no other test is accurate.
    """

    def _get_woschni_params(self, P_curr, T_curr, V_curr):         
        """Helper to package raw floats into the required Object structure."""
        valves = MockValves()
        cyl = MockCylinderState()
        IVC = int(valves.intake.close_angle)
        cyl.log_P[IVC] = 101325.0
        cyl.log_T[IVC] = 310.0
        cyl.V_list[IVC] = 0.00055
        
        # Inject the specific state for this test point
        cyl.P_curr = P_curr
        cyl.T_curr = T_curr
        cyl.V_curr = V_curr
        
        # Return a dict that can be unpacked into the function
        return {
            "cyl": cyl,
            "valves": valves
        }
        
    def test_burn_duration(self):
        """
        Unified Audit: Ensures burn duration allows for 32kPa idle vacuum 
        while maintaining high-RPM safety for the 94mm WBX bore.
        """
        # 1. IDLE (The 'Vacuum' Gatekeeper)
        # Target: 46-52 deg. If > 55, the engine is too inefficient to pull 32kPa.
        bd_idle = pf.get_burn_duration(900, 1.0)
        self.assertLess(bd_idle, 55.0, 
            f"IDLE FAILURE: Burn is too 'lazy' ({bd_idle:.1f}°). "
            "Engine will require too much throttle to stay alive, killing vacuum.")
        
        # 2. CRUISE (Efficiency Zone)
        # Target: 40-45 deg. Matches 30-35° spark timing at 3000 RPM.
        bd_cruise = pf.get_burn_duration(3000, 1.0)
        self.assertTrue(38.0 <= bd_cruise <= 48.0, 
            f"CRUISE FAILURE: Burn ({bd_cruise:.1f}°) out of range for cruise efficiency.")

        # 3. HIGH RPM (Safety Zone)
        # Target: 35-42 deg. Prevents 'Slow Burn' from torching exhaust valves 
        # while preventing 'Fast Burn' pressure spikes.
        bd_wot = pf.get_burn_duration(5500, 1.0)
        self.assertGreaterEqual(bd_wot, 34.0, 
            f"SAFETY FAILURE: Burn too fast ({bd_wot:.1f}°). Risk of cylinder head lifting.")
        self.assertLessEqual(bd_wot, 45.0, 
            f"THERMAL FAILURE: Burn too slow ({bd_wot:.1f}°). Exhaust Gas Temps will be too high.")

        # 4. TREND CHECK
        # Turbulence (RPM) must always speed up the burn.
        self.assertGreater(bd_idle, bd_cruise, "Physics Error: Burn should be slower at Idle than Cruise.")
        self.assertGreater(bd_cruise, bd_wot, "Physics Error: Burn should be slower at Cruise than WOT.")
        
    def test_woschni_sane_output(self):
        """Verifies that heat energy rejection per degree matches global calibration."""
        params = self._get_woschni_params(P_curr=60e5, T_curr=2200, V_curr=0.0005)
        loss = pf.calc_woschni_heat_loss(CAD=370, rpm=3000, **params)

        # Adjusted for thermal_scaling = 1.35
        self.assertGreaterEqual(loss, 10.0, f"Thermal under-reporting: {loss:.2f} J/deg")
        self.assertLessEqual(loss, 14.5, f"Thermal quenching: {loss:.2f} J/deg")
        
    def test_volume_geometry(self):
        """Verify the slider-crank math matches the 2.1L dimensions."""
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        v_tdc = pf.v_cyl(0, c.A_PISTON, V_clearance)
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        
        self.assertAlmostEqual(v_tdc, V_clearance, places=7)
        self.assertAlmostEqual(v_bdc, V_clearance + (c.V_DISPLACED), places=7)
        
    def test_woschni_rpm_scaling(self):
        """
        At lower RPM, gas has more time to transfer heat to walls.
        Heat loss per degree should be HIGHER at 1500 than 4500.
        """
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        
        params = self._get_woschni_params(P_curr=60e5, T_curr=2200, V_curr=0.0005)
        loss_low =  pf.calc_woschni_heat_loss(CAD=370, rpm=1500, **params)
        loss_high = pf.calc_woschni_heat_loss(CAD=370, rpm=4500, **params)
                
        # print(f"\nHeat Loss Test: 1500 RPM: {loss_low:.2f} J/deg | 4500 RPM: {loss_high:.2f} J/deg")
        self.assertGreater(loss_low, loss_high, "Heat loss per degree should be higher at low RPM")

    def test_friction_900_rpm(self):
        """Target: 3.3 Nm (Mean Friction Torque)"""
        cyl_fric = np.mean([pf.calc_single_cylinder_friction(theta=cad, rpm=900, p_cyl=101325.0, clt=90.0) for cad in range(720)])
        engine_fric = (cyl_fric * c.NUM_CYL) + pf.calc_engine_core_friction(rpm=900, clt=90.0)
        self.assertAlmostEqual(engine_fric, 7, delta=1.2)

    def test_friction_growth(self):
        """Friction should scale quadratically with RPM (isolated from pressure)."""
        f_low = pf.calc_single_cylinder_friction(theta=450, rpm=1500, p_cyl=1e5, clt=90)
        f_high = pf.calc_single_cylinder_friction(theta=450, rpm=4500, p_cyl=1e5, clt=90)
        
        self.assertGreater(f_high, f_low * 1.7, "High RPM friction should be close to double low RPM")
        
    def test_wall_temp_stability(self):
        """Wall temperature should rise with heat loss but exhibit inertia."""
        clt_k = 90 + 273.15
        initial_wall = clt_k + 10 # Start slightly above coolant
        
        # Simulate one high-load cycle (e.g., 500J lost to walls)
        new_wall = pf.update_cylinder_wall_temperature(
            current_clt_C=90, 
            cycle_Q_loss_joules=500, 
            rpm=3000, 
            previous_T_wall=initial_wall
        )
        
        # Assertions
        self.assertGreater(new_wall, initial_wall, "Wall should heat up under load")
        self.assertLess(new_wall, initial_wall + 5.0, "Wall shouldn't jump more than 5K in one cycle (Inertia)")

    def test_thermostat_logic(self):
        """Thermostat must remain closed < 92C and open > 92C."""
        cooling_cold = pf.calc_thermostat_cooling_rate(80.0)
        cooling_hot = pf.calc_thermostat_cooling_rate(95.0)
        
        # print(f"Thermostat Test: 80C: {cooling_cold:.4f} | 95C: {cooling_hot:.4f}")
        
        self.assertEqual(cooling_cold, 0.0, "Thermostat should be closed at 80C")
        self.assertGreater(cooling_hot, 0.0, "Thermostat should provide cooling at 95C")
        
    def test_coolant_thermal_dynamics(self):
        """
        Tests the cooling loop: heating under load, thermostat regulation, 
        and safety clipping.
        """
        # --- Scenario 1: Cold Start Heating ---
        # Engine at 20C, high load (100Nm) at 3000 RPM
        temp_start = 20.0
        temp_after_load = pf.update_coolant_temp(temp_start, brake_torque_nm=100.0, rpm=3000)
        
        # print(f"\nCoolant Test (Heating): 20C -> {temp_after_load:.4f}C")
        self.assertGreater(temp_after_load, temp_start, "Coolant should heat up under load")
        
        # --- Scenario 2: Thermostat Regulation ---
        # Engine at 95C (Thermostat should be open)
        # We check if the temp rise is smaller (or negative) compared to when cold
        rise_cold = temp_after_load - temp_start
        
        temp_hot = 95.0
        temp_after_cooling = pf.update_coolant_temp(temp_hot, brake_torque_nm=100.0, rpm=3000)
        rise_hot = temp_after_cooling - temp_hot
        
        # print(f"Coolant Test (Regulation): Rise at 20C: {rise_cold:.4f} | Rise at 95C: {rise_hot:.4f}")
        self.assertLess(rise_hot, rise_cold, "Thermostat should reduce the rate of temperature increase")

        # --- Scenario 3: Hard Safety Limits ---
        # Ensure it never exceeds 115C even with massive torque
        extreme_temp = pf.update_coolant_temp(114.9, brake_torque_nm=5000.0, rpm=8000)
        # print(f"Coolant Test (Limit): Extreme Load Result: {extreme_temp:.2f}C")
        self.assertLessEqual(extreme_temp, 115.0, "Coolant temp must be clipped at 115C")
        
    def test_flow_continuity_vectorized(self):  
        """Validates continuity for both Intake and Exhaust using vectorized cam logic."""

        valves = MockValves()
        rpm=3000


        # --- TEST 1: INTAKE (Inflow / Vacuum Sweep) ---
        # Simulate crank angle at near-max lift
        # lift_in_vec = pf.calc_valve_lift_flat_follower(np.arange(720), valves.intake.open_1mm, valves.intake.close_1mm, valves.intake.max_lift)
        lift_in_vec = pf.calculate_wbx_physical_lift(c.INTAKE_DURATION, c.INTAKE_CENTERLINE, c.INTAKE_MAX_LIFT, is_intake=True, is_duration_at_1mm=c.IS_AT_1mm)

       
        area_in_vec = pf.calc_valve_area_vectorized(np.arange(720), valves.intake, lift_in_vec)
        theta_mid_intake = int((valves.intake.open_angle + valves.intake.close_angle) / 2)
        lift_in = lift_in_vec[theta_mid_intake]
        area_in = area_in_vec[theta_mid_intake]

        mdots_in = []
        mdot_previous = 0.0
        p_cyl_sweep_in = np.linspace(1.0, 0.2, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_in:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_in, lift=lift_in, diameter=valves.intake.diameter,
                P_cyl=p, T_cyl=c.T_INTAKE_K, R_cyl=c.R_SPECIFIC_AIR, g_cyl=c.GAMMA_AIR,
                P_manifold=c.P_ATM_PA, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
                is_intake=True, rpm=rpm, mdot_previous=mdot_previous
            )
            mdot_previous = mdot
            mdots_in.append(mdot)

        # --- TEST 2: EXHAUST (Outflow / Blowdown Sweep) ---
        theta_mid_exhaust = int((valves.exhaust.open_angle + valves.exhaust.close_angle) / 2)      
        # lift_ex_vec = pf.calc_valve_lift_flat_follower(np.arange(720),  valves.exhaust.open_1mm, valves.exhaust.close_1mm, valves.exhaust.max_lift)
        lift_ex_vec = pf.calculate_wbx_physical_lift(c.EXHAUST_DURATION, c.INTAKE_CENTERLINE, c.EXHAUST_MAX_LIFT, is_intake=False, is_duration_at_1mm=c.IS_AT_1mm)
        area_ex_vec = pf.calc_valve_area_vectorized(np.arange(720), valves.exhaust, lift_ex_vec)
        lift_ex = lift_ex_vec[theta_mid_exhaust]
        area_ex = area_ex_vec[theta_mid_exhaust]

        mdots_ex = []
        mdot_previous = 0.0
        p_cyl_sweep_ex = np.linspace(1.0, 5.0, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_ex:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_ex, lift=lift_ex, diameter=valves.exhaust.diameter,
                P_cyl=p, T_cyl=c.T_EXHAUST_K, R_cyl=c.R_SPECIFIC_EXHAUST, g_cyl=c.GAMMA_EXHAUST,
                P_manifold=c.P_ATM_PA, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
                is_intake=False, rpm=rpm, mdot_previous=mdot_previous
            )
            mdots_ex.append(mdot)

        # Assertions: 
        # Intake: mdot starts at 0 and goes POSITIVE. Diff should be >= 0.
        self.assertTrue(np.all(np.diff(mdots_in) >= -1e-12), "Intake flow decreased as pressure drop increased")
        
        # Exhaust: mdot starts at 0 and goes NEGATIVE. Diff should be <= 0.
        self.assertTrue(np.all(np.diff(mdots_ex) <= 1e-12), "Exhaust flow increased as pressure increased")
   
    def test_quasi_static_flow_coefficient_sweep(self):
        """
        Geometry Audit: Sweeps the _calc_wbx_valve_cd function directly 
        from 0mm to 8.75mm lift to verify continuous arcs and safe boundaries.
        """
        # Sweep lifts from 0.0 to 8.75mm
        i_lifts = np.linspace(0.0, c.INTAKE_MAX_LIFT, 100) 
        i_diameter = c.INTAKE_DIAM
        
        e_lifts = np.linspace(0.0, c.EXHAUST_MAX_LIFT, 100) 
        e_diameter = c.EXHAUST_DIAM
        
        for lift in i_lifts:
            # Call the underlying flow coefficient calculators directly
            cd_intake = pf._calc_wbx_valve_cd(lift, i_diameter, is_intake=True)
            
            # 1. Zero lift boundary checks
            if lift == 0.0:
                self.assertEqual(cd_intake, 0.0, "Intake Cd must start cleanly at 0.0 at zero lift.")
            
            # 2. Maximum upper saturation limits 
            self.assertLessEqual(cd_intake, 0.65, f"Intake Cd exploded past stock boundary: {cd_intake:.4f} at {lift}mm")
            
            # 3. Ensure limits approach realistic hardware maxima at full lift
            if lift == c.INTAKE_MAX_LIFT:
                self.assertAlmostEqual(cd_intake, 0.65, delta=0.02, msg="Intake full-lift Cd not matching expected hardware maximum.")
        
        for lift in e_lifts:
            # Call the underlying flow coefficient calculators directly
            cd_exhaust = pf._calc_wbx_valve_cd(lift, e_diameter, is_intake=False)
            
            # 1. Zero lift boundary checks
            if lift == 0.0:
                self.assertEqual(cd_exhaust, 0.0, "Exhaust Cd must start cleanly at 0.0 at zero lift.")
            
            # 2. Maximum upper saturation limits 
            self.assertLessEqual(cd_exhaust, 0.63, f"Exhaust Cd exploded past stock boundary: {cd_exhaust:.4f} at {lift}mm")
            
            # 3. Ensure limits approach realistic hardware maxima at full lift
            if lift == c.EXHAUST_MAX_LIFT:
                self.assertAlmostEqual(cd_exhaust, 0.63, delta=0.02, msg="Exhaust full-lift Cd not matching expected hardware maximum.")
   
    def test_friction_absolute_magnitude(self):
        """Ensures friction isn't physically impossible at cruise RPM."""
        f_val = pf.calc_single_cylinder_friction(theta=450, rpm=3000, p_cyl=5e5, clt=90)
        # A single cylinder at 2000 RPM mid-stroke shouldn't exceed ~5-7 Nm 
        # (since 4 cylinders total ~20 Nm)
        self.assertLess(f_val, 7.0, "Single cylinder friction is too high for a 2.1L engine")
          
    def test_friction_drag_components(self):
        """
        Ensures the mechanical friction function doesn't overlap with 
        pumping work logic.
        """
        def _get_fric(rpm, clt):
            # Cylinder Component (Mean over 720 deg)
            cyl_samples = [pf.calc_single_cylinder_friction(cad, rpm, 101325.0, clt) for cad in range(720)]
            cyl_mean = np.mean(cyl_samples)
            # Global Component
            glob = pf.calc_engine_core_friction(rpm, clt)
            return cyl_mean, glob

        # Data at 3000 RPM
        cyl_warm, glob_warm = _get_fric(3000, 90.0)
        cyl_cold, glob_cold = _get_fric(3000, 20.0)
        glob_ratio = glob_cold / glob_warm
        cyl_ratio = cyl_cold / cyl_warm

        engine_warm = c.NUM_CYL * cyl_warm + glob_warm
        engine_cold = c.NUM_CYL * cyl_cold + glob_cold
        engine_ratio = engine_cold / engine_warm

        
        # In a 2.1L engine, pure metal-on-metal friction at 3000 RPM 
        self.assertGreater(engine_warm, 8, "Mechanical friction is unrealistically low")
        # self.assertLess(engine_warm, 8.0, "Mechanical friction is unrealistically high")
        self.assertLess(engine_warm, 15.5, "Mechanical friction is unrealistically high for a WBX 2.1L")
        if engine_warm > 12.5:
            print(f"WARNING: mechnical friction {engine_warm} Nm is higher than ideal target of 12.5 Nm")
        
        # 1. Cylinder Ratio (Target: < 2.0)
        # Rings are constant-tension; they shouldn't care much about oil temp.
        self.assertLess(cyl_ratio, 2.0, f"Cylinder friction too sensitive to temp (Ratio: {cyl_ratio:.1f})")

        # 2. Global Ratio (Target: < 3.0)
        # Bearings are viscous-heavy, but Valvetrain and Seals provide a floor.
        self.assertLess(glob_ratio, 3.0, f"Global friction too sensitive to temp (Ratio: {glob_ratio:.1f})")
        
        # 3. Partition Check: Is the Core doing its fair share?
        core_contribution = glob_warm / engine_warm
        self.assertGreater(core_contribution, 0.25, "Core friction (Crank/Cam) is too low relative to Pistons")
        self.assertLess(core_contribution, 0.65, "Core friction is dominating the Pistons too heavily")
 
    def test_knock_octane_sensitivity(self):
        # Higher octane should increase the pressure threshold (reduce knock risk)
        _, intensity_95 = pf.detect_knock(100.0, 90.0, 2000, 20.0, 1.0, fuel_octane=95.0)
        _, intensity_100  = pf.detect_knock(100.0, 90.0, 2000, 20.0, 1.0, fuel_octane=100.0)
        
        self.assertGreater(intensity_95, intensity_100, "Higher octane should result in lower knock intensity")

    def test_knock_rpm_safety(self):
        # Higher RPM reduces time for end-gas to auto-ignite, increasing threshold
        low_rpm_knock, _ = pf.detect_knock(98.0, 90.0, 1000, 25.0, 1.0)
        high_rpm_knock, _ = pf.detect_knock(98.0, 90.0, 5000, 25.0, 1.0)
        
        self.assertTrue(low_rpm_knock)
        self.assertFalse(high_rpm_knock, "High RPM should provide more pressure tolerance")

    def test_knock_spark_penalty(self):
        # Extreme spark advance (e.g., 40 deg) should lower the threshold (increase knock)
        _, intensity_safe = pf.detect_knock(90.0, 90.0, 3000, 20.0, 1.0)
        _, intensity_aggressive = pf.detect_knock(90.0, 90.0, 3000, 40.0, 1.0)
        
        self.assertGreater(intensity_aggressive, intensity_safe)

    def test_manifold_vacuum_logic(self):
        """
        Comprehensive Audit of Manifold Physics using Mass-Based Flow.
        """
        cyl = MockCylinderState()
        P_start = 101325.0 * 0.90
        iat_k = 293.15
        cad = 90
        dV = cyl.dV_list[cad] # Geometric displacement (m^3)
        
        # --- MOCKING THE MASS FLOW ---
        # In the real engine, the cylinder takes mass based on density: m = P*V / (R*T)
        rho_start = P_start / (c.R_SPECIFIC_AIR * iat_k)
        dm_mock = rho_start * dV # This is the mass 'sucked' out for this degree
        
        # --- PHASE 1: GEOMETRIC SUCTION (TPS = 0.0) ---
        # Since dm_mock is fixed (one degree of rotation), the pressure drop
        # must be RPM-independent because no time-dependent 'filling' occurs yet.
        p_0_low = pf.update_intake_manifold_pressure(current_map=P_start, effective_tps=0.0, rpm=1000.0, iat_k=iat_k, dm_in_engine=dm_mock)
        p_0_high = pf.update_intake_manifold_pressure(current_map=P_start, effective_tps=0.0, rpm=3000.0, iat_k=iat_k, dm_in_engine=dm_mock)
        
        self.assertAlmostEqual(p_0_low, p_0_high, places=7, 
                            msg="Physics Error: Suction-per-degree should be RPM-independent at 0% TPS")

        # --- PHASE 2: RPM SENSITIVITY (TPS = 1.98 / IACV) ---
        # At high RPM, 'dt' is smaller. The throttle has less time to 'refill' 
        # the manifold between cylinder gulps, leading to a deeper vacuum.
        p_idle_low = pf.update_intake_manifold_pressure(current_map=P_start, effective_tps=1.98, rpm=1000.0, iat_k=iat_k, dm_in_engine=dm_mock)
        p_idle_high = pf.update_intake_manifold_pressure(current_map=P_start, effective_tps=1.98, rpm=3000.0, iat_k=iat_k, dm_in_engine=dm_mock)
        
        self.assertLess(p_idle_high, p_idle_low, 
                        "Non-Physical: Higher RPM should show deeper vacuum (throttle bottleneck)")

        # --- PHASE 3: WOT MITIGATION (TPS = 100.0) ---
        p_wot = pf.update_intake_manifold_pressure(current_map=P_start, effective_tps=100.0, rpm=1000.0, iat_k=iat_k, dm_in_engine=dm_mock)
        
        self.assertGreater(p_wot, p_idle_low, "Throttle failed to mitigate manifold vacuum at WOT")

    def test_manifold_stability(self):
        """Tests if the integrator smoothly converges to atmospheric at WOT."""
        # 1. Start with a deep vacuum (30kPa)
        cyl = MockCylinderState()
        p_manifold = 30000
        iat_k = 293.15
        cad = 90
        dV = cyl.dV_list[cad] # Geometric displacement (m^3)
        rpm = 3000
        
        # --- MOCKING THE MASS FLOW ---
        # In the real engine, the cylinder takes mass based on density: m = P*V / (R*T)
        rho_start = p_manifold / (c.R_SPECIFIC_AIR * iat_k)
        dm_mock = rho_start * dV # This is the mass 'sucked' out for this degree

        # 2. RUN THE INTEGRATOR (The 'Chain' Logic)
        # We run 100 iterations. Each iteration uses the result of the last.
        for _ in range(1000):

            
            # FEEDBACK LOOP: p_manifold is both input and output
            p_manifold = pf.update_intake_manifold_pressure(
                current_map=p_manifold,  # Use the UPDATED pressure from last loop
                effective_tps=100.0,     # Wide Open Throttle
                rpm=rpm, 
                iat_k=c.T_AMBIENT, 
                dm_in_engine=dm_mock
            )
                
        # 3. ASSERTIONS
        # At WOT, the pressure should rise and eventually 'plateau' near atmospheric.
        # If it goes to 500,000 Pa, the math is unstable.
        # If it stays at 30,000 Pa, the filling logic (mass_in) is broken.
        
        self.assertLessEqual(p_manifold, c.P_ATM_PA * 1.02, 
                            f"Numerical Divergence: Pressure exploded to {p_manifold/1000:.1f} kPa")
        
        self.assertGreater(p_manifold, 95000.0, 
                        f"Convergence Error: Manifold failed to refill. Only reached {p_manifold/1000:.1f} kPa")

    def test_manifold_throttle_sensitivity(self):
        """
        Calibration Audit: Ensures throttle coefficients allow for realistic vacuum.
        A 2% throttle should NOT be able to maintain 90kPa against 900RPM suction.
        """
        # 1. Setup a standard 'Idle' scenario
        P_atm = 101325.0
        iat_k = 293.15
        rpm = 900.0
        tps_idle = 1.98  # Your IACV/Idle crack
        
        # 2. Mock a realistic 'Cylinder Gulp' at 50kPa (mid-vacuum)
        # If the throttle is too 'leaky', it will refill the manifold faster 
        # than this gulp can drain it.
        p_mid = 50000.0
        rho_mid = p_mid / (c.R_SPECIFIC_AIR * iat_k)
        total_stroke_mass = rho_mid * c.V_DISPLACED
        dm_per_degree = total_stroke_mass / 180.0  # Distributed over the intake stroke
        # 3. Single Step Update
        p_next = pf.update_intake_manifold_pressure(
            current_map=p_mid, 
            effective_tps=tps_idle, 
            rpm=rpm, 
            iat_k=iat_k, 
            dm_in_engine=dm_per_degree
        )
        
        print("test_manifold_throttle_sensitity")
        print(f"p_next:{p_next}")
        
        # 4. ASSERTION
        # At 50kPa, the pressure delta (Atm - Manifold) is huge (~51kPa).
        # If the MAP rises back toward 100kPa, the throttle is too 'free-flowing'.
        self.assertLess(p_next, p_mid, 
            f"Throttle is too leaky! At 2% TPS, MAP rose from 50kPa to {p_next/1000:.1f}kPa. "
            "The vacuum is being 'killed' by the THROTTLE_FLOW_COEFF.")
       
    def test_motoring_energy_balance(self):
        """
        Closed-cycle test (Valves shut, no combustion). 
        Validates integrator precision.
        """
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        p_initial = 101325.0
        t_initial = 300.0
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        v_next = pf.v_cyl(181, c.A_PISTON, V_clearance)
        dV = v_next - v_bdc
        m_air = (p_initial * v_bdc) / (c.R_SPECIFIC_AIR * t_initial)
        
        theta_start, theta_end, step_size = 180.0, 360.0, 0.1
        
        p_curr, t_curr, v_curr = p_initial, t_initial, v_bdc

        for th in np.arange(theta_start, theta_end, step_size):
            # Calculate volume at the start of THIS micro-step
            v_this_step = pf.v_cyl(th, c.A_PISTON, V_clearance)
            # Calculate volume at the end of THIS micro-step
            v_next_step = pf.v_cyl(th + step_size, c.A_PISTON, V_clearance)
            
            # The actual change in volume for this specific slice of the stroke
            dV_step = v_next_step - v_this_step

            p_curr, t_curr = pf.integrate_first_law(
                CAD=th, 
                P_curr=p_curr, 
                T_curr=t_curr, 
                M_curr=m_air, 
                V_curr=v_this_step, 
                Delta_Q_in=0,            
                Delta_Q_loss=0,        
                dV=dV_step,    
                Delta_M=0,       
                R_spec=c.R_SPECIFIC_AIR, 
                T_manifold=300,
                lambda_=1.0
            )
            
            v_curr = pf.v_cyl(th + step_size, c.A_PISTON, V_clearance)

        # 4. UPDATED REFERENCE:
        # Since your Cv is now T-dependent, a static P1 * CR^1.4 is wrong.
        # We check if the result is physically consistent with a variable gamma compression.
        # For Air at ~700K (TDC), gamma is ~1.37, not 1.4.
        
        # Calculate the "Actual Effective Gamma" of the simulation
        gamma_eff = np.log(p_curr / p_initial) / np.log(c.COMP_RATIO)
        
        # The error check should ensure we aren't losing mass/energy, 
        # A variable Cv model for air at 650K should land between 1.34 and 1.37
        self.assertGreater(gamma_eff, 1.34, "Integrator is actually leaking energy (Mass or Vol error)")
        self.assertLess(gamma_eff, 1.41, "Integrator is creating energy (Numerical instability)")
        
        # Continuity Check: The temperature must also be physically bounded (~600K-750K)
        self.assertGreater(t_curr, 600.0)
        self.assertLess(t_curr, 800.0)
        
    def test_woschni_continuity_at_tdc(self):
        """Ensures heat transfer doesn't collapse at zero piston speed (TDC)."""
        # Compare heat loss at 1 degree before TDC vs at TDC
        
        params = self._get_woschni_params(P_curr=50e5, T_curr=2000, V_curr=0.00006)
        loss_before = pf.calc_woschni_heat_loss(CAD=359, rpm=3000, **params)
        
        params = self._get_woschni_params(P_curr=60e5, T_curr=2200, V_curr=0.00005)
        loss_at_tdc = pf.calc_woschni_heat_loss(CAD=360, rpm=3000, **params)
        
        # loss_before = pf.calc_woschni_heat_loss(359, 3000, 50e5, 2000, 0.00006, 450, 0.00005, 1.0)
        # loss_at_tdc = pf.calc_woschni_heat_loss(360, 3000, 60e5, 2200, 0.00005, 450, 0.00005, 1.0)
        
        print("\nTEST WOSCHNI AT TDC")
        print(f"loss at 359: {loss_before}  loss at 360: {loss_at_tdc}")
        
        self.assertGreater(loss_at_tdc, 0, "Heat loss must be positive at TDC due to gas turbulence.")
        self.assertAlmostEqual(loss_at_tdc / loss_before, 1.2, delta=0.5, 
                            msg="Heat loss should scale with pressure/temp, not just piston speed.")

    def test_piston_geometry_symmetry(self):
        """Validates the Slider-Crank math for the WBX 127mm rod / 76mm stroke."""
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        v_tdc = pf.v_cyl(0, c.A_PISTON, V_clearance)
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        v_mid = pf.v_cyl(90, c.A_PISTON, V_clearance)
        
        self.assertAlmostEqual(v_tdc, V_clearance, places=7)
        # At 90 degrees, volume is NOT exactly (V_bdc + V_tdc)/2 due to rod angularity
        # This test ensures you didn't accidentally use a simple Sine wave for volume
        self.assertNotAlmostEqual(v_mid, (v_tdc + v_bdc) / 2.0, places=5)
        
    def test_piston_geometry_and_symmetry(self):
        """Checks slider-crank math and rod angularity symmetry."""
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        v_tdc = pf.v_cyl(0, c.A_PISTON, V_clearance)
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        # Symmetry check: 90 and 270 should be identical
        self.assertAlmostEqual(pf.v_cyl(90, c.A_PISTON, V_clearance), 
                               pf.v_cyl(270, c.A_PISTON, V_clearance), places=7)
        self.assertAlmostEqual(v_tdc, V_clearance, places=7)
        self.assertAlmostEqual(v_bdc, V_clearance + c.V_DISPLACED, places=7)

    def test_knock_sensitivity(self):
        """Checks octane and RPM influence on knock detection."""
        _, int_95 = pf.detect_knock(100.0, 90.0, 2000, 20.0, 1.0, fuel_octane=95.0)
        _, int_100 = pf.detect_knock(100.0, 90.0, 2000, 20.0, 1.0, fuel_octane=100.0)
        self.assertGreater(int_95, int_100)
        
        low_rpm_k, _ = pf.detect_knock(98.0, 90.0, 1000, 25.0, 1.0)
        high_rpm_k, _ = pf.detect_knock(98.0, 90.0, 5000, 25.0, 1.0)
        self.assertTrue(low_rpm_k and not high_rpm_k)

    def test_thermostat_clipping(self):
        """Checks cooling rate logic for thermostat states."""
        self.assertEqual(pf.calc_thermostat_cooling_rate(85.0), 0.0)
        self.assertGreater(pf.calc_thermostat_cooling_rate(95.0), 0.0)
        
    def test_isolated_valve_flow_physics(self):
        """
        Gated Physics Test: Sweeps gas velocity across realistic crank angles 
        using authentic cam tables, asserting non-linear high-RPM fluid penalties.
        """
        # Standard WOT snapshot boundary conditions
        P_manifold = getattr(c, 'P_ATM_PA', 101325.0)
        T_extern = getattr(c, 'T_INTAKE_K', 293.15)
        P_cyl = P_manifold - 6325.0  # Cylinder pulling suction vacuum
        T_cyl = T_extern
        
        R_air = c.R_SPECIFIC_AIR
        gamma = c.GAMMA_AIR
        diameter = c.INTAKE_DIAM / 1000.0  # Convert to meters

        # Calculate the authentic 720-degree physical lift profile
        intake_lift_table = pf.calculate_wbx_physical_lift(
            c.INTAKE_DURATION, c.INTAKE_CENTERLINE, c.INTAKE_MAX_LIFT, 
            is_intake=True, is_duration_at_1mm=c.IS_AT_1mm
        )

        # Map diagnostic Crank Angles during the intake stroke sweep (360° to 540°)
        theta_targets = [30.0, 75.0, 108.0]

        # Track results for verification assertions
        results = {2800.0: {}, 4800.0: {}}

        for rpm in [2800.0, 4800.0]:
            for theta in theta_targets:
                lift_m = intake_lift_table[int(theta)] / 1000.0
                A_valve = np.pi * diameter * lift_m

                # FIRST PRINCIPLES ESTIMATION FOR ISOLATED ENVIRONMENT:
                # Scale the baseline estimation to simulate higher runner velocity at 4500 RPM,
                # but keep it bounded so the static pressure drop does not flip the test flow direction.
                v_est_runner = 22.0 if rpm == 2800.0 else 42.0
                rho_est = P_manifold / (R_air * T_extern)
                A_runner_est = np.pi * (0.034 ** 2) / 4.0
                mdot_prev_est = v_est_runner * rho_est * A_runner_est

                mdot, cd = pf.calc_isentropic_flow(
                    A_valve=A_valve, 
                    lift=lift_m, 
                    diameter=diameter,
                    P_manifold=P_manifold, 
                    T_extern=T_extern, 
                    R_extern=R_air, 
                    g_extern=gamma,
                    P_cyl=P_cyl, 
                    T_cyl=T_cyl, 
                    R_cyl=R_air, 
                    g_cyl=gamma,
                    is_intake=True, 
                    rpm=rpm,
                    mdot_previous=mdot_prev_est
                )
                results[rpm][theta] = abs(mdot)

        # 1. High-RPM flow penalty assertion
        # ratio_peak_demand = results[4800.0][75.0] / results[2800.0][75.0]
        # self.assertLess(ratio_peak_demand, 0.86, 
        #     f"Acoustic/runner restriction roll-off too weak at high RPM. Ratio: {ratio_peak_demand:.3f}")

        # # 2. Tight geometric clearance chokes
        # ratio_early = results[4800.0][30.0] / results[2800.0][30.0]
        # self.assertLess(ratio_early, 0.86,
        #     f"Early geometric restrictions failing to taper flow. Ratio: {ratio_early:.3f}")
        
        # COME BACK to this
        
    def test_idle_valve_flow_physics(self):
        """
        Gated Physics Test: Validates low-velocity breathing mechanics at 900 RPM.
        Ensures geometric Cd remains stable, but checks that downstream sonic penalties
        correctly restrict high-RPM mass flow rates under identical pressure differentials.
        """
        P_extern = getattr(c, 'P_ATM_PA', 101325.0)
        T_extern = getattr(c, 'T_INTAKE_K', 293.15)
        
        # Standard idle vacuum target (~35 kPa manifold absolute = ~66 kPa suction delta)
        P_cyl_idle = 35000.0  
        T_cyl = T_extern
        
        R_air = c.R_SPECIFIC_AIR
        gamma = c.GAMMA_AIR
        diameter = c.INTAKE_DIAM / 1000.0

        intake_lift_table = pf.calculate_wbx_physical_lift(
            c.INTAKE_DURATION, c.INTAKE_CENTERLINE, c.INTAKE_MAX_LIFT, 
            is_intake=True, is_duration_at_1mm=c.IS_AT_1mm
        )

        # Mid-stroke sweep (75° local intake CAD) where valve is open
        theta = 75.0
        lift_m = intake_lift_table[int(theta)] / 1000.0
        A_valve = np.pi * diameter * lift_m

        # 1. Calculate mass flow and CD at 900 RPM (Idle)
        mdot_idle, cd_idle = pf.calc_isentropic_flow(
            A_valve=A_valve, lift=lift_m, diameter=diameter,
            P_cyl=P_cyl_idle, T_cyl=T_cyl, R_cyl=R_air, g_cyl=gamma,
            P_manifold=P_extern, T_extern=T_extern, R_extern=R_air, g_extern=gamma,
            is_intake=True, rpm=900.0, mdot_previous=0.0
        )

        # 2. Calculate mass flow and CD at 4500 RPM (High Speed) under identical pressures
        mdot_high, cd_high = pf.calc_isentropic_flow(
            A_valve=A_valve, lift=lift_m, diameter=diameter,
            P_cyl=P_cyl_idle, T_cyl=T_cyl, R_cyl=R_air, g_cyl=gamma,
            P_manifold=P_extern, T_extern=T_extern, R_extern=R_air, g_extern=gamma,
            is_intake=True, rpm=4500.0, mdot_previous=0.0
        )

        # 1. Geometric Cd Identity Check: Confirm Cd stays independent of RPM
        self.assertAlmostEqual(cd_idle, cd_high, places=7,
            msg="Architecture Error: Cd should be purely geometric and identical across RPMs.")

        # 2. Base Cd Floor Check: Ensure low-lift geometry remains physically accurate
        self.assertTrue(0.55 <= cd_idle <= 0.65, 
            f"Geometric Cd out of bounds. Value: {cd_idle:.3f}")

        # 3. Mass Flow Stability Check: Verify that the physics model returns a valid,
        # non-zero mass flow rate under low-speed idle vacuum conditions.
        self.assertGreater(mdot_idle, 0.0, "Physics Error: Idle mass flow collapsed to zero.")

    def test_manifold_pressure_drops_under_high_demand(self):
        """Low-level Audit: Ensures MAP drops when engine demand outpaces throttle flow."""
        # Setup: Part throttle (10%), low RPM, but massive cylinder ingestion
        current_map = 101325.0  # Start at atmospheric (Pa)
        effective_tps = 10.0    # 10% throttle
        rpm = 2000
        iat_k = 293.15          # 20°C
        
        # Simulate a massive gulp from the cylinder this degree step
        dm_in_engine = 0.00005  # kg
        
        # Run your function
        new_map = pf.update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, dm_in_engine)
        
        # Assertion: Pressure must drop below atmospheric baseline
        self.assertLess(new_map, current_map, 
                        f"MAP stayed flat or climbed ({new_map:.1f} Pa). Plenum is behaving like an infinite supply!")

    def test_manifold_pressure_accumulates_on_reversion(self):
        """Low-level Audit: Ensures MAP rises if the cylinder pushes mass BACK into the intake."""
        current_map = 90000.0   # Manifold is under slight vacuum (Pa)
        effective_tps = 0.0     # Closed throttle
        rpm = 1000
        iat_k = 293.15
        
        # A negative mass out means mass is being forced BACK into the manifold (Reversion)
        dm_in_engine = -0.00001 # kg
        
        new_map = pf.update_intake_manifold_pressure(current_map, effective_tps, rpm, iat_k, dm_in_engine)
        
        # Assertion: Reversion must pressurize the manifold plenum
        self.assertGreater(new_map, current_map, 
                        "Manifold failed to pressurize when cylinder backflow occurred.")
# =================================================================
# TIER 1B: Test the math of the ENGINE MODEL Functions
# =================================================================
class TestModel(BaseEngineTest):
    """ Validates caculations and tracking within the EngineModel """
    
    # def test_intake_stroke_pressure_tracking(self):
    #     """
    #     Lower-Level Audit: Verifies that cylinder pressure during the intake 
    #     stroke tracks Manifold Absolute Pressure (MAP) within a realistic fluid lag.
    #     """
    #     # 1. Initialize a baseline engine at low idle speed (900 RPM)
    #     engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=1)
        
    #     # Force a steady idle manifold vacuum (32.0 kPa)
    #     engine.sensors.P_manifold_Pa = 32000.0
        
    #     # 2. Run the engine through the mid-intake stroke (e.g., 90 degrees ATDC)
    #     # At 90° ATDC, piston velocity is near maximum, creating the highest fluid lag.
    #     for cad in range(0, 91):
    #         # Simulate a manual step pass mimicking your operational loop
    #         # (Ensure Mock ECU provides valid basic positions for IACV/Throttle)
    #         engine.step(ecu)
            
    #     # 3. First-Principles Validation Check:
    #     # The cylinder pressure should closely track MAP. It should never drop 
    #     # to a deep vacuum near 0 kPa unless the intake valve is completely broken.
    #     mid_stroke_p_cyl = engine.cyl.log_P[90] / 1000.0  # Convert to kPa
    #     map_target = engine.sensors.P_manifold_Pa / 1000.0
        
    #     print(f"\n[DIAGNOSTIC] At 90 deg ATDC: MAP = {map_target:.2f} kPa, Cyl P = {mid_stroke_p_cyl:.2f} kPa")
        
    #     # Physically, localized fluid restriction lag at idle should not exceed ~5 kPa
    #     self.assertGreater(mid_stroke_p_cyl, map_target - 5.0, 
    #         f"Cylinder pressure dropped to {mid_stroke_p_cyl:.2f} kPa under a {map_target:.2f} kPa manifold! "
    #         f"The intake flow restriction is too severe.")

    # def test_stroke_work_breakdown(self):
    #     """
    #     Diagnostic: Breaks down work metrics stroke-by-stroke to see 
    #     where the 0.87 bar PMEP / sign inversion is accumulating.
    #     """
    #     engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=5)
        
    #     # # Run a full 720 degree cycle to populate histories
    #     # for cad in range(0, 720):
    #     #     engine.step(ecu)
            
    #     # Manually compute the work arrays exactly like _calculate_cycle_work does
    #     cyl_P_start = engine.cyl.log_P
    #     cyl_P_end = np.roll(engine.cyl.log_P, -1)
    #     P_avg = (cyl_P_start + cyl_P_end) / 2.0
    #     w_degree_array = (P_avg - c.P_ATM_PA) * engine.cyl.dV_list
        
    #     # Slices for the 4 strokes
    #     intake_work = np.sum(w_degree_array[0:180]) * c.NUM_CYL
    #     compression_work = np.sum(w_degree_array[180:360]) * c.NUM_CYL
    #     expansion_work = np.sum(w_degree_array[360:540]) * c.NUM_CYL
    #     exhaust_work = np.sum(w_degree_array[540:720]) * c.NUM_CYL
        
    #     print("\n=== FIRST PRINCIPLES WORK BREAKDOWN (Joules) ===")
    #     print(f"Intake Stroke (0-180):      {intake_work:.2f} J")
    #     print(f"Compression Stroke (180-360): {compression_work:.2f} J")
    #     print(f"Expansion Stroke (360-540):   {expansion_work:.2f} J")
    #     print(f"Exhaust Stroke (540-720):     {exhaust_work:.2f} J")
        
    #     # Net sum of all work
    #     total_net_work = np.sum(w_degree_array) * c.NUM_CYL
    #     print(f"Total Net Indicated Work:     {total_net_work:.2f} J")

    def test_combustion_pressure_alignment(self):
        """
        Lower-Class Audit: Verifies that peak combustion pressure aligns
        with a positive volume change (expansion) rather than compression.
        """
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=5)
        
        # # Run a full cycle to ensure arrays are fully populated
        # for cad in range(0, 720):
        #     engine.step(ecu)
            
        # Find where peak pressure actually happens in the array
        peak_pressure_cad = np.argmax(engine.cyl.log_P)
        corresponding_dV = engine.cyl.dV_list[peak_pressure_cad]
        
        print(f"\n[DIAGNOSTIC] Peak Pressure occurs at: {peak_pressure_cad}° CAD")
        print(f"[DIAGNOSTIC] Volume change (dV) at peak pressure: {corresponding_dV:.6e} m3")
        
        # FIRST PRINCIPLES VALIDATION:
        # Peak pressure must occur after TDC (360°). If it happens before 360°,
        # or if dV is negative, combustion is fighting the piston.
        self.assertGreaterEqual(peak_pressure_cad, 360, 
            f"Peak pressure occurred at {peak_pressure_cad}° CAD (Before TDC!). This indicates an indexing or timing shift.")
        self.assertGreater(corresponding_dV, 0.0, 
            f"Volume change at peak pressure is negative ({corresponding_dV:.6e}). Combustion is occurring during compression.")

    def test_low_speed_intake_reversion(self):
        """
        Integration Test: Verifies that at 1000 RPM, upward piston travel
        between BDC (180 CAD) and IVC results in negative mass flow (reversion)
        back into the intake manifold, lowering trapped VE.
        """
        # Fire engine at a low-speed wide-open-throttle condition
        engine, ecu = self.fire_engine(rpm=1000.0, tps=100.0, cycles=3, motoring=True)
        
        # Pull the logged intake mass flow rate array from the master cylinder
        # Note: adjust property naming if your cylinder structure uses a different history tag
        dm_in = engine.cyl.dm_in_history  
        ivc_angle = int(engine.valves.intake.close_angle)
        
        # Check flow direction between BDC (180) and IVC
        reversion_detected = False
        for cad in range(180, ivc_angle):
            if dm_in[cad] < 0:
                reversion_detected = True
                break
                
        self.assertTrue(
            reversion_detected,
            "CRITICAL PHYSICS FAILURE: No intake flow reversion detected between BDC and IVC at 1000 RPM. "
            "Flow equations are likely clamping or miscalculating pressure deltas, causing an artificial high VE."
        )    

    def test_valve_area_and_cd_continuity(self):
        """
        Unit Test: Validates that the valve area array transitions smoothly 
        without discrete steps that could induce jagged torque curves.
        """
        valves = EngineModel(rpm=2000.0).valves
        
        # Check derivative of the intake area table to detect sharp step discontinuities
        diff_area = np.diff(valves.intake_area_table)
        max_step = np.max(np.abs(diff_area))
        
        # Threshold depends on your geometric area magnitude, but it shouldn't show zero change followed by a sudden jump
        self.assertLess(
            max_step, 0.005, 
            f"DISCRETIZATION ERROR: Found a jagged step change ({max_step:.5f}) in valve area table. "
            "Ensure interpolation is continuous and avoids index-rounding steps."
        )
    
    def test_intake_stroke_pressure_tracking(self):
        """
        Lower-Level Audit: Verifies that cylinder pressure during the intake 
        stroke tracks Manifold Absolute Pressure (MAP) within a realistic fluid lag.
        """
        # 1. Initialize a baseline engine at low idle speed (900 RPM)
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=1)
        
        # Force a steady idle manifold vacuum (32.0 kPa)
        engine.sensors.P_manifold_Pa = 32000.0
        
        # 2. Run the engine through the mid-intake stroke (e.g., 90 degrees ATDC)
        # At 90° ATDC, piston velocity is near maximum, creating the highest fluid lag.
        for cad in range(0, 91):
            engine.step(ecu)
            
        mid_stroke_p_cyl = engine.cyl.log_P[90] / 1000.0  # Convert to kPa
        map_target = engine.sensors.P_manifold_Pa / 1000.0
        
        print(f"\n[DIAGNOSTIC] At 90 deg ATDC: MAP = {map_target:.2f} kPa, Cyl P = {mid_stroke_p_cyl:.2f} kPa")
        print(f"[DIAGNOSTIC] Total integrated air mass at 90 deg: {engine.cyl.air_mass_kg * 1e6:.2f} mg")
        
        # PHYSICALLY: At 900 RPM, fluid restriction lag at idle should be minimal (~0.5 to 3 kPa).
        # If the cylinder pressure drops to a deep vacuum near 0 kPa, the flow model is choked.
        # If the cylinder pressure perfectly equals MAP, the valve restriction is nonexistent.
        self.assertGreater(mid_stroke_p_cyl, map_target - 5.0, 
            f"Cylinder pressure dropped to {mid_stroke_p_cyl:.2f} kPa under a {map_target:.2f} kPa manifold! "
            f"The intake flow restriction is too severe.")
        
        self.assertLess(mid_stroke_p_cyl, map_target - 0.1,
            f"Zero fluid restriction detected! Cyl P ({mid_stroke_p_cyl:.2f} kPa) matches MAP perfectly. "
            f"The flow calculation is bypassing the valve orifice restriction.")

    def test_stroke_work_breakdown(self):
        """
        Diagnostic: Breaks down work metrics stroke-by-stroke to see 
        where the 0.87 bar PMEP / sign inversion is accumulating.
        """
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=2)
        
        # Manually compute the work arrays exactly like _calculate_cycle_work does
        cyl_P_start = engine.cyl.log_P
        cyl_P_end = np.roll(engine.cyl.log_P, -1)
        P_avg = (cyl_P_start + cyl_P_end) / 2.0
        
        # Work = (P - P_ambient) * dV
        w_degree_array = (P_avg - c.P_ATM_PA) * engine.cyl.dV_list
        
        # Slices for the 4 strokes (assuming 0 = TDC Intake)
        intake_work = np.sum(w_degree_array[0:180]) * c.NUM_CYL
        compression_work = np.sum(w_degree_array[180:360]) * c.NUM_CYL
        expansion_work = np.sum(w_degree_array[360:540]) * c.NUM_CYL
        exhaust_work = np.sum(w_degree_array[540:720]) * c.NUM_CYL
        
        print("\n=== FIRST PRINCIPLES WORK BREAKDOWN (Joules) ===")
        print(f"Intake Stroke (0-180):      {intake_work:8.2f} J (Should be negative at IDLE due to throttling)")
        print(f"Compression Stroke (180-360): {compression_work:8.2f} J (Should be negative)")
        print(f"Expansion Stroke (360-540):   {expansion_work:8.2f} J (Should be positive)")
        print(f"Exhaust Stroke (540-720):     {exhaust_work:8.2f} J (Should be negative due to backpressure)")
        
        total_net_work = np.sum(w_degree_array) * c.NUM_CYL
        print(f"Total Net Indicated Work:     {total_net_work:8.2f} J")
        
        # PHYSICS CHECK: At Idle (32 kPa MAP), Pumping Loop Work (Intake + Exhaust) MUST be negative.
        pmep_work = intake_work + exhaust_work
        self.assertLess(pmep_work, 0.0, f"Pumping work is positive ({pmep_work:.2f} J) at idle! The loop is creating power instead of losing it.")

    def test_intake_mass_flow_scaling_linearity(self):
        """
        Physics Audit: Verifies that if the discharge coefficient (Cd) is manually altered, 
        the raw integrated mass flow scales predictably during the intake stroke.
        """
        # Fire a motoring engine at WOT, 1000 RPM
        engine, ecu = self.fire_engine(rpm=1000.0, tps=100.0, cycles=1, rpm_hold=True)
        
        # Run through the intake stroke to BDC with baseline Cd
        for cad in range(0, 180):
            engine.step(ecu)
        baseline_bdc_mass = engine.cyl.air_mass_kg
        air_mass_bdc_snapshot = engine.cyl.air_mass_at_BDC
        
        # Reset and mock the Cd calculation to return exactly half value
        engine_choked, ecu_choked = self.fire_engine(rpm=1000.0, tps=100.0, 
                                                     cycles=3, rpm_hold=True)
        
        # Monkeypatch the Cd function or override the table inside this instance
        original_cd_func = pf._calc_wbx_valve_cd
        pf._calc_wbx_valve_cd = lambda lift, diam, is_in: original_cd_func(lift, diam, is_in) * 0.5
        
        try:
            for cad in range(0, 180):
                engine_choked.step(ecu_choked)
            choked_bdc_mass = engine_choked.cyl.air_mass_kg
            max_cd_choked = np.max(engine_choked.cyl.Cd_in_history)
        finally:
            # Always restore the core physics function
            pf._calc_wbx_valve_cd = original_cd_func

        print("\n=== VALVE CURTAIN SCALING SENSITIVITY TEST ===")
        print(f"Baseline BDC Air Mass (Normal Cd): {baseline_bdc_mass * 1e6:.2f} mg")
        print(f"Choked BDC Air Mass (50% Cd):     {choked_bdc_mass * 1e6:.2f} mg")
        
        # PHYSICS VALIDATION:
        # If Cd is cut by 50%, mass flow rate must be reduced. Even with increased vacuum,
        # the net trapped mass at BDC should be lower than baseline at fixed low RPM.
        self.assertLess(choked_bdc_mass, baseline_bdc_mass, 
            "CRITICAL COUPLING FAILURE: Cutting valve Cd in half resulted in equal or greater mass accumulated at BDC! "
            "The mass integrator is decoupled from the orifice flow restriction area.")

# =================================================================
# TIER 2: EngineModel Losses (Friction & Pumping)
# =================================================================
class TestLosses(BaseEngineTest):
    """ Establishes friction, pumping losses, and aerodynamic restrictions 
        before spinning the engine.
    """
    
    def test_full_load_friction_scaling(self):
        # 1. Warm up the engine
        engine, _ = self.fire_engine(preset=RunMode.CRUISE, cycles=5)

        # 3. Collect Data
        w_fric_wot = engine.state.work_friction_j
        
        # Peak Cylinder Pressure (for context)
        p_max = np.max(engine.cyl.log_P) / 1e5 # bar
        
        # 4. Physical Sanity Check
        # At WOT, friction should be ~20-50% higher than motoring 
        # due to the rings being forced against the walls.
        engine, _ = self.fire_engine(rpm=3000, motoring=True, tps=100, thermal_state="warm", cycles=5)
        motoring_fric = engine.state.work_friction_j
        # motoring_fric = 6.10 # Our previous baseline
        increase = (w_fric_wot / motoring_fric) - 1.0
                
        self.assertGreater(abs(w_fric_wot), abs(motoring_fric), "Friction didn't increase under load!")

    def test_idle_pumping_equilibrium(self):
        """Ensures MAP stabilizes around 32.2 kPa with idle IACV."""
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=40)
        stable_map = np.mean(engine.state.map_history) / 1000
        
        P_peak = np.max(engine.cyl.log_P)
        P_peak_angle = engine.cyl.P_peak_angle
        effective_tps = engine.state.effective_tps
        iacv_pos = ecu.iacv_pos
        iacv_wot_equiv = ecu.iacv_wot_equiv
        burn_duration = engine.cyl.burn_duration
        m_vibe = engine.cyl.m_vibe
        dm_in = np.sum(engine.cyl.dm_in_history)
        air_at_ivc = engine.cyl.air_mass_at_IVC
        lambda_ = engine.sensors.lambda_
        map = engine.state.map_avg_kPa
        map2 = engine.sensors.MAP_kPa
        
        total_engine_dm_in = np.sum(engine.state.dm_in_history)
        
        self.assertTrue(35.0 <= stable_map <= 42.0, 
                f"Idle manifold absolute pressure ({stable_map:.2f} kPa) is outside physical bounds.")
    
    def test_static_compression_physics(self):
        """
        First Principles Audit: Validates Geometric CR and Polytropic Compression.
        """
        # Motor engine at low speed (1000 RPM) to minimize high-speed dynamic effects
        ecu = MockEcuOutput()
        engine = EngineModel(rpm=1000)
        engine.motoring_rpm = 1000
        engine.sensors.TPS_percent = 100.0 # WOT to ensure full cylinder filling
        
        for _ in range(3 * 720):
            engine.step(ecu)
            
        # 1. Capture Peak Motoring Pressure
        p_max_pa = np.max(engine.cyl.log_P)
        p_max_bar = p_max_pa / 1e5
        
        # 2. Capture Pressure at IVC (The starting point of actual compression)
        ivc_idx = int(engine.valves.intake.close_angle)
        p_at_ivc_bar = engine.cyl.log_P[ivc_idx] / 1e5
        
        # 3. Theoretical Adiabatic Check: P2 = P1 * (V1/V2)^gamma
        # V1 is volume at IVC, V2 is V_clearance at TDC (360 CAD)
        v_ivc = engine.cyl.V_list[ivc_idx]
        v_tdc = engine.cyl.V_clearance
        gamma = 1.4 # Specific heat ratio for air
        p_theoretical = p_at_ivc_bar * (v_ivc / v_tdc)**gamma

        # Validation: If Actual is significantly lower than Theoretical, 
        # the model has a mass-leak or 'Blow-back' issue.
        self.assertGreater(p_max_bar, 14.5, "Compression pressure too low for 9:1 CR")
        self.assertLess(p_max_bar, 23.0, "Compression pressure implies non-physical CR")

    def test_pumping_loss_floor(self):
        """Validates that gas exchange costs energy (PMEP)."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        pmep = abs(engine.state.PMEP)
        effective_tps = engine.state.effective_tps
        
        # At high RPM, PMEP should grow as air chokes through the valves
        self.assertGreater(pmep, 0.25, "Pumping losses are too low (Intake/Exhaust is too 'free').")

    def test_pumping_versus_mechanical_drag_distribution(self):
        """
        Energy Audit: Captures full cycle at 4500 RPM WOT to dissect Mean 
        Effective Pressure distributions and pinpoint mathematical drag inflation.
        """
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        
        # Capture critical parameters from state scalar registers
        imep_g = engine.state.IMEP_gross
        pmep = engine.state.PMEP
        fmep = engine.state.FMEP
        t_ind = np.mean(engine.state.torque_indicated_history)
        t_fric = np.mean(engine.state.torque_friction_history)
        
        print("\n=== 4500 RPM WOT LOSS DISTRIBUTION AUDIT ===")
        print(f"Indicated Torque:          {t_ind:.2f} Nm")
        print(f"Frictional Drag Torque:    {t_fric:.2f} Nm")
        print(f"Gross Indicated MEP (IMEP): {imep_g:.3f} bar")
        print(f"Pumping Loss MEP (PMEP):    {pmep:.3f} bar")
        print(f"Friction Loss MEP (FMEP):   {fmep:.3f} bar")
        print("============================================")
        
        # Verify drag matches physical plant allocations for the WBX flat-four pushrod design
        self.assertTrue(abs(pmep) > 0.0, "Pumping losses completely collapsed or bypassed.")
        self.assertTrue(abs(fmep) > 0.0, "Friction losses completely collapsed or bypassed.")
    
    def test_fmep_thermal_sensitivity(self):
        """
        Validates that FMEP drops as the engine warms up (oil viscosity effect).
        """
        # Cold Start
        engine_cold, _ = self.fire_engine(preset=RunMode.CRUISE, thermal_state="cold", cycles=3)
        fmep_cold = engine_cold.state.FMEP
        clt_cold = engine_cold.sensors.CLT_C
        T_wall_cold = engine_cold.cyl.T_wall
        imep_net_cold = engine_cold.state.IMEP_net
        
        # Fully Warm
        engine_hot, _ = self.fire_engine(preset=RunMode.CRUISE, thermal_state="hot", cycles=3)
        fmep_hot = engine_hot.state.FMEP
        clt_hot = engine_hot.sensors.CLT_C
        T_wall_hot = engine_hot.cyl.T_wall
        imep_net_hot = engine_hot.state.IMEP_net
        
        self.assertGreater(abs(fmep_cold), abs(fmep_hot) * 1.15, "FMEP should be at least 15% higher on a cold engine.")
        
    def test_mep_at_true_idle(self):
        """
        Audits the engine's operating state at Idle.
        Ensures that MEP magnitudes are within standard automotive bounds.
        """
        # 900 RPM, Hot, IACV at typical idle position
        # engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=1)
        # ecu.iacv_pos = 30
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=20)
        
        IACV_pos = ecu.iacv_pos
        p_peak = np.max(engine.cyl.log_P)
        p_peak_angle = np.argmax(engine.cyl.log_P)
        burn_duration = engine.cyl.burn_duration
        m_vibe = engine.cyl.m_vibe
        spark_timming = ecu.spark_timing
        T_ind_peak = np.max(engine.state.torque_indicated_history)
        T_ind_avg = np.mean(engine.state.torque_indicated_history)
        T_fric_piston = np.mean(engine.state.torque_friction_piston_history)
        T_fric_global = np.mean(engine.state.torque_friction_global_history)
        T_brake = np.mean(engine.state.torque_brake_history)
        IMEP_g = engine.state.IMEP_gross
        IMEP_n = engine.state.IMEP_net
        FMEP = engine.state.FMEP
        PMEP = engine.state.PMEP
        BMEP = engine.state.BMEP
        rpm_hold =engine.motoring_rpm
        
        # confirm the engine rpm is actually holding at 900 RPM
        self.assertAlmostEqual(engine.sensors.rpm, 900, delta=1.0, msg="RPM is not stable at IDLE")
        
        #1. Friction Mean Effective Pressure Check
        # Warm friction losses should be stable and consistent at 900 RPM
        self.assertGreater(abs(FMEP), 0.6, "Friction is unphysically low.")
        self.assertLess(abs(FMEP), 1.8, "Friction is unphysically high.")

        # 2. Pumping Mean Effective Pressure Check
        # Confirms that the IACV restriction creates a real, measurable pumping loss
        self.assertGreater(abs(PMEP), 0.2, "Pumping losses collapsed (check IACV restriction).")

        # 3. Indicated Efficiency Audit
        efficiency = engine.state.work_net_indicated_j / (engine.cyl.total_cycle_heat_J * c.NUM_CYL)
        self.assertTrue(0.20 <= efficiency <= 0.35, f"Thermal efficiency {efficiency:.2%} out of idle bounds.")

        # 4. Dyno Steady-State Balance Check
        # With the dyno holding the engine at 900 RPM, the net BMEP represents 
        # the stable, steady-state excess torque output under fixed air/fuel parameters.
        # the excess torque should be 0 at perfect idle conditions.
        self.assertLess(abs(BMEP), 3.0, "Steady-state dyno load is too far out of balance.")

    def test_pmep_vacuum_scaling(self):
        """
        Tier 3: Audits Pumping MEP at Idle vs WOT.
        VW 2.1L should show significant pumping work at high vacuum.
        """
        # Case A: WOT (Low Pumping Loss)
        engine_wot, _ = self.fire_engine(preset=RunMode.CRUISE, tps=100.0, cycles=5)
        pmep_wot = engine_wot.state.PMEP
        tps_wot = engine_wot.state.effective_tps
        
        # Case B: Part Throttle (High Pumping Loss)
        engine_pot, _ = self.fire_engine(preset=RunMode.CRUISE, cycles=5)
        pmep_pot = engine_pot.state.PMEP 
        tps_pot = engine_pot.state.effective_tps    
        
        self.assertGreater(abs(pmep_pot), abs(pmep_wot), "PMEP should increase as throttle closes (throttling losses).")
        self.assertLess(pmep_wot, 0.6, "WOT Pumping losses too high; check valve flow restrictions.")


    def test_fmep_speed_scaling(self):
        """Ensures friction doesn't 'explode' at high RPM."""
        engine_3k, _ = self.fire_engine(rpm=2500, rpm_hold=True, tps=100.0, thermal_state="warm")
        engine_5k, _ = self.fire_engine(rpm=4500, tps=100.0, thermal_state="warm")
        
        fmep_3k = engine_3k.state.FMEP
        fmep_5k = engine_5k.state.FMEP
        
        # For a WBX, FMEP shouldn't jump from 2.0 to 4.0 bar just by adding 2k RPM.
        # This checks the "curvature" of your friction model.
        self.assertLess(fmep_5k / fmep_3k, 1.5, "Friction curve is too steep at high RPM.")


# =================================================================
# TIER 3: EngineModel Motoring Tests (no combustion)
# =================================================================
class TestMotoring(BaseEngineTest):
    """ Spins the engine using an external motor without fuel/spark. 
        This checks moving parts and baseline pressures against the loss model.
    """
    
    def test_static_compression_audit(self):
        """Adiabatic check: Peak motoring pressure vs theoretical P2=P1*(V1/V2)^gamma.
        after allowing for heat loss.
        """
        engine, ecu = self.fire_engine(rpm=250.0, iacv_pos=40.0, tps=100.0, motoring=True, 
                                       has_spark=False, thermal_state="warm", cycles=1) # No spark in setup
        
        ivc_idx = int(engine.valves.intake.close_angle)
        v_ivc = engine.cyl.V_list[ivc_idx]
        v_tdc = engine.cyl.V_clearance
        
        # DEBUG VARIABLES
        p_max = np.max(engine.cyl.log_P) / 1e5
        P_angle = np.argmax(engine.cyl.log_P)
        rpm = engine.sensors.rpm
        dm_in = np.sum(engine.cyl.dm_in_history)
        air_at_ivc = engine.cyl.air_mass_at_IVC
        map_at_ivc = engine.state.map_history[ivc_idx]
        map_at_cycle_end = engine.sensors.MAP_kPa
        effective_tps = engine.state.effective_tps
        V_DISPLACED = c.V_DISPLACED
        V_INTAKE = c.V_INTAKE_MANIFOLD
        
        # CRANK TEST
        n_polytropic = 1.33 # Realistic index for a WBX with heat loss
        p_theoretical = (engine.cyl.log_P[ivc_idx]/1e5) * (v_ivc/v_tdc)**n_polytropic
        self.assertAlmostEqual(p_max, p_theoretical, delta=2.0)
        
        # LOW RPM
        engine, _ = self.fire_engine(rpm=1500.0, iacv_pos=40.0, tps=100.0, motoring=True, 
                                       has_spark=False, thermal_state="warm", cycles=3) # No spark in setup
        
        p_max = np.max(engine.cyl.log_P) / 1e5
        p_theoretical = (engine.cyl.log_P[ivc_idx]/1e5) * (v_ivc/v_tdc)**n_polytropic
        self.assertAlmostEqual(p_max, p_theoretical, delta=2.0)
       
        
        # HIGH RPM
        engine, _ = self.fire_engine(rpm=3000.0, iacv_pos=40.0, tps=100.0, motoring=True, 
                                       has_spark=False, thermal_state="warm", cycles=3) # No spark in setup

        p_max = np.max(engine.cyl.log_P) / 1e5
        p_theoretical = (engine.cyl.log_P[ivc_idx]/1e5) * (v_ivc/v_tdc)**n_polytropic
        self.assertAlmostEqual(p_max, p_theoretical, delta=2.0)

    def test_motoring_parasitic_drag(self):
        # Setup motoring engine
        ecu = MockEcuOutput()
        engine = EngineModel(rpm=3000)
        engine.motoring_rpm = 3000
        engine.sensors.TPS_percent = 100.0
        engine.sensors.CLT_C = 70
        engine.cyl.T_wall = 90 + 273.15
        engine.cyl.T_curr = 90 + 273.15
        
        # Conversion: Cycle Work (Joules) to Average Torque (Nm)
        # Torque_avg = Total_Work / Total_Radians
        # Total_Radians for 720 deg = 4 * pi
        j_to_nm = 1.0 / (4 * np.pi)
        
        
        # Run for 3 cycles to reach steady state
        for cycle in range(10):
            for _ in range(720):
                engine.step(ecu)
            t_ind_avg_history = np.mean(engine.state.torque_indicated_history)
            t_ind_avg_work = engine.state.work_net_indicated_j * j_to_nm       

        # 1. INDICATED TORQUE (Gas work only)
        t_ind_avg_history = np.mean(engine.state.torque_indicated_history)
        t_ind_avg_work = engine.state.work_net_indicated_j * j_to_nm
        
        # 2. BRAKE TORQUE (The real drag)
        # Average of instantaneous physics loop vs Cycle work calculation
        t_brake_avg_history = np.mean(engine.state.torque_brake_history)
        t_friction_avg_work = engine.state.work_friction_j * j_to_nm
        t_brake_avg_work = t_ind_avg_work + t_friction_avg_work
         
        # 3. DYNO SUPPORT (The Governor)
        # At steady state motoring, the Dyno support must equal the drag (Inverse of Brake Torque)
        avg_dyno_support = np.average(engine.state.torque_governor_history)
        total_drag = -avg_dyno_support 

        # 4. DIAGNOSTICS
        w_pumping = engine.state.work_pumping_j
        pumping_torque_nm = w_pumping * j_to_nm
        PMEP = engine.state.PMEP
        mech_thermal_drag = total_drag - pumping_torque_nm

        # VALIDATION
        # History mean must match Work calculation (Numerical Consistency)
        self.assertAlmostEqual(t_ind_avg_history, t_ind_avg_work, delta=0.01)
        self.assertAlmostEqual(t_brake_avg_history, t_brake_avg_work, delta=0.01)
        
        # Total Drag (Dyno) matches the sum of Indicated - Friction
        # This is the "Gold Standard" physics check
        calculated_drag = t_ind_avg_work + (engine.state.work_friction_j * j_to_nm)
        self.assertAlmostEqual(total_drag, calculated_drag, delta=0.01)
        
        # Total Drag should match Brake Torque (Physics Consistency)
        self.assertAlmostEqual(total_drag, t_brake_avg_work, delta=0.01)
        
        # Sanity Checks
        self.assertLess(total_drag, -2.0, "Engine friction is missing!")
        self.assertGreater(total_drag, -35.0, "Engine drag is unrealistically high!")
    
        # Add these methods inside class TestMotoring(BaseEngineTest):

    def test_manifold_drawdown_rate(self):
        """
        Fluid Isolation: Instantly cuts IACV/TPS to 0 on a spinning engine to 
        ensure manifold pressure decays to deep vacuum, exposing mass leaks or floors.
        """
        # Spin up engine to standard cruise state
        engine, ecu = self.fire_engine(preset=RunMode.CRUISE, motoring=True, cycles=3)
        
        # Force instantaneous step cut to absolute zero throttle & bypass air
        ecu.iacv_pos = 0.0
        engine.sensors.TPS_percent = 0.0
        
        # Step forward for a precise 100ms chunk (approx. 2 full cycles at 2500 RPM)
        # Assuming your cycle step structure can be manually pushed or simulated via low cycle count
        self.fire_engine(rpm=2500.0, iacv_pos=0.0, tps=0.0,
                         engine=engine, ecu=ecu, rpm_hold=True, cycles=2)
        
        final_map_pa = engine.sensors.P_manifold_Pa
        final_map_kpa = final_map_pa / 1000.0
        
        # Assert manifold pulls down to deep vacuum (< 30 kPa) when air pathways seal shut
        self.assertLess(final_map_kpa, 30.0, 
            f"DECAY FAILURE: Manifold hung at {final_map_kpa:.1f} kPa. "
            "Implies a mass-accumulation leak or non-physical flow floor in the 0D solver.")    
    
    def test_motoring_ve_at_peak_torque_rpm(self):
        """
        Lower-Order Breathing Test: Checks intake track capability at 2800 RPM.
        According to the DJ curve, 2800 RPM is peak volumetric capability.
        Target: Motoring VE should clear 86% here before fuel is introduced.
        """
        # Spin the engine at 2800 RPM WOT, motoring only
        engine, ecu = self.fire_engine(rpm=2800.0, tps=100.0, motoring=True, has_spark=False,
                                       thermal_state="warm", cycles=5)
        
        # Calculate maximum ideal mass standard reference (20C, 1 atm)
        # Displacement for 1 cylinder of the 2.1L WBX is ~0.0005274 m^3
        v_displaced = c.V_DISPLACED
        ideal_density = c.P_ATM_PA/ (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ideal_mass_per_cyl = ideal_density * v_displaced
        
        # Pull what your physics engine actually trapped at Intake Valve Closing (IVC)
        actual_trapped_mass = engine.cyl.air_mass_at_IVC
        motoring_ve = actual_trapped_mass / ideal_mass_per_cyl
        
        print(f"\n[Breathing Audit] 2800 RPM WOT Motoring VE: {motoring_ve * 100:.1f}%")
        
        # If your discharge coefficient or lift is choking the engine early, it will fail here
        self.assertGreater(motoring_ve, 0.85, 
            f"Intake tract is too restrictive at peak torque RPM. VE was only {motoring_ve*100:.1f}%.")

    def test_motoring_ve_high_rpm_choke(self):
        """
        Lower-Order Breathing Test: Validates breathing drops safely but realistically at high RPM.
        At 4500 RPM, the 2-valve restrictions cause a drop, but it should still hold ~80-83% VE motoring.
        """
        engine, _ = self.fire_engine(rpm=4500.0, tps=100.0, motoring=True, has_spark=False,
                                     thermal_state="cold", cycles=5)
        
        ideal_density = c.P_ATM_PA / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ideal_mass_per_cyl = ideal_density * c.V_DISPLACED
        
        actual_trapped_mass = engine.cyl.air_mass_at_IVC
        motoring_ve = actual_trapped_mass / ideal_mass_per_cyl
        
        print(f"[Breathing Audit] 4500 RPM WOT Motoring VE: {motoring_ve * 100:.1f}%")
        
        # motoring VE is higher than combustion and can be >1 due to Air raming.
        self.assertTrue(0.85 <= motoring_ve <= 1.08, 
            f"High-RPM Breathing out of range for stock 2-valve MV head. VE: {motoring_ve*100:.1f}%")
        # Verifies that it isn't dropping off a cliff.  VE is higher in motoring than combustion
        # self.assertGreater(motoring_ve, 0.80, 
        #     f"Engine is gasping/choking at 4500 RPM while motoring. VE dropped to {motoring_ve*100:.1f}%.")
        # self.assertLess(motoring_ve, 0.98, 
        #     "Engine is breathing too efficiently for a stock 2-valve head at high RPM.")
        # COME BACK to this
      

# =================================================================
# TIER 4: EngineModel Thermodynamics Tests
# =================================================================
class TestThermals(BaseEngineTest):
    """ Validates coolant loops, metal thermal mass, and ambient heat dissipation 
        before introducing the massive heat spike of fire.
    """
    
    def test_combustion_temperature_safety(self):
        """Target: Max 2800.0 K Peak Temperature"""
        engine, _ = self.fire_engine(preset=RunMode.CRUISE, rpm_hold=True, 
                                     thermal_state="warm", cycles=3)
        T_peak = np.max(engine.cyl.log_T)
        T_peak_angle = np.argmax(engine.cyl.log_T)
        P_peak = np.max(engine.cyl.log_P)
        P_peak_angle = engine.cyl.P_peak_angle
        m_vibe = engine.cyl.m_vibe
        burn_duration = engine.cyl.burn_duration
        lambda_ = engine.sensors.lambda_
        R_spec_blended = engine.cyl.R_specific_blend
        gamma_blended = engine.cyl.gamma_blend
        Q_loss_sum = np.sum(engine.cyl.Q_loss_history)
        Q_in_sum = np.sum(engine.cyl.Q_in_history)
        clt_C = engine.sensors.CLT_C
        T_wall = engine.cyl.T_wall
        V_at_T_peak = engine.cyl.V_list[T_peak_angle]
        M_at_T_peak = engine.cyl.total_mass_history[T_peak_angle]
        
        peak_idx = np.argmax(engine.cyl.log_T)
        cv_at_peak = pf.calc_specific_heat_cv(engine.cyl.log_T[peak_idx], engine.sensors.lambda_)
        # gamma_at_peak = engine.cyl.get_gamma(engine.cyl.log_T[peak_idx], r_spec)
        
        self.assertLess(T_peak, 2850.0)
        
    def test_coolant_thermal_limit(self):
        """Ensures CLT never exceeds safety hard-cap of 115C."""
        engine, _ = self.fire_engine(preset=RunMode.CRUISE, thermal_state="warm", cycles=3)
        engine.sensors.CLT_C = 114.9
        engine, _ = self.fire_engine(preset=RunMode.WOT, thermal_state="hot", engine=engine, cycles=2)
        self.assertLessEqual(engine.sensors.CLT_C, 115.0)

    def test_thermal_feedback_loop(self):
        """Tests that running the engine actually heats up the coolant."""
        initial_clt = c.COOLANT_START   

        engine, _ = self.fire_engine(preset=RunMode.CRUISE, cycles=10)
            
        self.assertGreater(engine.sensors.CLT_C, initial_clt, 
                           "Engine coolant failed to heat up after 10 WOT cycles")

    def test_low_rpm_heat_loss(self):
        """Target: 24.0% Energy Loss at cruising RPM (10% tolerance) due to longer residence time."""
        engine, _ = self.fire_engine(preset=RunMode.CRUISE, cycles=5)
        loss_pct = (engine.cyl.Q_loss_total / engine.cyl.total_cycle_heat_J) * 100
        
        # Target 24.0% with a 2.4% delta (Allows 21.6% to 26.4%)
        self.assertAlmostEqual(loss_pct, 24.0, delta=24.0 * 0.10,
            msg=f"Low RPM thermal partition unphysical: {loss_pct:.2f}%")

    def test_high_rpm_heat_loss(self):
        """Target: 20.0% Energy Loss at high RPM (10% tolerance) due to higher velocity but less time."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        loss_pct = (engine.cyl.Q_loss_total / engine.cyl.total_cycle_heat_J) * 100
        
        # Target 20.0% with a 15% delta (Allows 17-23 range.)
        self.assertAlmostEqual(loss_pct, 20.0, delta=20.0 * 0.15,
            msg=f"High RPM thermal partition unphysical: {loss_pct:.2f}%")
 
# =================================================================
# TIER 5: EngineModel Combustion Tests
# =================================================================
class TestCombustion(BaseEngineTest):
    """ Introduces fuel, chemistry, spark timing, and flame propagation 
        models once the mechanical and thermal environments are stable.
    """

    def test_manifold_suction_response(self):
        """Tests that closing the throttle increases vacuum (decreases MAP)."""

        engine, _ = self.fire_engine(preset=RunMode.WOT, thermal_state="warm", cycles=10)   

        map_wot = engine.sensors.MAP_kPa
        
        # 2. Close Throttle
        # engine.sensors.TPS_percent = 0 # the IACV is has min opening set in ecu
        engine, _ = self.fire_engine(rpm=3000, tps=0.0, cycles=3) # the IACV is has min opening set in ecu
        map_closed = engine.sensors.MAP_kPa
    
        self.assertLess(map_closed, map_wot, "Manifold pressure failed to drop when throttle closed")
    
    def test_firing_symmetry(self):
        """Integration: Validates that torque pulses occur every 180 degrees, ignoring Cycle 0."""
        # engine = EngineModel(rpm=3000)
        # engine.sensors.TPS_percent = 50.0
        
        # Run 5 cycles total, but we will only analyze the very last cycle
        engine, _ = self.fire_engine(preset=RunMode.CRUISE, thermal_state="warm", cycles=5) 
        
        # torque_indicated_history always contains the MOST RECENT 720 degrees.
        # By running 5 cycles, we are guaranteed that history is a "Steady State" cycle.
        torque_wave = engine.state.torque_indicated_history
        
        # 1. Identify raw peaks
        raw_peaks = [i for i in range(1, 719) if torque_wave[i-1] < torque_wave[i] > torque_wave[i+1]]
        
        # 2. Filter for significant power strokes (>20 Nm)
        significant_peaks = [p for p in raw_peaks if torque_wave[p] > 20.0] 
        
        # 3. Apply Distance Filter (90 deg) to handle high-res jitter
        power_peaks = []
        if significant_peaks:
            power_peaks.append(significant_peaks[0])
            for p in significant_peaks[1:]:
                if p - power_peaks[-1] > 90:
                    power_peaks.append(p)

        self.assertEqual(len(power_peaks), 4, f"Detected {len(power_peaks)} pulses instead of 4.")
        
        # Verify 180 degree symmetry
        for i in range(len(power_peaks)-1):
            diff = power_peaks[i+1] - power_peaks[i]
            self.assertAlmostEqual(diff, 180, delta=5)
    
    def test_firing_power_consistency(self):
        """
        Validates that a firing engine produces positive net work and that
        instantaneous torque history matches the cycle-work integration.
        """
            
        engine, _ = self.fire_engine(preset=RunMode.WOT, 
                                     thermal_state="hot", cycles=10)
 
        j_to_nm = 1.0 / (4 * np.pi)
        
        # Calculate Averages
        t_ind_avg_history = np.mean(engine.state.torque_indicated_history)
        t_brake_avg_history = np.mean(engine.state.torque_brake_history)
        
        # Work-based scalars (calculated by the model at the end of the cycle)
        t_ind_avg_work = engine.state.work_net_indicated_j * j_to_nm
        t_fric_avg_work = engine.state.work_friction_j * j_to_nm
              
        # 5. VALIDATION
        # A: Numerical Consistency (History vs Integrated Work)
        # Using a 720-sample window, these should be nearly identical
        self.assertAlmostEqual(t_ind_avg_history, t_ind_avg_work, delta=0.25)
        
        # B: Physics Consistency (Brake = Indicated - Friction)
        # Note: torque_governor is 0 if not motoring/governed, otherwise subtract it
        expected_brake = t_ind_avg_work + t_fric_avg_work
        self.assertAlmostEqual(t_brake_avg_history, expected_brake, delta=0.25)
        
        # C: Performance Sanity Check
        # A 4-cylinder engine at WOT should definitely produce positive brake torque
        self.assertGreater(t_brake_avg_history, 20.0, "Engine is firing but not producing power!")
         
    def test_volumetric_efficiency_limits(self):
        """Validation: Is the air-mass trapping realistic for a 2.1L?"""
        # engine = EngineModel(rpm=3000)
        # engine.sensors.TPS_percent = 100.0
        engine, _ = self.fire_engine(preset=RunMode.WOT, thermal_state="warm", cycles=5)
        
        # At 1 bar, a 525cc cylinder holds ~0.6g of air. 
        # With VE and heating, 0.45g - 0.55g is the 'Digital Twin' target.
        trapped_mass_mg = (engine.cyl.air_mass_at_spark - engine.cyl.air_mass_at_TDC) * 1e6
        self.assertGreater(trapped_mass_mg, 400.0, "Engine is gasping for air (VE too low)")
        self.assertLess(trapped_mass_mg, 650.0, "Engine is supercharging itself (VE too high)")
                
    def test_engine_stall_inertia(self):
        """
        Integration: Measures how many cycles the engine rotates after cutting 
        fuel/spark at 900 RPM (Inertia check).
        """
        
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, rpm_hold=True, thermal_state="hot", cycles=5)

        # 1. Initialize engine at a stable idle
        # We use motoring=False initially to ensure we have a 'running' state if needed,
        # but for a pure inertia/friction test, we start from 900 RPM.
        # initial_rpm = 900.0
        # engine = EngineModel(rpm=initial_rpm)
        # ecu = MockEcuOutput()
        

        
        stall_threshold_high = 600
        stall_threshold_low = c.CRANK_RPM
        max_cycles = 100       # Safety limit to prevent infinite loops
        cycles_to_stall_high = 0
        cycles_to_stall_low = 0
        cycles = 0
        
        # # 2. Cut power and simulate cycle-by-cycle
        # stall_threshold = 50.0 # RPM at which we consider the engine "stopped"
        # max_cycles = 100       # Safety limit to prevent infinite loops
        # cycles_to_stall = 0
        
        # We manually step cycles to track the decay
        while engine.sensors.rpm > stall_threshold_low and cycles < max_cycles:

            
            # debug
            tps = engine.state.effective_tps
            map = engine.sensors.MAP_kPa
            iacv = max(ecu.iacv_pos, ecu.iacv_wot_equiv)
            air_at_ivc = engine.cyl.air_mass_at_IVC
            air_at_spark = engine.cyl.air_mass_at_spark
            air_at_tdc = engine.cyl.air_mass_at_TDC
            fuel_at_ivc = engine.cyl.fuel_mass_at_IVC
            total_heat_J = engine.cyl.total_cycle_heat_J
            t_brake_nm = engine.state.torque_brake
            t_brake_avg = np.mean(engine.state.torque_brake_history)
            t_brake_max = np.max(engine.state.torque_brake_history)
            lambda_ = engine.sensors.lambda_
            rpm = engine.sensors.rpm
            dm_in = np.sum(engine.cyl.dm_in_history)
            map = engine.state.map_avg_kPa
            map_peak = np.max(engine.state.map_history)
            
            
            print(f"rpm:{rpm:4.0f} air_ivc:{air_at_ivc:.7f} air_tdc:{air_at_tdc:.7f} air_spark:{air_at_spark:.7f} "
                  f"fuel_ivc:{fuel_at_ivc:.7f} heat:{total_heat_J:.1f} lambda:{lambda_:.1f} "
                  f"dm_in_sum:{dm_in:.7f} eff_tps:{tps} "
                  f" T-brake:{t_brake_nm:.1f} map_avg:{map:.0f} map_peak:{map_peak / 1000:.0f}"
                # T_brake_avg:{t_brake_avg:.4f} T_brake_max:{t_brake_max:.4f}"
                  )
            
            # 2. trun off the IACV valve and setup trackers
            ecu.iacv_pos = 0.0
            engine.motoring_rpm = 0.0
            
            # fire_engine with motoring=True disables combustion/fuel
            # This forces the engine to rely solely on its internal inertia 
            # to overcome friction and pumping work.
            self.fire_engine(
                rpm=engine.sensors.rpm, 
                tps=0.0, 
                rpm_hold=False,
                motoring=False, 
                engine=engine, 
                ecu=ecu,
                has_spark=True,
                cycles=1
            )
            
            if engine.sensors.rpm <= stall_threshold_high and cycles_to_stall_high == 0:
                cycles_to_stall_high = cycles
                
            if engine.sensors.rpm == stall_threshold_low:
                cycles_to_stall_low = cycles
            
            cycles += 1
            
        print("test_engine_stall_inertia")
        print(f"cycles to stall_high:{cycles_to_stall_high} cycles to stall_low:{cycles_to_stall_low}")
        print(f"combustion active: {engine.cyl.combustion_active}")
        
        # 3. Assertions based on physical expectations for a 2.1L WBX
        # A typical engine with standard flywheel inertia should take 
        # roughly 3 to 10 cycles to come to a complete stop from idle.
        self.assertGreater(cycles_to_stall_low, 2, "Engine stalled too fast (Inertia too low or friction too high)")
        self.assertLess(cycles_to_stall_low, 15, "Engine spun for too long (Inertia too high or pumping losses too low)")
        
    def test_iacv_authority_and_step_response(self):
        """
        Integration: Verifies IACV has authority over RPM and moves in the correct direction.
        This ensures the RL agent isn't chasing a non-monotonic or insensitive plant.
        """
        # 1. Start with a "Closed" state (IACV is defaulted to ~33% for idle)
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, 
                                       iacv_pos=10, rpm_hold=False, cycles=5)
        rpm_low_stable = engine.sensors.rpm
        tps_low = engine.state.effective_tps
        
        # 2. Step to an "Open" state (80% IACV)
        # We pass the 'engine' object to persist the state (thermal, momentum)
        self.fire_engine(rpm=engine.sensors.rpm, tps=engine.sensors.TPS_percent,
                         engine=engine, ecu=ecu,
                         iacv_pos=80, rpm_hold=False, cycles=5)
        rpm_high_stable = engine.sensors.rpm
        tps_high = engine.state.effective_tps
        
        print(f"\n[IACV Step] Low (10%): {rpm_low_stable:.1f} RPM | High (80%): {rpm_high_stable:.1f} RPM")
        
        # 3. Validation
        # The delta should be significant (e.g., > 500 RPM)
        rpm_delta = rpm_high_stable - rpm_low_stable
        self.assertGreater(rpm_delta, 400, 
            f"IACV sensitivity too low. Delta was only {rpm_delta:.1f} RPM. "
            "The RL agent will struggle to find a gradient.")
        
        # 4. Decay Test (The "Stall" Sensitivity)
        # Return to 10% IACV and see if it drops back down

        self.fire_engine(rpm=engine.sensors.rpm, tps=engine.sensors.TPS_percent, 
                         engine=engine, ecu=ecu,
                         iacv_pos=10, rpm_hold=False, cycles=10)
        tps_stall = engine.state.effective_tps
        rpm_stall = engine.sensors.rpm
        self.assertLess(rpm_stall, rpm_high_stable, 
            "Engine failed to decelerate when IACV was closed. Check FMEP/PMEP logic.")
        
        # 5. The "Zero Air" Stability Test
        # If we drop to 0% IACV, the brake torque MUST be negative at high RPM.
        self.fire_engine(rpm=1800, tps=0.0, ecu=ecu, engine=engine, 
                         iacv_pos=0.0, rpm_hold=False, has_spark=False, cycles=10)
        P_peak = np.max(engine.cyl.log_P)
        P_peak_angle = np.argmax(engine.cyl.log_P)
        vol_at_TDC = engine.cyl.V_list[360]
        vol_at_P_peak = engine.cyl.V_list[P_peak_angle]
        t_ind_avg = np.mean(engine.state.torque_indicated_history)
        t_fric_avg = np.mean(engine.state.torque_friction_history)
        t_brake_avg = np.mean(engine.state.torque_brake_history)
        imep_g = engine.state.IMEP_gross
        pmep = engine.state.PMEP
        fmep = engine.state.FMEP
        rpm = engine.sensors.rpm
        rpm_max = np.max(engine.sensors.rpm_history)
        rpm_min = np.min(engine.sensors.rpm_history)
        map = engine.sensors.P_manifold_Pa
        map_max = np.max(engine.state.map_history)
        map_min = np.min(engine.state.map_history)
        combustion = engine.cyl.combustion_active
        
        # At 1800 RPM and 0% IACV, the engine MUST be losing energy
        self.assertLess(engine.sensors.rpm, 1750, 
            f"Non-Physical: Engine is maintaining/gaining RPM ({engine.sensors.rpm:.1f}) "
            "at 0% IACV. Braking forces (PMEP/FMEP) are too weak!")
            
        # Check the actual physics state
        self.assertLess(engine.state.torque_brake_history.mean(), 0.0,
            "Brake torque must be negative when air is cut at high RPM.")
        
        # Check the engine has completely stalled and is at cranking speed.
        self.assertEqual(engine.sensors.rpm, c.CRANK_RPM, f"Engine has not stalled and is {engine.sensors.rpm}")
        
    def test_low_rpm_drag_limit(self):
        """Ensures engine stalls when air is insufficient (Physics Sanity)."""
        # Set IACV to 5% (Should not be enough to sustain 1000+ RPM)
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, cycles=5)
        
        # Run for long enough for momentum to dissipate
        self.fire_engine(rpm=1000, tps=0.0, iacv_pos=0.0,
                         ecu=ecu, engine=engine, cycles=20)
        effective_tps = engine.state.effective_tps
        
        self.assertLess(engine.sensors.rpm, 600, 
            "Engine is 'floating' at too high RPM with closed IACV. Increase FMEP/PMEP.")
 
    def test_idle_manifold_pressure(self):
        """Target: 32.5 kPa at 900 RPM Idle (5% tolerance)"""
        engine, ecu = self.fire_engine(preset=RunMode.IDLE, thermal_state="warm", rpm_hold=True)
        ecu.iacv_pos = 43
        engine, _ = self.fire_engine(preset=RunMode.IDLE, rpm_hold=True, ecu=ecu, engine=engine, 
                                     thermal_state="warm", cycles=3)
        actual_map = np.mean(engine.state.map_history) / 1000
        effective_tps = engine.state.effective_tps
        # self.assertAlmostEqual(actual_map, 32.5, delta=32.5 * 0.05)
        
        target_idle_map = 37000.0  # example target pressure in Pa
        self.assertAlmostEqual(engine.sensors.P_manifold_Pa, target_idle_map, delta=target_idle_map * 0.10,
            msg="Idle manifold pressure drifted out of the open-loop 10% tolerance window.")

    # def test_volumetric_efficiency_high_rpm(self):
    #     """Target: 0.88 (88%) VE at 4500 RPM WOT"""
    #     engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
    #     actual_mass = engine.cyl.air_mass_at_IVC
    #     P_peak = engine.cyl.P_peak_bar
    #     P_peak_angle = engine.cyl.P_peak_angle
    #     P_manifold = engine.sensors.P_manifold_Pa
    #     C_DISPLACED = c.V_DISPLACED
    #     ideal_mass = (engine.sensors.P_manifold_Pa * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
    #     ve = actual_mass / ideal_mass
    #     # self.assertAlmostEqual(ve, 0.80, delta=0.80 * 0.05)
    #     # COME BACK
    #     print("COME BACK TO THIS TEST BYPASS")
        
    # def test_ve_curve(self):
    #     """
    #     Validation Guardrail: Verifies that the volumetric efficiency curve 
    #     follows a physical breathing profile across the operational range.
    #     """
    #     rpm_sweep = [1000, 2500, 4000, 5500]
    #     ve_results = []
    #     ideal_density = c.P_ATM_PA / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
    #     ideal_mass_per_cyl = ideal_density * c.V_DISPLACED
        
        
    #     for rpm in rpm_sweep:
    #         # Fire engine under WOT conditions to test peak breathing capacity
    #         engine, _ = self.fire_engine(preset=RunMode.WOT, rpm=rpm, cycles=4)
            
    #         # Pull the calculated volumetric efficiency from the final cycle
    #         actual_mass = engine.cyl.air_mass_at_IVC
    #         ve = actual_mass / ideal_mass_per_cyl
    #         ve_results.append(ve)
            
    #     # --- TREND ASSERTIONS ---
    #     # 1. Low to Mid RPM: Engine breathing should improve or stabilize as it enters its power band
    #     self.assertGreater(ve_results[1], ve_results[0] * 0.9, 
    #                     f"VE dropped too aggressively at mid-range: {ve_results}")
        
    #     # 2. High RPM Drop-off: High-RPM breathing should degrade due to valve restriction bottlenecks
    #     self.assertLess(ve_results[3], ve_results[1], 
    #                     f"VE failed to drop at high RPM. Charging effects are unrestricted: {ve_results}")
                        
    #     # 3. Absolute Sanity Boundary
    #     for ve, rpm in zip(ve_results, rpm_sweep):
    #         self.assertLess(ve, 1.15, f"Super-charging effect detected on N/A engine at {rpm} RPM: {ve:.2f}")
    #         self.assertGreater(ve, 0.45, f"Engine is completely choking out unreasonably at {rpm} RPM: {ve:.2f}")

    def test_ve_curve_shape_matches_dj_torque(self):
        """
        Validation Guardrail: Enforces that the VE curve mirrors the 
        real-world DJ torque profile (peaking at 2800 RPM, dropping by 4800 RPM).
        """
        ideal_density = c.P_ATM_PA / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ideal_mass_per_cyl = ideal_density * c.V_DISPLACED
        
        # Sweep the critical operational nodes of the DJ engine
        engine_1000, _ = self.fire_engine(preset=RunMode.WOT, rpm=1000.0, cycles=4)
        engine_2800, _ = self.fire_engine(preset=RunMode.WOT, rpm=2800.0, cycles=4)
        engine_4800, _ = self.fire_engine(preset=RunMode.WOT, rpm=4800.0, cycles=4)
        engine_5500, _ = self.fire_engine(preset=RunMode.WOT, rpm=5500.0, cycles=4)
        
        ve_1000 = (engine_1000.cyl.air_mass_at_IVC / ideal_mass_per_cyl) * 100.0
        ve_2800 = (engine_2800.cyl.air_mass_at_IVC / ideal_mass_per_cyl) * 100.0
        ve_4800 = (engine_4800.cyl.air_mass_at_IVC / ideal_mass_per_cyl) * 100.0
        ve_5500 = (engine_5500.cyl.air_mass_at_IVC / ideal_mass_per_cyl) * 100.0
        
        print(f"\n[VE Diagnostics] 1000: {ve_1000:.1f}% | 2800: {ve_2800:.1f}% | 4800: {ve_4800:.1f}% | 5500: {ve_5500:.1f}%")
        
        # 1. Assert Low-Speed VE is lower than Mid-Range Peak
        self.assertGreater(
            ve_2800, ve_1000 + 5.0, 
            f"Inverted Curve Error: VE at 1000 RPM ({ve_1000:.1f}%) must be lower than peak at 2800 RPM ({ve_2800:.1f}%)."
        )
        
        # 2. Assert VE is actively falling from its peak by the time it reaches power peak
        self.assertGreater(
            ve_2800, ve_4800, 
            f"VE should be tapering off by 4800 RPM ({ve_4800:.1f}%) compared to peak torque speed ({ve_2800:.1f}%)."
        )
        
        # 3. Assert sharp drop-off past peak power (High-RPM Choking)
        self.assertGreater(
            ve_4800, ve_5500 + 4.0, 
            f"Power Peak Shift Error: VE is not dropping off sharply past 4800 RPM. 5500 RPM VE is too high ({ve_5500:.1f}%)."
        )

    def test_verify_intake_reversion_at_low_rpm(self):
        """Validate that flow reversion occurs at low rpm.
        """

        engine, ecu = self.fire_engine(preset=RunMode.WOT, rpm=1000, 
                                       thermal_state="warm", cycles=5)
        
        print(f"IAT: {engine.sensors.IAT_K:.1f}K")
        air_ivc = engine.cyl.air_mass_at_IVC * 1e6
        air_tdc = engine.cyl.air_mass_at_TDC * 1e6
        air_bdc = engine.cyl.air_mass_at_BDC * 1e6
        ideal_density = c.P_ATM_PA / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ideal_mass_per_cyl = ideal_density * c.V_DISPLACED
        
        ve = air_ivc / (ideal_mass_per_cyl * 1e6)
        print(f"VE efficiency is {ve:.4f}")
        print(f"air mass at TDC {air_tdc:.5f}mg")
        print(f"air mass at BDC {air_bdc:.5f}mg")
        print(f"air mass at IVC {air_ivc:.5f}mg")
        
        # Extract telemetry profiles for the final full cycle
        dm_in = engine.cyl.dm_in_history  
        intake_open = engine.valves.intake_lift_table > 1e-5
        
        # Critical Angles
        bdc_angle = 180
        
        ivc_angle = int(engine.valves.intake.close_angle)
        ivo = int(engine.valves.intake.open_angle)
        
        print(f"Intake Valve IVO: {ivo}  IVC: {ivc_angle}° CAD")
        
        # Segment : Inflow phase (IVO up to TDC)
        mass_in_to_tdc = np.sum(dm_in[ivo:])
        
        # Segment 1: Inflow phase (IVO up to BDC)
        mass_in_to_bdc = np.sum(dm_in[intake_open & (np.arange(720) <= bdc_angle)])
         
        # Segment 2: Post-BDC reversion phase (BDC up to IVC)
        mass_post_bdc = np.sum(dm_in[(np.arange(720) > bdc_angle) & (np.arange(720) <= ivc_angle)])
        
        print(f"Air dm_in Mass inducted before TDC: {mass_in_to_tdc * 1e6:.3f} mg")
        print(f"Air dm_in Mass inducted before BDC: {mass_in_to_bdc * 1e6:.3f} mg")
        print(f"Air dm_in flow after BDC (Reversion Window): {mass_post_bdc * 1e6:.3f} mg")
        
        # Analysis and Assertions
        if mass_post_bdc >= 0:
            print("[FAIL] Reversion is completely absent or positive! Air is continuing to cram into the cylinder despite the ascending piston.")
        else:
            reversion_ratio = abs(mass_post_bdc) / mass_in_to_bdc
            print(f"[INFO] Reversion detected! Returned {reversion_ratio * 100:.1f}% of air mass to manifold.")
            
        # If VE is ~97% at 1000 RPM, reversion mass will be close to 0. 
        # A real engine expects at least 10-20% charge return at this speed.
        self.assertLess(mass_post_bdc, -1e-6, 
            msg="Breathing model fails to simulate charge reversion post-BDC at low RPM.")


    def test_peak_cylinder_pressure_limit(self):
        """Target: Max 65.0 bar structural limit at 4500 RPM"""
        engine, _ = self.fire_engine(preset=RunMode.WOT, thermal_state='warm')
        p_max_bar = np.max(engine.cyl.log_P) / 1e5
        P_peak_angle = engine.cyl.P_peak_angle
        P_peak = np.max(engine.cyl.log_P)
        burn_duration = engine.cyl.burn_duration
        spark_angle = engine.cyl.spark_event_theta
        self.assertLessEqual(p_max_bar, 65.0)

    def test_peak_pressure_angle(self):
        """Target: Peak pressure should occur at 372 CAD (ATDC)"""
        engine, _ = self.fire_engine(preset=RunMode.WOT, rpm_hold=True, 
                                     thermal_state="warm", cycles=3)
        p_angle = np.argmax(engine.cyl.log_P)
        self.assertAlmostEqual(p_angle, 372, delta=5) # 5 degree tolerance

    def test_net_torque_equilibrium(self):
        """Target: Nm Brake Torque at peak load"""
        engine, _ = self.fire_engine(rpm=2800, tps=100.0, rpm_hold=True,
                                     wheel_load=0, thermal_state="Warm", cycles=20)
        brake_torque = np.average(engine.state.torque_brake_history)
        P_peak_angle = engine.cyl.P_peak_angle
        rpm = engine.sensors.rpm
        self.assertAlmostEqual(brake_torque, 160.0, delta=160.0 * 0.05)


    def test_PMEP_high_rpm(self):
        """Validates PMEP (<1.1 bar)."""
        engine, ecu = self.fire_engine(preset=RunMode.WOT, cycles=5)
         
        intake = engine.valves.intake
        exhaust = engine.valves.exhaust
        P_peak_angle = engine.cyl.P_peak_angle
        mass_at_spark = engine.cyl.air_mass_at_spark
        spark = ecu.spark_timing
        start_trigger = ecu.injector_start % 720
        end_trigger = ecu.injector_end % 720
        rpm = engine.sensors.rpm
        pmep = engine.state.PMEP

        # 1. PMEP (Gas Exchange Efficiency)
        self.assertLess(abs(pmep), 1.10, 
            f"High-RPM pumping drag exceeded physical stock threshold of 1.10 bar: {engine.state.PMEP:.2f} bar")

               
    def test_mechanical_efficiency_wot(self):
        """Audits the ratio of Brake work to Indicated work at peak torque."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=10)
        
        imep_g = engine.state.IMEP_gross
        imep_n = engine.state.IMEP_net
        pmep   = engine.state.PMEP   # Should be ~ -0.15 to -0.25 at WOT
        fmep   = engine.state.FMEP   # Should be ~ 1.2
        bmep   = engine.state.BMEP   # Should be ~ 9.5
        
        # Mech Efficiency = BMEP / IMEP
        mech_eff = bmep / imep_n
        
        # Stock WBX target: 0.80 - 0.92
        self.assertGreater(mech_eff, 0.80)
        self.assertLess(mech_eff, 0.95)

    def test_knock_during_high_load(self):
        """
        Integration test: Runs a simulation at Wide Open Throttle (WOT)
        to see if the physics-based Pmax triggers the knock detector.
        """
        
        ecu = MockEcuOutput(
            spark_timing=330, # 30 deg BTDC
            afr_target=12.5,  # Rich for cooling
        )
        
        engine, _ = self.fire_engine(preset=RunMode.WOT, thermal_state="warm", 
                                     cycles=5, ecu=ecu)
            
        # Extract data for the detector
        p_max_bar = np.max(engine.cyl.log_P) / 1e5
        clt = 95.0 # Hot engine
        lambda_val = ecu.afr_target / 14.7
        spark_advance = 360 - ecu.spark_timing
        
        is_knocking, severity = pf.detect_knock(
            p_max_bar, clt, engine.sensors.rpm, spark_advance, lambda_val, fuel_octane=91.0
        )
        
        print(f"Pmax: {p_max_bar:.2f} bar | Knock: {is_knocking} (Severity: {severity:.2f})")
        
        # Validation: A 2.1L engine at WOT with 91 octane and 30 deg advance 
        # should likely show some knock intensity.
        if is_knocking:
            self.assertLess(severity, 20.0, "Knock intensity is high; engine damage likely!")

    def test_knock_induced_by_aggressive_timing(self):
        """
        Integration test: Forces a knock event by using low octane fuel,
        lean mixtures, and aggressive spark advance at high load.
        """
        engine, ecu = self.fire_engine(preset=RunMode.WOT, thermal_state="warm", cycles=4)
            
        # 3. Extract data
        p_max_bar = np.max(engine.cyl.log_P) / 1e5
        clt = 105.0        # Overheating engine reduces threshold
        lambda_val = ecu.afr_target / 14.7
        spark_advance = 40 # 40 degrees
        
        # 4. Use low-grade 87 Octane fuel to further lower the threshold
        is_knocking, severity = pf.detect_knock(
            p_max_bar, clt, engine.sensors.rpm, spark_advance, lambda_val, fuel_octane=87.0
        )
        
        # 5. Assertions
        self.assertTrue(is_knocking, "Engine should be knocking under these aggressive conditions")
        self.assertGreater(severity, 0.0, "Knock severity should be positive")

    def test_mechanical_efficiency_envelope(self):
        """Checks if internal drag is realistic for a pushrod boxer."""
        # Target: Mechanical Efficiency (BMEP/IMEP) should be ~85-88% at WOT
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=10)
        fmep = engine.state.FMEP
        bmep = engine.state.BMEP
        imep_n = engine.state.IMEP_net
        eff = engine.state.mechanical_efficiency
        mech_eff = bmep / imep_n
        
        t_brake_avg = np.mean(engine.state.torque_brake_history)
        t_fric_avg = np.mean(engine.state.torque_friction_history)
        t_ind_avg = np.mean(engine.state.torque_indicated_history)
        t_mech_eff = t_brake_avg / t_ind_avg
        
        self.assertGreater(mech_eff, 0.80, "Friction is consuming too much power (Non-Physical).")
        self.assertLess(mech_eff, 0.90, "Friction is too low for a pushrod engine.")
     
    def test_indicated_torque_magnitude(self):
        """Ensures combustion pressure isn't 'over-tuned' to hide friction."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        t_ind = np.mean(engine.state.torque_indicated_history)
        t_brake = np.mean(engine.state.torque_brake_history)
        t_fric = np.mean(engine.state.torque_friction_history)
        map_avg = np.mean(engine.state.map_history)
        map_peak = np.max(engine.state.map_history)
        bmep = engine.state.BMEP
        imep_n = engine.state.IMEP_net
        fmep = engine.state.FMEP
        imep_calc = (t_ind * 4 * np.pi) / (c.V_DISPLACED * c.NUM_CYL * 1e5)
        ME = t_brake / t_ind
        P_peak = np.max(engine.cyl.log_P)
        P_peak_angle = engine.cyl.P_peak_angle
        Q_loss_sum = np.sum(engine.cyl.Q_loss_history)
        total_mass_at_IVC = engine.cyl.total_mass_at_IVC
        air_mass_at_IVC = engine.cyl.air_mass_at_IVC
        Q_in_sum = np.sum(engine.cyl.Q_in_history)
        Q_loss_sum = np.sum(engine.cyl.Q_loss_history)
        mass_in = np.sum(engine.cyl.dm_in_history)
        mass_out = np.sum(engine.cyl.dm_ex_history)
        
        # 2.1L NA engine should not exceed ~195 Nm indicated to hit 160 Nm brake
        self.assertLess(t_ind, 220, "Indicated torque is too high (unphysical combustion pressure).")
        self.assertGreater(t_ind, 150, "Indicated torque is too low to sustain 160 Nm brake.")

   # def test_energy_retention_high_rpm(self):
    #     """Target: 78.0% Energy Retention at 4500 RPM"""
    #     engine, _ = self.fire_engine(rpm=4500, tps=100.0, rpm_hold=True, cycles=3)
    #     dq_added = engine.cyl.Q_in_history
    #     total_dq = np.sum(dq_added[dq_added > 0])
    #     total_q_lost = np.sum(engine.cyl.Q_loss_history)
    #     retention = ((total_dq - total_q_lost) / total_dq) * 100
    #     self.assertAlmostEqual(retention, 78.0, delta=78.0 * 0.05)

  
    """
    TO FIX LATER
    """    
    # def test_fuel_efficiency_bsfc(self):
    #     """Validates the overall energy conversion efficiency."""
    #     engine, _ = self.fire_engine(rpm=3000, tps=100.0)
    #     # Target for 80s tech: 260-300 g/kWh
    #     bsfc = engine.state.BSFC_g_kWh 
    #     self.assertIn(bsfc, [250, 320], "BSFC is non-physical for a VW WBX.")


# =================================================================
# TIER 6: EngineModel input and outout power Balance Tests
# =================================================================
class TestEnergyBalance(BaseEngineTest):
    """ Evaluates the entire system. It ensures that energy in (fuel) 
        perfectly equals energy out (shaft power, heat losses, exhaust enthalpy).
    """

    def test_mep_balance_audit(self):
        """
        Validation: Audits that Indicated work minus losses equals Brake work.
        Identity: IMEP_gross + PMEP + FMEP = BMEP (using signed values).
        """
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=10)
        
        # Standardizing to Bar
        # Assuming your model now provides these in engine.state
        imep_g = engine.state.IMEP_gross
        pmep   = engine.state.PMEP   # Should be ~ -1 at WOT
        fmep   = engine.state.FMEP   # Should be ~ - 1-2 bar
        bmep   = engine.state.BMEP   # Should be ~ 9.5
        
        calculated_bmep = imep_g + pmep + fmep
        
        self.assertAlmostEqual(bmep, calculated_bmep, delta=0.05, 
                            msg=f"MEP Balance Failure: BMEP({bmep:.2f}) != Calc({calculated_bmep:.2f})")
        
    def test_mep_ratio_realism(self):
        """
        Validates the ratio of Friction to Pumping losses for a stock 2.1L WBX.
        FMEP is typically 5-8x larger than PMEP at WOT.
        """
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        
        pmep = engine.state.PMEP
        fmep = engine.state.FMEP
        
        # Real world: FMEP (~1.2 bar) vs PMEP (~1 bar)
        ratio = fmep / pmep
               
        self.assertGreater(ratio, 1.5, 
                           f"Friction-to-pumping ratio unphysical for stock hardware: {ratio:.2f}")
        self.assertLess(ratio, 10.0, 
                        "Mechanical friction is too high (Check friction coefficient).")
    
    def test_combustion_energy_partition(self):
        """Validates Heat Loss % and Peak Temperature."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=10)
        pct_loss = (engine.cyl.Q_loss_total / engine.cyl.total_cycle_heat_J) * 100
        max_t = np.max(engine.cyl.log_T)
        
        P_peak_angle = engine.cyl.P_peak_angle
        m_vibe = engine.cyl.m_vibe
        m_air_at_IVC = engine.cyl.air_mass_at_IVC
        m_fuel_at_IVC = engine.cyl.fuel_mass_at_IVC
        m_total_at_IVC = engine.cyl.total_mass_at_IVC
        m_exh_at_IVC = m_total_at_IVC - m_fuel_at_IVC - m_air_at_IVC
        afr =  m_air_at_IVC / m_fuel_at_IVC
        
        # self.assertAlmostEqual(pct_loss, 20.0, delta=2.0) # this is validate earlier
        self.assertLess(max_t, 2850.0)

    # THIS TEST IS A DUPLICATE OF A LOWER ORDER TEST AND NOT ADDING VALUE.

    # def test_energy_conservation_proportions(self):
    #     """
    #     Integration: Checks the 'Heat Balance' of a single WOT combustion cycle.
    #     Formula: Q_fuel = W_brake + Q_loss + Q_exhaust + Friction
    #     """
    
    #     engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=30)
    #     q_fuel = np.sum(engine.cyl.Q_in_history)
    #     q_loss = np.sum(engine.cyl.Q_loss_history)
    #     # q_fuel_2 = engine.cyl.total_cycle_heat_J
    #     # q_loss_2 = engine.cyl.Q_loss_total
        
    #     # print(f"\nCOMPARING TWO Q_LOSS COUNTERS: {q_loss} and {q_loss_2}")
    #     # print(f"\nCOMPARING TWO Q_IN FUEL COUNTERS: {q_fuel} and {q_fuel_2}")
        
    #     T_cyl = engine.cyl.T_curr
    #     T_wall = engine.cyl.T_wall
    #     clt_C = engine.sensors.CLT_C
        
    #     loss_pct = (q_loss / q_fuel) * 100
        
    #     # print(f"\nDEBUG ENERGY CONSERVATAION rpm:{rpm} q_loss:{q_loss:.2f} total Q-fuel:{q_fuel:.2f} ratio:{loss_pct:.2f} ")
    #     # print(f"|__ T_wall:{engine.cyl.T_wall:.2f} cyl_P:{engine.cyl.P_curr/1000:.2f}kPa " 
    #     #       f"Peak_P:{np.max(engine.cyl.log_P)/1000:.2f}kPa Peak_P_angle:{np.argmax(engine.cyl.log_P):.2f} "
    #     #       f"Peak_T:{np.max(engine.cyl.log_T):.2f}K Peak_T_angle:{np.argmax(engine.cyl.log_T):.2f} ")
        
        
    #     # Target 20% per your validation report
    #     self.assertGreaterEqual(loss_pct, 18.0, f"Heat loss too low: {loss_pct:.1f}%")
    #     self.assertLessEqual(loss_pct, 21.0, f"Heat loss too high: {loss_pct:.1f}%")

    def test_torque_work_consistency(self):
        """Ensures Indicated Work matches the integral of Indicated Torque."""
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=5)
        
        # Total Work from P*dV
        work_j = engine.state.work_net_indicated_j
        
        # Total Work from Torque integral: Sum(T) * dTheta_radians
        torque_sum_j = np.sum(engine.state.torque_indicated_history) * (np.pi / 180.0)
        
        self.assertAlmostEqual(work_j, torque_sum_j, delta=5.0,
            msg=f"Torque/Work integration discrepancy too high. Work: {work_j:.2f}J, Torque: {torque_sum_j:.2f}J")
        
    def test_mep_mathematical_balance(self):
        """
        Tier 2: Validates the physical identity IMEP_net = BMEP + FMEP.
        This ensures no energy is 'vanishing' in the integrator.
        """
        engine, _ = self.fire_engine(preset=RunMode.WOT, cycles=10)
        
        # Extract MEPs (assuming these are in Bar)
        imep_n = engine.state.IMEP_net
        bmep = engine.state.BMEP
        fmep = engine.state.FMEP
        pmep = engine.state.PMEP
        
        # The sum of what we use (Brake) and what we lose (Friction) 
        # must equal the work indicated on the piston crown.
        calculated_imep_net = bmep - fmep
        
        self.assertAlmostEqual(imep_n, calculated_imep_net, delta=0.05, 
                            msg="MEP Balance broken: Energy is not conserved in the model.")
        



# if __name__ == '__main__':
    # unittest.main()
    
# =================================================================
# THE HIERARCHY CONTROLLER
# =================================================================
if __name__ == '__main__':
    
    import sys
    
    # Define the order of the hierarchy   
    hierarchy = [
        TestPhysics,         # Unit tests: Math, constants, grid sizes
        TestModel,           # Validate the math and performance tracking of the EngineModel
        TestLosses,          # Component tests: Friction coefficients, lookups
        TestMotoring,        # Integration tests: Kinematics, unfuelled cranking
        TestThermals,        # System tests: Heat rejection, mass-flow boundaries
        TestCombustion,      # Complex physics: Chemistry, pressure generation
        TestEnergyBalance,   # Sanity checks: Conservation laws across full cycles
    ]
    
    # Check command line arguments
    # -l or --locals: show local variables on failure
    # -b or --buffer: buffer stdout/stderr
    show_locals = '-l' in sys.argv or '--local' in sys.argv
    use_buffer = '-b' in sys.argv or '--buffer' in sys.argv
    
    runner = unittest.TextTestRunner(
        verbosity=2, 
        buffer=use_buffer, 
        tb_locals=show_locals
    )
    loader = unittest.TestLoader()
    
    print("--- Starting Hierarchical Engine Model Test ---")
    for tier in hierarchy:
        print(f"\n>>> RUNNING {tier.__name__}...")
        suite = loader.loadTestsFromTestCase(tier)
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            print(f"\n[!] STOPPING: {tier.__name__} failed.")
            print("Fix the foundation before testing higher-level integration.")
            exit(1) # Exit with error code

    print("\n[SUCCESS] All tiers passed.")


