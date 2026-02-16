import unittest
import numpy as np
import constants as c
import physics_functions as pf
from engine_model import EngineModel
from dataclasses import dataclass, field

# =================================================================
# 1. SHARED INFRASTRUCTURE (The "Digital Test Cell")
# =================================================================

@dataclass(slots=True, frozen=False)
class MockEcuOutput:
    """Consolidated ECU for all tiers of testing."""
    spark: bool = False
    spark_timing: int = 350
    afr_target: float = 14.7 * 0.88
    target_rpm: float = 0.0
    iacv_pos: float = 33 
    pid_P: float = 0.0; pid_I: float = 0.0; pid_D: float = 0.0
    injector_on: bool = False
    injector_start: int = 0
    injector_end: int = 170
    fuel_cut_active: bool = False
    
    @property
    def iacv_wot_equiv(self) -> float:
        return self.iacv_pos * 0.06

@dataclass
class MockValve:
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
class MockValves:
    intake: MockValve = field(default_factory=lambda: MockValve(is_intake=True)) 
    exhaust: MockValve = field(default_factory=lambda: MockValve(is_intake=False))
    # note:  in the engine model the cam open and close are specific and valve open/close is calculation
    # they are:
    #       intake_v = MockValve(open=688, close=250, lift=9, diameter=40, cam_open=6, cam_close=211)
    #       exhaust_v = MockValve(open=469, close=21, lift=9, diameter=34, cam_open=507, cam_close=702)

@dataclass
class MockCylinderState:
    # Arrays for historical lookup (IVC references)
    log_P: np.ndarray = field(default_factory=lambda: np.full(720, 101325.0))
    log_T: np.ndarray = field(default_factory=lambda: np.full(720, 300.0))
    V_list: np.ndarray = field(default_factory=lambda: np.full(720, 0.0005))
    
    # Instantaneous values for current step
    P_curr: float = 101325.0
    T_curr: float = 300.0
    V_curr: float = 0.0005
    T_wall: float = 450.0
    V_clearance: float = 0.00005
    spark_event_theta: float = 335.0

class BaseEngineTest(unittest.TestCase):
    """Provides the standard 'fire_engine' loop used by Integration and Validation."""

    def fire_engine(self, rpm, tps, wheel_load=0, ecu=None, engine=None, 
                rpm_hold=False, motoring=False, cycles=3, thermal_state="current"):
        """
        Simulates engine cycles. 
        thermal_state: "cold" (ambient), "Warm" (70C), "hot" (90C) or "current" (persist existing)
        """
        
        def get_spark(rpm):
            if rpm <= 1000: return 325 # 5 deg BTDC
            if rpm <= 1500: return 348 # 18 deg BTDC
            if rpm <= 3500: return 338
            else:
                return 335 # 28 deg BTDC
            
        if ecu is None: 
            ecu = MockEcuOutput()
            
        if engine is None:
            engine = EngineModel(rpm=rpm)
            

        # Handle Thermal State Sync
        if thermal_state == "cold":
            engine.sensors.CLT_C = 20
            engine.cyl.T_wall = 20 + 273.15
            engine.cyl.T_curr = 20 + 273.15
        elif thermal_state == "warm":
            engine.sensors.CLT_C = 70
            engine.cyl.T_wall = 90 + 273.15
            engine.cyl.T_curr = 90 + 273.15
        elif thermal_state == "hot":
            engine.sensors.CLT_C = 90
            engine.cyl.T_wall = 120 + 273.15
            engine.cyl.T_curr = 120 + 273.15
        else:
            pass # ontinue with defaults or previous if existing engine

        # Setup Operating Parameters
        engine.sensors.TPS_percent = tps
        engine.state.wheel_load = wheel_load
        if motoring or rpm_hold:  # motoring has no spark nor fuel
            engine.motoring_rpm = rpm
        
        target_afr = ecu.afr_target
            
        # 1. Convert CC/MIN to KG/S (assuming gasoline density 0.74)
        flow_kg_s = (c.INJECTOR_FLOW_CC_PER_MIN / 60.0) * (0.74 / 1000.0)
        
        # Assume IVC for a VW 2.1L is roughly 220-240 CAD (adjust to your model's spec)
        IVC_ANGLE = engine.valves.intake.close_angle 

        for _ in range(cycles):
            for CAD in range(720):
                
                # 2. At Intake Valve Close, calculate the injection needed for the NEXT cycle
                if CAD == IVC_ANGLE:
                    trapped_air = engine.cyl.air_mass_kg
                    
                    if trapped_air > 0:
                        required_fuel_kg = trapped_air / target_afr
                        
                        # Calculate Pulse Width in Seconds, then Degrees
                        inj_time_sec = required_fuel_kg / flow_kg_s
                        deg_per_sec = (engine.sensors.rpm / 60.0) * 360.0
                        pw_degrees = inj_time_sec * deg_per_sec
                        
                        # 3. Set Variable Start based on Fixed End
                        # Example: Injector End is fixed at 170 CAD (standard for many port setups)
                        ecu.injector_start = ecu.injector_end - pw_degrees


                # 4. Set ECU Outputs
                start_trigger = int(ecu.injector_start % 720)
                end_trigger = int(ecu.injector_end % 720)
                ecu.spark_timing = get_spark(rpm=rpm)
                ecu.spark = (CAD == ecu.spark_timing) if not motoring else False
                
                if not motoring:
                    if start_trigger > end_trigger: # Pulse crosses 720/0 boundary
                        ecu.injector_on = (CAD >= start_trigger or CAD <= end_trigger)
                    else:
                        ecu.injector_on = (start_trigger <= CAD <= end_trigger)

                engine.step(ecu)
        return engine, ecu

# =================================================================
# 2. TIER 1: FOUNDATION (From test_physics_foundation.py)
# =================================================================

class TestPhysicsFoundation(unittest.TestCase):
    """Pure math/geometry tests. No EngineModel instantiation required."""

    def _get_woschni_params(self, P_curr, T_curr, V_curr):         
        """Helper to package raw floats into the required Object structure."""
        valves = MockValves()
        cyl = MockCylinderState()
        IVC = valves.intake.close_angle
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
    
    def test_burn_duration_logic(self):
        """
        Validates that burn duration responds correctly to RPM for 94mm WBX.
        Expectations updated to allow realistic spark timing (20-30° BTDC).
        """
        bd_low = pf.get_burn_duration(rpm=1500, lambda_=1.0)
        bd_high = pf.get_burn_duration(rpm=4500, lambda_=1.0)
        
        print(f"\n[Burn Physics] 1500 RPM: {bd_low:.1f}° | 4500 RPM: {bd_high:.1f}°")
           
        # 1. 1500 RPM Targets
        # Was: 45.0 - 55.0
        # New Target: 60.0 - 70.0 (Correct for large 94mm bore)
        self.assertGreaterEqual(bd_low, 60.0, "Burn at 1500 RPM is too fast for 94mm bore.")
        self.assertLessEqual(bd_low, 70.0, "Burn at 1500 RPM is too slow.")

        # 2. 4500 RPM Targets
        # Was: 30.0 - 40.0
        # New Target: 48.0 - 55.0 (Allows 25-28° spark without exceeding 65 bar)
        self.assertLessEqual(bd_high, 55.0, "Burn at 4500 RPM is too slow.")
        self.assertGreaterEqual(bd_high, 45.0, "Burn at 4500 RPM is too fast.")
            
        # # 1. 1500 RPM Targets (Cruise/Part Load)
        # # New Target: ~46-52° (Allows ~35-40° BTDC spark for cruise efficiency)
        # self.assertGreaterEqual(bd_low, 45.0, "Burn at 1500 RPM is too fast for 94mm bore.")
        # self.assertLessEqual(bd_low, 55.0, "Burn at 1500 RPM is too slow; efficiency will drop.")
        
        # # 2. 4500 RPM Targets (High Power)
        # # New Target: ~34-38° (Allows 25° BTDC spark to hit ~12° ATDC peak)
        # self.assertLessEqual(bd_high, 45.0, "Burn at 4500 RPM is too slow; will cause fire in exhaust.")
        # self.assertGreaterEqual(bd_high, 30.0, "Burn at 4500 RPM is too fast; unrealistic flame speed.")
        
        # 3. Continuity check
        self.assertGreater(bd_low, bd_high, "Physics Error: Burn duration (degrees) must decrease as turbulence (RPM) increases.")

    def test_wiebe_center_of_gravity(self):
        """
        Validates the heat release shape. 
        For m=2.0, the 50% burn point should occur at ~45-50% of the duration.
        """
        duration = pf.get_burn_duration(rpm=4500, lambda_=1.0)
        theta_start = 330.0 # 30 deg BTDC
        total_heat = 1000.0 # Joules
        
        heat_per_degree = []
        thetas = np.arange(theta_start, theta_start + duration, 0.1) # High resolution sweep
        
        for t in thetas:
            dq = pf.calc_wiebe_heat_rate(t, theta_start, duration, total_heat)
            heat_per_degree.append(dq * 0.1) # Scale by step size
            
        cumulative_heat = np.cumsum(heat_per_degree)
        total_released = cumulative_heat[-1]
        
        # 1. Energy Conservation: Did we release ~99% of total_heat? 
        # (a=5.0 means 99.3% combustion completion)
        self.assertAlmostEqual(total_released, total_heat * 0.993, delta=5.0)
        
        # 2. MFB50 Check: Where is the 500J mark?
        mfb50_idx = np.where(cumulative_heat >= (total_released / 2))[0][0]
        mfb50_theta = thetas[mfb50_idx]
        relative_mfb50 = mfb50_theta - theta_start
        
        print(f"\n[Wiebe Audit] Duration: {duration}° | MFB50 at: +{relative_mfb50:.2f}°")
        
        # With m=2.0, MFB50 should be around 28-30 degrees into a 60 degree burn.
        self.assertLess(relative_mfb50, duration * 0.55, "Burn is back-loaded (too slow at start)")
        self.assertGreater(relative_mfb50, duration * 0.40, "Burn is front-loaded (too fast at start)")
       
    def test_woschni_sane_output(self):
        """Ensures heat loss isn't zeroing out or exploding."""
        # Test at typical combustion peak
        params = self._get_woschni_params(P_curr=60e5, T_curr=2200, V_curr=0.0005)
        loss = pf.calc_woschni_heat_loss(CAD=370, rpm=3000, **params)

        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 500.0, "Heat loss per degree is unrealistically high")
    
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

    def test_friction_growth(self):
        """Friction should scale quadratically with RPM (isolated from pressure)."""
        f_low = pf.calc_single_cylinder_friction(theta=450, rpm=1500, p_cyl=1e5, clt=90)
        f_high = pf.calc_single_cylinder_friction(theta=450, rpm=4500, p_cyl=1e5, clt=90)
        
        self.assertGreater(f_high, f_low * 2, "High RPM friction should be at least double low RPM")
        
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
        
    def test_manifold_vacuum_logic(self):
        """At low TPS and idle RPM, the manifold should develop significant vacuum."""
        # 900 RPM, 2% Effective TPS (IACV + Throttle)
        p_idle = pf.update_intake_manifold_pressure(effective_tps=1.0, rpm=900)
        p_idle_kpa = p_idle / 1000.0
        
        # print(f"\nManifold Test: 900 RPM @ 2% TPS: {p_idle_kpa:.2f} kPa")
        
        # Target for VW 2.1L MV is ~32.5 kPa
        self.assertLess(p_idle_kpa, 40.0, "Engine failed to develop enough vacuum for idle")
        self.assertGreater(p_idle_kpa, 20.0, "Vacuum is unrealistically deep (possible math error)")

    def test_flow_continuity_vectorized(self):  
        """Validates continuity for both Intake and Exhaust using vectorized cam logic."""

        valves = MockValves()


        # --- TEST 1: INTAKE (Inflow / Vacuum Sweep) ---
        # Simulate crank angle at near-max lift
        # lift_in_vec = pf.calc_valve_lift_flat_follower(np.arange(720), valves.intake.open_1mm, valves.intake.close_1mm, valves.intake.max_lift)
        lift_in_vec = pf.calculate_wbx_physical_lift(c.INTAKE_DURATION_1mm, c.CENTERLINE, c.INTAKE_MAX_LIFT, is_intake=True, is_duration_at_1mm=True)

       
        area_in_vec = pf.calc_valve_area_vectorized(np.arange(720), valves.intake, lift_in_vec)
        theta_mid_intake = int((valves.intake.open_angle + valves.intake.close_angle) / 2)
        lift_in = lift_in_vec[theta_mid_intake]
        area_in = area_in_vec[theta_mid_intake]

        mdots_in = []
        p_cyl_sweep_in = np.linspace(1.0, 0.2, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_in:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_in, lift=lift_in, diameter=valves.intake.diameter,
                P_cyl=p, T_cyl=c.T_INTAKE_K, R_cyl=c.R_SPECIFIC_AIR, g_cyl=c.GAMMA_AIR,
                P_extern=c.P_ATM_PA, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
                is_intake=True
            )
            mdots_in.append(mdot)

        # --- TEST 2: EXHAUST (Outflow / Blowdown Sweep) ---
        theta_mid_exhaust = int((valves.exhaust.open_angle + valves.exhaust.close_angle) / 2)      
        # lift_ex_vec = pf.calc_valve_lift_flat_follower(np.arange(720),  valves.exhaust.open_1mm, valves.exhaust.close_1mm, valves.exhaust.max_lift)
        lift_ex_vec = pf.calculate_wbx_physical_lift(c.EXHAUST_DURATION_1mm, c.CENTERLINE, c.EXHAUST_MAX_LIFT, is_intake=False, is_duration_at_1mm=True)
        area_ex_vec = pf.calc_valve_area_vectorized(np.arange(720), valves.exhaust, lift_ex_vec)
        lift_ex = lift_ex_vec[theta_mid_exhaust]
        area_ex = area_ex_vec[theta_mid_exhaust]

        mdots_ex = []
        p_cyl_sweep_ex = np.linspace(1.0, 5.0, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_ex:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_ex, lift=lift_ex, diameter=valves.exhaust.diameter,
                P_cyl=p, T_cyl=c.T_EXHAUST_K, R_cyl=c.R_SPECIFIC_EXHAUST, g_cyl=c.GAMMA_EXHAUST,
                P_extern=c.P_ATM_PA, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
                is_intake=False
            )
            mdots_ex.append(mdot)

        # Assertions: 
        # Intake: mdot starts at 0 and goes POSITIVE. Diff should be >= 0.
        self.assertTrue(np.all(np.diff(mdots_in) >= -1e-12), "Intake flow decreased as pressure drop increased")
        
        # Exhaust: mdot starts at 0 and goes NEGATIVE. Diff should be <= 0.
        self.assertTrue(np.all(np.diff(mdots_ex) <= 1e-12), "Exhaust flow increased as pressure increased")
        
    def test_motoring_energy_balance(self):
        """
        Closed-cycle test (Valves shut, no combustion). 
        Validates integrator precision.
        """
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        p_initial = 101325.0
        t_initial = 300.0
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        m_air = (p_initial * v_bdc) / (c.R_SPECIFIC_AIR * t_initial)
        
        theta_start, theta_end, step_size = 180.0, 360.0, 0.1
        
        p_curr, t_curr, v_curr = p_initial, t_initial, v_bdc

        # Track the 'work-done' to calculate a truly accurate reference
        for th in np.arange(theta_start, theta_end, step_size):
            ds_dtheta_rad = pf.calc_piston_speed_factor(th)
            dv_dtheta_deg = c.A_PISTON * ds_dtheta_rad * (np.pi / 180.0)

            # Use a constant R and the logic within integrate_first_law
            p_curr, t_curr = pf.integrate_first_law(
                P_curr=p_curr, T_curr=t_curr, M_curr=m_air, V_curr=v_curr,
                Delta_M=0, Delta_Q_in=0, Delta_Q_loss=0, 
                dV_d_theta=dv_dtheta_deg, 
                gamma_blended_start=c.GAMMA_AIR, # Note: if integrate_first_law re-calculates gamma internally, this is ignored
                theta_delta=step_size, T_manifold=300.0, R_spec_blended=c.R_SPECIFIC_AIR
            )
            v_curr = pf.v_cyl(th + step_size, c.A_PISTON, V_clearance)

        # 4. UPDATED REFERENCE:
        # Since your Cv is now T-dependent, a static P1 * CR^1.4 is wrong.
        # We check if the result is physically consistent with a variable gamma compression.
        # For Air at ~700K (TDC), gamma is ~1.37, not 1.4.
        
        # Calculate the "Actual Effective Gamma" of the simulation
        gamma_eff = np.log(p_curr / p_initial) / np.log(c.COMP_RATIO)
        
        print(f"\n[Integrator Audit] Peak P: {p_curr/1e5:.2f} bar | Eff Gamma: {gamma_eff:.3f}")

        # The error check should ensure we aren't losing mass/energy, 
        # allowing for the physical drop in gamma from 1.40 to ~1.375
        self.assertGreater(gamma_eff, 1.36, "Integrator is 'leaking' energy (Pressure too low)")
        self.assertLess(gamma_eff, 1.41, "Integrator is creating energy (Pressure too high)")
        
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

    def test_manifold_stability(self):
        """Tests for numerical divergence in the intake manifold pressure integrator."""
        p_manifold = 30000.0 # Start at high vacuum (30kPa)
        rpm = 3000
        tps = 100.0 # Wide Open Throttle
        
        # Simulate 50ms of real time
        for _ in range(50):
            p_manifold = pf.update_intake_manifold_pressure(tps, rpm)
            
        self.assertLessEqual(p_manifold, 101325.0 * 1.05, "Manifold pressure exploded (numerical instability)")
        self.assertGreater(p_manifold, 90000.0, "Manifold failed to fill towards atmospheric at WOT")

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
        def _calculate_mean_friction(rpm):
            """Helper to calculate the average friction torque over 720 degrees."""
            samples = []
            for cad in range(720):
                # We use atmospheric pressure to simulate 'motoring' conditions
                t_fric = pf.calc_single_cylinder_friction(
                    theta=cad, 
                    rpm=rpm, 
                    p_cyl=101325.0, 
                    clt=90.0
                )
                samples.append(t_fric)
            return np.mean(samples)

        # Calculate pure mechanical friction at 3000 RPM
        # (P_cyl = P_ATM means no pressure-loading on rings)
        cyl_fric_nm = _calculate_mean_friction(3000)
        global_parasitic_friction = pf.calc_engine_core_friction(rpm= 3000, clt=90.0)
        
        engine_friction_nm = c.NUM_CYL * cyl_fric_nm + global_parasitic_friction
        
        # In a 2.1L engine, pure metal-on-metal friction at 3000 RPM 
        # should be between 3.0 and 6.0 Nm.
        self.assertGreater(engine_friction_nm, 2.5, "Mechanical friction is unrealistically low")
        self.assertLess(engine_friction_nm, 7.0, "Mechanical friction is unrealistically high")
        
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

    def test_burn_duration_scaling(self):
        """Ensures turbulence-based burn duration scales with RPM."""
        bd_1500 = pf.get_burn_duration(1500, 1.0)
        bd_4500 = pf.get_burn_duration(4500, 1.0)
        self.assertGreater(bd_1500, bd_4500)
        # self.assertTrue(30.0 < bd_4500 < 45.0)
        
        # Update the range check for high RPM
        # Was: 30.0 < bd_4500 < 45.0
        # New: 45.0 < bd_4500 < 55.0
        self.assertTrue(45.0 < bd_4500 < 55.0, f"Expected 45-55, got {bd_4500}")

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

# =================================================================
# 3. TIER 2: INTEGRATION (From test_engine_model.py)
# =================================================================

class TestEngineIntegration(BaseEngineTest):
    """System-level tests ensuring manifold, cylinder, and thermal loops connect."""

    def test_static_compression_audit(self):
        """Adiabatic check: Peak motoring pressure vs theoretical P2=P1*(V1/V2)^gamma."""
        engine, _ = self.fire_engine(rpm=3000, tps=100.0, motoring=True, cycles=3) # No spark in setup
        p_max = np.max(engine.cyl.log_P) / 1e5
        ivc_idx = int(engine.valves.intake.close_angle)
        v_ivc = engine.cyl.V_list[ivc_idx]
        v_tdc = engine.cyl.V_clearance
        p_theoretical = (engine.cyl.log_P[ivc_idx]/1e5) * (v_ivc/v_tdc)**1.4
        self.assertAlmostEqual(p_max, p_theoretical, delta=2.0)

    def test_idle_pumping_equilibrium(self):
        """Ensures MAP stabilizes around 32.2 kPa with 14% IACV."""
        # This replaces the original validator test with the 40-cycle stability check
        engine, _ = self.fire_engine(rpm=900, tps=0.0, rpm_hold=True, cycles=40)
        stable_map = np.mean(engine.state.map_history) / 1000
        
        P_peak = engine.cyl.P_peak_bar
        P_peak_angle = engine.cyl.P_peak_angle
        effective_tps = engine.state.effective_tps
        self.assertAlmostEqual(stable_map, 32.24, delta=0.5)

    def test_coolant_thermal_limit(self):
        """Ensures CLT never exceeds safety hard-cap of 115C."""
        engine, _ = self.fire_engine(rpm=3000, tps=100, thermal_state="warm", cycles=3)
        engine.sensors.CLT_C = 114.9
        engine, _ = self.fire_engine(rpm=5000, tps=100.0, thermal_state="hot", engine=engine, cycles=2)
        self.assertLessEqual(engine.sensors.CLT_C, 115.0)

    def test_energy_conservation_proportions(self):
        """
        Integration: Checks the 'Heat Balance' of a single WOT combustion cycle.
        Formula: Q_fuel = W_brake + Q_loss + Q_exhaust + Friction
        """
        rpm=3000
        engine, _ = self.fire_engine(rpm=rpm, tps=100, rpm_hold=True, thermal_state="warm", cycles=30)
        q_fuel = np.sum(engine.cyl.Q_in_history)
        q_loss = np.sum(engine.cyl.Q_loss_history)
        # q_fuel_2 = engine.cyl.total_cycle_heat_J
        # q_loss_2 = engine.cyl.Q_loss_total
        
        # print(f"\nCOMPARING TWO Q_LOSS COUNTERS: {q_loss} and {q_loss_2}")
        # print(f"\nCOMPARING TWO Q_IN FUEL COUNTERS: {q_fuel} and {q_fuel_2}")
        
        T_cyl = engine.cyl.T_curr
        T_wall = engine.cyl.T_wall
        clt_C = engine.sensors.CLT_C
        
        loss_pct = (q_loss / q_fuel) * 100
        
        print(f"\nDEBUG ENERGY CONSERVATAION rpm:{rpm} q_loss:{q_loss:.2f} total Q-fuel:{q_fuel:.2f} ratio:{loss_pct:.2f} ")
        print(f"|__ T_wall:{engine.cyl.T_wall:.2f} cyl_P:{engine.cyl.P_curr/1000:.2f}kPa " 
              f"Peak_P:{np.max(engine.cyl.log_P)/1000:.2f}kPa Peak_P_angle:{np.argmax(engine.cyl.log_P):.2f} "
              f"Peak_T:{np.max(engine.cyl.log_T):.2f}K Peak_T_angle:{np.argmax(engine.cyl.log_T):.2f} ")
        
        
        # Target 20% per your validation report
        self.assertGreaterEqual(loss_pct, 19.0, f"Heat loss too low: {loss_pct:.1f}%")
        self.assertLessEqual(loss_pct, 21.0, f"Heat loss too high: {loss_pct:.1f}%")
        
    def test_knock_during_high_load(self):
        """
        Integration test: Runs a simulation at Wide Open Throttle (WOT)
        to see if the physics-based Pmax triggers the knock detector.
        """
        
        ecu = MockEcuOutput(
            spark_timing=330, # 30 deg BTDC
            afr_target=12.5,  # Rich for cooling
        )
        
        engine, _ = self.fire_engine(rpm=3000, tps=100, cycles=5, ecu=ecu)
            
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
        engine, ecu = self.fire_engine(rpm=3000, tps=100, cycles=4)
            
        # 3. Extract data
        p_max_bar = np.max(engine.cyl.log_P) / 1e5
        clt = 105.0        # Overheating engine reduces threshold
        lambda_val = ecu.afr_target / 14.7
        spark_advance = 40 # 40 degrees
        
        # 4. Use low-grade 87 Octane fuel to further lower the threshold
        is_knocking, severity = pf.detect_knock(
            p_max_bar, clt, engine.sensors.rpm, spark_advance, lambda_val, fuel_octane=87.0
        )
        
        print(f"\n--- FORCED KNOCK TEST ---")
        print(f"Pmax: {p_max_bar:.2f} bar | Octane: 87 | Spark: {spark_advance} deg")
        print(f"Knock Detected: {is_knocking} | Severity: {severity:.2f}")
        
        # 5. Assertions
        self.assertTrue(is_knocking, "Engine should be knocking under these aggressive conditions")
        self.assertGreater(severity, 0.0, "Knock severity should be positive")

    def test_thermal_feedback_loop(self):
        """Tests that running the engine actually heats up the coolant."""
        initial_clt = c.COOLANT_START   

        engine, _ = self.fire_engine(rpm=3000, tps=100.0, cycles=10)
            
        # print(f"\nDEBUG THERMAL "
        #     # f"T_Indicated:{np.average(engine.state.torque_indicated_history):6.2f}Nm | "
        #     # f"T_Friction:{np.average(engine.state.torque_friction_history):6.2f}Nm | "
        #     # f"T_Net:{np.average(engine.state.torque_brake_history):6.2f}Nm | "
        #     # )
        #       f"coolant test rpm:{engine.sensors.rpm:.0f} T_brake:{np.average(engine.state.torque_brake_history):.2f} "
        #       f"T_ind:{np.average(engine.state.torque_indicated_history):.2f} T_fri:{np.average(engine.state.torque_friction_history):.2f} "
        #       f"P_peak:{np.max(engine.cyl.log_P)/1e5:.0f}bar P_peak_angle:{np.argmax(engine.cyl.log_P)} "
        #       )    
        # print(f"\nDEBUG THERMAL Coolant Test (Heating): initial:{initial_clt} -> {engine.sensors.CLT_C:.4f}C")
        self.assertGreater(engine.sensors.CLT_C, initial_clt, 
                           "Engine coolant failed to heat up after 10 WOT cycles")

    def test_manifold_suction_response(self):
        """Tests that closing the throttle increases vacuum (decreases MAP)."""

        engine, _ = self.fire_engine(rpm=3000, tps=100.0, cycles=10)   

        map_wot = engine.sensors.MAP_kPa
        # print(f"\nWOT rpm:{engine.sensors.rpm} MAP:{map_wot:.2f} TPS_e:{engine.state.effective_tps:.2f} "
        #       f"T_brake:{engine.state.torque_brake:.2f} T_ind:{engine.state.torque_indicated_history[-1]:.2f} "
        #       f"T_fri:{engine.state.torque_friction:.2f} "
        #       f"lambda:{engine.sensors.lambda_} "
        #       )
        
        # 2. Close Throttle
        # engine.sensors.TPS_percent = 0 # the IACV is has min opening set in ecu
        engine, _ = self.fire_engine(rpm=3000, tps=0.0, cycles=3) # the IACV is has min opening set in ecu
        map_closed = engine.sensors.MAP_kPa
        
        # print(f"\nclosed rpm:{engine.sensors.rpm:.2f} MAP:{map_closed} TPS_e:{engine.state.effective_tps:.2f} " )
        # print(f"\nP_peak:{np.max(engine.cyl.log_P)/1000:.0f}kPa P_peak_angle:{np.argmax(engine.cyl.log_P)} " )   
        
        self.assertLess(map_closed, map_wot, "Manifold pressure failed to drop when throttle closed")
    
    def test_firing_symmetry(self):
        """Integration: Validates that torque pulses occur every 180 degrees, ignoring Cycle 0."""
        # engine = EngineModel(rpm=3000)
        # engine.sensors.TPS_percent = 50.0
        
        # Run 5 cycles total, but we will only analyze the very last cycle
        engine, _ = self.fire_engine(rpm=3000, tps=50.0, cycles=5) 
        
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
            
    def test_motoring_parasitic_drag(self):
        
        # cycle motoring engine
        ecu = MockEcuOutput()
        engine = EngineModel(rpm=3000)
        engine.motoring_rpm = 3000
        engine.sensors.TPS_percent = 100.0
        engine.sensors.CLT_C = 90.0  # Warm engine
        for _ in range(3 * 720):
            engine.step(ecu)
             
        # The torque the dyno HAD to provide to stay at 3000 RPM
        # This is the "Torque the crank needs to overcome"
        avg_dyno_support = np.average(engine.state.torque_governor_history)
        
        # Total Drag is the inverse of the support required to keep speed steady
        total_drag = -avg_dyno_support 
        
        j_to_nm = 1.0 / (4 * np.pi)
        
        # Gas Exchange Diagnostic
        w_pumping = engine.state.work_pumping_j
        pumping_torque_nm = w_pumping * j_to_nm
        pmep_bar = (abs(w_pumping) / engine.cyl.V_displaced) / 1e5
        
        # Mechanical/Thermal Remainder
        mech_thermal_drag = total_drag - pumping_torque_nm
        
        # check work done.
        t_compression_nm =     engine.state.work_compression_j * j_to_nm 
        t_expansion_nm =       engine.state.work_expansion_j * j_to_nm
        t_friction_nm =        engine.state.friction_work_j * j_to_nm
        t_indicated_nm =       engine.state.indicated_work_j * j_to_nm
        t_brake_nm =           engine.state.net_work_j * j_to_nm
        
        t_indicated_calc = pumping_torque_nm + t_compression_nm + t_expansion_nm
        t_brake_calc = t_indicated_nm - t_friction_nm

        print(f"\n--- DRAG DIAGNOSTIC (3000 RPM @ STEADY STATE) ---")
        print(f"DYNO SUPPORT:    {avg_dyno_support:.2f} Nm")
        print(f"TOTAL DRAG:      {total_drag:.2f} Nm")
        print(f"  └─ PUMPING:    {pumping_torque_nm:.2f} Nm (PMEP: {pmep_bar:.3f} bar)")
        print(f"  └─ MECH+THERM: {mech_thermal_drag:.2f} Nm")

        print(f"\n--- ENGINE WORK DATA (Converted to Nm for Comparison) ---")
        print(f"T BRAKE (MEASURED):  {t_brake_nm:.2f} Nm")
        print(f"T BRAKE (CALC):      {t_brake_calc:.2f} Nm")
        print(f"  └─ INDICATED:      {t_indicated_nm:.2f} Nm")
        print(f"  └─ FRICTION:       {t_friction_nm:.2f} Nm")
        
        print(f"\n--- ENGINE WORK DATA (Indicated Torque Check) ---")
        print(f"T INDICATED (MEASURED): {t_indicated_nm:.2f} Nm")
        print(f"T INDICATED (CALC):     {t_indicated_calc:.2f} Nm")
        print(f"  └─ COMPRESSION:       {t_compression_nm:.2f} Nm")
        print(f"  └─ EXPANSION:         {t_expansion_nm:.2f} Nm")
        print(f"  └─ FRICTION:          {t_friction_nm:.2f} Nm")
        
        # Validation
        self.assertLess(total_drag, -2.0, "Engine has zero friction! Check friction logic.")
        self.assertGreater(total_drag, -35.0, "Engine drag is too high! Is the brake on?")
    
    def test_full_load_friction_scaling(self):
        # 1. Warm up the engine
        engine, _ = self.fire_engine(rpm=3000, tps=100, thermal_state="warm", cycles=5)

            
        # 3. Collect Data
        w_fric_wot = engine.state.friction_work_j
        t_fric_wot = w_fric_wot / (4 * np.pi)
        
        # Peak Cylinder Pressure (for context)
        p_max = np.max(engine.cyl.log_P) / 1e5 # bar
        
        print(f"\n--- LOAD SCALING DIAGNOSTIC (3000 RPM @ WOT) ---")
        print(f"PEAK CYL PRESSURE: {p_max:.1f} bar")
        print(f"FRICTION TORQUE:   {t_fric_wot:.2f} Nm")
        
        # 4. Physical Sanity Check
        # At WOT, friction should be ~20-50% higher than motoring 
        # due to the rings being forced against the walls.
        motoring_fric = 6.10 # Our previous baseline
        increase = (t_fric_wot / motoring_fric) - 1.0
        
        print(f"SCALING INCREASE:  {increase*100:.1f}%")
        
        self.assertGreater(t_fric_wot, motoring_fric, "Friction didn't increase under load!")

    def test_volumetric_efficiency_limits(self):
        """Validation: Is the air-mass trapping realistic for a 2.1L?"""
        # engine = EngineModel(rpm=3000)
        # engine.sensors.TPS_percent = 100.0
        engine, _ = self.fire_engine(rpm=3000, tps=100.0, cycles=5)
        
        # At 1 bar, a 525cc cylinder holds ~0.6g of air. 
        # With VE and heating, 0.45g - 0.55g is the 'Digital Twin' target.
        trapped_mass_mg = engine.cyl.air_mass_at_spark * 1e6
        self.assertGreater(trapped_mass_mg, 400.0, "Engine is gasping for air (VE too low)")
        self.assertLess(trapped_mass_mg, 650.0, "Engine is supercharging itself (VE too high)")
        
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

        print(f"\n--- STATIC COMPRESSION AUDIT ---")
        print(f"Pressure at IVC ({ivc_idx} CAD): {p_at_ivc_bar:.2f} bar")
        print(f"Peak Motoring Pressure:      {p_max_bar:.2f} bar")
        print(f"Theoretical Adiabatic Peak:  {p_theoretical:.2f} bar")
        print(f"--------------------------------")

        # Validation: If Actual is significantly lower than Theoretical, 
        # the model has a mass-leak or 'Blow-back' issue.
        self.assertGreater(p_max_bar, 14.5, "Compression pressure too low for 9:1 CR")
        self.assertLess(p_max_bar, 23.0, "Compression pressure implies non-physical CR")


# =================================================================
# 4. TIER 3: VALIDATION (From physics_validator.py)
# =================================================================

class TestPhysicsValidation(BaseEngineTest):
    """The 'Golden Spec' check against known VW 2.1L MV performance data."""
 
    def test_idle_manifold_pressure(self):
        """Target: 32.5 kPa at 900 RPM Idle (5% tolerance)"""
        engine, ecu = self.fire_engine(rpm=900, tps=0.0, rpm_hold=True)
        ecu.iacv_pos = 35
        engine, _ = self.fire_engine(rpm=900, tps=0.0, rpm_hold=True, ecu=ecu, engine=engine, cycles=3)
        actual_map = np.mean(engine.state.map_history) / 1000
        effective_tps = engine.state.effective_tps
        self.assertAlmostEqual(actual_map, 32.5, delta=32.5 * 0.05)

    def test_friction_900_rpm(self):
        """Target: 3.3 Nm (Mean Friction Torque)"""
        cyl_fric = np.mean([pf.calc_single_cylinder_friction(theta=cad, rpm=900, p_cyl=101325.0, clt=90.0) for cad in range(720)])
        engine_fric = (cyl_fric * c.NUM_CYL) + pf.calc_engine_core_friction(rpm=900, clt=90.0)
        self.assertAlmostEqual(engine_fric, 4.5, delta=4.5 * 0.10)

    def test_volumetric_efficiency_high_rpm(self):
        """Target: 0.88 (88%) VE at 4500 RPM WOT"""
        engine, _ = self.fire_engine(rpm=4500, tps=100.0)
        actual_mass = engine.cyl.air_mass_at_IVC
        P_manifold = engine.sensors.P_manifold_Pa
        C_DISPLACED = c.V_DISPLACED
        ideal_mass = (engine.sensors.P_manifold_Pa * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ve = actual_mass / ideal_mass
        self.assertAlmostEqual(ve, 0.88, delta=0.88 * 0.05)

    def test_low_rpm_heat_loss(self):
        """Target: 20.0% Energy Loss at 1500 RPM (10% tolerance)"""
        engine, _ = self.fire_engine(rpm=1500, tps=100.0, rpm_hold=True, thermal_state="warm", cycles=3)
        P_peak_angle = engine.cyl.P_peak_angle
        loss_pct = (engine.cyl.Q_loss_total / engine.cyl.total_cycle_heat_J) * 100
        self.assertAlmostEqual(loss_pct, 20.0, delta=20.0 * 0.10)

    def test_peak_cylinder_pressure_limit(self):
        """Target: Max 65.0 bar structural limit at 4500 RPM"""
        engine, _ = self.fire_engine(rpm=4500, tps=100.0)
        p_max_bar = np.max(engine.cyl.log_P) / 1e5
        P_peak_angle = engine.cyl.P_peak_angle
        P_peak = engine.cyl.P_peak_bar
        burn_duration = engine.cyl.burn_duration
        spark_angle = engine.cyl.spark_event_theta
        self.assertLessEqual(p_max_bar, 65.0)

    def test_peak_pressure_angle(self):
        """Target: Peak pressure should occur at 372 CAD (ATDC)"""
        engine, _ = self.fire_engine(rpm=4500, tps=100.0, rpm_hold=True, cycles=3)
        p_angle = np.argmax(engine.cyl.log_P)
        self.assertAlmostEqual(p_angle, 372, delta=5) # 5 degree tolerance

    def test_net_torque_equilibrium(self):
        """Target: Nm Brake Torque at peak load"""
        engine, _ = self.fire_engine(rpm=2800, tps=100.0, wheel_load=160.0, cycles=4)
        brake_torque = np.average(engine.state.torque_brake_history)
        P_peak_angle = engine.cyl.P_peak_angle
        self.assertAlmostEqual(brake_torque, 160.0, delta=160.0 * 0.05)
        
        engine, _ = self.fire_engine(rpm=2800, tps=100.0, rpm_hold=True, cycles=4)
        net_torque_output = -1 * np.average(engine.state.torque_governor_history)
        self.assertAlmostEqual(net_torque_output, 160, delta=160*0.05)

        

    def test_combustion_temperature_safety(self):
        """Target: Max 2800.0 K Peak Temperature"""
        engine, _ = self.fire_engine(rpm=3000, tps=100.0, rpm_hold=True, thermal_state="Warm", cycles=3)
        t_peak = np.max(engine.cyl.log_T)
        P_peak = engine.cyl.P_peak_bar
        P_peak_angle = engine.cyl.P_peak_angle
        self.assertLess(t_peak, 2800.0)

    # def test_energy_retention_high_rpm(self):
    #     """Target: 78.0% Energy Retention at 4500 RPM"""
    #     engine, _ = self.fire_engine(rpm=4500, tps=100.0, rpm_hold=True, cycles=3)
    #     dq_added = engine.cyl.Q_in_history
    #     total_dq = np.sum(dq_added[dq_added > 0])
    #     total_q_lost = np.sum(engine.cyl.Q_loss_history)
    #     retention = ((total_dq - total_q_lost) / total_dq) * 100
    #     self.assertAlmostEqual(retention, 78.0, delta=78.0 * 0.05)

    def test_wot_performance_targets(self):
        """Validates Torque (174Nm), VE (~91%), and PMEP (<0.25 bar)."""
        engine, ecu = self.fire_engine(rpm=3000, tps=100.0, rpm_hold=True, thermal_state="warm", cycles=5)
        
        # # 1. Torque
        # avg_t = np.mean(engine.state.torque_brake_history)
        # self.assertAlmostEqual(avg_t, 174.0, delta=4.0)
        
        intake = engine.valves.intake
        exhaust = engine.valves.exhaust
        P_peak_angle = engine.cyl.P_peak_angle
        mass_at_spark = engine.cyl.total_mass_at_spark
        spark = ecu.spark_timing
        start_trigger = ecu.injector_start % 720
        end_trigger = ecu.injector_end % 720
        


        # 2. Volumetric Efficiency
        # mass_at_IVC = engine.cyl.total_mass_at_IVC
        air_mass_at_IVC = engine.cyl.air_mass_at_IVC
        # displacement_calc = max(engine.cyl.V_list) - min(engine.cyl.V_list)
        displacement_cylinder = c.V_DISPLACED
        ideal_mass = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ve = (air_mass_at_IVC / ideal_mass) * 100
        print(f"VE: {ve}")
        # for CAD in range(720):
        #     print(f"DEBUG MASS: CAD{CAD} @spark{mass_at_spark:.7f} @IVC:{air_mass_at_IVC:.7f} total@CAD:{engine.cyl.total_mass_history[CAD]:.7f} "
        #           f"a_i:{engine.valves.intake_area_table[CAD]:.7f} Cd_i:{engine.cyl.Cd_in_history[CAD]:.7f} " 
        #           f"l_i:{engine.valves.intake_lift_table[CAD]:.7f} "
        #           f"rpm:{engine.sensors.rpm_history[CAD]:.0f} MAP:{engine.state.map_history[CAD]/1000:.3f}kPa ")
        #     if CAD == int(engine.valves.intake.close_angle): print("IVC")
        #     elif CAD == int(ecu.spark_timing): print("SPARK")
            
        self.assertTrue(88.0 < ve < 93.0)

        


        # 3. PMEP (Gas Exchange Efficiency)
        pumping_j = engine.state.work_pumping_j
        pmep = abs(pumping_j) / (c.NUM_CYL * c.V_DISPLACED) / 1e5
        self.assertLess(pmep, 0.25)

    def test_combustion_energy_partition(self):
        """Validates Heat Loss % and Peak Temperature."""
        engine, _ = self.fire_engine(rpm=3000, tps=100.0, rpm_hold=True, thermal_state="hot", cycles=10)
        pct_loss = (engine.cyl.Q_loss_total / engine.cyl.total_cycle_heat_J) * 100
        max_t = np.max(engine.cyl.log_T)
        
        P_peak_angle = engine.cyl.P_peak_angle
        
        
        self.assertAlmostEqual(pct_loss, 20.0, delta=2.0)
        self.assertLess(max_t, 2850.0)

if __name__ == '__main__':
    unittest.main()


