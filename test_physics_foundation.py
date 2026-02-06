import unittest
import numpy as np
import physics_functions as pf
import constants as c

from dataclasses import dataclass, field
import numpy as np

@dataclass
class MockValve:
    open_angle: float
    close_angle: float

@dataclass
class MockValves:
    intake: MockValve = field(default_factory=lambda: MockValve(15, 202))
    exhaust: MockValve = field(default_factory=lambda: MockValve(517, 692))

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

class TestPhysicsFoundation(unittest.TestCase):
    
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
        Validates that burn duration responds correctly to RPM.
        In real physics, lower RPM = less turbulence = SLOWER burn (more degrees).
        """
        bd_low = pf.get_burn_duration(rpm=1500, lambda_=1.0)
        bd_high = pf.get_burn_duration(rpm=4500, lambda_=1.0)
        
        # print(f"\nBurn Duration Test: 1500 RPM: {bd_low:.2f}° | 4500 RPM: {bd_high:.2f}°")
        
        # This test will likely FAIL with your current code, 
        # revealing why low-end torque is too high.
        self.assertGreater(bd_low, bd_high, "Burn should take more degrees at low RPM")

    def test_woschni_sane_output(self):
        """Ensures heat loss isn't zeroing out or exploding."""
        # Test at typical combustion peak
        params = self._get_woschni_params(P_curr=60e5, T_curr=2200, V_curr=0.0005)
        loss = pf.calc_woschni_heat_loss(CAD=370, rpm=3000, **params)
        
        #     theta=370, rpm=3000, P_curr=60e5, T_curr=2200, 
        #     V_curr=0.0005, T_wall=450, V_clearance=0.00005, theta_delta=1.0
        # )
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
        
        
        # # Common state: Peak combustion (370 deg, 60 bar, 2200K)
        # common_args = {
        #     "theta": 370, "P_curr": 60e5, "T_curr": 2200, 
        #     "V_curr": 0.0005, "T_wall": 450, "V_clearance": V_clearance, 
        #     "theta_delta": 1.0
        # }
        
        # loss_low = pf.calc_woschni_heat_loss(rpm=1500, **common_args)
        # loss_high = pf.calc_woschni_heat_loss(rpm=4500, **common_args)
        
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
        
        # print(f"\nWall Temp Test: Initial: {initial_wall:.2f}K | After 1 Cycle: {new_wall:.2f}K")
        
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
        
        # 1. Setup Valve Objects per constants.py
        class MockValve:
            def __init__(self, v_type):
                self.diameter = c.VALVE_TIMING[v_type]['diameter'] / 1000.0
                self.max_lift = c.VALVE_TIMING[v_type]['max_lift'] / 1000.0
                self.open_angle = c.VALVE_TIMING[v_type]['open']
                self.close_angle = c.VALVE_TIMING[v_type]['close']

        intake_v = MockValve('intake')
        exhaust_v = MockValve('exhaust')

        # --- TEST 1: INTAKE (Inflow / Vacuum Sweep) ---
        # Simulate crank angle at near-max lift
        theta_mid_intake = (intake_v.open_angle + intake_v.close_angle) / 2
        area_in = pf.calc_valve_area_vectorized(np.array([theta_mid_intake]), intake_v)[0]
        lift_in = pf.calc_valve_lift_vectorized(np.array([theta_mid_intake]), intake_v)[0]

        mdots_in = []
        p_cyl_sweep_in = np.linspace(1.0, 0.2, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_in:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_in, lift=lift_in, diameter=intake_v.diameter,
                P_cyl=p, T_cyl=c.T_INTAKE_K, R_cyl=c.R_SPECIFIC_AIR, g_cyl=c.GAMMA_AIR,
                P_extern=c.P_ATM_PA, T_extern=c.T_AMBIENT, R_extern=c.R_SPECIFIC_AIR, g_extern=c.GAMMA_AIR,
                is_intake=True
            )
            mdots_in.append(mdot)

        # --- TEST 2: EXHAUST (Outflow / Blowdown Sweep) ---
        theta_mid_exhaust = (exhaust_v.open_angle + exhaust_v.close_angle) / 2
        area_ex = pf.calc_valve_area_vectorized(np.array([theta_mid_exhaust]), exhaust_v)[0]
        lift_ex = pf.calc_valve_lift_vectorized(np.array([theta_mid_exhaust]), exhaust_v)[0]

        mdots_ex = []
        p_cyl_sweep_ex = np.linspace(1.0, 5.0, 50) * c.P_ATM_PA
        for p in p_cyl_sweep_ex:
            mdot, _ = pf.calc_isentropic_flow(
                A_valve=area_ex, lift=lift_ex, diameter=exhaust_v.diameter,
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
        
    # def test_choked_flow_continuity(self):
    #     """
    #     Ensures mass flow continuity across the choked transition using R_mix.
    #     """
    #     a_valve = 0.0005  # m^2 (approx 25mm diameter effective area)
    #     rpm = 3000
    #     p_up = 101325.0   # 1 bar atmospheric
    #     t_up = 293.15     # 20°C
        
    #     # 1. Calculate R_mix for a Stoichiometric Mixture (Lambda 1.0)
    #     # AFR for gasoline is 14.7:1
    #     r_air = 287.05
    #     r_fuel_gas = 66.0  # Approx R for vaporized C8H18
    #     mass_air = 14.7
    #     mass_fuel = 1.0
    #     r_mix = (mass_air * r_air + mass_fuel * r_fuel_gas) / (mass_air + mass_fuel)
        
    #     # 2. Sweep pressure ratios (P_down / P_up)
    #     # The critical ratio for air (gamma=1.4) is ~0.528
    #     ratios = [0.8, 0.7, 0.6, 0.528, 0.4, 0.3]
    #     flows = []
        
    #     for r in ratios:
    #         p_down = p_up * r
    #         # Using the base flow function with the calculated R_mix
    #         mdot = pf.calc_mass_flow_base(
    #             A_valve=a_valve,
    #             P_up=p_up,
    #             T_up=t_up,
    #             P_down=p_down,
    #             R_spec=r_mix, # Dynamic mix constant
    #             gamma=c.GAMMA_AIR,
    #             rpm=rpm,
    #             Cd=0.7
    #         )
    #         flows.append(mdot)
            
    #     # 3. Assertions for numerical stability
    #     for i in range(len(flows)-1):
    #         # Flow magnitude should never decrease as P_down drops
    #         self.assertGreaterEqual(
    #             abs(flows[i+1]), 
    #             abs(flows[i]) * 0.999, # Allow tiny float tolerance
    #             f"Flow discontinuity detected at ratio {ratios[i+1]}"
    #         )
        
    #     # Once choked (ratio < 0.528), flow should remain constant
    #     self.assertAlmostEqual(flows[-1], flows[-2], places=5, 
    #                         msg="Choked flow is not capping correctly.")

    def test_motoring_energy_balance(self):
        """
        Closed-cycle test (Valves shut, no combustion). 
        Validates that the integrator follows the isentropic curve (P*V^gamma = const).
        """
        V_clearance = c.V_DISPLACED / (c.COMP_RATIO - 1.0)
        # 1. Setup Initial State (BDC)
        p_initial = 101325.0
        t_initial = 300.0
        # Use c.A_PISTON and c.R_SPECIFIC_AIR per previous attribute fixes
        v_bdc = pf.v_cyl(180, c.A_PISTON, V_clearance)
        m_air = (p_initial * v_bdc) / (c.R_SPECIFIC_AIR * t_initial)
        
        # 2. Setup Loop Parameters
        theta_start = 180.0  # BDC
        theta_end = 360.0    # TDC (Compression)
        step_size = 0.1      # Small steps are required for numerical convergence
        
        p_curr = p_initial
        t_curr = t_initial
        v_curr = v_bdc

        # 3. Integrate Cycle
        for th in np.arange(theta_start, theta_end, step_size):
            # Calculate the real dV/dtheta for this specific crank angle
            # physics_functions.calc_piston_speed_factor returns dS/dtheta (m/rad)
            ds_dtheta_rad = pf.calc_piston_speed_factor(th)
            # Convert m/rad to m^3/degree
            dv_dtheta_deg = c.A_PISTON * ds_dtheta_rad * (np.pi / 180.0)

            p_curr, t_curr = pf.integrate_first_law(
                P_curr=p_curr, T_curr=t_curr, M_curr=m_air, V_curr=v_curr,
                Delta_M=0, Delta_Q_in=0, Delta_Q_loss=0, 
                dV_d_theta=dv_dtheta_deg, gamma=c.GAMMA_AIR, 
                theta_delta=step_size, T_manifold=300.0, R_spec=c.R_SPECIFIC_AIR
            )
            # Update volume for next step using the geometric function
            v_curr = pf.v_cyl(th + step_size, c.A_PISTON, V_clearance)

        # 4. Compare vs Theoretical Isentropic Peak (P2 = P1 * CR^gamma)
        p_expected_tdc = p_initial * (c.COMP_RATIO ** c.GAMMA_AIR)
        error = abs(p_curr - p_expected_tdc) / p_expected_tdc
        
        # With 0.5 degree steps, error should drop from 81% to < 0.5%
        self.assertLess(error, 0.01, f"Isentropic integration error {error:.4%} is too high at TDC")
        
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
    
        
    def test_drag_components(self):
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

if __name__ == '__main__':
    unittest.main()