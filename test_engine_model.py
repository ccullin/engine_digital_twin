import unittest
import numpy as np
from engine_model import EngineModel
import constants as c

class TestEngineModelIntegration(unittest.TestCase):
  
    def setUp(self):
        # self.engine = EngineModel(rpm=2000)
        
        self.ecu = {
            "spark": False, "spark_timing": 350, "afr_target": 0.88 * 14.7,
            "iacv_pos": 25, "iacv_wot_equiv": 25 * 0.06, "trapped_air_mass_kg": 0.0,
            "ve_fraction": 0.0, "injector_on": False, "inject_start": 0.0, "inject_end": 170.0,
            "fuel_cut_active": False,
        }
    
    def fire_engine(self, rpm, tps, clt=c.COOLANT_START, wheel_load=0, cycles=3):
        def get_spark(rpm):
            """Provides a 'correct enough' spark for physics validation."""
            if rpm <= 1000:
                return 355  # 5 deg BTDC
            elif rpm >= 3000:
                return 325  # 35 deg BTDC
            else:
                # Linear interpolation between 900 and 3000 RPM
                slope = (35 - 5) / (3000 - 900)
                advance = 5 + slope * (rpm - 900)
                return int(360 - advance)
    
        self.engine = EngineModel(rpm=rpm)
        self.engine.state.wheel_load = wheel_load
        self.engine.sensors.TPS_percent = tps
        self.engine.sensors.CLT_C = clt
        ecu = self.ecu.copy()
        target_afr = ecu['afr_target']
        
        # 1. Convert CC/MIN to KG/S (assuming gasoline density 0.74)
        flow_kg_s = (c.INJECTOR_FLOW_CC_PER_MIN / 60.0) * (0.74 / 1000.0)
        
        # Assume IVC for a VW 2.1L is roughly 220-240 CAD (adjust to your model's spec)
        IVC_ANGLE = self.engine.valves.intake.close_angle 

        for _ in range(cycles):
            for CAD in range(720):
                
                # 2. At Intake Valve Close, calculate the injection needed for the NEXT cycle
                if CAD == IVC_ANGLE + 1:
                    trapped_air = self.engine.cyl.air_mass_kg
                    
                    if trapped_air > 0:
                        required_fuel_kg = trapped_air / target_afr
                        
                        # Calculate Pulse Width in Seconds, then Degrees
                        inj_time_sec = required_fuel_kg / flow_kg_s
                        deg_per_sec = (self.engine.sensors.rpm / 60.0) * 360.0
                        pw_degrees = inj_time_sec * deg_per_sec
                        
                        # 3. Set Variable Start based on Fixed End
                        # Example: Injector End is fixed at 170 CAD (standard for many port setups)
                        ecu['inject_start'] = ecu['inject_end'] - pw_degrees

                # 4. Set ECU Outputs
                ecu['spark_timing'] = get_spark(rpm=rpm)
                ecu["spark"] = (CAD == ecu['spark_timing'])
                
                # Handle pulse-width wrap-around (if start is negative or across 720)
                start_trigger = ecu['inject_start'] % 720
                end_trigger = ecu['inject_end'] % 720
                
                if start_trigger > end_trigger: # Pulse crosses 720/0 boundary
                    ecu["injector_on"] = (CAD >= start_trigger or CAD <= end_trigger)
                else:
                    ecu["injector_on"] = (start_trigger <= CAD <= end_trigger)

                self.engine.step(ecu)


    def test_thermal_feedback_loop(self):
        """Tests that running the engine actually heats up the coolant."""
        # self.engine = EngineModel(rpm=3000)
        initial_clt = c.COOLANT_START   
        # self.engine.sensors.TPS_percent = 100.0

        self.fire_engine(rpm=3000, tps=100.0, cycles=10)
            
        # print(f"\nDEBUG THERMAL "
        #     # f"T_Indicated:{np.average(self.engine.state.torque_indicated_history):6.2f}Nm | "
        #     # f"T_Friction:{np.average(self.engine.state.torque_friction_history):6.2f}Nm | "
        #     # f"T_Net:{np.average(self.engine.state.torque_brake_history):6.2f}Nm | "
        #     # )
        #       f"coolant test rpm:{self.engine.sensors.rpm:.0f} T_brake:{np.average(self.engine.state.torque_brake_history):.2f} "
        #       f"T_ind:{np.average(self.engine.state.torque_indicated_history):.2f} T_fri:{np.average(self.engine.state.torque_friction_history):.2f} "
        #       f"P_peak:{np.max(self.engine.cyl.log_P)/1e5:.0f}bar P_peak_angle:{np.argmax(self.engine.cyl.log_P)} "
        #       )    
        # print(f"\nDEBUG THERMAL Coolant Test (Heating): initial:{initial_clt} -> {self.engine.sensors.CLT_C:.4f}C")
        self.assertGreater(self.engine.sensors.CLT_C, initial_clt, 
                           "Engine coolant failed to heat up after 10 WOT cycles")

    def test_manifold_suction_response(self):
        """Tests that closing the throttle increases vacuum (decreases MAP)."""

        self.fire_engine(rpm=3000, tps=100.0, cycles=10)   

        map_wot = self.engine.sensors.MAP_kPa
        # print(f"\nWOT rpm:{self.engine.sensors.rpm} MAP:{map_wot:.2f} TPS_e:{self.engine.state.effective_tps:.2f} "
        #       f"T_brake:{self.engine.state.torque_brake:.2f} T_ind:{self.engine.state.torque_indicated_history[-1]:.2f} "
        #       f"T_fri:{self.engine.state.torque_friction:.2f} "
        #       f"lambda:{self.engine.sensors.lambda_} "
        #       )
        
        # 2. Close Throttle
        # self.engine.sensors.TPS_percent = 0 # the IACV is has min opening set in ecu
        self.fire_engine(rpm=3000, tps=0.0, cycles=3) # the IACV is has min opening set in ecu
        map_closed = self.engine.sensors.MAP_kPa
        
        # print(f"\nclosed rpm:{self.engine.sensors.rpm:.2f} MAP:{map_closed} TPS_e:{self.engine.state.effective_tps:.2f} " )
        # print(f"\nP_peak:{np.max(self.engine.cyl.log_P)/1000:.0f}kPa P_peak_angle:{np.argmax(self.engine.cyl.log_P)} " )   
        
        self.assertLess(map_closed, map_wot, "Manifold pressure failed to drop when throttle closed")
    
    def test_firing_symmetry(self):
        """Integration: Validates that torque pulses occur every 180 degrees, ignoring Cycle 0."""
        # self.engine = EngineModel(rpm=3000)
        # self.engine.sensors.TPS_percent = 50.0
        
        # Run 5 cycles total, but we will only analyze the very last cycle
        self.fire_engine(rpm=3000, tps=50.0, cycles=5) 
        
        # torque_indicated_history always contains the MOST RECENT 720 degrees.
        # By running 5 cycles, we are guaranteed that history is a "Steady State" cycle.
        torque_wave = self.engine.state.torque_indicated_history
        
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
        self.engine = EngineModel(rpm=3000)
        self.engine.motoring_rpm = 3000
        self.engine.sensors.TPS_percent = 100.0
        self.engine.sensors.CLT_C = 90.0  # Warm engine
        for _ in range(3 * 720):
            self.engine.step(self.ecu)
             
        # The torque the dyno HAD to provide to stay at 3000 RPM
        # This is the "Torque the crank needs to overcome"
        avg_dyno_support = np.average(self.engine.state.torque_governor_history)
        
        # Total Drag is the inverse of the support required to keep speed steady
        total_drag = -avg_dyno_support 
        
        j_to_nm = 1.0 / (4 * np.pi)
        
        # Gas Exchange Diagnostic
        w_pumping = self.engine.engine_data_dict["work_pumping_j"]
        pumping_torque_nm = w_pumping * j_to_nm * c.NUM_CYL  # pumping_torque is per cylinder while avg_dyno is for the engine
        pmep_bar = (abs(w_pumping) / self.engine.cyl.V_displaced) / 1e5
        
        # Mechanical/Thermal Remainder
        mech_thermal_drag = total_drag - pumping_torque_nm
        
        # check work done.
        t_compression_nm =     self.engine.engine_data_dict["work_compression_j"] * j_to_nm * c.NUM_CYL
        t_expansion_nm =       self.engine.engine_data_dict["work_expansion_j"] * j_to_nm * c.NUM_CYL
        t_friction_nm =        self.engine.engine_data_dict["friction_work_j"] * j_to_nm
        t_indicated_nm =       self.engine.engine_data_dict["indicated_work_j"] * j_to_nm
        t_brake_nm =           self.engine.engine_data_dict["net_work_j"] * j_to_nm
        
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
        self.fire_engine(rpm=3000, tps=100, clt=90, cycles=5)

            
        # 3. Collect Data
        w_fric_wot = self.engine.engine_data_dict["friction_work_j"]
        t_fric_wot = w_fric_wot / (4 * np.pi)
        
        # Peak Cylinder Pressure (for context)
        p_max = np.max(self.engine.cyl.log_P) / 1e5 # bar
        
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
        # self.engine = EngineModel(rpm=3000)
        # self.engine.sensors.TPS_percent = 100.0
        self.fire_engine(rpm=3000, tps=100.0, cycles=5)
        
        # At 1 bar, a 525cc cylinder holds ~0.6g of air. 
        # With VE and heating, 0.45g - 0.55g is the 'Digital Twin' target.
        trapped_mass_mg = self.engine.cyl.air_mass_at_spark * 1e6
        self.assertGreater(trapped_mass_mg, 400.0, "Engine is gasping for air (VE too low)")
        self.assertLess(trapped_mass_mg, 650.0, "Engine is supercharging itself (VE too high)")
        
    def test_static_compression_physics(self):
        """
        First Principles Audit: Validates Geometric CR and Polytropic Compression.
        """
        # Motor engine at low speed (1000 RPM) to minimize high-speed dynamic effects
        self.engine = EngineModel(rpm=1000)
        self.engine.motoring_rpm = 1000
        self.engine.sensors.TPS_percent = 100.0 # WOT to ensure full cylinder filling
        
        for _ in range(3 * 720):
            self.engine.step(self.ecu)
            
        # 1. Capture Peak Motoring Pressure
        p_max_pa = np.max(self.engine.cyl.log_P)
        p_max_bar = p_max_pa / 1e5
        
        # 2. Capture Pressure at IVC (The starting point of actual compression)
        ivc_idx = int(self.engine.valves.intake.close_angle)
        p_at_ivc_bar = self.engine.cyl.log_P[ivc_idx] / 1e5
        
        # 3. Theoretical Adiabatic Check: P2 = P1 * (V1/V2)^gamma
        # V1 is volume at IVC, V2 is V_clearance at TDC (360 CAD)
        v_ivc = self.engine.cyl.V_list[ivc_idx]
        v_tdc = self.engine.cyl.V_clearance
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
        
if __name__ == '__main__':
    unittest.main()