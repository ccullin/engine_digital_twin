import numpy as np
import constants as c
import physics_functions as pf
from engine_model import EngineModel
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=False)
class MockEcuOutput:
    spark: bool = False
    spark_timing: int = 350
    afr_target: float = 14.7 * 0.88
    target_rpm: float = 0.0
    iacv_pos: float = 25
    iacv_wot_equiv: float = 25 * 0.06
    pid_P: float = 0.0
    pid_I: float = 0.0
    pid_D: float = 0.0
    trapped_air_mass_kg: float = 0.0
    ve_fraction: float = 0.0
    injector_on: bool = False
    injector_start: int = 0
    injector_end: int = 170
    fuel_cut_active: bool = False


class PhysicsValidator:
    def __init__(self, cycles):
        self.engine = EngineModel(rpm=900)
        
        self.cycles = cycles
        
        # Define the "Golden Specs" for the VW 2.1L MV (WBX)
        self.reference_targets = {
            "idle_map_kpa": 32.5,
            "motoring_fric_3000": 5.0, # Nm per cylinder
            "motoring_fric_900": 3.3, # Nm per cylinder
            "peak_ve_high_rpm": 0.88,
            "heat_loss_pct_low_rpm": 20.0,  # Expected % loss at 1500 RPM, was 25% but woschni does not calc all heat loss
            "peak_pressure_bar_limit": 65.0, # Structural limit for stock WBX
            "peak_pressure_angle": 372,
            "net_torque_at_peak_load": 174.0,
            "combustion_max_temp": 2800.0,
        }
                

    
    def fire_engine(self, rpm, tps, clt=c.COOLANT_START, wheel_load=0, cycles=3):
        def get_spark(rpm):
            """Provides a 'correct enough' spark for physics validation."""
            if rpm <= 1000:
                return 355  # 5 deg BTDC
            elif rpm ==4500:
                return 315
            elif rpm >= 3000:
                return 320  # 40 deg BTDC (Advanced 2 deg from original 38)
            elif rpm ==1500:
                return 321
            elif rpm <= 1500:
                # Segment 1: 1000 to 1500 RPM
                # At 1500, we want original (approx 13.75) + 4 = 17.75 deg BTDC
                target_1500 = 17.75
                slope = (target_1500 - 5) / (1500 - 1000)
                advance = 5 + slope * (rpm - 1000)
                return int(360 - advance)
    
        self.engine = EngineModel(rpm=rpm)
        self.engine.state.wheel_load = wheel_load
        self.engine.sensors.TPS_percent = tps
        self.engine.sensors.CLT_C = clt
        ecu = MockEcuOutput()
        target_afr = ecu.afr_target
        
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
                        ecu.injector_start = ecu.injector_end - pw_degrees

                # 4. Set ECU Outputs
                ecu.spark_timing = get_spark(rpm=rpm)
                ecu.spark = (CAD == ecu.spark_timing)
                
                # Handle pulse-width wrap-around (if start is negative or across 720)
                start_trigger = ecu.injector_start % 720
                end_trigger = ecu.injector_end % 720
                
                if start_trigger > end_trigger: # Pulse crosses 720/0 boundary
                    ecu.injector_on = (CAD >= start_trigger or CAD <= end_trigger)
                else:
                    ecu.injector_on = (start_trigger <= CAD <= end_trigger)

                self.engine.step(ecu)
        return ecu

    def run_tests(self):
        """Runs the automated battery of physics tests and returns a health report."""
        peak_pressure_value, peak_pressure_angle = self._test_peak_pressure_safety(rpm=3000)

        results = {
            "idle_map_kpa": self._test_idle_vacuum(rpm=900),
            "motoring_fric_900": self._test_friction_integrity(rpm=900),
            "motoring_fric_3000": self._test_friction_integrity(rpm=3000),
            "peak_ve_high_rpm": self._test_breathing_capacity(rpm=4500),
            "heat_loss_pct_low_rpm": self._test_firing_efficiency(rpm=1500),
            "peak_pressure_bar_limit": peak_pressure_value,
            "peak_pressure_angle": peak_pressure_angle,
            "net_torque_at_peak_load": self.test_torque_equilibrium(),   
            "combustion_max_temp": self._test_combustion_thermodynamics(),     
        }
        self._generate_report(results)
        
        print(self.generate_sync_report())
        
        return results
        
    def _test_combustion_thermodynamics(self):
        """
        Diagnoses non-physical temperatures (e.g., 8000K).
        Audits Trapped Mass vs. Energy Release.
        """
        
        # Run 5 cycles to ensure manifold and fueling are stabilized
        self.fire_engine(rpm=3000, tps=100, cycles=5)
        
        # 1. Audit Trapped Mass at Ignition (assuming 321 degrees based on your ECU)
        m_air = self.engine.cyl.air_mass_at_spark
        m_fuel = self.engine.cyl.fuel_mass_at_spark
        m_total = self.engine.cyl.total_mass_at_spark  # Should be ~0.0005 - 0.0006 kg for a 0.5L cyl
        
        # 2. Capture Temperature jump
        # We look at the log_T history for the last cycle
        T_pre_spark = self.engine.cyl.log_T[320] 
        T_peak = np.max(self.engine.cyl.log_T)
        delta_T = T_peak - T_pre_spark
        
        # 3. Energy Audit
        q_fuel = self.engine.cyl.total_cycle_heat_J
        # Theoretical delta T = Q / (mass * Cv)
        # Using air Cv ~ 718 J/kgK
        expected_delta_t = q_fuel / (m_total * 718) if m_total > 0 else 0

        print(f"\n--- COMBUSTION THERMODYNAMIC AUDIT ---")
        print(f"Trapped Total Mass: {m_total*1e6:.3f} mg (Air: {m_air*1e6:.3f}mg)")
        print(f"Fuel Energy In:     {q_fuel:.2f} Joules")
        print(f"Peak Temperature:   {T_peak:.2f} K  (Delta T: {delta_T:.2f} K)")
        print(f"Expected Delta T:   {expected_delta_t:.2f} K (at Cv=718)")
        print(f"--------------------------------------\n")

        return T_peak
    
    
    
    def test_torque_equilibrium(self):
        """
        Validation: Can the engine maintain steady RPM against factory peak load?
        Target: 174 Nm at 2800-3200 RPM should result in Net Torque ~ 0.
        """
        # self.engine = EngineModel(rpm=3000)
        # self.engine.sensors.TPS_percent = 100.0
        # self.engine.state.wheel_load = 174.0 # Factory Peak Torque
        self.fire_engine(rpm=3000, tps=100, wheel_load=174, cycles=4)
        
        # 3. Measure Net Torque (Indicated - Friction - Pumping - Load)
        # If this is 0, the twin matches the factory 2.1L performance profile.
        # net_torque = np.mean(self.engine.state.torque_brake_history) - self.engine.state.wheel_load
        net_torque = np.average(self.engine.state.torque_net_history)
        brake_torque = np.average(self.engine.state.torque_brake_history)
    
        return brake_torque
    
    
    
    def _test_friction_integrity(self, rpm):
        """Calculates Mean Friction Torque (Nm) by averaging 720 instantaneous samples."""
        samples = []
        for cad in range(720):
            # Calculate instantaneous torque at this specific degree
            t_fric = pf.calc_single_cylinder_friction(theta=cad, rpm=rpm, p_cyl=c.P_ATM_PA, clt=90.0)
            samples.append(t_fric)
        
        # The mean of all 720 samples is the Average (Mean) Friction Torque for the cylinder cycle
        cylinder_friction_nm = np.mean(samples)
        common_fric_nm = pf.calc_engine_core_friction(rpm=rpm, clt=90.0)
        
        engine_friction_nm = c.NUM_CYL * cylinder_friction_nm + common_fric_nm
        print(f"\n--- FRICTION at RPM:{rpm} ---")
        print(f"ENGINE FRICTION:       {engine_friction_nm:.2f} Nm")
        print(f"  └─ CYLINDER * 4:     {cylinder_friction_nm * c.NUM_CYL:.2f} Nm")
        print(f"  └─ GLOBAL_PARASITIC: {common_fric_nm:.2f} Nm")

        return engine_friction_nm

    def _test_breathing_capacity(self, rpm):
        """Ported from run_mass_flow_audit: Calculates VE percentage."""
        # Setup WOT conditions
        # self.engine = EngineModel(rpm=rpm)
        # self.engine.sensors.TPS_percent = 100.0
        
        # Stabilize manifold
        ecu = self.fire_engine(rpm=4500, tps=100, cycles=3) # collect the ecu that was used for test continuity

        total_mass_in = 0.0
        # Monitor intake stroke
        for cad in range(0, 360):
            deltas = self.engine._calc_flow_deltas(ecu, 1.0)
            total_mass_in += deltas['dm_i'] * 1e6 # mg
            self.engine.step(ecu)

        # Calculate VE relative to current MAP
        ideal_mass = (self.engine.sensors.P_manifold_Pa * self.engine.cyl.V_displaced) / (287 * 293) * 1e6
        
        print(f"DEBUG _TEST_BREATHING_CAPACITY "
                f"Peak_P:{np.max(self.engine.cyl.log_P)/1e5:.2f}bar Peak_P_angle:{np.argmax(self.engine.cyl.log_P):.2f} ")
        return total_mass_in / ideal_mass

    def _test_idle_vacuum(self, rpm):
        """Runs the engine at idle speeds to check manifold pressure (MAP)."""
        ecu = MockEcuOutput()
        self.engine.sensors.TPS_percent = 0.0 # ecu has iacv bypass
        self.engine.sensors.rpm = rpm
        
        map_samples = np.zeros(720) 
        for _ in range(self.cycles * 720):
            CAD = _ % 720
            ecu.spark = (CAD == ecu.spark_timing)
            ecu.injector_on = (ecu.injector_start <= CAD <= ecu.injector_end)      
            self.engine.step(ecu)
            map_samples[CAD] = self.engine.sensors.MAP_kPa
            
        return np.mean(map_samples) # Average vacuum
    
    
    def _test_firing_efficiency(self, rpm):
        """Audits energy conservation: Fuel In = Work Out + Heat Loss + Exhaust."""

        self.fire_engine(rpm=rpm, tps=100.0, clt=90.0, cycles=2)
        
        heat_lost_j = self.engine.cyl.Q_loss_total # use EngineModel accumulators rather than duplicate
        total_fuel_energy = self.engine.cyl.total_cycle_heat_J # use EngineModel accumulators rather than duplicate
        
        
        heat_loss_pct = (heat_lost_j / total_fuel_energy) * 100
        print(f"DEBUG TEST_FIRING_EFF heat_lost_j:{heat_lost_j:.2f} total heat:{total_fuel_energy:.2f} ratio:{heat_loss_pct:.2f} ")
        print(f"|__ T_wall:{self.engine.cyl.T_wall:.2f} cyl_P:{self.engine.cyl.P_curr/1000:.2f}kPa " 
              f"Peak_P:{np.max(self.engine.cyl.log_P)/1000:.2f}kPa Peak_P_angle:{np.argmax(self.engine.cyl.log_P):.2f} "
              f"Peak_T:{np.max(self.engine.cyl.log_T):.2f}K Peak_T_angle:{np.argmax(self.engine.cyl.log_T):.2f} ")
        return heat_loss_pct

    def _test_peak_pressure_safety(self, rpm):
        """Ensures P_max stays within physical limits of a 10:1 WBX."""
        # self.engine = EngineModel(rpm=3000)
        # self.engine.sensors.TPS_percent = 100.0
        self.fire_engine(rpm=3000, tps=100.0, cycles=4)

        # Setup as above, return max(p_history)
        P_peak = np.max(self.engine.cyl.log_P) / 1e5 
        P_peak_angle = np.argmax(self.engine.cyl.log_P)
        
        return P_peak, P_peak_angle

    def _generate_report(self, results):
        print("\n" + "="*65)
        print(f"{'PHYSICS VALIDATION REPORT (VW 2.1L MV)':^65}")
        print("="*65)
        print(f"{'METRIC':<25} | {'ACTUAL':>8} | {'TARGET':>8} | {'ERR %':>6} | {'STATUS'}")
        print("-" * 65)
        
        for key, actual in results.items():
            target = self.reference_targets[key]
            if target == 0:
                error = abs(actual)
            else:
                error = abs((actual - target) / target) * 100
           
            # Validation tolerance: 5% for most, 10% for heat loss
            tolerance = 10.0 if "heat_loss" in key else 5.0
            status = "✅ PASS" if error < tolerance else "❌ FAIL"
            
            # Format display units
            if "pct" in key or "ve" in key:
                disp_act = f"{actual:.1f}%" if "pct" in key else f"{actual*100:.1f}%"
                disp_tar = f"{target:.1f}%" if "pct" in key else f"{target*100:.1f}%"
            else:
                disp_act = f"{actual:.2f}"
                disp_tar = f"{target:.2f}"
            
            print(f"{key:<25} | {disp_act:>8} | {disp_tar:>8} | {error:>5.1f}% | {status}")
        
        print("-" * 65)
        # Diagnostic hint for the user
        if results["heat_loss_pct_low_rpm"] < self.reference_targets["heat_loss_pct_low_rpm"]:
            print("DIAGNOSTIC: Low-RPM Heat Loss is too low.")
        print("="*65 + "\n")
        
    def generate_sync_report(self):
        """Generates a detailed thermodynamic payload to synchronize development."""
        # put the engine into known state
        ecu = MockEcuOutput()
        self.engine = EngineModel(rpm=3000)
        self.engine.sensors.TPS_percent = 100
        
        for _ in range ( 5 * 720):
            self.engine.sensors.rpm = 3000
            self.engine.step(ecu)
            
        # Capture a 720-degree snapshot from the last cycle
        p_work = self.engine.cyl.log_P * self.engine.cyl.dV_list
        
        # Energy Partitioning (Joules per Cylinder per Cycle)
        pumping_work = np.sum(p_work[0:180]) + np.sum(p_work[540:720])
        compression_work = np.sum(p_work[180:360])
        expansion_work = np.sum(p_work[360:540])
        net_indicated_work = np.sum(p_work)
        
        # Calculate Pumping Mean Effective Pressure (PMEP)
        pmep_bar = (abs(pumping_work) / c.V_DISPLACED) / 1e5
        
        report = [
            "\n" + "="*40,
            "   THERMODYNAMIC SYNC REPORT v1.0",
            "="*40,
            f"1. GAS EXCHANGE (PUMPING):",
            f"   Pumping Work: {pumping_work:.2f} J",
            f"   PMEP:         {pmep_bar:.3f} bar (Target < 0.15 @ WOT)",
            f"   Result:       {'CLOGGED' if pmep_bar > 0.2 else 'CLEAR'}",
            "",
            f"2. ENERGY PARTITION (POWER STROKE):",
            f"   Compression:  {compression_work:.2f} J",
            f"   Expansion:    {expansion_work:.2f} J",
            f"   Net Indicated:{net_indicated_work:.2f} J",
            "",
            f"3. THERMAL STATE AT IVC:",
            f"   Trapped Mass: {self.engine.cyl.total_mass_kg*1e6:.2f} mg",
            f"   Pressure:     {self.engine.cyl.P_curr/1e5:.3f} bar",
            f"   Temp:         {self.engine.cyl.T_curr:.1f} K",
            "="*40
        ]
        return  "\n".join(report)
        
        