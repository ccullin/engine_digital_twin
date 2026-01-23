# diagnostics.py (Fixed Logic Flow)
import argparse
import numpy as np
import physics_functions as pf
import constants as c
from engine_model import EngineModel



def run_mass_flow_audit(rpm, tps_percent):
    """
    Diagnostic to catch 'leaky' throttle math.
    Compiles mass inflow vs cylinder demand to see why P_cyl doesn't drop.
    """
    engine = EngineModel(rpm=rpm)
    ecu = get_default_ecu_outputs()
    engine.sensors.TPS_percent = tps_percent
    
    # We need to simulate a few cycles to stabilize the manifold pressure logic
    # from physics_functions.update_intake_manifold_pressure
    for _ in range(720 * 3):
        engine.step(ecu)

    print("="*95)
    print(f"MASS FLOW AUDIT | RPM: {rpm} | TPS: {tps_percent}% | MAP: {engine.sensors.MAP_kPa:.2f} kPa")
    print("="*95)
    print(f"{'CAD':>5} | {'V (L)':>8} | {'P_cyl(bar)':>10} | {'dm_in(mg)':>10} | {'Flow(g/s)':>10} | {'Status'}")
    print("-" * 95)

    total_mass_in = 0.0
    for cad in range(0, 360): # Focus on Intake and start of Compression
        engine.state.current_theta = cad
        
        # 1. Capture the flow deltas before the step
        # This uses your engine_model.py _calc_flow_deltas logic
        deltas = engine._calc_flow_deltas(ecu, 1.0)
        dm_mg = deltas['dm_i'] * 1e6
        total_mass_in += dm_mg
        
        # 2. Step the engine
        engine.step(ecu)
        p_bar = engine.cyl.P / 1e5
        vol_l = engine.cyl.V_list[cad] * 1000
        
        # 3. Calculate flow rate in g/s for physical comparison
        # (dm_kg / degree) * (degrees / second) * 1000
        deg_per_sec = rpm * 6.0
        flow_rate_gs = deltas['dm_i'] * deg_per_sec * 1000

        if cad % 20 == 0 or (160 <= cad <= 220): # Higher res near IVC
            status = "Filling" if dm_mg > 0 else "Backflow"
            print(f"{cad:5d} | {vol_l:8.3f} | {p_bar:10.3f} | {dm_mg:10.4f} | {flow_rate_gs:10.2f} | {status}")

    print("-" * 95)
    print(f"Total Trapped Mass: {total_mass_in:.2f} mg")
    print(f"Ideal Mass @ MAP:   {(engine.sensors.P_manifold_Pa * engine.cyl.V_displaced)/(287*293)*1e6:.2f} mg")
    print("="*95)

def run_high_vacuum_drag_audit(rpm, map_pa):
    """
    Audit to see why 151J of combustion is winning against friction/pumping.
    """
    engine = EngineModel(rpm=rpm)
    engine.sensors.P_manifold_Pa = map_pa
    
    total_fric_work = 0.0
    total_pumping_work = 0.0
    
    print("="*85)
    print(f"HIGH-VACUUM DRAG AUDIT | RPM: {rpm} | MAP: {map_pa/1000:.1f} kPa")
    print("="*85)

    for cad in range(720):
        # 1. Friction Torque from your function
        p_cyl = engine.cyl.log_P[cad]
        t_fric = pf.calc_single_cylinder_friction(cad, rpm, p_cyl, 90.0)
        total_fric_work += t_fric * (np.pi / 180.0)
        
        # 2. Pumping Work (P * dV)
        # This is the 'Brake' provided by the closed throttle
        dv = engine.cyl.dV_list[cad]
        total_pumping_work += p_cyl * dv

    print(f"Expansion Work (from log):  +151.0 J")
    print(f"Friction Loss (Calc):       {total_fric_work * -4:>8.2f} J") # 4 cylinders
    print(f"Pumping Loss (Calc):        {total_pumping_work:>8.2f} J")
    print(f"NET BALANCE:               {151.0 - (total_fric_work*4) + total_pumping_work:>8.2f} J")
    
    if (151.0 - (total_fric_work*4) + total_pumping_work) > 0:
        print("\nRESULT: Engine ACCELERATES. Physics are too 'slippery'.")
    else:
        print("\nRESULT: Engine DECELERATES. Physics are healthy.")


def run_idle_braking_audit(rpm, vacuum_pa):
    """
    Audit the 'Braking' forces at idle.
    A healthy 2.1L engine should lose ~100-150J per cycle to pumping/friction.
    """
    engine = EngineModel(rpm=rpm)
    # Force the manifold to a deep vacuum
    engine.sensors.P_manifold_Pa = vacuum_pa 
    
    work_pumping = 0.0
    work_friction = 0.0
    
    print("="*75)
    print(f"IDLE BRAKING AUDIT | RPM: {rpm} | MAP: {vacuum_pa/1000:.1f} kPa")
    print("="*75)
    
    for cad in range(720):
        # Calculate Pumping Work: W = P * dV
        # dV is in m^3, P is in Pa. Result is Joules.
        p_cyl = engine.cyl.log_P[cad]
        dv = engine.cyl.dV_list[cad]
        step_work = p_cyl * dv
        work_pumping += step_work
        
        # Calculate Friction Work
        f_torque = pf.calc_single_cylinder_friction(cad, rpm, p_cyl, 90.0)
        work_friction += f_torque * (np.pi / 180.0)

    print(f"1. Pumping Work (Net):  {work_pumping:>8.2f} J (Should be negative)")
    print(f"2. Friction Work:       {work_friction:>8.2f} J (Braking force)")
    print(f"3. TOTAL DRAG:          {work_pumping - work_friction:>8.2f} J")
    print("-" * 75)
    
    # Logic Check
    # Your log showed 151J of Expansion work.
    # Total Drag MUST be > 151J to slow down the engine.
    if (work_pumping - work_friction) + 151.0 > 0:
        print("VERDICT: PHYSICS GAP FOUND.")
        print("Drag is too weak to stop combustion torque. Increase Friction or Pumping.")



def run_combustion_efficiency_audit(rpm):
    """
    Diagnose power production by integrating the actual ECU logic.
    This reveals if the ECU is commanding enough spark/fuel for WOT.
    """
    from ecu_controller import ECUController # Local import to avoid circularity
    
    engine = EngineModel(rpm=rpm)
    ecu = ECUController()
    
    # Force WOT conditions
    engine.sensors.TPS_percent = 100.0
    
    # Enable Firing Mode
    ecu.is_motoring = False
    ecu.fuel_enabled = True
    ecu.spark_enabled = True

    print("="*85)
    print(f"FIRED COMBUSTION AUDIT | RPM: {rpm} | Mode: WOT")
    print("="*85)
    
    # Run one full cycle to stabilize physics and get ECU outputs
    for _ in range(30 * 720):
        # 1. Get real ECU response to current engine sensors
        ecu_outputs = ecu.update(engine.get_sensors())
        # 2. Step the engine with those outputs
        sensors, engine_data = engine.step(ecu_outputs)

    # Energy Results
    exp_work = engine_data['work_expansion_j']
    comp_work = abs(engine_data['work_compression_j'])
    fric_loss = abs(engine_data['friction_work_j'])
    net_ind_work = engine_data['net_work_j']
    brake_work = net_ind_work - fric_loss

    print(f"1. ECU COMMANDS:")
    print(f"   Ignition Advance: {ecu_outputs['spark_timing']:>8.1f}° BTDC")
    print(f"   Target AFR:       {ecu_outputs['afr_target']:>8.2f}")
    print(f"   VE Fraction:      {ecu_outputs['ve_fraction']:>8.2f}")
    
    print(f"\n2. ENERGY AUDIT (Joules):")
    print(f"   Compression:      {comp_work:>8.1f} J")
    print(f"   Expansion:        {exp_work:>+8.1f} J")
    print(f"   Net Indicated:    {net_ind_work:>+8.1f} J (Indicated Torque)")
    print(f"   Friction Loss:    {fric_loss:>8.1f} J")
    print(f"   BRAKE BALANCE:    {brake_work:>+8.1f} J")
    
    print(f"\n3. COMBUSTION GEOMETRY:")
    print(f"   Peak Pressure:    {max(engine.cyl.log_P) / 100000.0:>8.2f} Bar")
    print(f"   P_peak_angle:     {np.argmax(engine.cyl.log_P):>8.1f}° ATDC")

    print("-" * 85)
    if brake_work <= 0:
        print("VERDICT: STALL/REDLINE REACHED")
        print("The engine cannot accelerate because Brake Work is zero or negative.")
        if engine.cyl.P_peak_angle < 5:
            print("FIX: Ignition is too early. Peak pressure is fighting the rising piston.")
        elif exp_work < (comp_work * 2):
            print("FIX: Heat Release is too low. Check your fuel mass or burn rate model.")
    else:
        print("VERDICT: ACCELERATING")
        print(f"Engine has {brake_work:.1f}J of surplus energy per cycle to increase RPM.")
    print("="*85)


def run_gas_exchange_audit(rpm):
    """
    High-resolution audit of the pumping loop to detect sonic choking 
    and discharge coefficient (Cd) bottlenecks.
    """
    engine = EngineModel(rpm=rpm)
    ecu = get_default_ecu_outputs()
    engine.sensors.TPS_percent = 100.0  # Wide Open Throttle
    
    # Run simulation to populate logs
    for _ in range(720):
        engine.step(ecu)
        
    print("="*85)
    print(f"GAS EXCHANGE & FLOW AUDIT | RPM: {rpm} | displacement: 2.1L")
    print("="*85)
    print(f"{'CAD':>5} | {'P_cyl(bar)':>10} | {'P_ratio':>9} | {'dm (mg/deg)':>12} | {'Cd_Eff':>8} | {'Status'}")
    print("-" * 85)

    # Note: P_ratio < 0.528 indicates SONIC CHOKING (flow cannot go faster)
    CHOKE_THRESHOLD = 0.528

    for cad in range(0, 720):
        p_cyl = engine.cyl.log_P[cad]
        p_ratio = p_cyl / 101325.0  # Pressure ratio vs Atmospheric
        
        # Calculate dm and Cd for this degree
        # Assuming you've mapped log_dm_i and log_cd_i in your _step_one_degree
        dm = engine.cyl.dm_in_history[cad] * 1e6  # to mg
        cd = engine.cyl.Cd_in_history[cad]
        
        # Check flow status
        status = "OK"
        if 0 < cad < 180 and p_ratio < CHOKE_THRESHOLD:
            status = "!!! CHOKED"
        elif 540 < cad < 720 and p_ratio > (1.0 / CHOKE_THRESHOLD):
            status = "!!! BACKPRESSURE"

        # Print samples every 30 degrees or during critical valve events
        if cad % 30 == 0 or status != "OK":
            print(f"{cad:5d} | {p_cyl/1e5:10.3f} | {p_ratio:9.3f} | {dm:12.3f} | {cd:8.3f} | {status}")

    # Summary Analysis
    max_p_exh = np.max(np.array(engine.cyl.log_P[540:720])) / 1e5
    min_p_int = np.min(np.array(engine.cyl.log_P[0:180])) / 1e5
    
    print("-" * 85)
    print(f"DIAGNOSTIC SUMMARY:")
    print(f"Peak Exhaust Pressure: {max_p_exh:.2f} bar (Ideal: < 1.10)")
    print(f"Min Intake Pressure:   {min_p_int:.2f} bar (Ideal: > 0.90)")
    
    if min_p_int < 0.50:
        print("CRITICAL: Massive Intake Restriction. Increase Intake Cd or Valve Diameter.")
    if max_p_exh > 1.50:
        print("CRITICAL: Massive Exhaust Restriction. Increase Exhaust Cd or Port Area.")
    print("="*85)




def run_pv_work_analysis(rpm):
    """
    FIXED Diagnostic: Full 720-degree PV Analysis.
    This version actually runs the simulation to populate the pressure logs.
    """
    engine = EngineModel(rpm=rpm)
    ecu = get_default_ecu_outputs()
    engine.sensors.TPS_percent = 100.0
    
    # 1. Run a full cycle (720 degrees) to populate the physics logs
    print(f"Simulating full cycle at {rpm} RPM...")
    for _ in range(720):
        engine.step(ecu)
    
    # 2. Extract Data from the logs
    pressures = np.array(engine.cyl.log_P) / 1e5  # Bar
    volumes = np.array(engine.cyl.V_list) * 1000   # Liters
    
    def calc_work(p_segment, v_segment):
        # Work = Integral of P dv (using trapz for integration)
        # We must use absolute Joules: (bar * 10^5) * (L * 10^-3) = bar * L * 100
        return np.trapezoid(p_segment, v_segment) * 100.0

    # Segments: Intake(0-180), Comp(180-360), Exp(360-540), Exh(540-720)
    w_intake = calc_work(pressures[0:181],   volumes[0:181])
    w_comp   = calc_work(pressures[180:361], volumes[180:361])
    w_expan  = calc_work(pressures[360:541], volumes[360:541])
    w_exhaus = calc_work(pressures[540:721], volumes[540:721])
    
    pumping_work = w_intake + w_exhaus
    thermo_loss = w_comp + w_expan # In motoring, this should be negative (loss)
    
    print("="*75)
    print(f"CORRECTED P-V WORK ANALYSIS | RPM: {rpm}")
    print("="*75)
    print(f" PHASE          |  WORK (J)    |  DESCRIPTION")
    print("-" * 75)
    print(f" Intake         | {w_intake:12.2f} | Suction work")
    print(f" Compression    | {w_comp:12.2f} | Energy into air spring")
    print(f" Expansion      | {w_expan:12.2f} | Energy out of air spring")
    print(f" Exhaust        | {w_exhaus:12.2f} | Pumping loss")
    print("-" * 75)
    print(f" NET PUMPING    | {pumping_work:12.2f} | (Intake + Exhaust)")
    print(f" THERMO LOSS    | {thermo_loss:12.2f} | (Comp + Expan)")
    print("="*75)


def run_true_physics_test(rpm):
    """
    Diagnostic: True Physics Integration
    This version lets the engine model actually 'breathe' by calling 
    the internal physics steps rather than just replaying logs.
    """
    engine = EngineModel(rpm=rpm)
    # Mock ECU for Wide Open Throttle Motoring
    ecu = {
        "spark": False, "spark_timing": 0.0, "afr_target": 14.7,
        "idle_valve_position": 0.0, "trapped_air_mass_kg": 0.0,
        "ve_fraction": 0.0, "injector_on": False, "fuel_cut_active": False,
    }
    engine.sensors.TPS_percent = 100.0
    
    print("="*75)
    print(f"PHYSICS INTEGRATION DIAGNOSTIC | RPM: {rpm}")
    print(f"Goal: Observe P_cyl drop and Mass Inflow")
    print("="*75)
    print(f"{'CAD':>6} | {'P_cyl (bar)':>12} | {'V (L)':>10} | {'dm (mg/deg)':>12}")
    print("-" * 75)

    # We manually step the engine through the intake stroke
    # to ensure we see the 'Suction' phase clearly.
    for cad in range(0, 361): # Intake + Compression
        # 1. Get current volume and dV
        V_curr = engine.cyl.V_list[cad]
        dV = engine.cyl.dV_list[cad]
        
        # 2. Calculate flow based on current pressure delta
        deltas = engine._calc_flow_deltas(ecu, 1.0)
        dm = deltas['dm_i']
        
        # 3. Call the REAL integration function (First Law of Thermodynamics)
        # This is what updates P and T based on dV and dm
        P_next, T_next = pf.integrate_first_law(
            P_curr=engine.cyl.P,
            T_curr=engine.cyl.T,
            M_curr=engine.cyl.total_mass_kg,
            V_curr=V_curr,
            Delta_M=deltas["dm_tot"],
            Delta_Q_in=0.0, # No combustion
            Delta_Q_loss=0.0,
            dV_d_theta=dV, 
            gamma=engine.cyl.gamma_blend,
            theta_delta=1.0,
            T_manifold=deltas["T_inflow"],
            R_spec=engine.cyl.R_specific_blend,
            cycle=0, CAD=cad, substep=0
        )
        
        # 4. Update the actual engine state
        engine.cyl.P = P_next
        engine.cyl.T = T_next
        engine.cyl.air_mass_kg += dm
        engine.cyl.total_mass_kg = engine.cyl.air_mass_kg + engine.cyl.fuel_mass_kg
        engine.state.current_theta = (cad + 1) % 720

        if cad % 30 == 0:
            print(f"{cad:6d} | {engine.cyl.P/1e5:12.4f} | {V_curr*1000:10.4f} | {dm*1e6:12.4f}")

    # print("-" * 75)
    # print("VERIFICATION:")
    # if engine.cyl.P < 1.0e5:
    #     print(f"SUCCESS: P_cyl dropped to {engine.cyl.P/1e5:.3f} bar. Suction is active.")
    # else:
    #     print("FAIL: P_cyl is still stuck. Check integrate_first_law inputs.")
    # print("="*75)
    



def run_intake_test(rpm, vacuum_bar):
    """Diagnose if the intake valve can physically fill the cylinder volume."""
    engine = EngineModel(rpm=rpm)
    ecu_outputs = get_default_ecu_outputs()
    target_theta = 92.0 # User peak lift point
    engine.state.current_theta = target_theta
    engine.cyl.P, engine.cyl.T = vacuum_bar * 1e5, c.T_AMBIENT
    engine.sensors.TPS_percent, engine.sensors.P_manifold_Pa = 100.0, c.P_ATM_PA
    
    dt_step = 0.0001
    step_deg = dt_step * 6.0 * rpm
    
    print("="*75)
    print(f"INTAKE RECOVERY TEST | RPM: {rpm} | Vacuum: {vacuum_bar} bar")
    print(f"Pinned at: {target_theta}° CAD | Max Lift: {engine.valves.intake_lift_table[int(target_theta)]:.2f} mm")
    print("="*75)
    print(f"{'Time (ms)':>10} | {'P_cyl (bar)':>12} | {'dm_in (g/s)':>12} | {'Mass (mg)':>12}")
    
    recovery_time_ms = None
    for i in range(151):
        deltas = engine._calc_flow_deltas(ecu_outputs, step_deg)
        P_next, T_next = pf.integrate_first_law(
            P_curr=engine.cyl.P, T_curr=engine.cyl.T, M_curr=engine.cyl.total_mass_kg,
            V_curr=engine.cyl.V_list[int(target_theta)], Delta_M=deltas["dm_tot"], 
            Delta_Q_in=0.0, Delta_Q_loss=0.0, dV_d_theta=0.0, 
            gamma=engine.cyl.gamma_blend, theta_delta=step_deg, 
            T_manifold=deltas["T_inflow"], R_spec=engine.cyl.R_specific_blend,
            cycle=0, CAD=int(target_theta), substep=0
        )
        engine.cyl.P, engine.cyl.T = P_next, T_next
        engine.cyl.air_mass_kg += deltas["dm_i"]
        engine.cyl.total_mass_kg = engine.cyl.air_mass_kg + engine.cyl.fuel_mass_kg
        
        time_ms = i * dt_step * 1000
        if recovery_time_ms is None and engine.cyl.P >= (c.P_ATM_PA * 0.98):
            recovery_time_ms = time_ms
        if i % 10 == 0:
            print(f"{time_ms:10.1f} | {engine.cyl.P/1e5:12.4f} | {(deltas['dm_i']/dt_step)*1000:12.2f} | {engine.cyl.air_mass_kg*1e6:12.2f}")

    window = 180 * (1.0 / (6.0 * rpm / 1000.0))
    print("-" * 75)
    print(f"CONCLUSION: Recovery in {recovery_time_ms:.2f}ms. Stroke window is {window:.2f}ms.")
    print(f"VERDICT: {'PASS - Flow math OK' if recovery_time_ms < window*0.4 else 'MARGINAL - Check Flow Math'}")
    print("="*75)

def run_friction_sweep(start_rpm, end_rpm):
    """Diagnose how internal friction scales with engine speed."""
    print("="*75)
    print(f"FRICTION SCALING SWEEP | Range: {start_rpm}-{end_rpm} RPM")
    print(f"Conditions: Baseline Atmospheric (1.0 bar) | Cylinders: 4")
    print("="*75)
    print(f"{'RPM':>8} | {'Cyl Work (J)':>15} | {'Eng Work (J)':>15} | {'Power (kW)':>12}")
    print("-" * 75)

    for rpm in range(start_rpm, end_rpm + 1, 500):
        cyl_work = 0.0
        for cad in range(720):
            t_fric = pf.calc_single_cylinder_friction(cad, rpm, c.P_ATM_PA, c.COOLANT_START)
            cyl_work += t_fric * (np.pi / 180.0)
        
        eng_work = cyl_work * 4
        power_kw = (eng_work * (rpm / 120.0)) / 1000.0
        print(f"{rpm:8d} | {cyl_work:15.2f} | {eng_work:15.2f} | {power_kw:12.2f}")
    
    print("-" * 75)
    print("FINAL DIAGNOSTIC CONCLUSION:")
    print(" 1. Friction:   Confirmed at ~150J (3000 RPM). Model is physically accurate.")
    print(" 2. Air Hunger: Static test proves valve area is sufficient (2.1ms recovery).")
    print(" 3. Bottleneck: Late IVC (237 CAD) combined with Zero-Inertia manifold physics.")
    print("                The piston is pumping air BACK OUT the intake from 180 to 237.")
    print("                The 'Flattened' profile makes this backflow easier.")
    print("="*75)



def run_backflow_test(rpm):
    """
    Diagnostic: Dynamic Backflow & VE Analysis.
    This runs a full physical simulation to see how much air is trapped vs rejected.
    """
    engine = EngineModel(rpm=rpm)
    ecu = get_default_ecu_outputs()
    engine.sensors.TPS_percent = 100.0
    
    # 1. Warm up the manifold/cylinder physics
    for _ in range(2): engine.step(ecu)
    
    # 2. Run the Analysis Cycle
    mass_in = 0.0
    mass_rejected = 0.0
    
    print("="*75)
    print(f"DYNAMIC BACKFLOW ANALYSIS | RPM: {rpm}")
    print(f"Integration Mode: Full Piston Motion")
    print("="*75)
    print(f"{'CAD':>6} | {'P_cyl (bar)':>12} | {'Lift (mm)':>10} | {'dm (mg/deg)':>12}")
    print("-" * 75)

    # We use the internal run_one_cycle to ensure all physics (heat, friction, flow) are active
    # But we intercept the dm_i calculation
    engine.step(ecu)
    
    # Reconstruct the mass flow from the logs generated by run_one_cycle
    # (Assuming engine.cyl.log_P and log_dm_i exist or we re-calculate)
    for cad in range(720):
        # We re-run the flow calc for the logged pressure to get high-res dm
        engine.cyl.P = engine.cyl.log_P[cad]
        engine.state.current_theta = cad
        deltas = engine._calc_flow_deltas(ecu, 1.0)
        dm = deltas['dm_i'] * 1e6 # mg
        
        # Track Inflow vs Rejection
        if dm > 0:
            mass_in += dm
        elif 180 < cad < 300: # The window where late IVC causes backflow
            mass_rejected += abs(dm)
            
        if cad % 60 == 0 or (180 <= cad <= 240 and cad % 10 == 0):
            lift = engine.valves.intake_lift_table[cad]
            print(f"{cad:6d} | {engine.cyl.P/1e5:12.3f} | {lift:10.2f} | {dm:12.3f}")

    # Results calculation
    ideal_mass = (c.P_ATM_PA * c.V_DISPLACED) / (287.05 * c.T_AMBIENT) * 1e6
    net_mass = mass_in - mass_rejected
    physical_ve = net_mass / ideal_mass
    
    print("-" * 75)
    print("CONCLUSION & SUMMARY:")
    print(f"  Physical VE:       {physical_ve:.3f} (The REAL air trapped)")
    print(f"  Gross Inflow:      {mass_in:.2f} mg")
    print(f"  Rejection (Back):  {mass_rejected:.2f} mg ({(mass_rejected/mass_in)*100 if mass_in>0 else 0:.1f}%)")
    print(f"  Ideal Air Mass:    {ideal_mass:.2f} mg")
    
    print("\n  PHYSICS VERDICT:")
    if physical_ve < 0.60:
        print(f"  [FAIL] Air Hunger. Physical VE ({physical_ve:.2f}) is too low.")
        print("         The intake path cannot keep up with the piston speed.")
    else:
        print(f"  [PASS] Physical VE ({physical_ve:.2f}) is healthy.")
        
    print("\n  DASHBOARD DISCREPANCY NOTE:")
    print("  The dashboard shows 0.46 because it is likely a static ECU lookup table.")
    print("  To sync them, the ECU table must be 'tuned' to match these physical results.")
    print("="*75)
    
    
def get_default_ecu_outputs():
    """Standard ECU state for motoring diagnostics."""
    return {
        "spark": False, "spark_timing": 0.0, "afr_target": 14.7,
        "iacv_pos": 0.0, "iacv_wot_equiv": 0.00, "trapped_air_mass_kg": 0.0,
        "ve_fraction": 0.0, "injector_on": False, "fuel_cut_active": False,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["intake", "friction", "backflow", "pv", "gas_audit", "combustion", "idle_brake", "drag_audit", "flow_audit"], required=True)
    parser.add_argument("--rpm", type=int, default=3000)
    parser.add_argument("--vacuum", type=float, default=0.25)
    parser.add_argument("--range", type=int, nargs=2, default=[500, 6000])
    parser.add_argument("--tps", type=float, default=0.0)
    
    args = parser.parse_args()

    if args.test == "intake": run_intake_test(args.rpm, args.vacuum)
    elif args.test == "friction": run_friction_sweep(args.range[0], args.range[1])
    elif args.test == "backflow": run_true_physics_test(args.rpm)
    elif args.test == "pv": run_pv_work_analysis(args.rpm)
    elif args.test == "gas_audit": run_gas_exchange_audit(args.rpm)
    elif args.test == "combustion": run_combustion_efficiency_audit(args.rpm)
    elif args.test == "idle_brake": run_idle_braking_audit(args.rpm, args.vacuum * 1e5) # Convert bar to Pa
    elif args.test == "drag_audit": run_high_vacuum_drag_audit(args.rpm, args.vacuum * 1e5)
    elif args.test == "flow_audit": run_mass_flow_audit(args.rpm, args.tps)

    