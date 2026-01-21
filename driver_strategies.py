# driver_strategies.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import constants as c
from engine_model import FixedKeyDictionary
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Strategy classes — one per driver mode
# =============================================================================

class BaseStrategy:
    start_rpm = 900
        
    def driver_update(self, driver):
        """Return (tps, load, air pressure)"""
        tps = 0.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure

    
    def get_telemetry(self):
        """Return dict of extra keys for dashboard"""
        return {}
    
    # def plot_strategy_panel(self, dashboard_manager):
    #     ax, ax_bottom = dashboard_manager.get_strategy_axes()
    #     if ax is None:
    #         return
    #     ax.set_title(f"{self.__class__.__name__} Mode")
    #     ax.text(0.5, 0.5, "No custom visualization\n(Base telemetry active)", 
    #             ha='center', va='center', transform=ax.transAxes, fontsize=16)
    #     ax.axis('off')

    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        """Called every cycle — delegate to manager for shared realtime loop"""
        pass  # default: nothing


class IdleStrategy(BaseStrategy):
    """ Engine start from cranking RPM to idle RPM """
    def __init__(self):
        self.start_rpm = 250
    
    def driver_update(self, driver):
        tps = 0.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure

    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        ax_top, ax_topright, _, _ = dashboard_manager.get_strategy_axes()
        if ax_top is None or ax_topright is None or data is None:
            return

        sensors, engine_data, ecu_outputs = data

        if not hasattr(self, 'artists_created'):
            # --- Top Plot: PID Response (RPM vs Target & Valve %) ---
            ax_top.clear()
            ax_top.set_title("Idle PID Response")
            ax_top.set_xlabel("Crank Angle (deg)")
            ax_top.set_ylabel("RPM", color='blue')
            ax_top.grid(True, alpha=0.3)
            
            self.rpm_line, = ax_top.plot([], [], color='blue', label='Actual RPM')
            self.target_line, = ax_top.plot([], [], color='cyan', linestyle='--', label='Target')
            
            # Secondary axis for Valve Duty Cycle
            self.ax_valve = ax_top.twinx()
            self.ax_valve.set_ylabel("IACV Opening %", color='red')
            self.valve_line, = self.ax_valve.plot([], [], color='red', alpha=0.6, label='Valve %')
            
            # --- Bottom Plot: PID Component Split ---
            ax_topright.clear()
            ax_topright.set_title("PID Component Contribution")
            ax_topright.set_xlabel("Crank Angle (deg)")
            ax_topright.set_ylabel("Output Contribution")
            ax_topright.grid(True, alpha=0.3)

            self.p_trace, = ax_topright.plot([], [], color='orange', label='P (Prop)')
            self.i_trace, = ax_topright.plot([], [], color='purple', label='I (Integ)')
            self.d_trace, = ax_topright.plot([], [], color='brown', label='D (Deriv)')
            ax_topright.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_topright.legend(loc='upper right', fontsize='xx-small')
            
            self.artists_created = True

        # === DATA PROCESSING ===
        log_cad = engine_data.get('theta_list')
        rpm_history = sensors.rpm_history # Assumes 720 length array
        
        # Map ECU outputs (Assume your _idle_pid logic exposes these internal terms)
        # If not available in dict, these will default to 0 for plotting
        p_val = ecu_outputs['pid_P']
        i_val = ecu_outputs['pid_I']
        d_val = ecu_outputs['pid_D']
        valve_pct = ecu_outputs['idle_valve_position']
        target_rpm = ecu_outputs['target_rpm']

        # --- 1. Update RPM Trace (Top) ---
        self.rpm_line.set_data(log_cad, rpm_history)
        self.target_line.set_data([0, 720], [target_rpm, target_rpm])
        self.valve_line.set_data(log_cad, np.full_like(log_cad, valve_pct))
        
        ax_top.set_xlim(0, 720)
        ax_top.set_ylim(0, max(2000, np.max(rpm_history) * 1.2))
        self.ax_valve.set_ylim(0, 100)

        # --- 2. Update PID Components (Bottom) ---
        # Plotting static values across the cycle to see current "effort" levels
        self.p_trace.set_data(log_cad, np.full_like(log_cad, p_val))
        self.i_trace.set_data(log_cad, np.full_like(log_cad, i_val))
        self.d_trace.set_data(log_cad, np.full_like(log_cad, d_val))
        
        ax_topright.set_xlim(0, 720)
        ax_topright.set_ylim(-50, 100) # Range for PID terms

        # === IDLE DEBUG TELEMETRY (Left Table) ===
        # Calculate actual AFR error
        target_afr = ecu_outputs['afr_target']
        afr_error = sensors.afr - target_afr

        lines = [
            f"CYCLE:          {current_cycle:8.0f}",
            f"RPM AVG:        {np.mean(rpm_history):8.1f}",
            "----------------------------",
            f"MAP:            {sensors.MAP_kPa:8.2f} kPa",
            f"AFR ACTUAL:     {sensors.afr:8.2f}",
            f"AFR TARGET:     {target_afr:8.2f}",
            f"AFR ERROR:      {afr_error:+8.2f}",
            "----------------------------",
            f"IACV POSITION:  {valve_pct:8.1f} %",
            f"WOT EQUIV:      {ecu_outputs['idle_pos_wot']:8.4f}",
            f"IGNITION ADV:   {ecu_outputs['spark_timing']:8.1f} °",
            "",
            "ENERGY AUDIT (Joules):",
            f"Expansion (Ind):{engine_data['work_expansion_j']:+8.1f} J",
            f"Friction Loss:  {engine_data['friction_work_j']:8.1f} J",
            f"Pumping Loss:   {engine_data['work_pumping_j']:8.1f} J",
            "----------------------------",
            f"NET BALANCE:    {engine_data['net_work_j'] + engine_data['friction_work_j']:+8.1f} J",
        ]
        
        # Diagnosis Logic
        if sensors.afr > 20:
            lines.append("DIAG:  EXTREME LEAN / STALL")
        elif sensors.rpm < target_rpm - 100 and valve_pct > 95:
            lines.append("DIAG:  INSUFFICIENT AIR (MAXED)")
        elif abs(afr_error) < 0.5 and (engine_data['net_work_j'] + engine_data['friction_work_j']) < 0:
            lines.append("DIAG:  HIGH MECHANICAL DRAG")
        else:
            lines.append("DIAG:  IDLE STABLE")
        
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()


class WotStrategy(BaseStrategy):
    """ Engine start from cranking RPM to idle RPM """
    def __init__(self):
        self.start_rpm = 900
    
    def driver_update(self, driver):
        tps = 100.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure
    
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = dashboard_manager.get_strategy_axes()
        if any(v is None for v in (ax_topleft, ax_topright, ax_bottomleft, ax_bottomright, data)):
            return
        
        sensors, engine_data, ecu_outputs = data
        theta_vec = engine_data.get('theta_list')
        p_bar = engine_data['log_P'] / 1e5
        v_liters = engine_data['log_V'] * 1000.0

        # --- 1. Top Left: P-V Loop (High Pressure Scale) ---
        ax_topleft.clear()
        ax_topleft.set_title("Fired P-V Loop (WOT)")
        ax_topleft.plot(v_liters, p_bar, color='red', linewidth=2)
        ax_topleft.set_ylabel("Pressure (Bar)")
        ax_topleft.set_ylim(-1, 60) # Scaled for combustion
        ax_topleft.grid(True, alpha=0.3)

        # --- 2. Top Right: Cylinder Pressure Trace ---
        ax_topright.clear()
        ax_topright.set_title("Cylinder Pressure vs. CAD")
        ax_topright.plot(theta_vec, p_bar, color='black')
        ax_topright.set_ylabel("Bar")
        ax_topright.set_xlim(0, 720)
        ax_topright.axvline(360, color='red', linestyle='--', alpha=0.3) # TDC
        ax_topright.grid(True, alpha=0.3)

        # --- 3. Bottom Left: Cumulative Work ---
        ax_bottomleft.clear()
        ax_bottomleft.set_title("Cumulative Work Trace")
        ax_bottomleft.plot(theta_vec, engine_data['work_deg'], color='blue')
        ax_bottomleft.set_ylabel("Joules")
        ax_bottomleft.grid(True, alpha=0.3)

        # --- 4. Bottom Right: Heat Release / Burn Profile ---
        # Assuming your engine_data logs the heat addition
        ax_bottomright.clear()
        ax_bottomright.set_title("Heat Release (Q_in)")
        q_in = engine_data.get('log_Q_in', np.zeros(720))
        ax_bottomright.fill_between(theta_vec, q_in, color='orange', alpha=0.5)
        ax_bottomright.set_ylabel("Joules/Deg")
        ax_bottomright.set_xlim(300, 450) # Zoom into combustion window
        ax_bottomright.grid(True, alpha=0.3)

        # --- 5. THE TABLE (LINE Variable) ---
        net_balance = engine_data['net_work_j'] + engine_data['friction_work_j']
        
        lines = [
            f"WOT POWER AUDIT - RPM: {sensors.rpm:4.0f}",
            "----------------------------",
            f"Brake Torque:   {np.average(engine_data['torque_history']):8.1f} Nm",
            f"Peak Pressure:  {engine_data['peak_pressure_bar']:8.2f} Bar",
            f"P_Peak Angle:   {engine_data['P_peak_angle']:8.1f}°",
            "----------------------------",
            "ENERGY BREAKDOWN (J):",
            f"Indicated Work: {engine_data['net_work_j']:8.1f} J",
            f"Friction Loss:  {engine_data['friction_work_j']:8.1f} J",
            f"Pumping Loss:   {engine_data['work_pumping_j']:8.1f} J",
            f"NET BALANCE:    {net_balance:+8.1f} J",
            "----------------------------",
            "STATUS: " + ("ACCELERATING" if net_balance > 0 else "STALLED/DRAGGING")
        ]
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()
    
    
class MotoringStrategy(BaseStrategy):
    """ Motor the engine with spark and fuel disabled """
    def __init__(self):
        self.start_rpm = 250
        self.motoring_enabled = True
        
    def driver_update(self, driver):
        tps = 100.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure
   
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        # Unpack the two axes from the dashboard manager
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = dashboard_manager.get_strategy_axes()
        if any(v is None for v in (ax_topleft, ax_topright, ax_bottomleft, ax_bottomright, data)):
            return
        
        plot1 = ax_topleft
        plot2 = ax_bottomleft
        plot3 = ax_topright
        plot4 = ax_bottomright

        sensors, engine_data, ecu_outputs = data

        # === Setup Static Visuals for all plots ===
        if not hasattr(self, 'artists_created'):
            # --- Plot 1 (P-V Loop) ---
            plot1.clear()
            plot1.set_title("P-V Loop - 4-Stroke Cycle")
            plot1.set_xlabel("Cylinder Volume (L)")
            plot1.set_ylabel("Pressure (Bar)")
            plot1.grid(True, alpha=0.3)

            self.intake_line, = plot1.plot([], [], color='cyan', linewidth=2, label='Intake')
            self.comp_line,   = plot1.plot([], [], color='green', linewidth=2, label='Compression')
            self.power_line,  = plot1.plot([], [], color='red', linewidth=2, label='Expansion')
            self.exh_line,    = plot1.plot([], [], color='orange', linewidth=2, label='Exhaust')
            plot1.legend(loc='upper right', fontsize='x-small')

            # --- Plot 2 (Work vs. Theta) ---
            plot2.clear()
            plot2.set_title("Cumulative Work vs. CAD")
            plot2.set_xlabel("Crank Angle (deg)")
            plot2.set_ylabel("Work (Joules)")
            plot2.grid(True, alpha=0.3)

            self.work_trace, = plot2.plot([], [], color='red', linewidth=2)
            plot2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            # --- Plot 3 Valve Mechanical Sync ---
            plot3.clear()
            plot3.set_title("Valve Area & Lift")
            plot3.grid(True, alpha=0.3)

            # Area traces (Primary Y - Left)
            self.a_int_line, = plot3.plot([], [], color='blue', linewidth=2, label='Intake Area')
            self.a_exh_line, = plot3.plot([], [], color='orange', linewidth=2, label='Exhaust Area')
            plot3.set_ylabel("Area (m²)")

            # Lift traces (Secondary Y - Right)
            self.ax_lift = plot3.twinx()
            self.l_int_line, = self.ax_lift.plot([], [], color='blue', linestyle='--', alpha=0.5, label='Intake Lift')
            self.l_exh_line, = self.ax_lift.plot([], [], color='orange', linestyle='--', alpha=0.5, label='Exhaust Lift')
            self.ax_lift.set_ylabel("Lift (m)")

            # --- Plot 4: Pumping Loop Detail (Gas Exchange) ---
            plot4.clear()
            plot4.set_title("Gas Exchange Pressure (Bar)")
            plot4.set_xlabel("Crank Angle (deg)")
            plot4.grid(True, alpha=0.3)
            self.p_pump_trace, = plot4.plot([], [], color='black', linewidth=1.5)
            # Reference line at Atmospheric Pressure
            plot4.axhline(1.013, color='red', linestyle='--', alpha=0.5, label='Amb')
            
            self.artists_created = True


        # === DATA PROCESSING ===
                
        # --- PV and Work charts
        log_p = engine_data.get('log_P') # Pressure in Pa
        log_v = engine_data.get('log_V') # Volume in m^3
        
        # Calculate instantaneous work: dW = P * dV
        # Cumulative sum gives the work profile across the 720 degrees
        if log_p is not None and log_v is not None:
            dv = np.diff(log_v, prepend=log_v[0])
            work_array = engine_data.get('work_deg')
            log_cad = engine_data.get('theta_list') 
            v_liters = log_v * 1000.0
            p_bar = log_p / 1e5
            
            # --- 1. Update P-V Loop (Top Right) ---
            idx_intake = (log_cad < 180)
            idx_comp   = (log_cad >= 180) & (log_cad < 360)
            idx_power  = (log_cad >= 360) & (log_cad < 540)
            idx_exh    = (log_cad >= 540)

            self.intake_line.set_data(v_liters[idx_intake], p_bar[idx_intake])
            self.comp_line.set_data(v_liters[idx_comp], p_bar[idx_comp])
            self.power_line.set_data(v_liters[idx_power], p_bar[idx_power])
            self.exh_line.set_data(v_liters[idx_exh], p_bar[idx_exh])
            
            plot1.set_xlim(np.min(v_liters) * 0.9, np.max(v_liters) * 1.1)
            # plot1.set_ylim(-0.5, 4)
            plot1.set_ylim(-0.5, max(15, np.max(p_bar) * 1.1))

            # --- 2. Update Work vs Theta (Bottom Left) ---
            self.work_trace.set_data(log_cad, work_array)
            plot2.set_xlim(0, 720)
            plot2.set_xticks([0, 180, 360, 540, 720])
            
            # Auto-scale Y for work trace
            y_min, y_max = np.min(work_array), np.max(work_array)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            plot2.set_ylim(y_min - margin, y_max + margin)
        
        # --- Valve and Pressure charts
        area_int_vec = engine_data.get('intake_area_vec') 
        area_exh_vec = engine_data.get('exhaust_area_vec')
        lift_int_vec = engine_data["intake_lift_vec"]
        lift_exh_vec = engine_data["exhaust_lift_vec"]
        theta_vec = engine_data.get('theta_list')

        if area_int_vec is not None and area_exh_vec is not None:
            self.a_int_line.set_data(theta_vec, area_int_vec)
            self.a_exh_line.set_data(theta_vec, area_exh_vec)
            plot3.set_xlim(0, 720)
            plot3.set_ylim(0, max(np.max(area_int_vec), np.max(area_exh_vec)) * 1.2)
            
        if lift_int_vec is not None and lift_exh_vec is not None:
            self.l_int_line.set_data(theta_vec, lift_int_vec)
            self.l_exh_line.set_data(theta_vec, lift_exh_vec)
            self.ax_lift.set_ylim(0, max(np.max(lift_int_vec), np.max(lift_exh_vec)) * 1.1)
            
            # Calculate the points
            ivo_1mm, ivc_1mm = self.get_lift_timing(theta_vec, lift_int_vec, threshold=1.0) # 0.001m = 1mm
            evo_1mm, evc_1mm = self.get_lift_timing(theta_vec, lift_exh_vec, threshold=1.0)
                       
            # Create a string for the chart overlay
            timing_str = (
                f"1mm Timing:\n"
                f"IVO: {ivo_1mm:3.0f}°  IVC: {ivc_1mm:3.0f}°\n"
                f"EVO: {evo_1mm:3.0f}°  EVC: {evc_1mm:3.0f}°"
            )

            # Use coordinate transform 'axes fraction' so (0,0) is bottom-left and (1,1) is top-right
            if not hasattr(self, 'timing_text'):
                self.timing_text = plot3.text(0.5, 0.95, timing_str, 
                                                    transform=plot3.transAxes,
                                                    fontsize=9, family='monospace',
                                                    ha='center', va='top',
                                                    bbox=dict(boxstyle="round", alpha=0.6, facecolor='white'))
            else:
                self.timing_text.set_text(timing_str)
    

        # 2. Update Pumping Pressure (Bottom-Right)
        # We use the full 720 P list but zoom the Y-axis
        p_bar = engine_data['log_P'] / 1e5
        self.p_pump_trace.set_data(theta_vec, p_bar)
        plot4.set_xlim(0, 720)
        plot4.set_xticks([0, 180, 360, 540, 720])
        
        # CRITICAL ZOOM: 
        # We want to see the "pumping" detail. 
        # If it's a Motoring run, P rarely exceeds 15-20 bar (compression).
        # But the 'trap' happens between 0.5 and 3.0 bar.
        plot4.set_ylim(0, 2.0) 
        # plot4.set_ylim(0, max(p_bar)*1.1)
        
        # Temporary debug print in your update loop
        print(f"P_bar range: {np.min(p_bar):.3f} to {np.max(p_bar):.3f}")


        
            

        # === Energy Audit Overlay (Bottom Left Table) ===
        net_work = engine_data["net_work_j"]
        fric_work = engine_data["friction_work_j"]
        total_balance = net_work + fric_work

        lines = [
            # f"MOTORING CYCLE: {current_cycle:8.0f}",
            # f"RPM:            {sensors.rpm:8.1f}",
            # "----------------------------",
            # f"dm_in Total:  {np.sum(engine_data['dm_in']):8.4f} kg",
            # f"dm_out Total: {np.sum(engine_data['dm_out']):8.4f} kg",
            # f"dm Flow Var:  {np.std(engine_data['dm_in']):8.6f}", # High variance = oscillation
            # "----------------------------",
            f"T_ind_avg:      {np.average(engine_data['torque_I_history']):8.1f} Nm",
            f"T_fric_avg:     {np.average(engine_data['torque_F_history']):8.1f} Nm",
            f"T_brake_avg:    {np.average(engine_data['torque_history']):8.1f} Nm",
            "----------------------------",
            f"Peak Press:     {engine_data['peak_pressure_bar']:8.2f} Bar",
            f"P_peak_angle:      {engine_data['P_peak_angle']:3.0f}",
            "",
            "ENERGY AUDIT (Joules):",
            f"Compression:    {engine_data['work_compression_j']:8.1f} J",
            f"Expansion:      {engine_data['work_expansion_j']:+8.1f} J",
            f"Pumping Loss:   {engine_data['work_pumping_j']:8.1f} J",
            f"Friction Loss:  {fric_work:8.1f} J",
            "----------------------------",
            f"NET BALANCE:    {total_balance:+8.1f} J",
        ]
        
        # Status Logic
        work_exp = engine_data.get('work_expansion_j', 0)
        work_comp = abs(engine_data.get('work_compression_j', 0))
        
        if work_comp > 0 and work_exp < (work_comp * 0.8):
            lines.append("STATUS:         CRITICAL LEAK")
        elif total_balance > 5.0: # Threshold for numerical "ghost" energy
            lines.append("STATUS:         PHYSICS ERROR (+)")
        else:
            lines.append("STATUS:         HEALTHY")
        
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()
        
    def get_lift_timing(self, theta, lift, threshold=1.0): #mm
        # Find indices where lift is above threshold
        above = np.where(lift >= threshold)[0]
        if len(above) == 0:
            return None, None
            
        # Check for wrapping: are the indices contiguous?
        # np.diff(above) > 1 finds the "gap" in the array of indices
        diffs = np.diff(above)
        gap_indices = np.where(diffs > 1)[0]
        
        if len(gap_indices) > 0:
            # WRAPPED CASE: e.g., [0, 1, 2 ... 718, 719]
            # The true "Opening" is the first index AFTER the gap
            # The true "Closing" is the index BEFORE the gap
            opened = theta[above[gap_indices[0] + 1]]
            closed = theta[above[gap_indices[0]]]
        else:
            # STANDARD CASE: e.g., [100, 101, ... 200]
            opened = theta[above[0]]
            closed = theta[above[-1]]
            
        return opened, closed


class RoadtestStrategy(BaseStrategy):
    """Realistic open-road driving: hills, varying throttle, altitude effects"""
    
    CYCLE_LENGTH = 400  # cycles for one full "drive" loop (~80–40 sec depending on RPM)
    
    def __init__(self):
        # Pre-compute profiles for speed (no per-cycle computation overhead)
        cycles = np.arange(RoadtestStrategy.CYCLE_LENGTH)
        phase = cycles / RoadtestStrategy.CYCLE_LENGTH * 2 * np.pi
        
        # Elevation: rolling hills + big climb
        self.elevation_m = 200 + 100 * np.sin(phase * 3) + 150 * np.sin(phase * 0.8 + 1)
        
        # Grade (%) from elevation gradient
        self.grade_pct = np.gradient(self.elevation_m) * 20  # scaled to realistic range
        self.grade_pct = np.clip(self.grade_pct, -10, 12)
        
        # Throttle: base cruise oscillation + extra on climbs
        self.base_tps = 30 + 40 * (0.5 + 0.5 * np.sin(phase * 2.5))
        self.climb_boost = 40 * (self.grade_pct > 4)
        self.tps_profile = np.clip(self.base_tps + self.climb_boost, 10, 100)
        
        # Ambient pressure drop with altitude
        self.ambient_pressure_pa = c.P_ATM_PA - 11.3 * (self.elevation_m - 200)
        self.ambient_pressure_pa = np.maximum(self.ambient_pressure_pa, 75000)  # ~2500m max
        
        # Road load (engine torque Nm)
        m_kg = 1500
        g = 9.81
        roll_N = 0.012 * m_kg * g
        grade_N = m_kg * g * np.sin(np.arctan(self.grade_pct / 100))
        # Simple aero + base
        aero_base_N = 100 + 300 * np.power(np.linspace(20, 80, RoadtestStrategy.CYCLE_LENGTH)/50, 2)
        total_load_N = roll_N + grade_N + aero_base_N
        self.load_profile = np.clip(total_load_N * 0.15, 0, 550)  # tuned to realistic engine torque range

    def driver_update(self, driver):
        cycle_idx = driver.cycle % RoadtestStrategy.CYCLE_LENGTH
        self.current_cycle_idx = cycle_idx  # for any future get_telemetry use

        tps = self.tps_profile[cycle_idx]
        load = self.load_profile[cycle_idx]
        pressure = self.ambient_pressure_pa[cycle_idx]

        return tps, load, pressure

    def get_telemetry(self):
        idx = getattr(self, 'current_cycle_idx', 0)
        return {
            "road_grade_%": round(self.grade_pct[idx], 1),
            "elevation_m": round(self.elevation_m[idx]),
            "road_load_Nm": round(self.load_profile[idx]),
        }
    
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        ax, ax_bottom = dashboard_manager.get_strategy_axes()
        if ax is None:
            return

        progress = current_cycle % RoadtestStrategy.CYCLE_LENGTH
        current_idx = int(progress)

        # === FIRST CALL: Create all artists once ===
        if not hasattr(self, 'artists_created'):
            ax.clear()
            ax.set_title("Road Test - Mountain Drive Profile")
            ax.set_xlabel("Cycle (loop = 400 cycles)")
            ax.set_ylabel("Throttle % / Load (Nm)")
            ax.set_ylim(0, max(110, self.load_profile.max() + 50))
            ax.grid(True, alpha=0.3)

            cycles_x = np.arange(RoadtestStrategy.CYCLE_LENGTH)

            # Terrain fill — store the PolyCollection directly (no [0])
            self.terrain_fill = ax.fill_between(cycles_x, self.elevation_m - 200, -100,
                                                color='lightgray', alpha=0.4)

            # Throttle line
            self.throttle_line, = ax.plot(cycles_x, self.tps_profile,
                                          color='orange', linewidth=3, label='Throttle %')

            # Load line
            self.load_line, = ax.plot(cycles_x, self.load_profile,
                                      color='green', linewidth=3, label='Road Load (Nm)')

            # Progress marker (vertical red line)
            self.progress_line = ax.axvline(progress, color='red', linestyle='--', linewidth=2)

            # Grade on twin axis
            self.ax2 = ax.twinx()
            self.grade_line, = self.ax2.plot(cycles_x, self.grade_pct,
                                             color='brown', linewidth=2, alpha=0.7, label='Grade %')
            self.ax2.set_ylabel("Grade (%)", color='brown')
            self.ax2.tick_params(axis='y', labelcolor='brown')

            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            self.artists_created = True

        else:
            # === SUBSEQUENT CALLS: Only update the moving part ===
            self.progress_line.set_xdata([progress, progress])

        # === Always update bottom-left strategy overlay table ===
        lines = [
            f"Cycle:         {current_cycle:8.0f}",
            f"Progress:      {progress:.0f}/{RoadtestStrategy.CYCLE_LENGTH}",
            "",
            f"Grade:         {self.grade_pct[current_idx]:+8.1f} %",
            f"Elevation:     {self.elevation_m[current_idx]:8.0f} m",
            f"Road Load:     {self.load_profile[current_idx]:8.0f} Nm",
            f"Throttle:      {self.tps_profile[current_idx]:8.1f} %",
            f"Amb Press:     {self.ambient_pressure_pa[current_idx]/100:8.0f} hPa",
        ]
        dashboard_manager.update_strategy_overlay(lines)

        dashboard_manager.draw()


class DynoStrategy(BaseStrategy):
    """ a variable load Dyno with a full rpm sweep """
    
    # PID paramters for controller to hold rpm while at WOT.
    DYNO_KP = 1.0  # Proportional Gain 1.5 last know best.
    DYNO_KI = 0.003 # Integral Gain
    DYNO_KD = 3.5  # Derivative Gain was 1.5 last known best
    DYNO_BACK_CALC_GAIN = 0.0   # Usually 0.3–1.0 — how aggressively we unwind
    MAX_DYNO_TORQUE = 2000     # if your engine peaks at ~500 Nm → 1.5× headroom

    # Dyno start and RPM stability parameters
    DYNO_START_RPM = 2000
    DYNO_FINISH_RPM = c.RPM_LIMIT - 500
    DYNO_STEP_SIZE_RPM = 200 # How far to step the RPM target after settling
    DYNO_RPM_TOLERANCE = 5.0
    SETTLE_CYCLES_REQUIRED = 1 * 720  # it is measured in degrees
           
    def __init__(self, rl_dyno_mode=False):
        super().__init__()
        
        # rl support
        self.rl_dyno_mode = rl_dyno_mode
        self.external_target_rpm = 2000.0  # Controlled by RL
        self.force_next_step = False       # Flag for RL to trigger data capture
        
        # pid cycles
        self.in_dyno_sweep = False
        self.settled_cycles = 0
        self.current_step_peak = 0.0

        # pid parameters
        self.pid_error_p = 0.0
        self.pid_derivative = 0.0
        self.pid_error_integral = 0.0
        self._filtered_drpm = 0.0

        # pid Scoring for comparing different pid paramaters
        self.total_settle_cycles = 0.0
        self.steps_completed = 0.0
        self.total_peak_error = 0.0
        self.max_peak_error = 0.0

        # pid output
        self.pid_output = 0.0
        self.dyno_target_rpm = DynoStrategy.DYNO_START_RPM
        self.settled_torque_values = np.zeros(720)

        # data for Dyno Chart plot
        self.torque_data = []  # list of (rpm, filtered_torque)
        self.dyno_curve_data = []  # list of (rpm_target, avg_torque_Nm, avg_power_kW)
        self.line_torque = None
        self.line_power = None
        
        # Stablization flags for downstream services
        self.point_settled = False
        self.last_settled_rpm = 0.0
        self.last_settled_torque = 0.0
        self.last_settled_power = 0.0
        self.last_settled_std = 0.0
        self.dyno_complete = False
           
    def driver_update(self, driver):
        tps = 100.0 # always WOT in dyno mode.
        
        # inject RL target RPM
        if self.rl_dyno_mode:
            self.dyno_target_rpm = self.external_target_rpm
        
        # Start DYNO once target rpm reached (uses instantaneous rpm with hysteresis)  
        low_threshold = c.IDLE_RPM  # e.g., 1800
        high_threshold = DynoStrategy.DYNO_START_RPM - 100 # e.g., 1900    
        if not self.in_dyno_sweep:
            # We only wake up if we cross the high bar
            if driver.rpm >= high_threshold:
                self.in_dyno_sweep = True
        else:
            # Once we are on, we stay on unless we fall below the low bar
            if driver.rpm < low_threshold:
                self.in_dyno_sweep = False
                print(f"DEBUG: fell out of dyno sweep at rpm: {driver.rpm:4.0f}")
        
        
        
        # control_rpm = np.mean(driver.cycle_rpm)
        # if control_rpm >= (DynoStrategy.DYNO_START_RPM - 100):
        # # if control_rpm >= (DynoStrategy.DYNO_START_RPM - 100) and control_rpm <= DynoStrategy.DYNO_FINISH_RPM:
        #     self.in_dyno_sweep = True
        #     tps = 100.0
        # else:
        #     self.in_dyno_sweep = False
        #     tps = 100.0  # keep WOT until end
   
        # Dyno load
        self._set_dyno_target(driver)
        load = self._pid_control(driver) if self.in_dyno_sweep else 0.0
  
        # print(
        #     f"theta: {driver.cycle:4.0f}/{driver.theta:3.0f} | "
        #     # f"P_err: {self.pid_error_p:5.1f} | "
        #     # f"Integral: {self.pid_error_integral:5.0f} | "
        #     # f"derivative: {self.pid_derivative:5.0f} | "
        #     # f"pid out: {self.pid_output:6.2f} | "
        #     f"RPM: {driver.rpm:4.0f} | "
        #     f"target_rpm: {self.dyno_target_rpm:4.0f} | "
        #     f"mean_rpm: {np.mean(driver.cycle_rpm):4.0f} | "
        #     f"in_dyno_sweep: {self.in_dyno_sweep}"
        #     )
        
        return tps, load, c.P_ATM_PA
 
    def get_telemetry(self):
        return {
            "dyno_settled": (self.settled_cycles >= DynoStrategy.SETTLE_CYCLES_REQUIRED),
            "target_rpm": self.dyno_target_rpm,
            "in_dyno_sweep": self.in_dyno_sweep,
        }   

    def _set_dyno_target(self, driver):
        
        if not self.in_dyno_sweep:
            # print(f"== NOT IN DYNO MODE == "
            #       f"target: {self.dyno_target_rpm:4.0f} | "
            #       f"mean_rpm: {np.mean(driver.cycle_rpm):4.0f} | "
            #       f"instant_rpm {driver.rpm:4.0f} | "
            #       )
            return
        
        if self._rpm_stablised(driver): 
            if self.rl_dyno_mode:
                self.point_settled = True   # TRIGGER EVENT for RL training: Point is settled
                # self._handle_rl_progression(driver)
            else:
                self._handle_automatic_progression(driver)
        self._handle_rl_progression(driver) # RL can skip RPMs, manual mode cannot.
            
    def _rpm_stablised(self, driver):
        """
        RPM must be withiin error tolerance for set number of cycles
        """
        control_rpm = np.mean(driver.cycle_rpm)
        error = abs(self.dyno_target_rpm - control_rpm)
        
        # Track peak error for the stabilization period
        if error > self.current_step_peak:
            self.current_step_peak = error

        # Point settle detection
        if error < DynoStrategy.DYNO_RPM_TOLERANCE:
            self.settled_torque_values[self.settled_cycles%720] = self.pid_output
            self.settled_cycles += 1      
        else:
            self.settled_cycles = 0

        # Window settle detection 
        if self.settled_cycles >= DynoStrategy.SETTLE_CYCLES_REQUIRED:
            rpm_stablised = True
        else:
            rpm_stablised = False

        return rpm_stablised
       
    def _store_stabilized_data(self):
        """
        Clean the data and compute final values for this point
        """
        raw_torque = np.mean(self.settled_torque_values)
        std_dev = np.std(self.settled_torque_values)
        # Trimmed mean (drop top/bottom 10%)
        boundary = int(0.10 * DynoStrategy.SETTLE_CYCLES_REQUIRED) #trim 10% top and bottom
        trimmed = np.sort(self.settled_torque_values)[boundary:-boundary]
        avg_brake_torque = np.mean(trimmed)
        avg_power_kW = avg_brake_torque * self.dyno_target_rpm / 9549.0
        
        # === STORE CLEAN POINT DATA (for RL env and others) ===
        self.last_settled_rpm = self.dyno_target_rpm
        self.last_settled_torque = avg_brake_torque
        self.last_settled_power = avg_power_kW
        self.last_settled_std = std_dev

        # Append to curve for dashboard/history
        self.dyno_curve_data.append((self.dyno_target_rpm, avg_brake_torque, avg_power_kW))

        # Stats tracking
        self.total_peak_error += self.current_step_peak
        if self.current_step_peak > self.max_peak_error:
            self.max_peak_error = self.current_step_peak

    def _handle_rl_progression(self, driver):
        """
        Logic for when RL is in control. 
        It waits for 'force_next_step' to log a data point.
        """
        if self.force_next_step:
            # Store the stablised rpm and torque data
            self._store_stabilized_data()
            # Update cycle counters
            self.total_settle_cycles += DynoStrategy.SETTLE_CYCLES_REQUIRED
            self.steps_completed += 1
            self.force_next_step = False 
            # Note: We do NOT increment dyno_target_rpm here; 
            # the RL agent will update external_target_rpm for the next step.
      
    def _handle_automatic_progression(self, driver):
        """
        Logic for when running in manual dyno mode. 
        automatically increments RPM target.
        """
        
        # Store the stablised rpm and torque data
        self._store_stabilized_data()
        
        # Update cycle counters
        self.total_settle_cycles += DynoStrategy.SETTLE_CYCLES_REQUIRED
        self.steps_completed += 1

        # Dashboard update
        if getattr(driver, 'dashboard_manager', None):
            ax, ax_bottom = driver.dashboard_manager.get_strategy_axes()
            if ax:
                if not hasattr(self, 'line_torque') or self.line_torque is None:
                    self._create_dyno_plot(ax)
                self._update_dyno_plot(ax)
                driver.dashboard_manager.draw()
                    
        # Prepare for next point
        self.settled_cycles = 0
        self.current_step_peak = 0.0
        self.settled_torque_values.fill(0)  # optional: clear array

        # Advance rpm target
        if self.dyno_target_rpm < DynoStrategy.DYNO_FINISH_RPM:
            self.dyno_target_rpm += DynoStrategy.DYNO_STEP_SIZE_RPM
        else:
            self.in_dyno_sweep = False
            self.dyno_complete = True
            driver.tps = 0.0

    def _pid_control(self, driver):
        instant_rpm = driver.rpm
        mean_rpm = np.mean(driver.cycle_rpm)

        # 25° low-pass for P and I
        window_size = 25
        start_idx = (driver.theta - window_size) % 720
        if start_idx < driver.theta:
            smooth_rpm = np.mean(driver.cycle_rpm[start_idx:driver.theta])
        else:
            wrapped = np.concatenate((driver.cycle_rpm[start_idx:], driver.cycle_rpm[:driver.theta]))
            smooth_rpm = np.mean(wrapped)

        # ------------------------------------------------------------------
        # 2. Errors
        # ------------------------------------------------------------------
        error_raw = instant_rpm - self.dyno_target_rpm
        error_smooth = smooth_rpm - self.dyno_target_rpm

        # ------------------------------------------------------------------
        # 3. Proportional – on lightly filtered RPM
        # ------------------------------------------------------------------
        P = DynoStrategy.DYNO_KP * error_raw
        self.pid_error_p = P

        # ------------------------------------------------------------------
        # 4. Derivative – on instantaneous rate (proper sign & filtered)
        # ------------------------------------------------------------------
        prev_rpm = driver.cycle_rpm[(driver.theta - 1) % 720]
        rpm_rate = instant_rpm - prev_rpm # rate of change
        
        # Light filtering of derivative to kill noise but keep phase
        alpha = 0.3
        self._filtered_drpm = alpha * rpm_rate + (1 - alpha) * self._filtered_drpm
        
        D = - DynoStrategy.DYNO_KD * self._filtered_drpm # opposes acceleration
        self.pid_derivative = D

        # ------------------------------------------------------------------
        # 5. Integral and PID sum
        # ------------------------------------------------------------------
        pid_unclamped = P + self.pid_error_integral + D
        pid_clamped = max(0.0, min(DynoStrategy.MAX_DYNO_TORQUE, pid_unclamped))

        saturation_error = pid_clamped - pid_unclamped
        # integral with unwind with backgain
        self.pid_error_integral += DynoStrategy.DYNO_KI * error_smooth + DynoStrategy.DYNO_BACK_CALC_GAIN * saturation_error

        # ------------------------------------------------------------------
        # 6. Iapply brake torque directlt
        # ------------------------------------------------------------------
        self.pid_output = pid_clamped
        
        return pid_clamped

    def print_final_pid_score(self):
        # Same as your original — copied here for completeness
        if self.steps_completed == 0:
            print("\n" + "="*50)
            print("           DYNO PID PERFORMANCE SCORE")
            print("="*50)
            print(f"   KP= {DynoStrategy.DYNO_KP}  KI= {DynoStrategy.DYNO_KI}  KD= {DynoStrategy.DYNO_KD}  Back Gain= {DynoStrategy.DYNO_BACK_CALC_GAIN}")
            print(f"   FINAL SCORE         : *** No steps settled *** ")
            return

        avg_settle_time = self.total_settle_cycles / self.steps_completed / 720.0 * 60
        avg_peak = self.total_peak_error / self.steps_completed

        # Final composite score (lower = better)
        score = (
            1.0 * avg_peak +           # penalize overshoot
            8.0 * avg_settle_time +    # fast settling is critical
            0.5 * self.max_peak_error  # penalize any really bad step
        )
        
        print(f"cycles: {self.total_settle_cycles}, steps: {self.steps_completed}")

        print("\n" + "="*50)
        print("           DYNO PID PERFORMANCE SCORE")
        print("="*50)
        print(f"   KP= {DynoStrategy.DYNO_KP}  KI= {DynoStrategy.DYNO_KI}  KD= {DynoStrategy.DYNO_KD}  Back Gain= {DynoStrategy.DYNO_BACK_CALC_GAIN}")
        print(f"   Steps completed     : {self.steps_completed}")
        print(f"   Avg peak error      : {avg_peak:.1f} RPM")
        print(f"   Worst peak error    : {self.max_peak_error:.1f} RPM")
        print(f"   Avg settle time     : {avg_settle_time:.2f} sec per 100 RPM")
        print(f"   FINAL SCORE         : {score:.1f}  ← lower = better")
        print("="*50)
        if score < 300:
            print("   → EXCELLENT - Professional dyno quality")
        elif score < 500:
            print("   → GOOD - Usable for tuning")
        elif score < 800:
            print("   → ACCEPTABLE - Needs improvement")
        else:
            print("   → POOR - Unstable or slow")

    def _create_dyno_plot(self, ax):
        ax.clear()
        ax.set_title("Steady-State Dyno Curve")
        ax.set_xlabel("RPM")
        ax.set_ylabel("Torque (Nm)", color='g')
        ax.set_ylim(0, 360)
        ax.tick_params(axis='y', labelcolor='g')

        self.line_torque, = ax.plot([], [], 'go-', linewidth=2, markersize=8, label='Torque (Nm)')

        # Create twin axis ONCE and store it
        self.ax2 = ax.twinx()
        self.ax2.set_ylabel("Power (kW)", color='m')
        self.ax2.set_ylim(0, 105)
        self.ax2.tick_params(axis='y', labelcolor='m')
        self.line_power, = self.ax2.plot([], [], 'm^--', linewidth=2, markersize=8, label='Power (kW)')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        self.ax2.legend(loc='upper right')

    def _update_dyno_plot(self, ax):
        if not self.dyno_curve_data:
            return

        rpms, torques, powers = zip(*self.dyno_curve_data)

        self.line_torque.set_data(rpms, torques)
        self.line_power.set_data(rpms, powers)

        # Rescale torque axis
        ax.relim()
        ax.autoscale_view()

        # Rescale stored power axis (no new twin!)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_ylim(bottom=0)  # keep floor at 0

        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()
