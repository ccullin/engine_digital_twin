# driver_strategies.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

from typing import TYPE_CHECKING
import constants as c
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

if TYPE_CHECKING:
    from engine_model import EngineSensors, EngineTelemetry
    from ecu_controller import EcuOutput


# =============================================================================
# Strategy classes — one per driver mode
# =============================================================================

class BaseStrategy:
        
    def driver_update(self, driver):
        """Return (tps, load, air pressure)"""
        tps = 0.0
        load = 0.0
        pressure = c.P_ATM_PA
        impulse_target_rpm = 0.0
        return tps, load, pressure, impulse_target_rpm
 
    def get_telemetry(self):
        """Return dict of extra keys for dashboard"""
        return {}
    
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        """Called every cycle — delegate to manager for shared realtime loop"""
        pass  # default: nothing
    
    def dump_telemetry(self, current_cycle=None, data=None):
        sensors, telem, ecu_outputs = data
        sensors: EngineSensors
        telem: EngineTelemetry
        ecu_outputs: EcuOutput
        
        
        
        # We target the window where the 'Ghost Air' is entering
        print(f"\n--- CYCLE {current_cycle} VE ANALYSIS ---")
        print(f"{'CAD':>4} | {'T_cyl':>6}  | {'P_cyl':>8} | {'P_man':>8} | {'dm_in':>11} | {'dm_ex':>11} | {'dm_tot':>11}  | {'Total_m':>11} | {'Cd':>6} | {'IVC_mass':>10} ")
        

        # Scan from BDC to IVC
        for cad in range(720):
            cad_in_range = (0 < cad <=30) or (180 <= cad <= 240) or (690 <= cad <= 720)
            if cad_in_range:
                p_cyl = telem.cyl.log_P[cad] / 1000.0
                p_man = telem.state.map_history[cad] / 1000.0
                dm_in = telem.cyl.dm_in_history[cad] * 1e6 # Convert to mg
                dm_ex = telem.cyl.dm_ex_history[cad] * 1e6
                dm_tot = dm_in + dm_ex
                m_tot = telem.cyl.total_mass_history[cad] * 1e6
                Cd = telem.cyl.Cd_in_history[cad]
                m_TDC = telem.cyl.air_mass_at_TDC 
                m_IVC = (telem.cyl.air_mass_at_IVC - m_TDC)
                T_cyl = telem.cyl.log_T[cad] 
                T_wall = telem.cyl.T_wall
                clt_c = sensors.CLT_C
                iat_k = sensors.IAT_K
            
                
                # Print specifically where flow is entering (dm > 0) 
                # despite pressure being equalizing/reversing
                if dm_in > 0 or dm_ex < 0 or cad % 5 == 0:
                    print(f"{cad:4d} | {T_cyl:7.2f}K | {p_cyl:7.2f}kPa | {p_man:7.2f}kPa | {dm_in:9.4f}mg | {dm_ex:9.4f}mg | {dm_tot:9.4f}mg  | {m_tot:9.4f}mg | {Cd:7.2f} | {m_IVC:9.6f}")
        ideal_air_mass = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        ve = (m_IVC / ideal_air_mass) * 100
        print(f"mass_TDC:{m_TDC:9.6f}kg ideal_mass:{ideal_air_mass:9.6f}kg")
        print(f"FINAL CYCLE VE: {ve:.2f}% T_wall:{T_wall:.1f} CLT:{clt_c:.1f}C IAT:{iat_k:.1f}K")


class IdleStrategy(BaseStrategy):
    """
    Engine start from cranking RPM to idle RPM
    """
    def __init__(self):
        self.start_rpm = 250
    
    def driver_update(self, driver):
        tps = 0.0
        load = 0.0
        pressure = c.P_ATM_PA
        impulse_target_rpm = 0.0
        return tps, load, pressure, impulse_target_rpm

    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        ax_top, ax_topright, ax_bottomleft, ax_bottomright = dashboard_manager.get_strategy_axes()
        if ax_top is None or ax_topright is None or data is None:
            return
        
        plot1 = ax_top
        plot2 = ax_topright
        plot3 = ax_bottomleft
        plot4 = ax_bottomright

        sensors, engine_data, ecu_outputs = data
        sensors: EngineSensors
        engine_data: EngineTelemetry
        ecu_outputs: EcuOutput

        if not hasattr(self, 'artists_created'):
            # --- PLOT 1: PID Response (RPM vs Target & Valve %) ---
            plot1.clear()
            plot1.set_title("Idle PID Response")
            plot1.set_xlabel("Crank Angle (deg)")
            plot1.set_ylabel("RPM", color='blue')
            plot1.grid(True, alpha=0.3)
            
            self.rpm_line, = plot1.plot([], [], color='blue', label='Actual RPM')
            self.target_line, = plot1.plot([], [], color='cyan', linestyle='--', label='Target')
            
            # Secondary axis for Valve Duty Cycle
            self.ax_valve = plot1.twinx()
            self.ax_valve.set_ylabel("IACV Opening %", color='red')
            self.valve_line, = self.ax_valve.plot([], [], color='red', alpha=0.6, label='Valve %')
            
            # --- PLOT 2: PID Component Split ---
            plot2.clear()
            plot2.set_title("PID Component Contribution")
            plot2.set_xlabel("Crank Angle (deg)")
            plot2.set_ylabel("Output Contribution")
            plot2.grid(True, alpha=0.3)

            self.p_trace, = plot2.plot([], [], color='orange', label='P (Prop)')
            self.i_trace, = plot2.plot([], [], color='purple', label='I (Integ)')
            self.d_trace, = plot2.plot([], [], color='brown', label='D (Deriv)')
            plot2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            plot2.legend(loc='upper right', fontsize='xx-small')
            
            # --- PLOT 3: COMBUSTION STABILITY
            plot3.clear()
            plot3.set_title("Combustion Stability (Last 20 Cycles)")
            plot3.set_xlabel("Cycle Number")
            plot3.set_ylabel("AFR", color='lime')
            self.afr_line, = plot3.plot([], [], color='lime', label='AFR')
            
            self.ax_spark = plot3.twinx()
            self.ax_spark.set_ylabel("Spark Advance (°BTDC)", color='blue')
            self.spark_line, = self.ax_spark.plot([], [], color='blue', label='Spark')
            
            # --- PLOT 4: IACV Trend ---
            plot4.clear()
            plot4.set_title("IACV Authority Trend")
            plot4.set_xlabel("Cycle Number")
            plot4.set_ylabel("Valve %", color='red')
            self.valve_trend_line, = plot4.plot([], [], color='red', linewidth=2)
            plot4.set_ylim(0, 100)
        
            
            self.artists_created = True
            
            

        # === DATA PROCESSING ===
        log_cad = engine_data.theta_list
        rpm_history = sensors.rpm_history # Assumes 720 length array
        
        # Map ECU outputs (Assume your _idle_pid logic exposes these internal terms)
        # If not available in dict, these will default to 0 for plotting
        p_val = ecu_outputs.pid_P
        i_val = ecu_outputs.pid_I
        d_val = ecu_outputs.pid_D
        valve_pct = ecu_outputs.iacv_pos
        target_rpm = ecu_outputs.target_rpm

        # --- 1. Update RPM Trace (Top) ---
        self.rpm_line.set_data(log_cad, rpm_history)
        self.target_line.set_data([0, 720], [target_rpm, target_rpm])
        self.valve_line.set_data(log_cad, np.full_like(log_cad, valve_pct))
        
        plot1.set_xlim(0, 720)
        plot1.set_ylim(0, max(2000, np.max(rpm_history) * 1.2))
        self.ax_valve.set_ylim(0, 100)

        # --- 2. Update PID Components (Bottom) ---
        # Plotting static values across the cycle to see current "effort" levels
        self.p_trace.set_data(log_cad, np.full_like(log_cad, p_val))
        self.i_trace.set_data(log_cad, np.full_like(log_cad, i_val))
        self.d_trace.set_data(log_cad, np.full_like(log_cad, d_val))
        
        plot2.set_xlim(0, 720)
        plot2.set_ylim(-50, 100) # Range for PID terms
        
        
        # --- DATA FOR COMBUSTION STABILITY AND IDLE VALVE. 
        if not hasattr(self, 'cycle_history'):
            self.max_cycles = 100
            self.cycle_history = deque(maxlen=self.max_cycles)
            self.afr_history = deque(maxlen=self.max_cycles)
            self.spark_history = deque(maxlen=self.max_cycles)
            self.valve_history = deque(maxlen=self.max_cycles)
        
        self.cycle_history.append(current_cycle)
        self.afr_history.append(sensors.afr)
        self.spark_history.append(ecu_outputs.spark_timing)
        self.valve_history.append(ecu_outputs.iacv_pos)
        
        # Update AFR vs Spark Plot
        self.afr_line.set_data(self.cycle_history, self.afr_history)
        self.spark_line.set_data(self.cycle_history, self.spark_history)
        
        plot3.set_xlim(min(self.cycle_history), max(self.cycle_history))
        plot3.set_ylim(min(self.afr_history)-1, max(self.afr_history)+1)
        self.ax_spark.set_ylim(min(self.spark_history)-5, max(self.spark_history)+5)

        # Update IACV Trend Plot
        self.valve_trend_line.set_data(self.cycle_history, self.valve_history)
        plot4.set_xlim(min(self.cycle_history), max(self.cycle_history))
        # Keep y-limit steady at 0-100 to visualize how close to the floor it is
        

        # === IDLE DEBUG TELEMETRY (Left Table) ===
        # Calculate actual AFR error
        target_afr = ecu_outputs.afr_target
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
            f"IACV POSITION:  {valve_pct:8.2f} %",
            f"WOT EQUIV:      {ecu_outputs.iacv_wot_equiv:8.2f} %",
            f"IGNITION ADV:   {ecu_outputs.spark_timing:8.1f} °",
            "",
            "ENERGY AUDIT (Joules):",
            f"Expansion (Ind):{engine_data.state.work_expansion_j:+8.1f} J",
            f"Friction Loss:  {engine_data.state.work_friction_j:8.1f} J",
            f"Pumping Loss:   {engine_data.state.work_pumping_j:8.1f} J",
            "----------------------------",
            f"NET BALANCE:    {engine_data.state.work_engine_720 + engine_data.state.work_friction_j:+8.1f} J",
        ]
        
        # Diagnosis Logic
        if sensors.afr > 20:
            lines.append("DIAG:  EXTREME LEAN / STALL")
        elif sensors.rpm < target_rpm - 100 and valve_pct > 95:
            lines.append("DIAG:  INSUFFICIENT AIR (MAXED)")
        elif abs(afr_error) < 0.5 and (engine_data.state.work_engine_720 + engine_data.state.work_friction_j) < 0:
            lines.append("DIAG:  HIGH MECHANICAL DRAG")
        else:
            lines.append("DIAG:  IDLE STABLE")
        
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()


class WotStrategy(BaseStrategy):
    """
    Engine start from cranking RPM to idle RPM
    """
    def __init__(self):
        self.start_rpm = 900
    
    def driver_update(self, driver):
        tps = 100.0
        load = 0.0
        pressure = c.P_ATM_PA
        impulse_target_rpm = 0.0
        return tps, load, pressure, impulse_target_rpm
    
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        axes = dashboard_manager.get_strategy_axes()
        if axes is None or data is None:
            return
        
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = axes
        
        plot1 = ax_topleft
        plot2 = ax_bottomleft
        plot3 = ax_topright
        plot4 = ax_bottomright

        sensors, engine_data, ecu_outputs = data
        sensors: EngineSensors
        engine_data: EngineTelemetry
        ecu_outputs: EcuOutput
        
        theta_vec = engine_data.theta_list
        p_bar = engine_data.cyl.log_P / 1e5
        v_litres = engine_data.cyl.V_list * 1000.0
        
        if not hasattr(self, 'artists_created'):
            # --- 1. Top Left: P-V Loop (High Pressure Scale) ---
            plot1.clear()
            plot1.set_title("Fired P-V Loop (WOT)")
            # plot1.plot(v_liters, p_bar, color='red', linewidth=2)
            plot1.set_ylabel("Pressure (Bar)")
            plot1.set_ylim(-1, 60) # Scaled for combustion
            plot1.grid(True, alpha=0.3)
            
            self.intake_line, = plot1.plot([], [], color='cyan', linewidth=2, label='Intake')
            self.comp_line,   = plot1.plot([], [], color='green', linewidth=2, label='Compression')
            self.power_line,  = plot1.plot([], [], color='red', linewidth=2, label='Expansion', ls='dashed')
            self.exh_line,    = plot1.plot([], [], color='orange', linewidth=2, label='Exhaust')
            plot1.legend(loc='upper right', fontsize='x-small')
            
            # --- 1. Update P-V Loop (Top Right) ---
            idx_intake = (theta_vec < 180)
            idx_comp   = (theta_vec >= 180) & (theta_vec < 360)
            idx_power  = (theta_vec >= 360) & (theta_vec < 540)
            idx_exh    = (theta_vec >= 540)

            self.intake_line.set_data(v_litres[idx_intake], p_bar[idx_intake])
            self.comp_line.set_data(v_litres[idx_comp], p_bar[idx_comp])
            self.power_line.set_data(v_litres[idx_power], p_bar[idx_power])
            self.exh_line.set_data(v_litres[idx_exh], p_bar[idx_exh])
            
            plot1.set_xlim(np.min(v_litres) * 0.9, np.max(v_litres) * 1.1)
            plot1.set_ylim(-0.5, max(15, np.max(p_bar) * 1.1))

            # --- 2. Top Right: Cylinder Pressure Trace ---
            plot2.clear()
            plot2.set_title("Cylinder Pressure vs. CAD")
            plot2.plot(theta_vec, p_bar, color='black')
            plot2.set_ylabel("Bar")
            plot2.set_xlim(0, 720)
            plot2.axvline(360, color='red', linestyle='--', alpha=0.3) # TDC
            plot2.grid(True, alpha=0.3)

            # --- 3. Bottom Left: Cumulative Work ---
            plot3.clear()
            plot3.set_title("Cumulative Work Trace")
            plot3.plot(theta_vec, engine_data.state.work_engine_720, color='blue')
            plot3.set_ylabel("Joules")
            plot3.grid(True, alpha=0.3)

            # --- 4. Peak Pressure and Angle ---
            plot4.clear()
            plot4.set_title("Peak Pressure and Angle")
            plot4.set_xlabel("cycle")
            plot4.set_ylabel("Pressure (bar)", color='g')
            plot4.set_ylim(0, 200)
            plot4.tick_params(axis='y', labelcolor='g')
            self.P_peak_line, = plot4.plot([], [], color='green', linewidth=2, markersize=8, label='Pressure (bar)', ls='-')

            # Create twin axis ONCE and store it
            plot4_ax2 = plot4.twinx()
            plot4_ax2.set_ylabel("Peak Angle", color='magenta')
            plot4_ax2.set_ylim(300, 400)
            plot4_ax2.tick_params(axis='y', labelcolor='magenta')
            self.P_peak_angle_line, = plot4_ax2.plot([], [], color='magenta', linewidth=2, markersize=8, label='Paek Angle)', ls='dashed')
            

            self.artists_created = True

        # === DATA PROCESSING ===
        if not hasattr(self, 'P_peak_data'):
            self.P_peak_data = []
            self.P_peak_angle_data = []
            self.cycle_indices = []
        
        P_peak = engine_data.cyl.P_peak_bar
        P_peak_angle = engine_data.cyl.P_peak_angle
        self.P_peak_data.append(P_peak)
        self.P_peak_angle_data.append(P_peak_angle)
        self.cycle_indices.append(len(self.P_peak_data))
            
        plot4.set_xlim(0, max(10, len(self.cycle_indices)))
        self.P_peak_line.set_data(self.cycle_indices, self.P_peak_data)
        self.P_peak_angle_line.set_data(self.cycle_indices, self.P_peak_angle_data)

        # --- 5. THE TABLE (LINE Variable) ---
        net_balance = engine_data.state.work_engine_720 + engine_data.state.work_friction_j
        
        lines = [
            f"WOT POWER AUDIT - RPM: {sensors.rpm:4.0f}",
            "----------------------------",
            f"Brake Torque:   {np.average(engine_data.state.torque_brake_history):8.1f} Nm",
            f"Peak Pressure:  {engine_data.cyl.P_peak_bar:8.2f} Bar",
            f"P_Peak Angle:   {engine_data.cyl.P_peak_angle:8.1f}°",
            "----------------------------",
            "ENERGY BREAKDOWN (J):",
            f"Indicated Work: {engine_data.state.indicated_work_j:8.1f} J",
            f"Friction Loss:  {engine_data.state.work_friction_j:8.1f} J",
            f"(Pumping Loss): {engine_data.state.work_pumping_j:8.1f} J",
            f"Brake Work:     {engine_data.state.work_engine_720:+8.1f} J",
            "----------------------------",
            "STATUS: " + ("ACCELERATING" if net_balance > 0 else "STALLED/DRAGGING")
        ]
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()
    
    
class MotoringStrategy(BaseStrategy):
    """
    Motor the engine with spark and fuel disabled
    """
    
    def __init__(self, rpm=None):
        super().__init__()
        self.start_rpm = rpm if rpm is not None else 3000
        self.impulse_target_rpm = self.start_rpm
        
    def driver_update(self, driver):
        tps = 100
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure, self.impulse_target_rpm
   
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        # Unpack the two axes from the dashboard manager
        axes = dashboard_manager.get_strategy_axes()
        if axes is None or data is None:
            return
        
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = axes
        
        plot1 = ax_topleft
        plot2 = ax_bottomleft
        plot3 = ax_topright
        plot4 = ax_bottomright

        sensors, engine_data, ecu_outputs = data
        sensors: EngineSensors
        engine_data: EngineTelemetry
        ecu_outputs: EcuOutput

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
            self.power_line,  = plot1.plot([], [], color='red', linewidth=2, label='Expansion', ls='dashed')
            self.exh_line,    = plot1.plot([], [], color='orange', linewidth=2, label='Exhaust')
            plot1.legend(loc='upper right', fontsize='x-small')
            
            # During Plot Initialization
            self.poly_text = plot1.text(0.05, 0.95, '', transform=plot1.transAxes, 
                            verticalalignment='top', fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

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
            plot3.set_title("Valve Area & Cd")
            plot3.grid(True, alpha=0.3)

            # Area traces (Primary Y - Left)
            self.a_int_line, = plot3.plot([], [], color='blue', linewidth=2, label='Intake Area')
            self.a_exh_line, = plot3.plot([], [], color='orange', linewidth=2, label='Exhaust Area')
            plot3.set_ylabel("Area (m²)")

            # # Lift traces (Secondary Y - Right)
            # self.ax_lift = plot3.twinx()
            # self.l_int_line, = self.ax_lift.plot([], [], color='blue', linestyle='--', alpha=0.5, label='Intake Lift')
            # self.l_exh_line, = self.ax_lift.plot([], [], color='orange', linestyle='--', alpha=0.5, label='Exhaust Lift')
            # self.ax_lift.set_ylabel("Lift (m)")
            
            # Cd traces (Secondary Y - Right)
            self.ax_Cd = plot3.twinx()
            self.Cd_i_line, = self.ax_Cd.plot([], [], color='blue', linestyle='--', alpha=0.5, label='Intake Lift')
            self.Cd_e_line, = self.ax_Cd.plot([], [], color='orange', linestyle='--', alpha=0.5, label='Exhaust Lift')
            self.ax_Cd.set_ylabel("Cd")

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
        log_p = engine_data.cyl.log_P # Pressure in Pa
        log_v = engine_data.cyl.V_list # Volume in m^3
        
        # Calculate instantaneous work: dW = P * dV
        # Cumulative sum gives the work profile across the 720 degrees
        if log_p is not None and log_v is not None:
            dv = np.diff(log_v, prepend=log_v[0])
            work_array = engine_data.state.work_engine_720
            log_cad = engine_data.theta_list 
            v_liters = log_v * 1000.0
            p_bar = log_p / 1e5
            
            log10_p = np.log10(log_p)  # Use raw Pa for the log calculation
            log10_v = np.log10(log_v)  # Use raw m^3 for the log calculation
            
            # --- 1. Update P-V Loop (Top Right) ---
            idx_intake = (log_cad < 180)
            idx_comp   = (log_cad >= 180) & (log_cad < 360)
            idx_power  = (log_cad >= 360) & (log_cad < 540)
            idx_exh    = (log_cad >= 540)

            # self.intake_line.set_data(v_liters[idx_intake], p_bar[idx_intake])
            # self.comp_line.set_data(v_liters[idx_comp], p_bar[idx_comp])
            # self.power_line.set_data(v_liters[idx_power], p_bar[idx_power])
            # self.exh_line.set_data(v_liters[idx_exh], p_bar[idx_exh])
            
            # plot1.set_xlim(np.min(v_liters) * 0.9, np.max(v_liters) * 1.1)
            # plot1.set_ylim(-0.5, max(15, np.max(p_bar) * 1.1))
            
            # Update Lines with Log Data
            self.intake_line.set_data(log10_v[idx_intake], log10_p[idx_intake])
            self.comp_line.set_data(log10_v[idx_comp], log10_p[idx_comp])
            self.power_line.set_data(log10_v[idx_power], log10_p[idx_power])
            self.exh_line.set_data(log10_v[idx_exh], log10_p[idx_exh])
            
            # Adjust Axis Limits for Log Scale
            # Since these are logs, the ranges will be small (e.g., -4 to -3 for Volume)
            plot1.set_xlim(np.min(log10_v) - 0.1, np.max(log10_v) + 0.1)
            plot1.set_ylim(np.min(log10_p) - 0.1, np.max(log10_p) + 0.1)
            
            # Labels (Important to remind you it's a Log plot)
            plot1.set_xlabel("log10(Volume [m^3])")
            plot1.set_ylabel("log10(Pressure [Pa])")
            
            # Overlay slope text
            mask = (log_cad >= 200) & (log_cad <= 320)
            if np.any(mask):
                # Linear fit: log10(P) = -n * log10(V) + C
                # Note: polyfit returns [slope, intercept]
                n_comp, _ = np.polyfit(log10_v[mask], log10_p[mask], 1)
                
                # In the P*V^n = C relation, slope = -n. 
                # So we take the negative of the slope.
                n_value = -n_comp
                
                # Clear previous text if you've stored a reference, 
                # otherwise use a static coordinate in the plot.
                # Placing it in the top-right of the subplot (Log P-V)
                self.poly_text.set_text(f'n (comp): {n_value:.3f}')

            # --- 2. Update Work vs Theta (Bottom Left) ---
            self.work_trace.set_data(log_cad, work_array)
            plot2.set_xlim(0, 720)
            plot2.set_xticks([0, 180, 360, 540, 720])
            
            # Auto-scale Y for work trace
            y_min, y_max = np.min(work_array), np.max(work_array)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            plot2.set_ylim(y_min - margin, y_max + margin)
        
        # --- Valve and Pressure charts
        area_int_vec = engine_data.valves.intake_area_table 
        area_exh_vec = engine_data.valves.exhaust_area_table
        lift_int_vec = engine_data.valves.intake_lift_table
        lift_exh_vec = engine_data.valves.exhaust_lift_table
        theta_vec = engine_data.theta_list
        Cd_i_vec = engine_data.cyl.Cd_in_history
        Cd_e_vec = engine_data.cyl.Cd_ex_history

        if area_int_vec is not None and area_exh_vec is not None:
            self.a_int_line.set_data(theta_vec, area_int_vec)
            self.a_exh_line.set_data(theta_vec, area_exh_vec)
            plot3.set_xlim(0, 720)
            plot3.set_ylim(0, max(np.max(area_int_vec), np.max(area_exh_vec)) * 1.2)
            
        # if lift_int_vec is not None and lift_exh_vec is not None:
        #     self.l_int_line.set_data(theta_vec, lift_int_vec)
        #     self.l_exh_line.set_data(theta_vec, lift_exh_vec)
        #     self.ax_lift.set_ylim(0, max(np.max(lift_int_vec), np.max(lift_exh_vec)) * 1.1)
            
        if Cd_i_vec is not None and Cd_e_vec is not None:
            self.Cd_i_line.set_data(theta_vec, Cd_i_vec)
            self.Cd_e_line.set_data(theta_vec, Cd_e_vec)
            self.ax_Cd.set_ylim(0.0, max(np.max(Cd_i_vec), np.max(Cd_e_vec)) * 1.5)
            
            # Calculate the points
            ivo_1mm, ivc_1mm = self.get_lift_timing(theta_vec, lift_int_vec, threshold=1.0) # 0.001m = 1mm
            evo_1mm, evc_1mm = self.get_lift_timing(theta_vec, lift_exh_vec, threshold=1.0)
                       
            # Create a string for the chart overlay
            timing_str = (
                f"1mm Timing:\n"
                f"IVO: {ivo_1mm:3.0f}°  IVC: {ivc_1mm:3.0f}°\n"
                f"EVO: {evo_1mm:3.0f}°  EVC: {evc_1mm:3.0f}°\n"
                "\n"
                f"0mm Timing:\n"
                f"IVO: {engine_data.valves.intake.open_angle:3.0f}°  IVC: {engine_data.valves.intake.close_angle:3.0f}°\n"
                f"EVO: {engine_data.valves.exhaust.open_angle:3.0f}°  EVC: {engine_data.valves.exhaust.close_angle:3.0f}°"
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
        p_bar = engine_data.cyl.log_P / 1e5
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
        # print(f"P_bar range: {np.min(p_bar):.3f} to {np.max(p_bar):.3f}")


        
            

        # === Energy Audit Overlay (Bottom Left Table) ===
        fric_work = engine_data.state.work_friction_j
        net_work = engine_data.state.work_net_indicated_j - fric_work
        # total_balance = net_work + fric_work

        # lines = [
        #     # f"MOTORING CYCLE: {current_cycle:8.0f}",
        #     # f"RPM:            {sensors.rpm:8.1f}",
        #     # "----------------------------",
        #     # f"dm_in Total:  {np.sum(engine_data['dm_in']):8.4f} kg",
        #     # f"dm_out Total: {np.sum(engine_data['dm_out']):8.4f} kg",
        #     # f"dm Flow Var:  {np.std(engine_data['dm_in']):8.6f}", # High variance = oscillation
        #     # "----------------------------",
        #     f"T_ind_avg:      {np.average(engine_data.state.torque_indicated_history):8.1f} Nm",
        #     f"T_fric_avg:     {np.average(engine_data.state.torque_friction_history):8.1f} Nm",
        #     f"T_brake_avg:    {np.average(engine_data.state.torque_brake_history):8.1f} Nm",
        #     "----------------------------",
        #     f"Peak Press:     {engine_data.cyl.P_peak_bar:8.2f} Bar",
        #     f"P_peak_angle:      {engine_data.cyl.P_peak_angle:3.0f}",
        #     "",
        #     "ENERGY AUDIT (Joules):",
        #     f"Compression:    {engine_data.state.work_compression_j:8.1f} J",
        #     f"Expansion:      {engine_data.state.work_expansion_j:+8.1f} J",
        #     f"Pumping Loss:   {engine_data.state.work_pumping_j:8.1f} J",
        #     f"Friction Loss:  {fric_work:8.1f} J",
        #     "----------------------------",
        #     f"Brake Work:    {net_work:+8.1f} J",
        # ]
        
        # --- Additional Calculations ---
        air_mass_ideal = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        # trapped_mass should be captured at IVC in your cycle update  
        mass_IVC = engine_data.cyl.air_mass_at_IVC
        mass_TDC = engine_data.cyl.air_mass_at_TDC      
        vol_eff = ((mass_IVC - mass_TDC) / air_mass_ideal) * 100 
        print(f"DEBUG motoring: vol_eff:{vol_eff} air_mass:{engine_data.cyl.air_mass_at_IVC}  mass_ideal:{air_mass_ideal}")

        # Mean Effective Pressures (in Bar)
        imep_net = (engine_data.state.work_compression_j + engine_data.state.work_expansion_j) / c.V_DISPLACED / 1e5
        pmep = engine_data.state.work_pumping_j / c.V_DISPLACED / 1e5

        # --- Updated Table Lines ---
        lines = [
            # --- Performance & Efficiency ---
            f"Vol. Efficiency: {vol_eff:8.1f} %",        # NEW: Compliments Valve/Gas Exchange plots
            f"Polytropic n:    {n_value:8.3f}",         # NEW: Compliments Log PV plot
            f"Net IMEP:        {imep_net:8.2f} Bar",    # NEW: Standardizes Work to pressure
            "----------------------------",
            
            # --- Mechanical Limits (KEEP THESE) ---
            f"Peak Press:     {engine_data.cyl.P_peak_bar:8.2f} Bar", # Critical for head gasket/conrod limits
            f"P_peak_angle:   {engine_data.cyl.P_peak_angle:8.0f} ATDC", # Validates phasing
            f"T_fric_avg:     {np.average(engine_data.state.torque_friction_history):8.1f} Nm", # Validates oil/bearing model
            "----------------------------",
            
            # --- Energy Audit (KEEP THESE) ---
            f"Pumping Loss:   {engine_data.state.work_pumping_j:8.1f} J",  # Cost of "breathing"
            f"Friction Loss:  {fric_work:8.1f} J",                         # Cost of "moving"
            f"Cycle Heat Loss:{abs(engine_data.state.work_compression_j) - engine_data.state.work_expansion_j:8.1f} J", # NEW: The "Log PV gap"
            "----------------------------",
            
            # --- Final Output ---
            f"Brake Work:    {net_work:8.1f} J",
        ]
        
        # Status Logic
        work_exp = engine_data.state.work_expansion_j
        work_comp = abs(engine_data.state.work_compression_j)
        
        if work_comp > 0 and work_exp < (work_comp * 0.8):
            lines.append("STATUS:         CRITICAL LEAK")
        elif net_work > 5.0: # Threshold for numerical "ghost" energy
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
    """
    Realistic open-road driving: hills, varying throttle, altitude effects
    """
    
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
        impulse_target_rpm = 0.0

        return tps, load, pressure, impulse_target_rpm

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

class ImpulseDynoStrategy(BaseStrategy):
    """
    A full rpm sweep impulse dyno
    """
    
    DYNO_START_RPM = 1000
    DYNO_FINISH_RPM = c.RPM_LIMIT
    
    def __init__(self, rpm=None, rl_dyno_mode=False):
        super().__init__()
    
        self.start_rpm = rpm if rpm is not None else 900
        
        # impulse dyno values
        self.rpm_range = np.arange(DynoStrategy.DYNO_START_RPM, DynoStrategy.DYNO_FINISH_RPM + 1, 100)
        rpm_steps = len(self.rpm_range)
        self.rpm_index = 0
        self.governor_torque_data = np.zeros(rpm_steps)
        self.power_data = np.zeros(rpm_steps)
        self.dyno_complete = False
        
        # rl support
        self.rl_dyno_mode = rl_dyno_mode
        self.external_target_rpm = 0.0  # Controlled by RL
        self.force_next_step = False       # Flag for RL to trigger data capture
 
        # data for Dyno Chart plot
        self.line_torque = None
        self.line_power = None
        
        #plot 2 - 4
        self.pmep_data = []
        self.fmep_data = []
        self.ve_data = []
        self.Cd_i_data = []
        self.Cd_e_data = []
        
        #plot4B
        self.fric_cyl_data = []
        self.fric_global_data = []
        
        #plot4C stacked bars
        self.t_fric_stack = []
        self.t_fric_piston_stack = []
        self.t_fric_global_stack = []
        self.t_inidcated_stack = []
        
    
    def driver_update(self, driver):
        tps = 100.0 # always WOT in dyno mode.
        load = 0.0 # no load is needed for an impulse dyno
        ambient_pressure = c.P_ATM_PA
        
        if self.rl_dyno_mode:
            # inject RL target RPM
            impulse_target_rpm = self.external_target_rpm
        else:
            # interate through rpm_range
            impulse_target_rpm = self.set_target_rpm()
        
        self.record_torque(driver) # need to set the rpm and record torque on the next engine cycle
           
        # print(f"DEBUG DRIVER UPDATE {tps} {load} {ambient_pressure} {impulse_target_rpm}")
        return tps, load, ambient_pressure, impulse_target_rpm
        
    def set_target_rpm(self):
        rpm = self.rpm_range[self.rpm_index]
        return rpm
    
    def record_torque(self, driver):
        if driver.cycle %3 == 0 and driver.cycle > 0 and driver.theta == 719:
            self.governor_torque_data[self.rpm_index] = abs(driver.governor_torque) # index was updated for next rpm
            self.power_data[self.rpm_index] = abs(driver.governor_torque) * self.rpm_range[self.rpm_index] / 9548.8
            if self.rpm_index < len(self.rpm_range) - 1:
                self.rpm_index += 1
            else:
                self.dyno_complete = True


        
    # def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
    #     """Standardized update called every cycle by the main loop"""
    #     axes = dashboard_manager.get_strategy_axes()
    #     if axes is None or data is None:
    #         return
        
    #     ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = axes
    #     plot1 = ax_topleft
    #     plot2 = ax_topright
    #     plot3 = ax_bottomleft
    #     plot4 = ax_bottomright
        
    #     sensors, engine_data, ecu_outputs = data
    #     sensors: EngineSensors
    #     engine_data: EngineTelemetry
    #     ecu_outputs: EcuOutput

    #     # === INITIALIZE ARTISTS (Standard Pattern) ===
    #     if not hasattr(self, 'artists_created'):
    #         self._setup_plots(plot1, plot2, plot3, plot4)
    #         self.artists_created = True

    #     rpms = self.rpm_range
        
    #     # 1. Update Plot 1 (Power Curve)
    #     active_torques = self.governor_torque_data
    #     active_powers = self.power_data
    
    #     self.line_torque.set_data(rpms, active_torques)
    #     self.line_power.set_data(rpms, active_powers)
    #     plot1.set_xlim(min(rpms)-200, max(rpms)+500)

  
    #     if current_cycle %3 == 0 and current_cycle > 0 and not self.dyno_complete:
    #         # 2. Update Plot 2 (Losses)
    #         # Assuming you've collected these in lists during the sweep
    #         current_PMEP = -(engine_data.state.work_pumping_j / (c.V_DISPLACED * c.NUM_CYL * 1e5))
    #         current_FMEP = engine_data.state.work_friction_j / (c.V_DISPLACED * c.NUM_CYL * 1e5)
    #         self.pmep_data.append(current_PMEP)
    #         self.fmep_data.append(abs(current_FMEP))
        
    #         current_len = len(self.pmep_data)
    #         # Slice the rpms to match the data we actually have
    #         rpms_view = rpms[:current_len]
            
    #         self.line_pmep.set_data(rpms_view, self.pmep_data)
    #         self.line_fmep.set_data(rpms_view, self.fmep_data)
    #         plot2.set_ylim(0, max(max(self.fmep_data, default=1), 2))
    #         plot2.set_xlim(min(rpms), max(rpms))

    #         # Update Plot 3 (Breathing)
    #         ideal_air_mass = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
    #         current_ve = (engine_data.cyl.air_mass_at_IVC - engine_data.cyl.air_mass_at_TDC) / ideal_air_mass
    #         self.ve_data.append(current_ve * 100)
        
    #         plot3.set_xlim(min(rpms), max(rpms))
    #         self.line_ve.set_data(rpms_view, self.ve_data)

    #         # PLOT 4B (Friction)
    #         self.fric_cyl_data.append(np.mean(engine_data.state.torque_friction_piston_history))
    #         self.fric_global_data.append(np.mean(engine_data.state.torque_friction_global_history))
    #         self.line_fric_cyl.set_data(rpms_view, self.fric_cyl_data)
    #         self.line_fric_global.set_data(rpms_view, self.fric_global_data)
            
    #         # 1. Find the peak in each list (or 0 if empty)
    #         peak_cyl = max(self.fric_cyl_data, default=0)
    #         peak_global = max(self.fric_global_data, default=0)

    #         # 2. Find the highest of the two, scale it, and ensure a minimum of 2.0
    #         y_limit = max(max(peak_cyl, peak_global) * 1.1, 2.0)

    #         plot4.set_ylim(0, y_limit)
    #         plot4.set_xlim(min(rpms), max(rpms))

        
        
    #     # PLOT 4A (Isentropic Cd values)
    #     # self.line_Cd_i.set_data(np.arange(720), engine_data.cyl.Cd_in_history)
    #     # self.line_Cd_e.set_data(np.arange(720), engine_data.cyl.Cd_ex_history)

        
        
    #     # 4. Update the overlay text box
    #     max_torque = np.max(active_torques)
    #     max_t_idx = np.argmax(active_torques)
    #     max_t_rpm = rpms[max_t_idx]
    #     max_power = np.max(active_powers)
    #     max_p_idx = np.argmax(active_powers)
    #     max_p_rpm = rpms[max_p_idx]
    #     dyno_str = (
    #         f"Dyno report:\n"
    #         f"Max Torque: {max_torque:5.0f} Nm  at: {max_t_rpm:.0f} rpm\n"
    #         f"Max Power: {max_power:5.1f} kW  at: {max_p_rpm:.0f} rpm\n"
    #     )
        
    #     if not hasattr(self, 'dyno_text'):
    #         self.dyno_text = plot3.text(0.5, 0.85, dyno_str, 
    #                                             transform=plot1.transAxes,
    #                                             fontsize=9, family='monospace',
    #                                             ha='center', va='top',
    #                                             bbox=dict(boxstyle="round", alpha=0.6, facecolor='white'))
    #     else:
    #         self.dyno_text.set_text(dyno_str)
        
        
    #     # === UPDATE TEXT OVERLAY ===
    #     lines = []
    #     dashboard_manager.update_strategy_overlay(lines)
    #     dashboard_manager.draw()
    
    # def _setup_plots(self, plot1, plot2, plot3, plot4):
        
    #     # --- PLOT1 DYNO TORQUE and POWER CHART
    #     plot1.clear()
    #     plot1.set_title("Steady-State Dyno Curve")
    #     plot1.set_xlabel("RPM")
    #     plot1.set_ylabel("Torque (Nm)", color='g')
    #     plot1.set_ylim(100, 260)
    #     plot1.tick_params(axis='y', labelcolor='g')
    #     plot1.grid(True, alpha=0.3)

    #     self.line_torque, = plot1.plot([], [], color='green', linewidth=2, markersize=8, label='Torque (Nm)', ls='-')
        
    #     plot1.legend(loc='upper left')

    #     # Create twin axis ONCE and store it
    #     plot1_ax2 = plot1.twinx()
    #     plot1_ax2.set_ylabel("Power (kW)", color='magenta')
    #     plot1_ax2.set_ylim(10, 90)
    #     plot1_ax2.tick_params(axis='y', labelcolor='magenta')

    #     self.line_power, = plot1_ax2.plot([], [], color='magenta', linewidth=2, markersize=8, label='Power (kW)', ls='dashed')
        
    #     plot1_ax2.legend(loc='upper right')

        
    #     # PLOT 2: Internal Losses (PMEP and FMEP)
    #     plot2.clear()
    #     plot2.set_title("Pumping & Friction Losses")
    #     plot2.set_xlabel("RPM")
    #     plot2.set_ylabel("Pressure (Bar)")
    #     plot2.grid(True, alpha=0.3)

    #     self.line_pmep, = plot2.plot([], [], label='PMEP (Pumping)', color='orange', lw=2)
    #     self.line_fmep, = plot2.plot([], [], label='FMEP (Friction)', color='red', lw=1, ls='--')
        
    #     plot2.legend(loc='upper left', fontsize='small')

    #     # PLOT 3: Breathing Dynamics (VE)
    #     plot3.clear()
    #     plot3.set_title("Breathing Dynamics")
    #     plot3.set_xlabel("RPM")
    #     plot3.set_ylabel("VE (%)")
    #     plot3.set_ylim(60, 110) # Typical VE range
    #     self.line_ve, = plot3.plot([], [], label='VE %', color='blue', lw=2)
        
    #     # Twin axis for Mach Number and Cd values
    #     # self.ax_mach = plot3.twinx()
    #     # self.ax_mach.set_ylabel("Mach Number and Cd value", color='purple')        
    #     # self.ax_mach.set_ylim(0, 1.0) # Mach 1.0 is the physical limit
    #     # self.ax_mach.axhline(0.45, color='purple', alpha=0.2, ls='--') # The "Cliff" threshold
    #     # self.line_mach, = self.ax_mach.plot([], [], label='Intake Mach', color='purple', lw=1, ls=':')
        
    #     # # PLOT 4A: Cd values
    #     # plot4.clear()
    #     # plot4.set_title("Isentropic Cd factors")
    #     # plot4.set_xlabel("CAD")
    #     # plot4.set_ylabel("Cd")
    #     # plot4.set_xlim(0, 720)
    #     # plot4.set_ylim(0, 1.0) # Mach 1.0 is the physical limit

    #     # self.line_Cd_i, = plot4.plot([], [], label='Intake Cd', color='blue')
    #     # self.line_Cd_e, = plot4.plot([], [], label='Exhaust Cd', color='cyan')
        
    #     # plot4.legend(loc='upper right', fontsize='small')

    #     # PLOT 4B: Friction
    #     plot4.clear()
    #     plot4.set_title("Friction")
    #     plot4.set_xlabel("RPM")
    #     plot4.set_ylabel("Friction(Nm)")

    #     self.line_fric_cyl,    = plot4.plot([], [], label='Fric (Cylinder)', color='orange', lw=2)
    #     self.line_fric_global, = plot4.plot([], [], label='Fric (Core)', color='red', lw=1, ls='--')
        
        
    #     plot4.legend(loc='upper left', fontsize='small')
  
  
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        """Standardized update called every cycle by the main loop"""
        axes = dashboard_manager.get_strategy_axes()
        if axes is None or data is None:
            return
        
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = axes
        plot1, plot2, plot3, plot4 = ax_topleft, ax_topright, ax_bottomleft, ax_bottomright
        
        sensors, engine_data, ecu_outputs = data
        sensors: EngineSensors
        engine_data: EngineTelemetry
        ecu_outputs: EcuOutput
        
        # === INITIALIZE ARTISTS & DATA BUFFERS ===
        if not hasattr(self, 'artists_created'):
            self._setup_plots(plot1, plot2, plot3, plot4)
            self.t_brake_stack = []
            self.t_fric_stack = []
            self.t_pump_stack = []
            self.artists_created = True

        rpms = self.rpm_range
        active_torques = self.governor_torque_data
        active_powers = self.power_data
    
        # 1. Update Plot 1 (Power Curve)
        self.line_torque.set_data(rpms, active_torques)
        self.line_power.set_data(rpms, active_powers)
        plot1.set_xlim(min(rpms)-200, max(rpms)+500)

        # 2. Process Stats (Every 3 cycles to maintain performance)
        if current_cycle % 3 == 0 and current_cycle > 0 and not self.dyno_complete:
            # --- CALCULATE LOSSES ---
            # MEP values (Bar)
            current_PMEP = -(engine_data.state.work_pumping_j / (c.V_DISPLACED * c.NUM_CYL * 1e5))
            current_FMEP = engine_data.state.work_friction_j / (c.V_DISPLACED * c.NUM_CYL * 1e5)
            self.pmep_data.append(current_PMEP)
            self.fmep_data.append(abs(current_FMEP))
        
            current_len = len(self.pmep_data)
            rpms_view = rpms[:current_len]
            
            # --- UPDATE PLOT 2 (MEP Lines) ---
            self.line_pmep.set_data(rpms_view, self.pmep_data)
            self.line_fmep.set_data(rpms_view, self.fmep_data)
            plot2.set_ylim(0, max(max(self.fmep_data, default=1), 2.5))
            plot2.set_xlim(min(rpms), max(rpms))

            # --- UPDATE PLOT 3 (VE) ---
            ideal_air_mass = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
            current_ve = (engine_data.cyl.air_mass_at_IVC - engine_data.cyl.air_mass_at_TDC) / ideal_air_mass
            self.ve_data.append(current_ve * 100)
            self.line_ve.set_data(rpms_view, self.ve_data)
            plot3.set_xlim(min(rpms), max(rpms))

            # --- UPDATE PLOT 4 (STACKED TORQUE BREAKDOWN) ---
            # 1. Capture current torque components (Nm)
            
            t_brake = np.mean(engine_data.state.torque_brake_history)
            t_fric_piston = abs(np.mean(engine_data.state.torque_friction_piston_history))
            t_fric_global = abs(np.mean(engine_data.state.torque_friction_global_history))
            t_indicated = np.mean(engine_data.state.torque_indicated_history)
        
            # Physics: Convert PMEP Bar to Torque Nm: T = (P * Vd) / (4 * pi)
            t_pumping = (current_PMEP * 1e5 * (c.V_DISPLACED * c.NUM_CYL)) / (4 * np.pi)
            
            self.t_brake_stack.append(t_brake)
            self.t_fric_stack.append(t_fric_piston + t_fric_global)
            self.t_fric_piston_stack.append(t_fric_piston)
            self.t_fric_global_stack.append(t_fric_global)
            self.t_inidcated_stack.append(t_indicated)
            self.t_pump_stack.append(t_pumping)

            # 2. Redraw the stacked bar
            plot4.clear()
            plot4.set_title("Torque Breakdown (Stack)")
            plot4.set_ylabel("Torque (Nm)")
            plot4.grid(True, alpha=0.2, axis='y')
            
            # Calculate width based on sweep resolution
            bar_width = (max(rpms) - min(rpms)) / len(rpms) * 0.9

               
            # Foundation Brake
            plot4.bar(rpms_view, self.t_brake_stack, bar_width, label='Brake', color='green', alpha=0.8)
            # Stack Friction (piston)
            combined_bottom = np.array(self.t_brake_stack)
            plot4.bar(rpms_view, self.t_fric_stack, bar_width, bottom=combined_bottom, label='Friction', color='red', alpha=0.7)
            # Stack Friction (global)
            combined_bottom = np.array(self.t_brake_stack) + np.array(self.t_fric_stack)
            plot4.bar(rpms_view, self.t_pump_stack, bar_width, bottom=combined_bottom, label='Pumping', color='orange', alpha=0.7)
            # Stack Pumping
            # combined_bottom = np.array(self.t_brake_stack) + np.array(self.t_fric_piston_stack) + np.array(self.t_fric_global_stack)
            # plot4.bar(rpms_view, self.t_pump_stack, bar_width, bottom=combined_bottom, label='Pumping', color='orange', alpha=0.6)
            # Stack Indicated torque balance 
            # combined_bottom = np.array(self.t_brake_stack) + np.array(self.t_fric_piston_stack) + np.array(self.t_fric_global_stack)
            # plot4.bar(rpms_view, (self.t_inidcated_stack), bar_width, bottom=combined_bottom, label='indicated', color='blue')
            

            # 3. Finalize Plot 4 visuals
            plot4.set_xlim(min(rpms), max(rpms))
            indicated_peak = np.max(combined_bottom + np.array(self.t_pump_stack)) if len(self.t_pump_stack) > 0 else 200
            plot4.set_ylim(0, max(indicated_peak * 1.3, 200))
            plot4.legend(loc='upper left', fontsize='x-small', ncol=3)

        # 4. Update the overlay text box
        max_torque = np.max(active_torques)
        max_t_rpm = rpms[np.argmax(active_torques)]
        max_power = np.max(active_powers)
        max_p_rpm = rpms[np.argmax(active_powers)]
        dyno_str = (
            f"Dyno report:\n"
            f"Max Torque: {max_torque:5.0f} Nm  at: {max_t_rpm:.0f} rpm\n"
            f"Max Power: {max_power:5.1f} kW  at: {max_p_rpm:.0f} rpm\n"
        )
        
        if not hasattr(self, 'dyno_text'):
            self.dyno_text = plot3.text(0.5, 0.85, dyno_str, transform=plot1.transAxes,
                                       fontsize=9, family='monospace', ha='center', va='top',
                                       bbox=dict(boxstyle="round", alpha=0.6, facecolor='white'))
        else:
            self.dyno_text.set_text(dyno_str)
        
        dashboard_manager.update_strategy_overlay([])
        dashboard_manager.draw()
    
    def _setup_plots(self, plot1, plot2, plot3, plot4):
        # --- PLOT 1: DYNO CURVE ---
        plot1.clear()
        plot1.set_title("Steady-State Dyno Curve")
        plot1.set_ylabel("Torque (Nm)", color='g')
        plot1.set_ylim(0, 260)
        plot1.grid(True, alpha=0.3)
        self.line_torque, = plot1.plot([], [], color='green', linewidth=2, label='Torque')
        
        ax1_twin = plot1.twinx()
        ax1_twin.set_ylabel("Power (kW)", color='magenta')
        ax1_twin.set_ylim(0, 100)
        self.line_power, = ax1_twin.plot([], [], color='magenta', linewidth=2, ls='--', label='Power')

        # --- PLOT 2: MEP LOSSES ---
        plot2.clear()
        plot2.set_title("Pumping & Friction (Bar)")
        self.line_pmep, = plot2.plot([], [], label='PMEP', color='orange', lw=2)
        self.line_fmep, = plot2.plot([], [], label='FMEP', color='red', lw=1, ls='--')
        plot2.legend(loc='upper left', fontsize='small')

        # --- PLOT 3: VOLUMETRIC EFFICIENCY ---
        plot3.clear()
        plot3.set_title("Volumetric Efficiency (%)")
        plot3.set_ylim(60, 110)
        self.line_ve, = plot3.plot([], [], label='VE %', color='blue', lw=2)

        # --- PLOT 4: TORQUE BREAKDOWN ---
        plot4.clear()
        plot4.set_title("Torque Breakdown (Nm)")
        # Note: Bars are created dynamically in update_dashboard
        
    def dump_telemetry(self, current_cycle=None, data=None):
        sensors, telem, ecu_outputs = data
        sensors: EngineSensors
        telem: EngineTelemetry
        ecu_outputs: EcuOutput
        
        # 1. Prepare Summary Scalars (Calculate these once per cycle)
        v_disp_total = c.V_DISPLACED * c.NUM_CYL
        # Convert Joules/m^3 to Bar (1e5 Pa = 1 Bar)
        pmep_bar = (telem.state.work_pumping_j / v_disp_total) / 1e5
        fmep_bar = (telem.state.work_friction_j / v_disp_total) / 1e5
        imep_bar = (telem.state.work_net_indicated_j / v_disp_total) / 1e5
        bmep_bar = imep_bar - fmep_bar
        mech_eff = (bmep_bar / imep_bar * 100) if imep_bar > 0 else 0

        # 2. Header (Focus on gas exchange dynamics)
        header = (f"{'CAD':>4} | {'P_cyl[kP]':>9} | {'P_man[kP]':>9} | {'dm_in[mg]':>10} | "
                f"{'m_cyl[mg]':>10} | {'Cd':>6} | {'T_cyl[K]':>8}")
        print(header)
        print("-" * len(header))  

        for cad in range(720):
            # Scan window: Focus on Overlap (0-30) and Intake/IVC (180-240)
            cad_in_range = (0 <= cad <= 45) or (140 <= cad <= 240) or (680 <= cad <= 720)
            if cad_in_range:
                p_cyl = telem.cyl.log_P[cad] / 1000.0
                p_man = telem.state.map_history[cad] / 1000.0
                dm_in = telem.cyl.dm_in_history[cad] * 1e6
                m_cyl = telem.cyl.total_mass_history[cad] * 1e6
                Cd = telem.cyl.Cd_in_history[cad]
                T_cyl = telem.cyl.log_T[cad]
                
                # Print logic: sample every 5 deg unless mass is actively moving
                if abs(dm_in) > 0.01 or cad % 5 == 0:
                    print(f"{cad:4d} | {p_cyl:9.2f} | {p_man:9.2f} | "
                            f"{dm_in:10.3f} | {m_cyl:10.3f} | "
                            f"{Cd:6.2f} | {T_cyl:8.1f}")

        # 3. Final Performance Summary
        ideal_air_mass = (c.P_ATM_PA * c.V_DISPLACED) / (c.R_SPECIFIC_AIR * c.T_AMBIENT)
        m_trapped = telem.cyl.air_mass_at_IVC - telem.cyl.air_mass_at_TDC
        ve = (m_trapped / ideal_air_mass) * 100
        
        print("-" * len(header))
        print(f"--- CYCLE {current_cycle} SUMMARY ---")
        print(f"VE: {ve:6.2f}% | Pumping: {pmep_bar:5.2f} bar | Friction: {fmep_bar:5.2f} bar")
        print(f"IMEP: {imep_bar:5.2f} bar | BMEP: {bmep_bar:5.2f} bar | Mech Eff: {mech_eff:5.1f}%")
        print(f"Mass Trapped: {m_trapped*1e6:6.2f}mg | Target (100% VE): {ideal_air_mass*1e6:6.2f}mg")

                        

    
class DynoStrategy(BaseStrategy):
    """
    A variable load Dyno with a full rpm sweep
    """
    
    # PID paramters for controller to hold rpm while at WOT.
    DYNO_KP = 1.0  # Proportional Gain 1.5 last know best.
    DYNO_KI = 0.003 # Integral Gain
    DYNO_KD = 3.5  # Derivative Gain was 1.5 last known best
    DYNO_BACK_CALC_GAIN = 0.0   # Usually 0.3–1.0 — how aggressively we unwind
    MAX_DYNO_TORQUE = 2000     # if your engine peaks at ~500 Nm → 1.5× headroom

    # Dyno start and RPM stability parameters
    DYNO_START_RPM = 1500
    DYNO_FINISH_RPM = c.RPM_LIMIT - 500
    DYNO_STEP_SIZE_RPM = 200 # How far to step the RPM target after settling
    DYNO_RPM_TOLERANCE = 5.0
    SETTLE_CYCLES_REQUIRED = 1 * 720  # it is measured in degrees
           
    def __init__(self, rpm=None, rl_dyno_mode=False):
        super().__init__()
        
        self.start_rpm = rpm if rpm is not None else 900
        
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
        
        # data for other plots
        self.settled_rpm_log = []
        self.settled_torque_log = []
        self.settled_cycles_log = []
        self.rejection_mass_log = [] # mass that is pushed back into the intake manifold
        self.rejection_rpm_log = [] # log of live rpms not bounded by 720
        
        
        # Stablization flags for downstream services
        self.point_settled = False
        self.last_settled_rpm = 0.0
        self.last_settled_torque = 0.0
        self.last_settled_power = 0.0
        self.last_settled_std = 0.0
        self.dyno_complete = False
           
    
    def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
        """Standardized update called every cycle by the main loop"""
        axes = dashboard_manager.get_strategy_axes()
        if axes is None or data is None:
            return
        
        ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = axes
        plot1 = ax_topleft
        plot2 = ax_topright
        plot3 = ax_bottomleft
        plot4 = ax_bottomright
        
        sensors, engine_data, ecu_outputs = data
        sensors: EngineSensors
        engine_data: EngineTelemetry
        ecu_outputs: EcuOutput

        # === INITIALIZE ARTISTS (Standard Pattern) ===
        if not hasattr(self, 'artists_created'):
            self._setup_plots(plot1, plot2, plot3, plot4)
            self.artists_created = True

        # === UPDATE THE POWER CURVE ===
        if self.dyno_curve_data:
            rpms, torques, powers = zip(*self.dyno_curve_data)
            self.line_torque.set_data(rpms, torques)
            self.line_power.set_data(rpms, powers)
            plot1.set_xlim(min(rpms)-200, max(rpms)+500)
            
            
        # === UPDATE THE BACKFLOW PLOT
        dm_history = engine_data.cyl.dm_in_history
        gross_inflow = np.sum(dm_history[dm_history > 0])
        # We focus on the window between BDC (180) and your known IVC (210)
        backflow_window = dm_history[180:210]
        rejected_mass = np.sum(np.abs(backflow_window[backflow_window < 0]))
        rejection_percent = (rejected_mass / gross_inflow * 100) if gross_inflow > 0 else 0
        self.rejection_mass_log.append(rejection_percent)
        self.rejection_rpm_log.append(sensors.rpm)
        
        self.line_backflow.set_data(self.rejection_rpm_log, self.rejection_mass_log)
        plot2.set_xlim(min(self.rejection_rpm_log)-200, max(self.rejection_rpm_log)+500)
        
        # === UPDATE TEXT OVERLAY ===
        lines = [
            f"TARGET RPM:     {self.dyno_target_rpm:8.0f}",
            f"CURRENT RPM:    {np.mean(sensors.rpm_history):8.1f}",
            f"SETTLE STATUS:  {self.settled_cycles}/{DynoStrategy.SETTLE_CYCLES_REQUIRED}",
            "----------------------------",
            f"BRAKE TORQUE:   {self.pid_output:8.1f} Nm",
            f"PID P:          {self.pid_error_p:8.1f}",
            f"PID I:          {self.pid_error_integral:8.1f}",
            f"PID D:          {self.pid_derivative:8.1f}",
            "----------------------------",
            f"STEPS DONE:     {self.steps_completed:8.0f}",
            f"PEAK ERR:       {self.current_step_peak:8.1f} RPM",
        ]
        dashboard_manager.update_strategy_overlay(lines)
        dashboard_manager.draw()
    
    def driver_update(self, driver):
        tps = 100.0 # always WOT in dyno mode.
        impulse_rpm = 0
        
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
   
        # Dyno load
        self._set_dyno_target(driver)
        load = self._pid_control(driver) if self.in_dyno_sweep else 0.0
        
        return tps, load, c.P_ATM_PA, impulse_rpm

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
        self.settled_rpm_log.append(self.last_settled_rpm)
        self.settled_torque_log.append(self.last_settled_torque)
        self.settled_cycles_log.append(self.settled_cycles)
        
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


    
    def _setup_plots(self, plot1, plot2, plot3, plot4):
        
        # --- PLOT1 DYNO TORQUE and POWER CHART
        plot1.clear()
        plot1.set_title("Steady-State Dyno Curve")
        plot1.set_xlabel("RPM")
        plot1.set_ylabel("Torque (Nm)", color='g')
        plot1.set_ylim(100, 260)
        plot1.tick_params(axis='y', labelcolor='g')
        self.line_torque, = plot1.plot([], [], color='green', linewidth=2, markersize=8, label='Torque (Nm)', ls='-')

        # Create twin axis ONCE and store it
        plot1_ax2 = plot1.twinx()
        plot1_ax2.set_ylabel("Power (kW)", color='magenta')
        plot1_ax2.set_ylim(10, 90)
        plot1_ax2.tick_params(axis='y', labelcolor='magenta')
        self.line_power, = plot1_ax2.plot([], [], color='magenta', linewidth=2, markersize=8, label='Power (kW)', ls='dashed')

        plot1.grid(True, alpha=0.3)
        plot1.legend(loc='upper left')
        plot1_ax2.legend(loc='upper right')
        
        # --- Backflow Ratio
        plot2.set_title("Intake Backflow vs RPM")
        plot2.set_ylabel("Rejection %")
        plot2.set_ylim(0, 5)
        self.line_backflow, = plot2.plot([], [], color='orange', label='Backflow %')
        

                