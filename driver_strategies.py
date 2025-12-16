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
    #     ax = dashboard_manager.get_strategy_axes()
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


class WotStrategy(BaseStrategy):
    """ Engine wide open throttle from idle RPM """
        
    def driver_update(self, driver):
        tps = 100.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure
    
    
class RoadTestStategy(BaseStrategy):
    """ road test over varying landcapes """
    
    def diver_update(self, driver):
        driver.tps = 0.0
        driver.load = 0.0
        driver.air_pressure = c.P_ATM_PA

    

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
    DYNO_FINISH_RPM = c.RPM_LIMIT + 1000
    DYNO_STEP_SIZE_RPM = 100 # How far to step the RPM target after settling
    DYNO_RPM_TOLERANCE = 5.0
    SETTLE_CYCLES_REQUIRED = 50
         
         
    def __init__(self):
        self.in_dyno_sweep = False
        self.settled_cycles = 0
        self.current_step_peak = 0.0

        self.pid_error_p = 0.0
        self.pid_derivative = 0.0
        self.pid_error_integral = 0.0
        self._filtered_drpm = 0.0
        self.pid_output = 0.0

        self.dyno_target_rpm = 2000

        self.settled_torque_values = np.zeros(DynoStrategy.SETTLE_CYCLES_REQUIRED)

        # Scoring
        self.total_settle_cycles = 0.0
        self.steps_completed = 0.0
        self.total_peak_error = 0.0
        self.max_peak_error = 0.0
        
        # data for Dyno Chart plot
        self.torque_data = []  # list of (rpm, filtered_torque)
        self.dyno_curve_data = []  # list of (rpm_target, avg_torque_Nm, avg_power_kW)
        self.line_torque = None
        self.line_power = None
        
        
        
    def driver_update(self, driver):
        # Dyno-specific TPS logic
        control_rpm = np.mean(driver.cycle_rpm)
        if control_rpm >= (DynoStrategy.DYNO_START_RPM - 100) and control_rpm <= DynoStrategy.DYNO_FINISH_RPM:
            self.in_dyno_sweep = True
            tps = 100.0
        else:
            self.in_dyno_sweep = False
            tps = 100.0  # keep WOT until end

        # Dyno load
        self._set_dyno_target(driver)
        if self.in_dyno_sweep:
            load = self._pid_control(driver)
        else:
            load = 0.0
        
        print(
            f"theta: {driver.cycle:4.0f}/{driver.theta:3.0f} | "
            f"P_err: {self.pid_error_p:5.1f} | "
            f"Integral: {self.pid_error_integral:5.0f} | "
            f"derivative: {self.pid_derivative:5.0f} | "
            f"pid out: {self.pid_output:6.2f} | "
            f"RPM: {driver.rpm:4.0f} | "
            f"target_rpm: {self.dyno_target_rpm:4.0f} | "
            f"mean_rpm: {np.mean(driver.cycle_rpm):4.0f} | "
            f"in_dyno_sweep: {self.in_dyno_sweep}"
            )
        
        return tps, load, c.P_ATM_PA


    
    
    def get_telemetry(self):
        return {
            "dyno_settled": (self.settled_cycles >= DynoStrategy.SETTLE_CYCLES_REQUIRED),
            "target_rpm": self.dyno_target_rpm,
            "in_dyno_sweep": self.in_dyno_sweep,
        }   
        
    # def update_dashboard(self, dashboard_manager, current_cycle=None, data=None):
    #     # data = (sensors_dict, engine_data_dict, ecu_outputs_dict) passed from main
    #     if data:
    #         sensors, engine_data, ecu = data
    #         torque_Nm = engine_data['torque_brake']
    #         rpm = engine_data['rpm']
    #         power_kW = torque_Nm * rpm / 9549
            
    #         self.history_rpm.append(rpm)
    #         self.history_torque.append(torque_Nm)
    #         self.history_power.append(power_kW)
            
    #     dashboard_manager.create_or_update_figure(
    #         key="dyno_chart",
    #         create_func=self._create_dyno_plot,
    #         update_func=self._update_dyno_plot,
    #         data = self # ← strategy instance has all the history lists
    #     ) 

    def _set_dyno_target(self, driver):
        control_rpm = np.mean(driver.cycle_rpm)
        error = abs(self.dyno_target_rpm - control_rpm)
        
        if not self.in_dyno_sweep:
            return

        # trake peak error for the stabiliosation period
        if error > self.current_step_peak:
            self.current_step_peak = error

        if error < DynoStrategy.DYNO_RPM_TOLERANCE:
            self.settled_torque_values[self.settled_cycles] = self.pid_output
            self.settled_cycles += 1
        else:
            self.settled_cycles = 0

        if self.settled_cycles >= DynoStrategy.SETTLE_CYCLES_REQUIRED:
            self.total_settle_cycles += DynoStrategy.SETTLE_CYCLES_REQUIRED
            self.steps_completed += 1

            #------------------------------------------------------------------
            # clean the data and update dyno plot 
            #------------------------------------------------------------------
            raw_torque = np.mean(self.settled_torque_values)
            std_dev = np.std(self.settled_torque_values)
            # Trimmed mean (drop top/bottom 10%)
            boundary = int(0.10 * DynoStrategy.SETTLE_CYCLES_REQUIRED)
            trimmed_raw_torque = np.sort(self.settled_torque_values)[boundary : -boundary]
            avg_brake_torque = np.mean(trimmed_raw_torque)
            avg_power_kW = avg_brake_torque * self.dyno_target_rpm / 9549.0
            
            self.dyno_curve_data.append((self.dyno_target_rpm, avg_brake_torque, avg_power_kW))
            
            if driver.dashboard_manager:
                ax = driver.dashboard_manager.get_strategy_axes()
                if ax:
                    if not hasattr(self, 'line_torque') or self.line_torque is None:
                        self._create_dyno_plot(ax)
                    self._update_dyno_plot(ax)
                    driver.dashboard_manager.draw()

            print(
                f"Target RPM {self.dyno_target_rpm:4.0f} | "
                f"Torque(Nm) {raw_torque:5.0f} | "
                f"Filtered T {avg_brake_torque:5.0f} | "
                f"T_std_dev {std_dev:5.1f}"
            )

            self.total_peak_error += self.current_step_peak
            if self.current_step_peak > self.max_peak_error:
                self.max_peak_error = self.current_step_peak

            # Reset for next step
            self.settled_cycles = 0
            self.current_step_peak = 0.0

            # Advance target...
            if self.dyno_target_rpm < DynoStrategy.DYNO_FINISH_RPM:
                self.dyno_target_rpm += DynoStrategy.DYNO_STEP_SIZE_RPM
            else:
                print("DYNO RUN COMPLETED")
                self.in_dyno_sweep = False
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

    # def plot_strategy_panel(self, ax):
    #     ax.clear()
    #     ax.set_title("Steady-State Torque & Power Curve")
    #     ax.set_xlabel("RPM")
    #     ax.set_ylabel("Torque (Nm)", color='g')

    #     if not self.dyno_curve_data:
    #         ax.text(0.5, 0.5, "Collecting data...", ha='center', va='center', transform=ax.transAxes)
    #         return

    #     rpms, torques, powers = zip(*self.dyno_curve_data)

    #     ax.plot(rpms, torques, 'go-', label='Torque (Nm)')
    #     ax_twin = ax.twinx()
    #     ax_twin.plot(rpms, powers, 'm^--', label='Power (kW)')
    #     ax_twin.set_ylabel("Power (kW)", color='m')

    #     ax.grid(True, alpha=0.3)
    #     ax.legend(loc='upper left')
    #     ax_twin.legend(loc='upper right')

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
        self.ax2.set_ylim(0, 90)
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




# class DriverInput:
#     def __init__(self, mode="idle"):
#         self.mode = mode
#         self.tps = 0.0  # foot off the accelerator
#         self.load = 0.0  # sitting in neutral
#         self.air_pressure = c.P_ATM_PA
#         self.phase = 0

#         if mode == "idle":
#             self.start_rpm = c.CRANK_RPM
#         else:
#             self.start_rpm = c.IDLE_RPM

#         # --- Dyno Parameters for Stepped Sweep and PID ---
#         DynoStrategy.DYNO_START_RPM = 2000
#         DynoStrategy.DYNO_FINISH_RPM = c.RPM_LIMIT + 1000
#         self.dyno_target_rpm = 2000 # Current RPM setpoint for the dyno
#         DynoStrategy.DYNO_RPM_TOLERANCE = 5.0             # Tolerance for "settled" RPM (e.g., +/- 5 RPM)
#         self.in_dyno_sweep = False 
#         DynoStrategy.SETTLE_CYCLES_REQUIRED = 100     # Number of consecutive cycles required to be settled
#         self.settled_cycles = 0             # Counter for consecutive settled cycles


#         # --- Dyno PID Controller State Variables ---
#         self.pid_error_integral = 0.0
#         self.pid_derivative = 0.0
#         self.pid_output = 0.0                # The PID controller's adjustment to the load
#         driver.cycle_rpm = np.zeros(720)
#         # ------------------------------------------------
        
#         # --- Dyno output table --------------------------
#         self.settled_torque_map = {}   # RPM: average torque during stable period
#         self.settled_torque_values = np.zeros(DynoStrategy.SETTLE_CYCLES_REQUIRED)
#         #-------------------------------------------------
        
#         # --- PID performance tracking -------------------
#         self.current_step_peak = 0.0
#         self.abs_error_sum = 0.0
#         self.peak_error = 0.0
#         self.total_settle_cycles = 0.0
#         self.steps_completed = 0.0
#         self.total_peak_error = 0.0
#         self.max_peak_error = 0.0
#         # -------------------------------------------------
        
#         self.driver_dict = FixedKeyDictionary({
#             "throttle_pos": self.tps,
#             "wheel_load": self.load,
#             "ambient_pressure": self.air_pressure,
#             "dyno_settled": False,
#             "target_rpm": self.dyno_target_rpm,
#         })

#     # ---------------------------------------------------------------------------------
#     def get_driver_dict(self):
#         return self.driver_dict

#     # ---------------------------------------------------------------------------------
#     def _set_tps(self):
#         global RPM, driver.theta

#         if self.mode == "idle":
#             self.tps = 0.0

#         elif self.mode == "wot":
#             self.tps = 100.0

#         elif self.mode == "dyno":
#             # Set throttle to WOT (100%) during the sweep to measure max torque
#             self.tps = 100.0
            
#             # The sweep is considered 'in progress' if RPM has reached the start point
#             control_rpm = np.mean(driver.cycle_rpm)
#             if control_rpm >= (DynoStrategy.DYNO_START_RPM - 100) and control_rpm <= DynoStrategy.DYNO_FINISH_RPM:
#             # if RPM >= DynoStrategy.DYNO_START_RPM and RPM <= DynoStrategy.DYNO_FINISH_RPM:
#                 self.in_dyno_sweep = True
#             else:
#                 # print("/n *** DROPPED OUT OF DYNO SWEEP DUE TO RPM STALL *** /n")
#                 self.in_dyno_sweep = False # Shut down after sweep

#         elif self.mode == "roadtest":
#             # Simple roadtest cycle logic
#             self.phase = driver.theta // 300
#             if self.phase == 0:
#                 self.tps = 0.0 # Idle/Coast
#             elif self.phase == 1:
#                 self.tps = 25.0 # Part Throttle
#             elif self.phase == 2:
#                 self.tps = 100.0 # WOT pull
#             else:
#                 self.tps = 0.0
#         else:
#             self.tps = 0.0 

#         return self.tps


                
#     # ---------------------------------------------------------------------------------
#     def _set_load(self, engine_output_torque):
#         """
#         sets the wheel load for the Engine Model
        
#         :param engine_output_torque: Current out put torque of the Engine Model
#         """
#         global RPM

#         if self.mode == "idle" or self.mode == "wot":
#             self.load = 0.0

#         elif self.mode == "dyno":
#             self._set_dyno_target() # Check if we need to advance the target

#             if self.in_dyno_sweep:
#                 # Calculate the adjustment torque using PID
#                 self.load = self._pid_control()
#             else:
#                 # Zero load when sweep is over
#                 self.load = 0.0

#         elif self.mode == "roadtest":
#             # Temporary simple load, needs to be replaced with full Road Load Equation (Aero + Rolling)
#             self.load = 50.0

#         else:
#             self.load = 0.0 
        


#         return self.load

#     # ---------------------------------------------------------------------------------
#     # --- ORIGINAL _set_air_pressure METHOD (Restored) ---
#     # ---------------------------------------------------------------------------------
#     def _set_air_pressure(self):
#         # In this simple model, ambient pressure remains constant at sea level.
#         self.air_pressure = c.P_ATM_PA
#         return self.air_pressure

#     # ---------------------------------------------------------------------------------
#     # --- UPDATED get_environment ---
#     # ---------------------------------------------------------------------------------
#     def get_environment(self, rpm, engine_output_torque, cycle, theta):
#         # --- Update Global Variables ---
#         global RPM, driver.theta, CYCLE, ENGINE_TORQUE
#         RPM = rpm
#         driver.theta = theta
#         CYCLE = cycle
#         ENGINE_TORQUE = engine_output_torque
        
#         driver.cycle_rpm[driver.theta] = RPM

        
#         # --- Update Dyno Sweep State Initialization ---
#         # The sweep officially starts when the RPM reaches the start point
#         # if self.mode == "dyno" and RPM >= DynoStrategy.DYNO_START_RPM and self.dyno_target_rpm < DynoStrategy.DYNO_FINISH_RPM:
#         #      self.in_dyno_sweep = True
        
#         self._set_tps()
#         self._set_load(engine_output_torque)
#         self._set_air_pressure()

#         self.driver_dict.update(
#             {
#                 "throttle_pos"      : self.tps,
#                 "wheel_load"        : self.load,
#                 "ambient_pressure"  : self.air_pressure,
#                 "dyno_settled"      : (self.settled_cycles >= DynoStrategy.SETTLE_CYCLES_REQUIRED),
#                 "target_rpm"        : self.dyno_target_rpm
#             }
#         )
        
#         return self.get_driver_dict()
    
  