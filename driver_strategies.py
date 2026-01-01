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
    """ Engine start from cranking RPM to idle RPM """
    def __init__(self):
        self.start_rpm = 900
    
    def driver_update(self, driver):
        tps = 100.0
        load = 0.0
        pressure = c.P_ATM_PA
        return tps, load, pressure

class RoadTestStrategy(BaseStrategy):
    """Realistic open-road driving: hills, varying throttle, altitude effects"""
    
    CYCLE_LENGTH = 400  # cycles for one full "drive" loop (~80–40 sec depending on RPM)
    
    def __init__(self):
        # Pre-compute profiles for speed (no per-cycle computation overhead)
        cycles = np.arange(RoadTestStrategy.CYCLE_LENGTH)
        phase = cycles / RoadTestStrategy.CYCLE_LENGTH * 2 * np.pi
        
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
        aero_base_N = 100 + 300 * np.power(np.linspace(20, 80, RoadTestStrategy.CYCLE_LENGTH)/50, 2)
        total_load_N = roll_N + grade_N + aero_base_N
        self.load_profile = np.clip(total_load_N * 0.15, 0, 550)  # tuned to realistic engine torque range

    def driver_update(self, driver):
        cycle_idx = driver.cycle % RoadTestStrategy.CYCLE_LENGTH
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
        ax = dashboard_manager.get_strategy_axes()
        if ax is None:
            return

        progress = current_cycle % RoadTestStrategy.CYCLE_LENGTH
        current_idx = int(progress)

        # === FIRST CALL: Create all artists once ===
        if not hasattr(self, 'artists_created'):
            ax.clear()
            ax.set_title("Road Test - Mountain Drive Profile")
            ax.set_xlabel("Cycle (loop = 400 cycles)")
            ax.set_ylabel("Throttle % / Load (Nm)")
            ax.set_ylim(0, max(110, self.load_profile.max() + 50))
            ax.grid(True, alpha=0.3)

            cycles_x = np.arange(RoadTestStrategy.CYCLE_LENGTH)

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
            f"Progress:      {progress:.0f}/{RoadTestStrategy.CYCLE_LENGTH}",
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
            ax = driver.dashboard_manager.get_strategy_axes()
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
