# driver_input.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import constants as c
from engine_model import FixedKeyDictionary
import numpy as np

# Global scope variables (MUST BE SET IN get_environment)
RPM = 0
THETA = 0
CYCLE = 0
ENGINE_TORQUE = 0

# BEST SO FAR KP= 1.0  KI= 0.08  KD= 3.0  Back Gain= 0.5
DYNO_KP = 1.0  # Proportional Gain 1.5 last know best.
DYNO_KI = 0.003 # Integral Gain
DYNO_KD = 3.5  # Derivative Gain was 1.5 last known best
DYNO_BACK_CALC_GAIN = 0.0   # Usually 0.3–1.0 — how aggressively we unwind
MAX_DYNO_TORQUE = 2000     # if your engine peaks at ~500 Nm → 1.5× headroom
# DYNO_KP = 1.5  # Proportional Gain
# DYNO_KI = 0.05 # Integral Gain
# DYNO_KD = 1.0  # Derivative Gain 
# WORKED ON LAST THETA OF CYCLE
# DYNO_KP = 2.0   # Proportional Gain
# DYNO_KI = 0.01 # Integral Gain
# DYNO_KD = 0.05   # Derivative Gain 
DYNO_STEP_SIZE_RPM = 100 # How far to step the RPM target after settling


class DriverInput:
    def __init__(self, mode="idle"):
        self.mode = mode
        self.tps = 0.0  # foot off the accelerator
        self.load = 0.0  # sitting in neutral
        self.air_pressure = c.P_ATM_PA
        self.phase = 0

        if mode == "idle":
            self.start_rpm = c.CRANK_RPM
        else:
            self.start_rpm = c.IDLE_RPM

        # --- Dyno Parameters for Stepped Sweep and PID ---
        self.DYNO_START_RPM = 2000
        self.DYNO_FINISH_RPM = c.RPM_LIMIT + 1000
        self.DYNO_TARGET_RPM = 2000 # Current RPM setpoint for the dyno
        self.DYNO_RPM_TOLERANCE = 5.0             # Tolerance for "settled" RPM (e.g., +/- 5 RPM)
        self.in_dyno_sweep = False 
        self.SETTLE_CYCLES_REQUIRED = 100     # Number of consecutive cycles required to be settled
        self.settled_cycles = 0             # Counter for consecutive settled cycles


        # --- Dyno PID Controller State Variables ---
        self.pid_error_integral = 0.0
        self.pid_derivative = 0.0
        self.pid_output = 0.0                # The PID controller's adjustment to the load
        self.cycle_rpm = np.zeros(720)
        # ------------------------------------------------
        
        # --- Dyno output table --------------------------
        self.settled_torque_map = {}   # RPM: average torque during stable period
        self.settled_torque_values = np.zeros(self.SETTLE_CYCLES_REQUIRED)
        #-------------------------------------------------
        
        # --- PID performance tracking -------------------
        self.current_step_peak = 0.0
        self.abs_error_sum = 0.0
        self.peak_error = 0.0
        self.total_settle_cycles = 0.0
        self.steps_completed = 0.0
        self.total_peak_error = 0.0
        self.max_peak_error = 0.0
        # -------------------------------------------------
        
        self.driver_dict = FixedKeyDictionary({
            "throttle_pos": self.tps,
            "wheel_load": self.load,
            "ambient_pressure": self.air_pressure,
            "dyno_settled": False,
            "target_rpm": self.DYNO_TARGET_RPM,
        })

    # ---------------------------------------------------------------------------------
    def get_driver_dict(self):
        return self.driver_dict

    # ---------------------------------------------------------------------------------
    def _set_tps(self):
        global RPM, THETA

        if self.mode == "idle":
            self.tps = 0.0

        elif self.mode == "wot":
            self.tps = 100.0

        elif self.mode == "dyno":
            # Set throttle to WOT (100%) during the sweep to measure max torque
            self.tps = 100.0
            
            # The sweep is considered 'in progress' if RPM has reached the start point
            control_rpm = np.mean(self.cycle_rpm)
            if control_rpm >= (self.DYNO_START_RPM - 100) and control_rpm <= self.DYNO_FINISH_RPM:
            # if RPM >= self.DYNO_START_RPM and RPM <= self.DYNO_FINISH_RPM:
                self.in_dyno_sweep = True
            else:
                # print("/n *** DROPPED OUT OF DYNO SWEEP DUE TO RPM STALL *** /n")
                self.in_dyno_sweep = False # Shut down after sweep

        elif self.mode == "roadtest":
            # Simple roadtest cycle logic
            self.phase = THETA // 300
            if self.phase == 0:
                self.tps = 0.0 # Idle/Coast
            elif self.phase == 1:
                self.tps = 25.0 # Part Throttle
            elif self.phase == 2:
                self.tps = 100.0 # WOT pull
            else:
                self.tps = 0.0
        else:
            self.tps = 0.0 

        return self.tps


    # ---------------------------------------------------------------------------------
    def _pid_control(self):
        """
        Calculates the torque adjustment needed to maintain DYNO_TARGET_RPM.
        """
        
        # ------------------------------------------------------------------
        # 1. Choose the right RPM signals for each term
        # ------------------------------------------------------------------
        instant_rpm = RPM
        mean_rpm = np.mean(self.cycle_rpm)
        
        
        # Light low-pass for P and I (20–30° window = ~3–5 ms at 6000 RPM)
        window_size = 25
        start_idx = (THETA - window_size) % 720          # <-- modulo handles wrap!
        if start_idx < THETA:
            # Normal case: no wrap
            smooth_rpm = np.mean(self.cycle_rpm[start_idx:THETA])
        else:
            # Wrapped case: concatenate end + beginning
            wrapped = np.concatenate((self.cycle_rpm[start_idx:], self.cycle_rpm[:THETA]))
            smooth_rpm = np.mean(wrapped)
        # ------------------------------------------------------------------
        # 2. Errors
        # ------------------------------------------------------------------
        filtered_error = mean_rpm - self.DYNO_TARGET_RPM
        error_smooth = smooth_rpm - self.DYNO_TARGET_RPM          # P + I
        error_raw    = instant_rpm - self.DYNO_TARGET_RPM         # not used directly

        # ------------------------------------------------------------------
        # 3. Proportional – on lightly filtered RPM
        # ------------------------------------------------------------------
        P = DYNO_KP * error_raw

        # ------------------------------------------------------------------
        # 4. Derivative – on instantaneous rate (proper sign & filtered)
        # ------------------------------------------------------------------
        # Store previous instant RPM for next call
        prev_rpm = self.cycle_rpm[(THETA - 1)%720]
        rpm_rate = instant_rpm - prev_rpm                         # deg-to-deg change

        # Light filtering of derivative to kill noise but keep phase
        alpha = 0.3
        self._filtered_drpm = getattr(self, '_filtered_drpm', 0.0)
        self._filtered_drpm = alpha * rpm_rate + (1 - alpha) * self._filtered_drpm

        D = -DYNO_KD * self._filtered_drpm                        # opposes acceleration
        
        # ------------------------------------------------------------------
        # 5–7. PID sum → clamp → back-calculation (unchanged, perfect)
        # ------------------------------------------------------------------
        pid_unclamped = P + self.pid_error_integral + D
        pid_clamped   = max(0.0, min(MAX_DYNO_TORQUE, pid_unclamped))

        saturation_error = pid_clamped - pid_unclamped
        self.pid_error_integral += DYNO_KI * error_smooth + DYNO_BACK_CALC_GAIN * saturation_error
        
        # ------------------------------------------------------------------
        # 8. Apply brake torque directly (NO engine torque subtraction!)
        # ------------------------------------------------------------------
        self.load = pid_clamped          # ← this is the opposing torque
        self.pid_output = pid_clamped
        
        # ------------------------------------------------------------------
        # 9. update PID performance score
        # ------------------------------------------------------------------
        self.abs_error_sum += error_raw
        if error_raw > self.peak_error:
            self.peak_error = error_raw
        

        # Optional: debug print every 10° or so
        # if THETA % 72 == 0:
        #     print(
        #         f"theta: {CYCLE}/{THETA:3.0f} | "
        #         f"I={self.pid_error_integral:8.0f} | "
        #         f"P={P:6.1f} | D={D:6.1f} | "
        #         f"unc={pid_unclamped:7.1f} → clamped={pid_clamped:6.1f} | "
        #         f"pid_load={self.pid_output:6.0f} | "
        #         f"T_engine: {ENGINE_TORQUE:6.0f} | "
        #         f"RPM={RPM:4.0f}→{mean_rpm:4.0f} target={self.DYNO_TARGET_RPM}"
        #         )
        
        return self.pid_output
        # # PROPORTIONAL RESPONSE
        # #----------------------
        # p_error = RPM - self.DYNO_TARGET_RPM
        
        # # DERIVATIVE RESPONSE 
        # #---------------------      
        # # derivative = error - self.pid_last_error
        # derivative = RPM - self.DYNO_TARGET_RPM  # using the current RPM as a look ahead on the mean
        # self.pid_derivative = derivative
        # # self.pid_last_error = error
        
        # # INTERGRAL REPONSE
        # #---------------------
        # # 1. set up the mean of last 180 degree of rpm
        # #--
        # start_theta = (THETA - 180)%720
        # if THETA > start_theta:
        #     rpm_sample = self.cycle_rpm[start_theta : THETA]
        # else:
        #     rpm_sample = np.concatenate( (self.cycle_rpm[start_theta:] , self.cycle_rpm[:THETA]) )
        # i_error =  np.mean(rpm_sample) - self.DYNO_TARGET_RPM
        # #-- 
        # # i_error =  np.mean(self.cycle_rpm) - self.DYNO_TARGET_RPM
        # # error = self.DYNO_TARGET_RPM - current_rpm

        # # 2. as the dyno is a unidirectional PID, i.e. it can brake but not accelerate the engine
        # # we need to only accumulate integral error on the brake side.
        # # Anti-windup (limit integral contribution)
        
        # #--
        # # self.pid_180_integral_error[THETA%180] = i_error
        # # i_error = np.mean(self.pid_180_integral_error)
        # #--
        
        # # --
        # trial_integral_error = self.pid_error_integral + i_error
        # trial_pid_output = (DYNO_KP * p_error) + (DYNO_KI * trial_integral_error) + (DYNO_KD * derivative)
        # if (ENGINE_TORQUE + trial_pid_output) < 0:
        #     pass # dont accumulate integral error as PID cannot accelerate the engine
        # else:
        #     self.pid_error_integral += i_error
        # #--
        
        # # self.pid_error_integral = np.sum(self.pid_180_integral_error)
        # # self.pid_error_integral += i_error
        # # self.pid_error_integral = max(min(self.pid_error_integral, 5000), -5000)
 

        # # 2. Calculate PID Output (Torque Adjustment)
        # self.pid_output = (DYNO_KP * p_error) + (DYNO_KI * self.pid_error_integral) + (DYNO_KD * derivative)
        
        # # self.pid_output = max(-1000.0, min(1000.0, self.pid_output)) # clamp +- 1000
        
        # load_next = (ENGINE_TORQUE - self.pid_output)
        # load_next = load_next if load_next > 0 else 0
        
        # print(
        #     f"theta: {CYCLE}/{THETA:3.0f} | "
        #     f"P_err: {p_error:5.1f} | "
        #     f"Integral: {self.pid_error_integral:5.0f} | "
        #     f"derivative: {self.pid_derivative:5.0f} | "
        #     f"pid out: {self.pid_output:6.2f} | "
        #     f"T_engine: {ENGINE_TORQUE:6.0f} | "
        #     # f"self.load next: {load_next:6.0f} | "
        #     f"self.load prev: {self.load:6.0f} | "
        #     f"RPM: {RPM:4.0f} | "
        #     f"target_rpm: {self.DYNO_TARGET_RPM:4.0f} | "
        #     f"mean_rpm: {np.mean(self.cycle_rpm):4.0f} | "
        #     # f"in_dyno_sweep: {self.in_dyno_sweep}"
        #     )
        
        # return self.pid_output

    # ---------------------------------------------------------------------------------
    def _set_dyno_target(self):
        control_rpm = np.mean(self.cycle_rpm)
        error = abs(self.DYNO_TARGET_RPM - control_rpm)

        if not self.in_dyno_sweep:
            return

        # Track peak error for this step
        if error > self.current_step_peak:
            self.current_step_peak = error

        if error < self.DYNO_RPM_TOLERANCE:
            self.settled_torque_values[self.settled_cycles] = self.pid_output
            self.settled_cycles += 1
        else:
            self.settled_cycles = 0
            # Note: you had self.pid_peak_error here — remove or replace with current_step_peak

        if self.settled_cycles >= self.SETTLE_CYCLES_REQUIRED:
            # --- Record stats for this completed step ---
            self.total_settle_cycles += self.SETTLE_CYCLES_REQUIRED
            self.steps_completed += 1
            
            stable_torque = np.mean(self.settled_torque_values)
            std_dev = np.std(self.settled_torque_values)
            filtered_stable_torque = np.sort(self.settled_torque_values)[10:-10]
            std_dev_filtered = np.std(filtered_stable_torque)
            filtered_stable_torque = np.mean(filtered_stable_torque)
            
            print(
                f"Target RPM {self.DYNO_TARGET_RPM:4.0f} | "
                f"raw Torque {stable_torque:5.0f} | "
                f"raw std_dev {std_dev:5.0f} | "
                f"Filtered T {filtered_stable_torque:5.0f} | "
                f"filteres std_dev {std_dev_filtered:5.0f} | "
                )
            
            
            
            
            
            self.total_peak_error += self.current_step_peak
            if self.current_step_peak > self.max_peak_error:
                self.max_peak_error = self.current_step_peak

            # Reset for next step
            self.settled_cycles = 0
            self.current_step_peak = 0.0

            # Advance target...
            if self.DYNO_TARGET_RPM < self.DYNO_FINISH_RPM:
                self.DYNO_TARGET_RPM += DYNO_STEP_SIZE_RPM
                if self.DYNO_TARGET_RPM > self.DYNO_FINISH_RPM:
                    self.DYNO_TARGET_RPM = self.DYNO_FINISH_RPM
            else:
                print("DYNO RUN COMPLETED")
                self.in_dyno_sweep = False
                self.tps = 0.0


        
                
                
    # ---------------------------------------------------------------------------------
    def _set_load(self, engine_output_torque):
        """
        sets the wheel load for the Engine Model
        
        :param engine_output_torque: Current out put torque of the Engine Model
        """
        global RPM

        if self.mode == "idle" or self.mode == "wot":
            self.load = 0.0

        elif self.mode == "dyno":
            self._set_dyno_target() # Check if we need to advance the target

            if self.in_dyno_sweep:
                # Calculate the adjustment torque using PID
                self.load = self._pid_control()
            else:
                # Zero load when sweep is over
                self.load = 0.0

        elif self.mode == "roadtest":
            # Temporary simple load, needs to be replaced with full Road Load Equation (Aero + Rolling)
            self.load = 50.0

        else:
            self.load = 0.0 
        


        return self.load

    # ---------------------------------------------------------------------------------
    # --- ORIGINAL _set_air_pressure METHOD (Restored) ---
    # ---------------------------------------------------------------------------------
    def _set_air_pressure(self):
        # In this simple model, ambient pressure remains constant at sea level.
        self.air_pressure = c.P_ATM_PA
        return self.air_pressure

    # ---------------------------------------------------------------------------------
    # --- UPDATED get_environment ---
    # ---------------------------------------------------------------------------------
    def get_environment(self, rpm, engine_output_torque, cycle, theta):
        # --- Update Global Variables ---
        global RPM, THETA, CYCLE, ENGINE_TORQUE
        RPM = rpm
        THETA = theta
        CYCLE = cycle
        ENGINE_TORQUE = engine_output_torque
        
        self.cycle_rpm[THETA] = RPM

        
        # --- Update Dyno Sweep State Initialization ---
        # The sweep officially starts when the RPM reaches the start point
        # if self.mode == "dyno" and RPM >= self.DYNO_START_RPM and self.DYNO_TARGET_RPM < self.DYNO_FINISH_RPM:
        #      self.in_dyno_sweep = True
        
        self._set_tps()
        self._set_load(engine_output_torque)
        self._set_air_pressure()

        self.driver_dict.update(
            {
                "throttle_pos"      : self.tps,
                "wheel_load"        : self.load,
                "ambient_pressure"  : self.air_pressure,
                "dyno_settled"      : (self.settled_cycles >= self.SETTLE_CYCLES_REQUIRED),
                "target_rpm"        : self.DYNO_TARGET_RPM
            }
        )
        
        return self.get_driver_dict()
    
    # ---------------------------------------------------------------------------------
    # --- PID SCORE ---
    # ---------------------------------------------------------------------------------
    def print_final_pid_score(self):
        if self.steps_completed == 0:
            print("\n" + "="*50)
            print("           DYNO PID PERFORMANCE SCORE")
            print("="*50)
            print(f"   KP= {DYNO_KP}  KI= {DYNO_KI}  KD= {DYNO_KD}  Back Gain= {DYNO_BACK_CALC_GAIN}")
            print(f"   FINAL SCORE         : *** No steps settled *** ")
            return

        avg_settle_time = self.total_settle_cycles / self.steps_completed / 720.0 * 60  # approx seconds at 6000 RPM
        avg_peak = self.total_peak_error / self.steps_completed

        # Final composite score (lower = better)
        score = (
            1.0 * avg_peak +           # penalize overshoot
            8.0 * avg_settle_time +    # fast settling is critical
            0.5 * self.max_peak_error  # penalize any really bad step
        )

        print("\n" + "="*50)
        print("           DYNO PID PERFORMANCE SCORE")
        print("="*50)
        print(f"   KP= {DYNO_KP}  KI= {DYNO_KI}  KD= {DYNO_KD}  Back Gain= {DYNO_BACK_CALC_GAIN}")
        print(f"   Steps completed     : {self.steps_completed}")
        print(f"   Avg peak error      : {avg_peak:.1f} RPM")
        print(f"   Worst peak error    : {self.max_peak_error:.1f} RPM")
        print(f"   Avg settle time     : {avg_settle_time:.2f} sec per 100 RPM")
        print(f"   FINAL SCORE         : {score:.1f}  ← lower = better")
        print("="*50)

        # Target benchmarks
        if score < 300:
            print("   → EXCELLENT - Professional dyno quality")
        elif score < 500:
            print("   → GOOD - Usable for tuning")
        elif score < 800:
            print("   → ACCEPTABLE - Needs improvement")
        else:
            print("   → POOR - Unstable or slow")
    