# driver_input.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import constants as c

# Global scope variables (MUST BE SET IN get_environment)
RPM = 0
CYCLE_COUNT = 0


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

        self.driver_dict = {
            "throttle_pos": self.tps,
            "wheel_load": self.load,
            "ambient_pressure": self.air_pressure,
        }

        # --- New Dyno Parameters ---
        self.DYNO_LOAD_NM = (
            100  # Constant resistive torque (Load) applied by the dyno, in Nm
        )
        self.DYNO_START_RPM = (
            2000  # RPM at which the dyno brake is released and the sweep begins
        )
        self.DYNO_FINISH_RPM = c.RPM_LIMIT  # RPM at which the sweep stops
        self.in_dyno_sweep = False  # State tracking if the sweep has started
        # ---------------------------

    def _set_tps(self):
        global RPM

        if self.mode == "idle":
            self.tps = 0.0

        elif self.mode == "wot":
            self.tps = 100.0

        elif self.mode == "dyno":
            # When RPM reaches the start point, set TPS to 100% and initiate the sweep
            self.tps = 100.0
            # if RPM > self.DYNO_START_RPM and RPM < self.DYNO_FINISH_RPM:
            if RPM < self.DYNO_FINISH_RPM:
                # if RPM < self.DYNO_FINISH_RPM:
                self.in_dyno_sweep = True
            else:
                # Before or after the sweep, keep throttle closed to idle/stop
                self.in_dyno_sweep = False

        elif self.mode == "circuit":
            # ... (Existing circuit logic) ...
            self.tps = 100.0 if RPM > 1200 else 0.0
            self.phase = CYCLE_COUNT // 300
            if self.phase == 0:
                self.tps = 0.0
            elif self.phase == 1:
                self.tps = 25.0
            elif self.phase == 2:
                self.tps = 100.0
            else:
                self.tps = 0.0
        else:
            self.tps = 0.0  # catch all and just idle

        return self.tps

    def _set_load(self):
        # Determine the wheel load based on the driver mode

        if self.mode == "idle" or self.mode == "wot":
            # Zero load in neutral for idle/wot
            self.load = 0.0

        elif self.mode == "dyno":
            R_START = 1000.0  # RPM
            R_END = 2000.0  # RPM
            TAU_TARGET = self.DYNO_LOAD_NM  # e.g., 100.0 Nm

            # In dyno mode, apply a constant resistive load (torque) during the sweep.
            if self.in_dyno_sweep:
                if self.in_dyno_sweep:
                    if RPM < R_START:
                        # Below the ramp start, load is zero.
                        self.load = 0.0
                elif RPM < R_END:
                    # Linearly ramp the load from 0 to TAU_TARGET.
                    scale_factor = (RPM - R_START) / (R_END - R_START)
                    self.load = TAU_TARGET * scale_factor
                else:
                    # Above the ramp end, load is the full target value.
                    self.load = TAU_TARGET
            else:
                # No load when not in the active sweep phase
                self.load = 0.0

        elif self.mode == "circuit":
            # For circuit, you'd typically apply a load that simulates vehicle inertia
            # and aerodynamic drag, but we'll set it to a simple baseline for now.
            self.load = 50.0

        else:
            self.load = 0.0  # Default to no load

        return self.load

    def _set_air_pressure(self):
        # update self.air_pressure
        return self.air_pressure

    def get_environment(self, rpm, cycle):
        # --- Critical Fix: Must use the global keyword to update module-level variables ---
        global RPM, CYCLE_COUNT
        RPM = rpm
        CYCLE_COUNT = cycle
        # ---------------------------------------------------------------------------------

        self._set_tps()
        self._set_load()
        self._set_air_pressure()

        self.driver_dict.update(
            {
                "throttle_pos": self.tps,
                "wheel_load": self.load,
                "ambient_pressure": self.air_pressure,
            }
        )
        return self.driver_dict
