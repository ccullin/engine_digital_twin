# driver_input.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import constants as c
from engine_model import FixedKeyDictionary
import numpy as np
import driver_strategies as strategies


class DriverInput:

    def __init__(self, mode="idle"):
        self.rpm = 0
        self.cycle = 0 # crank cycle number , which is 720 degrees of rotation
        self.theta = 0 # crank degree of rotation 0-719

        
        self.mode = mode
        self.tps = 0.0  # foot off the accelerator
        self.load = 0.0  # sitting in neutral
        self.air_pressure = c.P_ATM_PA # sea level
        self.cycle_rpm = np.zeros(720) # array of the RPM measured at each theta (crank degree)
        
        self.strategy = self._create_strategy(mode)
        
        self.driver_dict = FixedKeyDictionary({
            "throttle_pos": self.tps,
            "wheel_load": self.load,
            "ambient_pressure": self.air_pressure,
        })

    # ---------------------------------------------------------------------------------
    def get_driver_dict(self):
        self.driver_dict.update(
            {
                "throttle_pos"      : self.tps,
                "wheel_load"        : self.load,
                "ambient_pressure"  : self.air_pressure,
            }
        )
        return self.driver_dict

    # ---------------------------------------------------------------------------------
    def _create_strategy(self, mode):
        if mode == "idle":
            return strategies.IdleStrategy()
        elif mode == "wot":
            return strategies.WotStrategy()
        elif mode == "dyno":
            return strategies.DynoStrategy()
        elif mode == "roadtest":
            return strategies.RoadTestStrategy()
        else:
            return strategies.IdleStrategy()
    
    # ---------------------------------------------------------------------------------
    # def _set_tps(self):

    #     if self.mode == "idle":
    #         self.tps = 0.0

    #     elif self.mode == "wot":
    #         self.tps = 100.0

    #     elif self.mode == "dyno":
    #         # Set throttle to WOT (100%) during the sweep to measure max torque
    #         self.tps = 100.0
            
    #         # The sweep is considered 'in progress' if RPM has reached the start point
    #         control_rpm = np.mean(self.cycle_rpm)
    #         if control_rpm >= (self.DYNO_START_RPM - 100) and control_rpm <= self.DYNO_FINISH_RPM:
    #         # if RPM >= self.DYNO_START_RPM and RPM <= self.DYNO_FINISH_RPM:
    #             self.in_dyno_sweep = True
    #         else:
    #             # print("/n *** DROPPED OUT OF DYNO SWEEP DUE TO RPM STALL *** /n")
    #             self.in_dyno_sweep = False # Shut down after sweep

    #     elif self.mode == "roadtest":
    #         # Simple roadtest cycle logic
    #         self.phase = THETA // 300
    #         if self.phase == 0:
    #             self.tps = 0.0 # Idle/Coast
    #         elif self.phase == 1:
    #             self.tps = 25.0 # Part Throttle
    #         elif self.phase == 2:
    #             self.tps = 100.0 # WOT pull
    #         else:
    #             self.tps = 0.0
    #     else:
    #         self.tps = 0.0 

    #     return self.tps

    # ---------------------------------------------------------------------------------
    # def _set_load(self, engine_output_torque):
    #     """
    #     sets the wheel load for the Engine Model
        
    #     :param engine_output_torque: Current out put torque of the Engine Model
    #     """
    #     global RPM

    #     if self.mode == "idle" or self.mode == "wot":
    #         self.load = 0.0

    #     elif self.mode == "dyno":
    #         self._set_dyno_target() # Check if we need to advance the target

    #         if self.in_dyno_sweep:
    #             # Calculate the adjustment torque using PID
    #             self.load = self._pid_control()
    #         else:
    #             # Zero load when sweep is over
    #             self.load = 0.0

    #     elif self.mode == "roadtest":
    #         # Temporary simple load, needs to be replaced with full Road Load Equation (Aero + Rolling)
    #         self.load = 50.0

    #     else:
    #         self.load = 0.0 
        


    #     return self.load

    # ---------------------------------------------------------------------------------
    # --- ORIGINAL _set_air_pressure METHOD (Restored) ---
    # ---------------------------------------------------------------------------------
    # def _set_air_pressure(self):
    #     # In this simple model, ambient pressure remains constant at sea level.
    #     self.air_pressure = c.P_ATM_PA
    #     return self.air_pressure

    # ---------------------------------------------------------------------------------
    # --- UPDATED get_environment ---
    # ---------------------------------------------------------------------------------
    def get_environment(self, rpm, cycle, theta):
        # --- Update Global Variables ---
        self.rpm = rpm
        self.theta = theta
        self.cycle = cycle
        self.cycle_rpm[theta] = rpm
               
        self.tps, self.load, self.air_pressure = self.strategy.driver_update(self) 
        
        return self.get_driver_dict()
        