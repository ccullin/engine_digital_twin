# driver_input.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import constants as c
import numpy as np
import driver_strategies as strategies

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=False)
class DriverOutput:
    """The interface between the engine physics and the ECU"""
    throttle_pos:float = 0.0
    wheel_load:float = 0.0
    ambient_pressure:float = 0.0
    impulse_target_rpm: float = 0.0
    
class DriverInput:
    def __init__(self, mode="idle", rpm=None, start_rpm=900):
        self.rpm = rpm
        self.cycle = 0 # crank cycle number , which is 720 degrees of rotation
        self.theta = 0 # crank degree of rotation 0-719

        
        self.mode = mode
        self.tps = 0.0  # foot off the accelerator
        self.load = 0.0  # sitting in neutral
        self.air_pressure = c.P_ATM_PA # sea level
        self.impulse_target_rpm = 0.0
        self.cycle_rpm = np.full(720, start_rpm) # array of the RPM measured at each theta (crank degree)
        
        self.strategy = self._create_strategy(mode)
        
        self.output = DriverOutput
        
    # ---------------------------------------------------------------------------------
    def get_driver_output(self):
        
        self.output.throttle_pos = self.tps
        self.output.wheel_load = self.load
        self.output.ambient_pressure = self.air_pressure
        self.output.impulse_target_rpm = self.impulse_target_rpm
        
        return self.output

    # ---------------------------------------------------------------------------------
    def _create_strategy(self, mode):
        if mode == "idle":
            return strategies.IdleStrategy()
        elif mode == "wot":
            return strategies.WotStrategy()
        elif mode == "dyno":
            return strategies.DynoStrategy()
        elif mode == "roadtest":
            return strategies.RoadtestStrategy()
        elif mode == "motor":
            return strategies.MotoringStrategy(rpm=self.rpm)
        elif mode =="impulse":
            return strategies.ImpulseDynoStrategy()

    
   
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
    def get_environment(self, rpm, engine_state, cycle, theta):
        # --- Update Global Variables ---
        self.rpm = rpm
        self.theta = theta
        self.cycle = cycle
        self.cycle_rpm[theta] = rpm
        self.governor_torque = np.average(engine_state.torque_governor_history)
               
        self.tps, self.load, self.air_pressure, self.impulse_target_rpm = self.strategy.driver_update(self) 
        
        return self.get_driver_output()
        