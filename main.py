# main.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import argparse
from engine_model import EngineModel
from ecu_controller import ECUController
from driver_input import DriverInput
from logger import Logger
# from dashboard import Dashboard  # ← Remove old dashboard
from dashboard_manager import DashboardManager  # ← New real-time manager
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["idle", "wot", "dyno", "roadtest", "rl"], default="wot")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--cycles", type=int, default=30)
args = parser.parse_args()

class SimulationManager:
    def __init__(self, driver, ecu, engine, logger=None, dashboard_manager=None):
        self.driver = driver
        self.ecu = ecu
        self.engine = engine
        self.logger = logger
        self.dashboard_manager = dashboard_manager 
        self.cycle_count = 0
        self.degree_count = 0
        self.stop_simulation = False

    def run_one_cycle(self, cycle_count):
        self.degree_count = 0
        while self.degree_count < 720 and not self.stop_simulation:
            
            # 1. get driver inputs (TPS (accelerator) wheel load (hill or dyno) and ambient air pressure (altitude))
            driver_dict = self.driver.get_environment(self.engine.rpm, cycle_count, self.degree_count)

            # 2. update engine with driver inputs
            self.engine.tps_sensor = driver_dict["throttle_pos"]
            self.engine.P_ambient_sensor = driver_dict["ambient_pressure"]
            self.engine.wheel_load = driver_dict["wheel_load"]
            
            # 3. get the ecu reponse to current engine sensors
            ecu_outputs_dict = self.ecu.update(self.engine.get_sensors())
            
            # 4. get engine reponse to update ECU inputs
            sensors_dict, engine_data_dict = self.engine.step(ecu_outputs_dict)
            
            

            self.degree_count += 1

        return sensors_dict, engine_data_dict, ecu_outputs_dict


if __name__ == "__main__":
    driver = DriverInput(mode=args.mode)
    ecu = ECUController()
    engine = EngineModel(rpm=driver.strategy.start_rpm)

    logger = None
    dashboard_manager = None

    if not args.debug:
        logger = Logger(engine.get_sensors(), engine.get_engine_data(), ecu.get_outputs())
        dashboard_manager = DashboardManager()  
        
    # Give the current strategy access to the dashboard
    driver.dashboard_manager = dashboard_manager
    system = SimulationManager(driver, ecu, engine, logger, dashboard_manager)
    
    cycle_count = 0

    try:
        exit_now = False
        while cycle_count < args.cycles and not exit_now:
            exit_now = dashboard_manager.stopped if dashboard_manager else False
            
            sensors_dict, engine_data_dict, ecu_outputs_dict = system.run_one_cycle(cycle_count)
            data = (sensors_dict, engine_data_dict, ecu_outputs_dict)
            cycle_count += 1

            # Optional: old-style bulk update every 10 cycles (can keep or remove)
            # if not args.debug and cycle_count % 10 == 0:
            if not args.debug:
                # if logger:
                #     logger.log(...)
                           
                # === REAL-TIME DASHBOARD UPDATE for base telemetry ===
                if dashboard_manager:
                    dashboard_manager.update_base_telemetry(cycle_count, data)
                    if hasattr(driver.strategy, 'update_dashboard'):
                        driver.strategy.update_dashboard(
                            dashboard_manager,
                            current_cycle=cycle_count,
                            data=data
                        )
                    dashboard_manager.draw()


    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    finally:
        if not args.debug:
            if logger:
                logger.close()
            if dashboard_manager:
                dashboard_manager.close()  # clean shutdown
        print("Real-time simulation complete.")
