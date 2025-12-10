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
from dashboard import Dashboard

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["idle", "wot", "dyno", "circuit", "rl"], default="wot"
)
parser.add_argument("--debug", action="store_true", default=False)

args = parser.parse_args()


class SimulationManager:
    """
    Real-time, crank-synchronous simulation.
    One loop = one crank degree (or every N degrees).
    """

    def __init__(self, driver, ecu, engine, logger=None, dashboard=None):
        self.driver = driver
        self.ecu = ecu
        self.engine = engine
        # self.logger = logger
        self.dashboard = dashboard
        self.cycle_count = 0
        self.degree_count = 0
        self.stop_simulation = False

    # =================================================================================
    def run_one_cycle(self, cycle_count):

        # 1. Get driver inputs (TPS, load, etc.)
        driver_dict = self.driver.get_environment(self.engine.rpm, self.degree_count)
        self.engine.tps_sensor = driver_dict["throttle_pos"]
        self.engine.P_ambient_sensor = driver_dict["ambient_pressure"]
        self.engine.wheel_load = driver_dict["wheel_load"]

        system.degree_count = 0
        while system.degree_count < 720 and not system.stop_simulation:

            # 2. ECU runs — sees current crank angle and sensors
            ecu_outputs_dict = self.ecu.update(self.engine.get_sensors())

            # 3. Engine advances ONE degree with fresh ECU commands
            sensors_dict, engine_data_dict = self.engine.step(ecu_outputs_dict)

            system.degree_count += 1

        # 4. after cycle update Logging & Dashboard
        if not args.debug:
            # if self.logger:
            # self.logger.log(sensors_dict, engine_data_dict, ecu_outputs_dict)
            if self.dashboard:
                self.dashboard.update(sensors_dict, engine_data_dict, ecu_outputs_dict)

        return sensors_dict, engine_data_dict


# --- Main Execution Script ---
if __name__ == "__main__":
    driver = DriverInput(mode=args.mode)
    ecu = ECUController()
    engine = EngineModel(rpm=driver.start_rpm)

    if args.debug:
        system = SimulationManager(driver, ecu, engine)
    else:
        logger = Logger(engine.get_sensors(), engine.get_data(), ecu.get_outputs())
        dashboard = Dashboard()
        system = SimulationManager(driver, ecu, engine, logger, dashboard)
        dashboard.set_system_reference(system)

    cycle_count = 0
    max_cycles = 10000

    try:
        while system.cycle_count < max_cycles and not system.stop_simulation:
            system.run_one_cycle(system.cycle_count)
            system.cycle_count += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    finally:
        if not (args.debug):
            print(f"stop simulation: {system.stop_simulation}")
            logger.close()
            dashboard.close()
        print("Real-time simulation complete.")
