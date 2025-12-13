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
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["idle", "wot", "dyno", "roadtest", "rl"], default="wot"
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

 
        system.degree_count = 0
        while system.degree_count < 720 and not system.stop_simulation:
            
            # 1. Get driver inputs (TPS, load, etc.)
            engine_rpm = self.engine.rpm  # getting absolute latest
            # engine_output = self.engine.engine_data_dict['brake_torque_nm'] # getting absolute latest
            engine_output = self.engine.torque_brake # getting absolute latest
            driver_dict = self.driver.get_environment(engine_rpm, engine_output, cycle_count, system.degree_count)
            self.engine.tps_sensor = driver_dict["throttle_pos"]
            self.engine.P_ambient_sensor = driver_dict["ambient_pressure"]
            self.engine.wheel_load = driver_dict["wheel_load"]


            # 2. ECU runs — sees current crank angle and sensors
            ecu_outputs_dict = self.ecu.update(self.engine.get_sensors())

            # 3. Engine advances ONE degree with fresh ECU commands
            sensors_dict, engine_data_dict = self.engine.step(ecu_outputs_dict)
            
            # print(
            #     f"MAIN LOOP: "
            #     f" theta: {system.degree_count:3.0f} | "
            #     # f"error: {error:5.1f} | "
            #     f"cycle_mean_rpm: {np.mean(self.driver.cycle_rpm):4.0f} | "
            #     )

            system.degree_count += 1
            
        # if args.debug:
            # print(
            #     f"720° SUMMARY | Cycle {cycle_count:3d} |"
            #     f" RPM {sensors_dict['RPM']:4.0f} |"
            #     # f"  MAP {sensors_dict['MAP_kPa']:5.1f} kPa |"
            #     f"  Wheel load: {driver_dict['wheel_load']:+6.1f} Nm |"
            #     f"  Torque net Eng: {engine_data_dict['brake_torque_nm']:6.1f} Nm |"
            #     f"  Power : {engine_data_dict['brake_power_kw']:+6.2f} kW |"
            #     f"  dyno_settled: {driver_dict['dyno_settled']} |" 
            #     f"  target_rpm: {driver_dict['target_rpm']} |"
            # )
            
            # print(
            #     f"peak_error: {self.driver.pid_peak_error:6.2f} | "
            #     f"cycle_mean_rpm: {np.mean(self.driver.cycle_rpm):5.0f}"
            #     )
        self.driver.pid_peak_error = 0

        cycle_output_dicts = sensors_dict, engine_data_dict, ecu_outputs_dict

        return cycle_output_dicts


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
    max_cycles = 100

    try:
        while system.cycle_count < max_cycles and not system.stop_simulation:
            sensors_dict, engine_data_dict, ecu_outputs_dict = system.run_one_cycle(system.cycle_count)
            system.cycle_count += 1
            
            # update Logging & Dashboard very 10 cycles
            if not args.debug and system.cycle_count % 10 == 0: # Update dashboard every 10 cycles
                # if logger:
                #     logger.log(sensors_dict, engine_data_dict, ecu_outputs_dict)
                if dashboard:
                    dashboard.update(sensors_dict, engine_data_dict, ecu_outputs_dict)
        
        # if args.mode == "dyno":
        #     driver.print_final_pid_score() 

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    finally:
        if not (args.debug):
            print(f"stop simulation: {system.stop_simulation}")
            logger.close()
            dashboard.close()
        print("Real-time simulation complete.")
