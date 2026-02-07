# logger.py

import csv
import time
from dataclasses import fields
import sys

LOGFILE = "engine_log.csv"


class Logger:
    def __init__(self, sensors, engine_data, ecu_data):
        self.start_time = time.time()
        self.csv_file = open(LOGFILE, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        
        sensor_keys = [f.name for f in fields(sensors)]

        HEADER_KEYS = ["time_s", *sensor_keys, *engine_data.keys(), *ecu_data.keys()]

        self.writer.writerow(HEADER_KEYS)
        # print(HEADER_KEYS)

    # ---------------------------------------------------------------------------
    def log(self, sensors, engine_data, ecu):
        t = time.time() - self.start_time

        # 1. Gather all data into a single list, maintaining the order defined by the headers.
        ROW_VALUES = [t, *sensors.values(), *engine_data.values(), *ecu.values()]

        # 2. Apply decimal formatting using a list comprehension
        # Floats are formatted to 3 decimal places (e.g., 12.345)
        final_row = [
            "{:.1f}".format(v) if isinstance(v, float) else str(v) for v in ROW_VALUES
        ]
        self.writer.writerow(final_row)
        self.csv_file.flush()
        # print(ROW)

    def close(self):
        self.csv_file.close()
