import matplotlib.pyplot as plt
import numpy as np
from driver_input import DriverInput
from ecu_controller import ECUController
from engine_model import EngineModel

# --- SETUP ---
TARGET_RPM = 2800
# Sweep from 10 to 50 degrees to see the full physical response
spark_sweep = np.linspace(0, 50, 51) 
results = []

driver = DriverInput(mode="dyno")
ecu = ECUController(rl_ecu_spark_mode=False)
engine = EngineModel(rpm=900)


def cycle_until_settled(timeout=100):
    """Ported from wot_env.py to ensure steady-state measurements"""
    strategy = driver.strategy
    cycles_waited = 0
    
    # We must run full cycles (720 deg) to trigger the dyno's settling logic
    while not strategy.point_settled and cycles_waited < timeout:
        run_one_cycle()
        cycles_waited += 1
        
    if cycles_waited >= timeout:
        print(f"Warning: Dyno failed to settle at {ecu.external_spark_advance}°")
        return False

    # Reset for next point
    strategy.point_settled = False
    return True

def run_one_cycle():
    """Standard 720 degree physics loop"""
    degree_count = 0
    while degree_count < 720:
        # Get dyno load from PID controller
        driver_dict = driver.get_environment(engine.rpm, 0, degree_count)
        
        engine.tps_sensor = driver_dict["throttle_pos"]
        engine.P_ambient_sensor = driver_dict["ambient_pressure"]
        engine.wheel_load = driver_dict["wheel_load"]
        
        ecu_outputs = ecu.update(engine.get_sensors())
        engine.step(ecu_outputs)
        degree_count += 1




# print(f"Short warm up to get dyno upto speed")
# run_one_cycle()

# NOW enable RL spark and dyno control
ecu.rl_ecu_spark_mode = True
driver.strategy.rl_dyno_mode = True
driver.strategy.external_target_rpm = TARGET_RPM


print(f"Starting Steady-State Sweep at {TARGET_RPM} RPM...")
for spark in spark_sweep:
    ecu.external_spark_advance = spark
    
    driver.strategy.external_target_rpm = TARGET_RPM
    
    # Wait for the PID to stabilize RPM before taking the measurement
    if cycle_until_settled():
        # Record data only once settled
        res = {
            'spark': spark,
            'torque': np.mean(engine.engine_data_dict['torque_history']),
            'knock': engine.sensors_dict['knock_intensity'],
            'peak_p': engine.engine_data_dict['peak_pressure_bar']
        }
        results.append(res)
        
        print(f"SETTLED | Spark: {spark:4.1f}° | Torque: {res['torque']:6.1f} Nm | "
              f"Knock: {res['knock']:4.2f} | Peak P: {res['peak_p']:5.1f} bar")

# --- PLOTTING ---
sparks = [r['spark'] for r in results]
torques = [r['torque'] for r in results]
knocks = [r['knock'] for r in results]

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Spark Advance (BTDC)')
ax1.set_ylabel('Torque (Nm)', color='tab:blue')
ax1.plot(sparks, torques, 'b-o', label='Torque')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Knock Intensity', color='tab:red')
ax2.plot(sparks, knocks, 'r--', label='Knock')
ax2.axhline(y=5.0, color='black', linestyle=':', label='Failure Threshold')

plt.title(f'EngineModel Physical Response @ {TARGET_RPM} RPM (Steady State)')
fig.tight_layout()
plt.show()