# constants.py

import numpy as np

# =================================== ENGINE GEOMETRY ===================================
BORE = 0.094  # m       (86 mm typical 2.0 L NA)
STROKE = 0.076  # m
LEN_CONROD = 0.143  # m
RADIUS_CRANK = STROKE / 2.0  # m
A_PISTON = np.pi * (BORE / 2.0) ** 2  # m²
COMP_RATIO = 10.5
V_DISPLACED = A_PISTON * STROKE  # m³ per cylinder
NUM_CYL = 4

# =================================== VALVE DATA ========================================
# VALVE_TIMING = {
#     'intake': {
#         'open_btdc':  5,    # IVO = 5° Before TDC
#         'close_abdc': 55,   # IVC = 55° After BDC
#         'max_lift':   9.5,  # mm
#         'diameter':   32.0  # mm
#     },
#     'exhaust': {
#         'open_bbdc':  55,   # EVO = 55° Before BDC
#         'close_atdc': 10,   # EVC = 10° After TDC
#         'max_lift':   9.0,
#         'diameter':   28.0
#     }
# }

# =================================== FUEL INJECTOR and SPARK DATA ========================
INJECTOR_FLOW_CC_PER_MIN = 500.0  # Injector flow rate in cc/min
FUEL_DENSITY_KG_CC = 7.5e-4  # Fuel density in kg/cc
INJECTOR_DEAD_TIME_MS = 0.0  # MUST BE ZERO as the digital ICE MODEL has no lag.
BURN_DURATION_DEG = 30.0

# =================================== ENVIRONMENT & INITIAL CONDITIONS ==========
P_ATM_PA = 101325.0  # at sea level (Pa)
T_AMBIENT = 293.0  # K (20°C)
T_INTAKE_K = 308.0  # K (35°C)
T_EXHAUST_K = 900.0
T_WALL = 450.0
COOLANT_START = 20.0
CRANK_RPM = 250.0


# =================================== RPM LIMITS ======================================
IDLE_RPM = 900.0
RPM_LIMIT = 6000


# M_GAS_INITIAL   = 1.2e-4        # kg (reasonable trapped mass at cold crank)

# =================================== PHYSICS CONSTANTS ================================
R_SPECIFIC_AIR = 287.0
GAMMA_AIR = 1.4
LHV_FUEL_GASOLINE = 43.5e6  # J/kg (or 43,500 kJ/kg)
# R_SPECIFIC_MIX  = 290.0         # approximate for mixture

# Wiebe
WEIBE_A = 5.0
WB_M = 2.0
BURN_DELAY = 8.0  # °CA after spark

# =================================== DYNAMICS =========================================
MOMENT_OF_INERTIA = 0.18  # kg·m² (typical 2.0 L 4-cyl + flywheel)
RPM_MAX_FLOW_REF = 6500.0
K_CD_RPM_CORR = 0.35

# =================================== SIMULATION SETTINGS ==============================
THETA_MIN = 0.0
THETA_MAX = 720.0
THETA_DELTA = 1.0

# Wall heat transfer
C_WALL = 2.0
C_TURB = 0.0  # simplified
