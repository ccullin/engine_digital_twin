# constants.py

import numpy as np

# =================================== ENGINE GEOMETRY ===================================
BORE = 0.094  # m       (86 mm typical 2.0 L NA)
STROKE = 0.076  # m
LEN_CONROD = 0.143  # m
RADIUS_CRANK = STROKE / 2.0  # m
A_PISTON = np.pi * (BORE / 2.0) ** 2  # m²
COMP_RATIO = 9.0
V_DISPLACED = A_PISTON * STROKE  # m³ per cylinder
NUM_CYL = 4
V_INTAKE_MANIFOLD = V_DISPLACED * NUM_CYL * 1.5


# =================================== VALVE AND CAM  ========================
# INTAKE_DURATION_1mm  = 224.0 # degrees advertised.  (assume 1mm)
# EXHAUST_DURATION_1mm = 230.0
INTAKE_DURATION  = 238.0 # at 1mm.  see https://brickwerks.co.uk/tech/t3-technical-section/t3-data/mv-engine-data/
EXHAUST_DURATION = 230.0 # at 1mm
# INTAKE_DURATION  = 184.0 # @1mm
# EXHAUST_DURATION = 180.0 # @1mm
IS_AT_1mm        = True
INTAKE_CENTERLINE    = 109 #ATDC
EXHAUST_CENTERLINE   = 115 #BTDC
# CENTERLINE           = 108.0 # degrees ATDC
INTAKE_DIAM          = 40   # mm
INTAKE_MAX_LIFT      = 10.2 #8.75  # mm
EXHAUST_DIAM         = 34    # mm
EXHAUST_MAX_LIFT     = 10.2 #8.35  # mm



# =================================== FUEL INJECTOR and SPARK DATA ========================
# INJECTOR_FLOW_CC_PER_MIN = 200.0  # Factory Injector flow rate in cc/min
INJECTOR_FLOW_CC_PER_MIN = 470.0  # Work around to allow for spraying on back of value due to amount of fuel required.
FUEL_DENSITY_KG_CC = 7.5e-4  # Fuel density in kg/cc
INJECTOR_DEAD_TIME_MS = 0.0  # MUST BE ZERO as the digital ICE MODEL has no lag.
BURN_DURATION_DEG = 50.0

# =================================== ENVIRONMENT & INITIAL CONDITIONS ==========
P_ATM_PA = 101325.0  # at sea level (Pa)
T_AMBIENT = 293.0  # K (20°C)
T_INTAKE_K = 293.0  # K
T_EXHAUST_K = 900.0
COOLANT_START = 20.0
FUEL_OCTANE = 95
T_FUEL_K = T_AMBIENT


# =================================== RPM LIMITS ======================================
CRANK_RPM = 250.0
IDLE_RPM = 900.0
RPM_LIMIT = 6000


# M_GAS_INITIAL   = 1.2e-4        # kg (reasonable trapped mass at cold crank)

# =================================== PHYSICS CONSTANTS ================================
R_SPECIFIC_AIR = 287.0
R_SPECIFIC_EXHAUST  = 271.0 # 290.0         # approximate for mixture
R_SPECIFIC_FUEL = 45 # J/Kg.K
GAMMA_AIR = 1.4
GAMMA_EXHAUST = 1.33
GAMMA_FUEL = 1.05
LHV_FUEL_GASOLINE = 43.5e6  # J/kg (or 43,500 kJ/kg)



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

# =================================== Engine Output targets ==============================
MAX_TORQUE = 159 #Nm at 3200 rpm
MAX_POWER = 67 # Kw at 4800 rpm

