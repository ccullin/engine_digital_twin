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

# =================================== VALVE DATA ========================================
# 15 CAD to open to 1mm with FAT CAM profile of 0.7
#  8 CAD to open to 1mm with FAT CAM profile of 0.5

""" WORKS FOR DEBUGGING """
# VALVE_TIMING = {
#     'intake': {
#         'open':  0-10,       # IVO = 10 ATDC specs are at 1mm lift and it takes 15 CAD to get to 1mm
#         'close': 180+24,    # IVC = 39 After BDC
#         'max_lift':   9.6,   # 9.6 mm
#         'diameter':   40.0  # 40 mm
#     },
#     'exhaust': {
#         'open':  540-30 ,    # EVO = 40° Before BDC
#         'close': 0 + 5,     # EVC = 3° BTDC TDC
#         'max_lift':   9.6,   # 9.6 mm
#         'diameter':  34     # 34 mm # DEBUGGING WORKED AT 36 with 10mm lift
#     }
# }

"""
    FACTORY SPECS - WBX 2.1 MV
    Valve timing at 1mm lift.
    Inlet opens. 10° BTDC (710 CAD)
    Inlet closes. 48° ABDC (228 CAD)
    Exhaust opens. 50° BBDC (490 CAD)
    Exhaust closes. 0° ATDC (0 CAD)
"""
# direct setting at lift=0mm not 1mm
# VALVE_TIMING = {
#     'intake': {
#         'open':       705,  
#         'close':      228, 
#         'max_lift':   9.0,  # needs to allow for hydralic lift compression)
#         'diameter':  40.0   # 40 mm
#     },
#     'exhaust': {
#         'open':       490,  
#         'close':        5,  
#         'max_lift':   9.0,  # needs to allow for hydralic lift compression)
#         'diameter':    34   # 34 mm # DEBUGGING WORKED AT 36 with 10mm lift
#     }
# }


#set at 1mm height
# VALVE_TIMING = {
#     'intake': {
#         'cam_open':       6,  # 684
#         'cam_close':      211,  # 210
#         'max_lift':   8.9,  # needs to allow for hydralic lift compression)
#         'diameter':  40.0   # 40 mm
#     },
#     'exhaust': {
#         'cam_open':       507,  # 460 matches factory spec allowing for cam ramp
#         'cam_close':      702,  # 35   30 matches factory spec allowing for cam ramp
#         'max_lift':   8.9,  # needs to allow for hydralic lift compression)
#         'diameter':    34   # 34 mm # DEBUGGING WORKED AT 36 with 10mm lift
#     }
# }

""" using:
IVO: 716 (4 BTDC)
IVC: 220 (40 ABDC) (duration=224, centreline at 108)
EVO: 500 (40 BBDC)
EVC:   4 ( 4 ATDC) (duration=224, centreline at 612)
measured at 1mm
"""
VALVE_TIMING = {
    'intake': {
        'open_1mm':    716,  
        'close_1mm':   220,
        'max_lift':      8,  
        'diameter':     40.0   
    },
    'exhaust': {
        'open_1mm':    500,  
        'close_1mm':     4, 
        'max_lift':      9,  
        'diameter':     34.0   
    }
}


# =================================== FUEL INJECTOR and SPARK DATA ========================
INTAKE_DURATION_1mm  = 224.0 # degrees advertised.  at 1mm
EXHAUST_DURATION_1mm = 230
CENTERLINE           = 108.0    # degrees ATDC
INTAKE_MAX_LIFT      = 8 # mm
EXHAUST_MAX_LIFT     = 9 # mm
# DURATION             = 224.0 # degrees advertised.  seat to seat

# LOBE_SPERATION       = 108.0    # degrees
# ROCKER_RATIO         = 1.1      # factory ratio

# FOLLOWER_DIAM        = 24.0     # mm
# BASE_CIRCLE          = 34.0     # mm



#set at 1mm height
# VALVE_TIMING = {
#     'intake': {
#         'open_1mm':   710,  # @ 1mm lift
#         'close_1mm':  228,  # @ 1mm lift
#         'max_lift':      9.0,  # needs to allow for hydralic lift compression)
#         'diameter':     40.0   # 40 mm
#     },
#     'exhaust': {
#         'open_1mm':   490,  # @ 1mm lift
#         'close_1mm':    0,  # @ 1mm lift
#         'max_lift':      9.0,  # needs to allow for hydralic lift compression)
#         'diameter':     34.0   # 34 mm # DEBUGGING WORKED AT 36 with 10mm lift
#     }
# }


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
CRANK_RPM = 250.0
FUEL_OCTANE = 95
T_FUEL_K = T_AMBIENT


# =================================== RPM LIMITS ======================================
IDLE_RPM = 900.0
RPM_LIMIT = 5600


# M_GAS_INITIAL   = 1.2e-4        # kg (reasonable trapped mass at cold crank)

# =================================== PHYSICS CONSTANTS ================================
R_SPECIFIC_AIR = 287.0
R_SPECIFIC_EXHAUST  = 290.0         # approximate for mixture
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

