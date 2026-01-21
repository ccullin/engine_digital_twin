# efi_tables.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class EFITables:

    def __init__(self):
        # === Table Grid Definition ===
        # self.rpm_bins = np.array([100, 600, 2000, 4000, 6000, 8000])
        # self.map_bins = np.array([30, 50, 75, 95, 105, 150])  # kPa
        
        self.rpm_bins = np.array([800, 1600, 2400, 3200, 4000, 4800, 5600])
        self.map_bins = np.array([30, 50, 70, 90, 110])  # kPa
        

        # === VE Table: Current, Safe Default, Min/Max Bounds ===
        self.ve_table = self._proven_ve_table()
        # self.ve_table = self._learned_ve_table()
        self.ve_min = self._ve_min_bounds()
        self.ve_max = self._ve_max_bounds()

        # === Spark Table (BTDC) ===
        self.spark_table = self._proven_spark_table()
        # self.spark_table = self._learned_spark_table()
        self.spark_min = self._spark_min_bounds()
        self.spark_max = self._spark_max_bounds()

        # === AFR Target Table ===
        self.afr_table = self._proven_afr_table()
        # self.afr_table = self._learned_afr_table()
        self.afr_min = self._afr_min_bounds()
        self.afr_max = self._afr_max_bounds()
        
        # === Injector Timing Table ===
        self.injector_table = self._proven_injector_end_table()
        # self.injector_table = self._learned_injector_table()
        self.injector_min = self._injector_min_bounds()
        self.injector_max = self._injector_max_bounds()
        
        # Interpolators
        self._build_interpolators()

    def _build_interpolators(self):
        # === Interpolators (updated when tables change) ===
        self.ve_interp = RegularGridInterpolator(
            (self.rpm_bins, self.map_bins), self.ve_table,
            bounds_error=False, fill_value=None
        )
        self.spark_interp = RegularGridInterpolator(
            (self.rpm_bins, self.map_bins), self.spark_table,
            bounds_error=False, fill_value=None
        )
        self.afr_interp = RegularGridInterpolator(
            (self.rpm_bins, self.map_bins), self.afr_table,
            bounds_error=False, fill_value=None
        )
        
        self.injector_interp = RegularGridInterpolator(
            (self.rpm_bins, self.map_bins), self.injector_table,
            bounds_error=False, fill_value=None
        )

    # =========================================================================
    # Safe Default Tables (conservative — used on reset)
    # =========================================================================
    def _safe_ve_table(self):
        return np.full((6, 6), 70.0)  # Flat 70% — safe, low power

    def _safe_spark_table(self):
        return np.full((6, 6), 20.0)  # Very retarded — safe starting point

    def _safe_afr_table(self):
        return np.full((6, 6), 13.5)  # Moderately rich — safe and smooth
    
    def _safe_injector_table(self):
        return np.array([
            [150, 140, 120, 100,  90,  80],  #  100 RPM
            [155, 140, 120, 100,  90,  80],  #  600 RPM
            [160, 150, 130, 110, 100,  90],  # 2000 RPM
            [170, 160, 150, 130, 120, 110],  # 4000 RPM
            [180, 170, 160, 150, 140, 130],  # 6000 RPM
            [190, 180, 170, 160, 150, 140],  # 8000 RPM
        ])
    
    # =========================================================================
    # PROVEN Tables (known working state - used as benchmark)
    # =========================================================================

    def _proven_ve_table(self):
        # Rows (RPM): 800, 1600, 2400, 3200, 4000, 4800, 5600
        # Cols (MAP): 30, 50, 70, 90, 110 kPa
        return np.array([
            # 30      50      70      90      110    (MAP kPa)
            [45.0,   65.0,  100.0,   80.0,   82.0],  # 800  rpm (Idle)
            [48.0,   70.0,   80.0,   85.0,   87.0],  # 1600 rpm
            [50.0,   72.0,   82.0,   88.0,   90.0],  # 2400 rpm
            [52.0,   75.0,   85.0,   90.0,   92.0],  # 3200 rpm (Peak Torque)
            [52.0,   75.0,   84.0,   88.0,   90.0],  # 4000 rpm
            [50.0,   72.0,   80.0,   84.0,   85.0],  # 4800 rpm
            [45.0,   68.0,   75.0,   78.0,   80.0],  # 5600 rpm (Power drop-off)
        ])
        
    def _proven_afr_table(self):
        # Rows: 800, 1600, 2400, 3200, 4000, 4800, 5600 RPM
        # Cols: 30, 50, 70, 90, 110 kPa MAP
        return np.array([
            [14.0, 12.0, 12.0, 13.2, 12.8],  # 800 rpm
            [14.7, 14.7, 14.0, 13.0, 12.5],  # 1600 rpm
            [15.0, 14.7, 14.0, 13.0, 12.5],  # 2400 rpm
            [15.5, 15.0, 14.0, 13.0, 12.5],  # 3200 rpm (Cruise/Power transition)
            [15.5, 15.0, 14.0, 13.0, 12.5],  # 4000 rpm
            [15.5, 14.7, 13.5, 12.8, 12.5],  # 4800 rpm
            [15.5, 14.7, 13.5, 12.8, 12.5],  # 5600 rpm
        ])
        
    def _proven_spark_table(self):
        # Rows: 800, 1600, 2400, 3200, 4000, 4800, 5600 RPM
        # Cols: 30, 50, 70, 90, 110 kPa MAP
        return np.array([
            [10, 12, 16, 8,  6],   # 800 rpm (Low advance for idle stability)
            [22, 20, 18, 16, 14],  # 1600 rpm
            [28, 26, 22, 18, 16],  # 2400 rpm
            [32, 30, 26, 22, 20],  # 3200 rpm
            [34, 32, 28, 24, 22],  # 4000 rpm
            [36, 34, 30, 26, 24],  # 4800 rpm
            [36, 34, 30, 28, 26],  # 5600 rpm
        ])
        


    def _proven_injector_end_table(self):
        # Rows (RPM): 800, 1600, 2400, 3200, 4000, 4800, 5600
        # Cols (MAP): 30, 50, 70, 90, 110 kPa
        return np.array([
            # 30     50     70     90    110   (MAP kPa)
            [188,   175,   160,   145,   125], # 800  RPM
            [192,   180,   165,   150,   132], # 1600 RPM
            [200,   188,   173,   158,   140], # 2400 RPM
            [208,   195,   180,   165,   150], # 3200 RPM
            [215,   202,   188,   175,   163], # 4000 RPM
            [222,   210,   195,   185,   175], # 4800 RPM
            [230,   218,   205,   195,   188], # 5600 RPM
        ])







    # def _proven_ve_table(self):
    #     return np.array([
    #         [ 40, 100, 100, 100, 100, 100],  # 100 rpm (Low Cranking VE)
    #         [ 40,  50,  60,  70, 100, 100],  # 600 rpm
    #         [ 40,  70,  85,  95, 100, 100],  # 2000 rpm
    #         [ 40,  80,  95, 102, 105, 105],  # 4000 rpm
    #         [ 40,  75,  90,  98, 102, 102],  # 6000 rpm
    #         [ 40,  65,  80,  90,  95,  95],  # 8000 rpm
    #     ])


    # def _proven_spark_table(self):
    #     # return np.array([
    #     #     [ 8,  8,  8,  8,  8,  8],  #  100 rpm (CRANKING Safe Retarded Spark)
    #     #     [ 8, 30, 25, 20,    18,   15],  # 600 rpm (Extrapolated the 150 kPa column as retarded for load)
    #     #     [38, 33, 28, 26.31, 26.31, 18],  # 2000 rpm
    #     #     [42, 37, 32, 27.27, 27.27, 27.27],  # 4000 rpm
    #     #     [38, 33, 28, 24, 24, 24],  # 6000 rpm
    #     #     [35, 30, 25, 23, 21, 18],  # 8000 rpm
    #     # ])
        
    #     return np.array([
    #         [ 8,  8,  8, 20, 30, 30],  #  100 rpm (CRANKING Safe Retarded Spark)
    #         [ 8, 30, 25, 20, 30, 30],  # 600 rpm (Extrapolated the 150 kPa column as retarded for load)
    #         [38, 33, 28, 30, 30, 30],  # 2000 rpm
    #         [42, 37, 32, 30, 30, 30],  # 4000 rpm
    #         [38, 33, 28, 25, 23, 20],  # 6000 rpm
    #         [35, 30, 25, 23, 21, 18],  # 8000 rpm
    #     ])
  
    
    # def _proven_afr_table(self):
    #     return np.array([
    #         [11.5, 11.5, 11.5, 12,  12,  12],  # 100 rpm (CRANKING Very rich)
    #         [11.5, 14.7, 14.7, 13.0, 13.0, 13.0],  # 600 rpm (Idle/Low Load)
    #         [14.7, 14.7, 14.7, 13.5, 13.0, 13.0],  # 2000 rpm
    #         [14.7, 14.7, 14.0, 12.8, 12.5, 12.5],  # 4000 rpm
    #         [14.7, 14.0, 13.0, 12.5, 12.2, 12.2],  # 6000 rpm
    #         [14.0, 13.5, 12.5, 12.0, 12.0, 12.0],  # 8000 rpm
    #     ])
        
    # def _proven_injector_end_table(self):
    #     return np.array([
    #         [150, 140, 120, 100,  90,  80],  #  100 RPM
    #         [155, 140, 120, 100,  90,  80],  #  600 RPM
    #         [160, 150, 130, 110, 100,  90],  # 2000 RPM
    #         [170, 160, 150, 130, 120, 110],  # 4000 RPM
    #         [180, 170, 160, 150, 140, 130],  # 6000 RPM
    #         [190, 180, 170, 160, 150, 140],  # 8000 RPM
    #     ])
        
    # =========================================================================
    # RL LEARNED Tables
    # =========================================================================
    def _learned_ve_table(self):
        return np.array([
            [ 40,   70,  70 ,  70 ,  70 ,  70 ],
            [ 40,   70,   70,   70,   70,   70 ],
            [ 40,   70,   70,   70,  107,  70 ],
            [ 40,   70,   70,   70,  115,   70 ],
            [ 40,   70,   70,   70,   70,   70 ],
            [ 40,   70,   70,   70,   70,   70 ],
        ])

    def _learned_spark_table(self):
        return np.array([
            [15,  15,  15,  15,  15,  15, ],
            [20,  20,  20,  20,  20,  20, ],
            [20,  20,  20,  20,  21, 20, ],
            [20,  20,  20,  20,  21, 20, ],
            [20,  20,  20,  20,  20,  20, ],
            [20,  20,  20,  20,  20,  20, ],  # 8000 rpm
        ])
    
    def _learned_afr_table(self):
        return np.array([
            [13. , 13. , 13. , 13. , 13. , 13.0],
            [13.5, 13.5, 13.5, 13.5, 14. , 14.0],
            [13. , 13.5, 13.5, 14. , 14.2, 14.5],
            [12.8, 13.5, 13.5, 13.5, 14. , 14.2],
            [12.5, 13.2, 13.5, 13.8, 14.2, 14.5],
            [12.2, 13. , 13.5, 14. , 14.5, 15.0],  # 8000 rpm
        ])
        
    def _learned_injector_table(self):
        return np.array([
            [150, 140, 120, 100,  90,  80],  #  100 RPM
            [155, 140, 120, 100,  90,  80],  #  600 RPM
            [160, 150, 130, 110, 100,  90],  # 2000 RPM
            [170, 160, 150, 130, 120, 110],  # 4000 RPM
            [180, 170, 160, 150, 140, 130],  # 6000 RPM
            [190, 180, 170, 160, 150, 140],  # 8000 RPM
        ])


    # =========================================================================
    # Physics-Based Hard Bounds (explicit values)
    # =========================================================================
    def _ve_min_bounds(self):
        return np.array([
            [60, 60, 60, 60, 60, 60],
            [40, 40, 40, 40, 40, 40],
            [35, 35, 35, 35, 35, 35],
            [35, 35, 35, 35, 35, 35],
            [35, 35, 35, 35, 35, 35],
            [35, 35, 35, 35, 35, 35],
        ])

    def _ve_max_bounds(self):
        return np.array([
            [110, 110, 110, 110, 110, 110],
            [90,   90,  90,  90,  90,  90],
            [105, 105, 110, 115, 115, 115],
            [105, 110, 115, 115, 115, 115],
            [100, 105, 110, 112, 112, 112],
            [90,   95, 100, 105, 105, 105],
        ])

    def _spark_min_bounds(self):
        return np.array([
            [5,  5,  5,  5,  5,  5],
            [8,  8,  8,  8,  8,  8],
            [15, 15, 15, 15, 15, 15],
            [20, 20, 20, 20, 20, 20],
            [18, 18, 18, 18, 18, 18],
            [15, 15, 15, 15, 15, 15],
        ])

    def _spark_max_bounds(self):
        return np.array([
            [15, 15, 15, 15, 15, 15],
            [35, 35, 32, 30, 28, 25],
            [42, 40, 38, 35, 33, 30],
            [45, 43, 40, 37, 35, 32],
            [42, 40, 37, 34, 32, 30],
            [38, 36, 34, 32, 30, 28],
        ])

    def _afr_min_bounds(self):
        return np.array([
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [11.0, 12.0, 13.0, 13.5, 14.0, 14.0],
            [11.5, 12.5, 13.5, 14.0, 14.2, 14.5],
            [11.5, 12.5, 13.0, 13.5, 14.0, 14.2],
            [11.8, 12.8, 13.2, 13.8, 14.2, 14.5],
            [12.0, 13.0, 13.5, 14.0, 14.5, 15.0],
        ])

    def _afr_max_bounds(self):
        return np.array([
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [13.5, 15.0, 15.5, 16.0, 16.0, 16.0],
            [13.0, 14.0, 15.0, 15.5, 15.8, 16.0],
            [12.8, 13.5, 14.5, 15.0, 15.5, 15.8],
            [12.5, 13.2, 14.0, 14.8, 15.2, 15.5],
            [12.2, 13.0, 13.8, 14.5, 15.0, 15.5],
        ])
        
    def _injector_min_bounds(self):
        """
        Latest possible end of injection (degrees BTDC)
        - Lower number = ends later in cycle
        - Conservative: don't end too late (risk of fuel not entering cylinder)
        """
        return np.array([
            [120, 110, 100,  90,  80,  70],  # 100 rpm  — cranking: end relatively early
            [130, 120, 110, 100,  90,  80],  # 600 rpm
            [140, 130, 120, 110, 100,  90],  # 2000 rpm
            [150, 140, 130, 120, 110, 100],  # 4000 rpm
            [160, 150, 140, 130, 120, 110],  # 6000 rpm
            [170, 160, 150, 140, 130, 120],  # 8000 rpm — still needs time for atomization
        ])

    def _injector_max_bounds(self):
        """
        Earliest possible end of injection (degrees BTDC)
        - Higher number = ends very early
        - Allows maximum vaporization time
        - But too early risks fuel hanging in port or short-circuiting
        """
        return np.array([
            [200, 200, 200, 200, 200, 200],  # 100 rpm
            [210, 210, 210, 210, 210, 210],  # 600 rpm
            [220, 220, 220, 220, 220, 220],  # 2000 rpm
            [230, 230, 230, 230, 230, 230],  # 4000 rpm
            [230, 230, 230, 230, 230, 230],  # 6000 rpm
            [220, 220, 220, 220, 220, 220],  # 8000 rpm — slightly less due to shorter cycle time
        ])
         
    def update_table(self, name: str, values: np.ndarray):
        values = np.asarray(values)
        if name == "ve":
            clipped = np.clip(values, self.ve_min, self.ve_max)
            self.ve_table[:] = clipped
            self.ve_interp.values[:] = clipped
        if name == "spark":
            clipped = np.clip(values, self.spark_min, self.spark_max)
            self.spark_table[:] = clipped
            self.spark_interp.values[:] = clipped
        if name == "afr":
            clipped = np.clip(values, self.afr_min, self.afr_max)
            self.afr_table[:] = clipped
            self.afr_interp.values[:] = clipped
        if name == "injector":
            clipped = np.clip(values, self.injector_min, self.injector_max)
            self.injector_table[:] = clipped
            self.injector_interp.values[:] = clipped
            
    def reset_to_safe(self):
        self.ve_table[:] = self._safe_ve_table()
        self.spark_table[:] = self._safe_spark_table()
        self.afr_table[:] = self._safe_afr_table()
        self.injector_table[:] = self._safe_injector_table()
        self._build_interpolators()
        
    def reset_to_prooven(self):
        self.ve_table[:] = self._prooven_ve_table()
        self.spark_table[:] = self._prooven_spark_table()
        self.afr_table[:] = self._prooven_afr_table()
        self.injector_table[:] = self._prooven_injector_table()
        self._build_interpolators()
        
    def reset_to_learned(self):
        self.ve_table[:] = self._learned_ve_table()
        self.spark_table[:] = self._learned_spark_table()
        self.afr_table[:] = self._learned_afr_table()
        self.injector_table[:] = self._learned_injector_table()
        self._build_interpolators()

    def lookup(self, rpm: float, map_kpa: float) -> dict:
        points = (rpm, map_kpa)
        return {
            "ve": float(self.ve_interp(points)),
            "spark": float(self.spark_interp(points)),
            "afr": float(self.afr_interp(points)),
            "injector": float(self.injector_interp(points)),
        }
    
    def apply_global_multipliers(self, ve_mult, spark_offset, afr_offset):
        """
        Applies the RL agent's 3 actions to the entire base map.
        ve_mult: Global multiplier for the Fuel (VE) table (e.g., 0.8 to 1.3).
        spark_offset: Degrees of timing added/subtracted (e.g., -10 to +15).
        afr_offset: Adjustment to target AFR (e.g., -1.5 to +1.5).
        """
        # 1. Update VE Table (Clipped by physics bounds)
        # Note the () after _safe_ve_table — it's a function call!
        new_ve = self._safe_ve_table() * ve_mult
        self.ve_table[:] = np.clip(new_ve, self.ve_min, self.ve_max)
        
        # 2. Update Spark Table
        new_spark = self._safe_spark_table() + spark_offset
        self.spark_table[:] = np.clip(new_spark, self.spark_min, self.spark_max)
        
        # 3. Update AFR Table
        new_afr = self._safe_afr_table() + afr_offset
        self.afr_table[:] = np.clip(new_afr, self.afr_min, self.afr_max)
        
        # CRITICAL: Rebuild/Update interpolators so the engine "sees" the new tables
        self._build_interpolators()
        
    def apply_local_adjustments(self, rpm, map_kpa, ve_mult, spark_offset, afr_offset):
        """
        Applies adjustments ONLY to the cells closest to current RPM/MAP.
        """
        # Find the 6x6 grid indices for current state
        # (This keeps the rest of the map 'Safe' while we tune this point)
        new_ve = self.ve_table.copy()
        new_spark = self.spark_table.copy()
        new_afr = self.afr_table.copy()

        # Simple approach: Find nearest neighbor cell and apply change
        rpm_idx = (np.abs(self.rpm_bins - rpm)).argmin()
        map_idx = (np.abs(self.map_bins - map_kpa)).argmin()

        # Apply the agent's 'suggested' offset to that specific cell
        new_ve[rpm_idx, map_idx] *= ve_mult
        new_spark[rpm_idx, map_idx] += spark_offset
        new_afr[rpm_idx, map_idx] += afr_offset

        # Update the live tables with clipped values
        self.update_table("ve", new_ve)
        self.update_table("spark", new_spark)
        self.update_table("afr", new_afr)