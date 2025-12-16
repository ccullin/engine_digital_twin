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
        self.rpm_bins = np.array([100, 600, 2000, 4000, 6000, 8000])
        self.map_bins = np.array([30, 50, 75, 95, 105, 150])  # kPa

        # === VE Table: Current, Safe Default, Min/Max Bounds ===
        self.ve_table = self._prooven_ve_table()
        # self.ve_table = self._safe_ve_table()
        self.ve_min = self._ve_min_bounds()
        self.ve_max = self._ve_max_bounds()

        # === Spark Table (BTDC) ===
        self.spark_table = self._prooven_spark_table()
        # self.spark_table = self._safe_spark_table()
        self.spark_min = self._spark_min_bounds()
        self.spark_max = self._spark_max_bounds()

        # === AFR Target Table ===
        self.afr_table = self._prooven_afr_table()
        # self.afr_table = self._safe_afr_table()
        self.afr_min = self._afr_min_bounds()
        self.afr_max = self._afr_max_bounds()
        
        # === Injector Timing Table ===
        self.injector_table = self._prooven_injector_table()
        # self.injector_table = self._safe_injector_table()
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
    
    # =========================================================================
    # PROVEN Tables (known working state - used as benchmark)
    # =========================================================================
    def _prooven_ve_table(self):
        return np.array([
            [100, 100, 100, 100, 100, 100],  # 100 rpm (Low Cranking VE)
            [ 70,  50,  60,  70,  75,  75],  # 600 rpm
            [ 50,  70,  85,  95, 100, 100],  # 2000 rpm
            [ 55,  80,  95, 102, 105, 105],  # 4000 rpm
            [ 60,  75,  90,  98, 102, 102],  # 6000 rpm
            [ 50,  65,  80,  90,  95,  95],  # 8000 rpm
        ])

    def _prooven_spark_table(self):
        return np.array([
            [ 8,  8,  8,  8,  8,  8],  #  100 rpm (CRANKING Safe Retarded Spark)
            [ 8, 30, 25, 20, 18, 15],  # 600 rpm (Extrapolated the 150 kPa column as retarded for load)
            [38, 33, 28, 23, 21, 18],  # 2000 rpm
            [42, 37, 32, 27, 25, 22],  # 4000 rpm
            [38, 33, 28, 25, 23, 20],  # 6000 rpm
            [35, 30, 25, 23, 21, 18],  # 8000 rpm
        ])
    
    def _prooven_afr_table(self):
        return np.array([
            [11.5, 11.5, 11.5, 11.5, 11.5, 11.5],  # 100 rpm (CRANKING Very rich)
            [11.5, 14.7, 14.7, 14.7, 14.0, 14.0],  # 600 rpm (Idle/Low Load)
            [14.7, 14.7, 14.7, 13.5, 13.0, 13.0],  # 2000 rpm
            [14.7, 14.7, 14.0, 12.8, 12.5, 12.5],  # 4000 rpm
            [14.7, 14.0, 13.0, 12.5, 12.2, 12.2],  # 6000 rpm
            [14.0, 13.5, 12.5, 12.0, 12.0, 12.0],  # 8000 rpm
        ])
        
    def _prooven_injector_table(self):
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
        self.ve_table[:] = self._safe_ve()
        self.spark_table[:] = self._safe_spark()
        self.afr_table[:] = self._safe_afr()
        self.injector_table[:] = self._safe_injector()
        self._build_interpolators()

    def lookup(self, rpm: float, map_kpa: float) -> dict:
        points = (rpm, map_kpa)
        return {
            "ve": float(self.ve_interp(points)),
            "spark": float(self.spark_interp(points)),
            "afr": float(self.afr_interp(points)),
            "injector": float(self.injector_interp(points)),
        }