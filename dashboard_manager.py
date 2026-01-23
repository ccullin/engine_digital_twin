# ddashboard_manager.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import platform
import subprocess
import os



class DashboardManager:
    """Shared real-time plotting infrastructure"""
    
    def __init__(self):
        self.fig = None
        self.base_ax = None
        self.strategy_ax = None
        self.base_text = None  # for text table
        self.enabled = True
        self.stopped = False
        
        # set window Focus to the plot
        self._set_focus("plot")



    def get_or_create_figure(self):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(16, 9))

            # 2 rows, 2 columns:
            # Top-left:    Base universal telemetry (text)
            # Bottom-left: Strategy-specific telemetry table (text)
            # Right span:  Full-height strategy chart
            # gs = GridSpec(2, 2, figure=self.fig,
            #               width_ratios=[1, 3],
            #               height_ratios=[1, 1],
            #               wspace=0.4,
            #               hspace=0.4)  # increased hspace for separation
            
            # added addition column for 2 x 2 strategy plots.
            gs = GridSpec(2, 3, figure=self.fig,
                          width_ratios=[1, 1.5, 1.5],
                          height_ratios=[1, 1],
                          wspace=0.4,
                          hspace=0.4)

            # Top-left: Universal base telemetry
            self.base_ax = self.fig.add_subplot(gs[0, 0])
            self.base_ax.axis('off')
            self.base_text = self.base_ax.text(0.05, 0.95, "", va='top', ha='left',
                                              fontsize=10, family='monospace')

            # Bottom-left: Strategy-specific overlay table
            self.overlay_ax = self.fig.add_subplot(gs[1, 0])
            self.overlay_ax.axis('off')
            self.overlay_text = self.overlay_ax.text(0.05, 0.95, "", va='top', ha='left',
                                                    fontsize=10, family='monospace')
            self.overlay_text.set_text("─ STRATEGY TELEMETRY ─\n(No data yet)")

            # Right side: strategy panel
            # self.strategy_ax = self.fig.add_subplot(gs[:, 1]) # single combined chart on RHS
            # self.strategy_ax = self.fig.add_subplot(gs[0, 1]) # top chart on RHS
            # self.strategy_ax_bottom = self.fig.add_subplot(gs[1, 1]) # bottom chart on RHS
            
            self.strategy_ax_topleft = self.fig.add_subplot(gs[0, 1]) 
            self.strategy_ax_topright = self.fig.add_subplot(gs[0, 2]) 
            self.strategy_ax_bottomleft = self.fig.add_subplot(gs[1, 1]) 
            self.strategy_ax_bottomright = self.fig.add_subplot(gs[1, 2]) 

            self.strategy_ax_topleft.set_title("Mode-Specific Panel")

            # Clean padding
            self.fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07)

            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
            self.fig.canvas.mpl_connect("close_event", self.on_close_event)
            
            self.toolbar = self.fig.canvas.toolbar  # Keep reference

        return self.fig, self.base_ax, self.strategy_ax_topleft, self.strategy_ax_topright, self.strategy_ax_bottomleft, self.strategy_ax_bottomright
    
    def update_base_telemetry(self, cycle_count, data):
        if not self.enabled:
            return

        sensors, engine_data_dict, ecu_outputs_dict = data


        # Build formatted telemetry text
        lines = [
            "╔══════════════════════════════════╗",
            "║          BASE TELEMETRY          ║",
            "╚══════════════════════════════════╝",
            "",
            f"Cycle:          {cycle_count:8.0f}",
            "",
            "─ sensors ─",
            f"Throttle:       {sensors.TPS_percent:8.0f} %",
            f"RPM:            {sensors.rpm:8.0f}",
            f"RPM avg:        {np.mean(sensors.rpm_history):8.0f}",
            f"MAP:            {sensors.MAP_kPa:8.1f} kPa",
            f"AFR:            {sensors.afr:8.3f}",
            f"CLT:            {sensors.CLT_C:8.1f} °C",
            f"Knock:          {sensors.knock:8.3f}",
            "",
            "─ ECU Outputs ─",
            f"Idle Valve:     {ecu_outputs_dict['iacv_pos']:8.2e} %",
            f"ve_fraction:    {ecu_outputs_dict['ve_fraction']:8.2f}",
            f"Ignition Adv:   {ecu_outputs_dict['spark_timing']:8.1f} °BTDC",
            f"Target AFR:     {ecu_outputs_dict['afr_target']:8.2f}",
        ]

        telemetry_text = "\n".join(lines)

        fig = self.get_or_create_figure()
        self.base_text.set_text(telemetry_text)
    
    def update_strategy_overlay(self, text_lines):
        """Update the bottom-left strategy telemetry table"""
        if not self.enabled or self.overlay_text is None:
            return

        header = [
            "╔══════════════════════════════════╗",
            "║       STRATEGY TELEMETRY         ║",
            "╚══════════════════════════════════╝",
            ""
        ]
        content = header + text_lines
        telemetry_text = "\n".join(content)
        self.overlay_text.set_text(telemetry_text)

    def get_strategy_axes(self):
        """Return axes for strategy to plot on"""
        if not self.enabled:
            return None
        _, _, ax_topleft, ax_topright, ax_bottomleft, ax_bottomright = self.get_or_create_figure()
        # strategy_ax.cla()  # clear previous strategy plot
        return ax_topleft, ax_topright, ax_bottomleft, ax_bottomright

    # def create_or_update_figure(self, key, create_func, update_func, data=None):
    #     if key not in self.figures:
    #         self.figures[key] = create_func()
    #     update_func(self.figures[key], data)

    def show(self):
        plt.ioff()
        plt.show()
    
    def draw(self):
        if self.enabled and self.fig:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except:
                pass  # ignore if window closed

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            
    
    def _set_focus(self, target):
        """Switches focus to the plot window or back to the terminal."""
        if platform.system() != "Darwin":
            return

        # 1. Prioritize known terminal programs
        # Check for environment variable TERM_PROGRAM first for tools like iTerm or VS Code
        app_name = os.environ.get("TERM_PROGRAM", "Terminal")

        # 2. Fallback check for common terminal application names
        if app_name == "iTerm.app" or app_name == "iTerm":
            terminal_app = "iTerm"
        elif app_name == "vscode":
            # VS Code often needs the main app name
            terminal_app = "Visual Studio Code"
        else:
            # Default to macOS native Terminal
            terminal_app = "Terminal"

        if target == "plot":
            # Activate the Matplotlib/Python plot window
            apple_script = (
                f'tell application "System Events" to set frontmost of first process whose name is "Python" and '
                f'((name of window 1) contains "Figure 1") to true'
            )
        elif target == "terminal":
            # Explicitly activate the chosen terminal application
            apple_script = f'tell application "{terminal_app}" to activate'
        else:
            return

        try:
            # Use osascript with a timeout to prevent locking up
            subprocess.call(
                ["osascript", "-e", apple_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            # Silence the warning/error if the focus fails
            pass

    def set_system_reference(self, system_instance):
        self.system = system_instance

    def on_key_press(self, event):
        """Handles key presses in the matplotlib window."""
        if event.key == "q":
            self.close_and_cleanup()


    def on_close_event(self, event):
        """Handles clicking the 'X' button on the window."""
        self.close_and_cleanup()

    # def close_and_cleanup(self):
    #     """Centralized cleanup and focus return logic."""

    #     # The window close action (either 'q' or 'X' button)
    #     # must be performed before returning focus.
    #     # plt.close(self.fig)

    #     self.stopped = True
    #     self._set_focus("terminal")
    
    def close_and_cleanup(self):
        """Safely close figure and prevent Tkinter toolbar crash on macOS"""
        # if self.fig is None:
        #     self.stopped = True
        #     self._set_focus("terminal")
        #     return

        # --- FIX 1: Disconnect axes observers (prevents late toolbar updates) ---
        try:
            if hasattr(self.fig, "_axobservers"):
                self.fig._axobservers.callbacks.clear()
        except:
            pass

        # --- FIX 2: Explicitly hide/destroy toolbar before figure close ---
        try:
            if hasattr(self, "toolbar") and self.toolbar is not None:
                self.toolbar.set_visible(False)  # Hide it
                # On some backends, destroy() helps
                if hasattr(self.toolbar, "destroy"):
                    self.toolbar.destroy()
        except:
            pass

        self.stopped = True
        self._set_focus("terminal")

