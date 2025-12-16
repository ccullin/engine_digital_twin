# ddashboard_manager.py
#
# MIT License
#
# Copyright (c) 2025 Chris Cullin

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
            self.fig = plt.figure(figsize=(16, 9))  # wider window

            # Adjusted GridSpec: give more space to right panel
            # width_ratios=[1, 3] → base telemetry narrow, chart wide
            # Add hspace/wspace for breathing room
            gs = GridSpec(2, 2, figure=self.fig, 
                          width_ratios=[1, 3],    # ← more room for chart
                          height_ratios=[1, 1],
                          wspace=0.4,             # ← horizontal space between panels
                          hspace=0.3)             # ← vertical space

            # Top-left: Base telemetry
            self.base_ax = self.fig.add_subplot(gs[0, 0])
            self.base_ax.axis('off')
            self.base_text = self.base_ax.text(0.05, 0.95, "", va='top', ha='left',
                                              fontsize=10, family='monospace')

            # Right panel: Full height, wider
            self.strategy_ax = self.fig.add_subplot(gs[:, 1])
            self.strategy_ax.set_title("Mode-Specific Panel")

            # More padding around everything
            self.fig.tight_layout(pad=4.0)  # ← increased outer padding

            # Key events
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
            self.fig.canvas.mpl_connect("close_event", self.on_close_event)

        return self.fig, self.base_ax, self.strategy_ax
    
    def update_base_telemetry(self, cycle_count, data):
        if not self.enabled:
            return

        sensors_dict, engine_data_dict, ecu_outputs_dict = data


        # Build formatted telemetry text
        lines = [
            "╔══════════════════════════════════╗",
            "║          BASE TELEMETRY          ║",
            "╚══════════════════════════════════╝",
            "",
            f"Cycle:          {cycle_count:8.0f}",
            "",
            "─ Sensors ─",
            f"RPM:            {sensors_dict['RPM']:8.0f}",
            f"MAP:            {sensors_dict['MAP_kPa']:8.1f} kPa",
            f"AFR:            {sensors_dict['actual_AFR']:8.3f}",
            f"CLT:            {sensors_dict['CLT_C']:8.1f} °C",
            f"Knock:          {sensors_dict['Knock']:8.3f}",
            "",
            "─ ECU Outputs ─",
            f"Idle Valve:     {ecu_outputs_dict['idle_valve_position']:8.1f} %",
            f"ve_fraction:    {ecu_outputs_dict['ve_fraction']:8.2f}",
            f"Ignition Adv:   {ecu_outputs_dict['spark']:8.1f} °BTDC",
            f"Target AFR:     {ecu_outputs_dict['afr_target']:8.2f}",
        ]

        telemetry_text = "\n".join(lines)

        fig = self.get_or_create_figure()
        self.base_text.set_text(telemetry_text)

    def get_strategy_axes(self):
        """Return axes for strategy to plot on"""
        if not self.enabled:
            return None
        _, _, strategy_ax = self.get_or_create_figure()
        # strategy_ax.cla()  # clear previous strategy plot
        return strategy_ax

    # def create_or_update_figure(self, key, create_func, update_func, data=None):
    #     if key not in self.figures:
    #         self.figures[key] = create_func()
    #     update_func(self.figures[key], data)

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

    def close_and_cleanup(self):
        """Centralized cleanup and focus return logic."""

        # The window close action (either 'q' or 'X' button)
        # must be performed before returning focus.
        # plt.close(self.fig)

        self.stopped = True
        self._set_focus("terminal")

