import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import constants as c
import platform
import subprocess
import os


class Dashboard:
    def __init__(self):
        plt.style.use("dark_background")
        plt.ion()

        self.system = None

        self.fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
        self.fig.suptitle(
            "ICE TWIN — 2.0L NA DIGITAL TWIN — RL COCKPIT",
            fontsize=18,
            color="cyan",
            fontweight="bold",
        )

        # Connect the key press event handler
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close_event)

        # --- MODIFICATION 1: Adjust Gridspec for more spacing and a wider gauge area ---
        # Increased hspace and wspace slightly for more breathing room
        gs = self.fig.add_gridspec(10, 12, hspace=0.6, wspace=0.6)

        # Big dyno - SHRUNK HORIZONTALLY (0:9 -> 0:8)
        self.ax_dyno = self.fig.add_subplot(gs[0:5, 0:8])
        self.ax_dyno.set_xlim(c.IDLE_RPM, c.RPM_LIMIT)
        self.ax_dyno.set_ylim(-50, 260)
        self.ax_dyno.set_ylabel("Torque (Nm)", color="yellow", fontsize=14)
        self.ax_dyno.set_xlabel("RPM", color="white")
        self.ax_dyno.grid(alpha=0.3)

        self.ax_pwr = self.ax_dyno.twinx()
        self.ax_pwr.set_ylim(0, 200)
        self.ax_pwr.set_ylabel("Power (kW)", color="orange", fontsize=14)

        # Gauge panel - ENLARGED (0:4, 9:12 -> 0:6, 8:12) to prevent text overlap
        self.ax_gauge = self.fig.add_subplot(gs[0:5, 8:12])
        self.ax_gauge.set_facecolor("#0e1117")
        self.ax_gauge.axis("off")

        # Mini charts - ADDED TITLES AND GRIDS for clarity

        self.ax_throttle = self.fig.add_subplot(gs[6:7, 0:4])
        self.ax_throttle.set_title("TPS & Idle Valve (%)", color="gold")
        self.ax_throttle.grid(alpha=0.3)

        self.ax_spark = self.fig.add_subplot(gs[6:7, 5:8])
        self.ax_spark.set_title("Spark Advance (°BTDC)", color="magenta")
        self.ax_spark.grid(alpha=0.3)

        self.ax_lambda = self.fig.add_subplot(gs[6:7, 9:12])
        self.ax_lambda.set_title("Lambda Actual", color="white")
        self.ax_lambda.grid(alpha=0.3)

        self.ax_safety = self.fig.add_subplot(gs[8:10, 0:6])
        self.ax_safety.set_title("Knock & Peak Pressure", color="red")
        self.ax_safety.grid(alpha=0.3)

        self.ax_temp = self.fig.add_subplot(gs[8:10, 7:12])
        self.ax_temp.set_title("Coolant Temp (°C)", color="red")
        self.ax_temp.grid(alpha=0.3)

        # Lines
        (self.line_trq,) = self.ax_dyno.plot([], [], color="yellow", lw=3)
        (self.line_pwr,) = self.ax_pwr.plot([], [], color="orange", lw=3, alpha=0.9)

        (self.line_tps,) = self.ax_throttle.plot([], [], color="gold", lw=3)
        (self.line_idle,) = self.ax_throttle.plot([], [], color="deepskyblue", lw=2.5)
        (self.line_spark,) = self.ax_spark.plot([], [], color="magenta", lw=3)
        (self.line_lam,) = self.ax_lambda.plot([], [], color="white", lw=3)
        (self.line_knock,) = self.ax_safety.plot([], [], color="red", lw=3)
        (self.line_ppeak,) = self.ax_safety.plot([], [], color="gray", lw=2, alpha=0.7)
        (self.line_clt,) = self.ax_temp.plot([], [], color="red", lw=3)

        # Session best
        self.best_fill_trq = None
        self.best_fill_pwr = None

        # History
        self.history_x = []
        self.history = {
            k: []
            for k in [
                "rpm",
                "torque",
                "power",
                "map",
                "tps",
                "idle",
                "spark",
                "lambda",
                "afr",
                "clt",
                "knock",
                "ppeak",
                "reward",
            ]
        }

        self.rpm_bins = np.linspace(c.IDLE_RPM, c.RPM_LIMIT, 120)
        self.best_torque = np.full(len(self.rpm_bins) - 1, -np.inf)
        self.best_power = np.full(len(self.rpm_bins) - 1, -np.inf)

        plt.show(block=False)
        self.fig.canvas.flush_events()

        # Focus
        self._set_focus("plot")

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
        if self.system:
            self.system.stop_simulation = True

        # The window close action (either 'q' or 'X' button)
        # must be performed before returning focus.
        plt.close(self.fig)

        self._set_focus("terminal")

    def update(self, sensors, engine, ecu, reward=None, new_best=False):

        idx = len(self.history_x)
        self.history_x.append(idx)

        data = {
            "rpm": sensors.get("RPM", 0),
            "torque": engine["brake_torque_nm"],
            "power": engine["brake_power_kw"],
            "tps": sensors.get("TPS_percent", 0),
            "idle": ecu.get("idle_valve_position", 0),
            "spark": ecu.get("spark_advance_btdc", 10),
            "lambda": sensors.get("lambda_actual", 1.0),
            "afr": sensors.get("actual_AFR", 14.7),
            "clt": sensors.get("CLT_C", 20),
            "knock": sensors.get("Knock", 0),
            "ppeak": engine.get("peak_pressure_bar", 0),
            "reward": reward if reward is not None else 0,
        }

        for k, v in data.items():
            self.history[k].append(v)
            if len(self.history[k]) > 2500:
                self.history[k].pop(0)
        if len(self.history_x) > 2500:
            self.history_x.pop(0)

        x = np.array(self.history_x)

        # Update big dyno
        self.line_trq.set_data(self.history["rpm"], self.history["torque"])
        self.line_pwr.set_data(self.history["rpm"], self.history["power"])

        # Update mini charts
        self.line_tps.set_data(x, self.history["tps"])
        self.line_idle.set_data(x, self.history["idle"])
        self.line_spark.set_data(x, self.history["spark"])
        self.line_lam.set_data(x, self.history["lambda"])
        self.line_knock.set_data(x, self.history["knock"])
        self.line_ppeak.set_data(x, self.history["ppeak"])
        self.line_clt.set_data(x, self.history["clt"])

        # Session best
        if new_best or idx % 50 == 0:
            # if self.best_fill_trq:
            #     self.best_fill_trq.remove()
            # if self.best_fill_pwr:
            #     self.best_fill_pwr.remove()
            bins = np.digitize(self.history["rpm"], self.rpm_bins)
            curr_trq = np.full(len(self.rpm_bins) - 1, -np.inf)
            curr_pwr = np.full(len(self.rpm_bins) - 1, -np.inf)
            for i in range(1, len(self.rpm_bins)):
                m = bins == i
                if m.any():
                    curr_trq[i - 1] = max(
                        curr_trq[i - 1], max(np.array(self.history["torque"])[m])
                    )
                    curr_pwr[i - 1] = max(
                        curr_pwr[i - 1], max(np.array(self.history["power"])[m])
                    )
            self.best_torque = np.maximum(self.best_torque, curr_trq)
            self.best_power = np.maximum(self.best_power, curr_pwr)
            # self.best_fill_trq = self.ax_dyno.fill_between(
            #     self.rpm_bins[:-1], self.best_torque, alpha=0.3, color="red"
            # )
            # self.best_fill_pwr = self.ax_pwr.fill_between(
            #     self.rpm_bins[:-1], self.best_power, alpha=0.3, color="red"
            # )

        # Limits and autoscale
        for ax in [
            self.ax_dyno,
            self.ax_pwr,
            self.ax_throttle,
            self.ax_spark,
            self.ax_lambda,
            self.ax_safety,
            self.ax_temp,
        ]:
            ax.relim()
            ax.autoscale_view()
        self.ax_dyno.set_xlim(c.IDLE_RPM, c.RPM_LIMIT)
        self.ax_dyno.set_ylim(-50, 260)
        self.ax_pwr.set_ylim(0, 200)
        self.ax_throttle.set_ylim(0, 105)
        self.ax_spark.set_ylim(0, 45)
        self.ax_lambda.set_ylim(0.7, 1.3)
        self.ax_lambda.axhline(1.0, color="lime", ls="--", alpha=0.6)
        self.ax_safety.set_ylim(
            0,
            max(
                10, max(self.history["knock"]) * 1.5 + max(self.history["ppeak"]) * 0.1
            ),
        )
        self.ax_temp.set_ylim(20, 120)

        # --- MODIFICATION 2: Digital gauges - ADJUSTED TEXT PLACEMENT ---
        self.ax_gauge.clear()
        self.ax_gauge.axis("off")
        rpm_val = data["rpm"]
        rpm_col = "lime" if rpm_val < 9000 else "red"

        center_x = 0.5  # Use a centered X-coordinate for all elements

        # RPM
        self.ax_gauge.text(
            center_x,
            0.88,
            f"{rpm_val:,.0f}",
            fontsize=48,
            color=rpm_col,
            fontweight="bold",
            ha="center",
        )
        self.ax_gauge.text(
            center_x, 0.78, "RPM", fontsize=16, color=rpm_col, alpha=0.8, ha="center"
        )

        # Lambda
        self.ax_gauge.text(
            center_x,
            0.60,
            f"λ {data['lambda']:.3f}",
            fontsize=36,
            color="white",
            ha="center",
        )

        # AFR
        self.ax_gauge.text(
            center_x,
            0.48,
            f"AFR {data['afr']:.1f}",
            fontsize=26,
            color="springgreen" if abs(data["lambda"] - 1.0) < 0.05 else "orange",
            ha="center",
        )

        # Spark
        self.ax_gauge.text(
            center_x,
            0.30,
            f"{data['spark']:.1f}°",
            fontsize=40,
            color="magenta",
            ha="center",
        )
        self.ax_gauge.text(
            center_x, 0.22, "Spark", fontsize=16, color="magenta", ha="center"
        )

        # Knock
        knock_col = "red" if data["knock"] > 1 else "lime"
        self.ax_gauge.text(
            center_x,
            0.10,
            f"Knock {data['knock']:.1f}",
            fontsize=28,
            color=knock_col,
            ha="center",
        )

        # Reward
        if reward is not None:
            col = "cyan" if reward >= 0 else "orange"
            self.ax_gauge.text(
                0.5,
                0.00,
                f"REWARD {reward:+.2f}",
                fontsize=24,
                color=col,
                fontweight="bold",
                ha="center",
            )
        # --- END MODIFICATION 2 ---

        # Draw
        plt.draw()
        self.fig.canvas.flush_events()

    def close(self):
        # This function is now redundant but kept for completeness
        plt.ioff()
        plt.close(self.fig)
