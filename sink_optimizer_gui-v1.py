import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


# ============================================================
#  SVR(RBF) optimized equation (from your exported text)
#  If you later export a new equation, replace these constants.
# ============================================================

GAMMA = 0.001
B_INTERCEPT = 0.0259672843577

MEAN = {"A": 2.0, "B": 10.0, "C": 75.0, "D": 15.0}
SCALE = {"A": 0.816496580928, "B": 1.63299316186, "C": 20.4124145232, "D": 2.44948974278}

SV = np.array([
    [-1.22474487139, -1.22474487139, -1.22474487139, -1.22474487139],
    [-1.22474487139,  0.0,            0.0,            0.0           ],
    [-1.22474487139,  1.22474487139,  1.22474487139,  1.22474487139],
    [ 0.0,           -1.22474487139,  0.0,            1.22474487139],
    [ 0.0,            0.0,            1.22474487139, -1.22474487139],
    [ 0.0,            1.22474487139, -1.22474487139,  0.0           ],
    [ 1.22474487139, -1.22474487139,  1.22474487139,  0.0           ],
    [ 1.22474487139,  1.22474487139,  0.0,           -1.22474487139],
], dtype=float)

W = np.array([
     1.85789373672,
   -33.9190427369,
     0.0665714229322,
    22.2173877335,
    32.4691159539,
     9.63739585163,
   -20.979431688,
   -11.3498902739,
], dtype=float)

FEATURES = ["A", "B", "C", "D"]

# Bounds (from your DOE min/max)
BOUNDS = {
    "A": (1.0, 3.0),     # fill time [sec]
    "B": (8.0, 12.0),    # pack time [sec]
    "C": (50.0, 100.0),  # pack pressure [%]
    "D": (12.0, 18.0),   # cool time [sec]
}

# DOE discrete levels (L9)
LEVELS = {
    "A": [1.0, 2.0, 3.0],
    "B": [8.0, 10.0, 12.0],
    "C": [50.0, 75.0, 100.0],
    "D": [12.0, 15.0, 18.0],
}


def standardize(x_dict):
    z = np.array([(x_dict[k] - MEAN[k]) / SCALE[k] for k in FEATURES], dtype=float)
    return z


def svr_predict_sink(x_dict):
    """Predict sink mark index using SVR(RBF) equation."""
    z = standardize(x_dict)                 # (4,)
    diff = SV - z                           # (n_sv, 4)
    r2 = np.sum(diff * diff, axis=1)        # (n_sv,)
    k = np.exp(-GAMMA * r2)                 # (n_sv,)
    y = float(np.dot(W, k) + B_INTERCEPT)
    return y


def optimize_sink(fixed, mode="continuous", grid_per_dim=81):
    """
    fixed: dict like {"B": 12.0, "C": 100.0}
    mode: "continuous" or "discrete"
    grid_per_dim: used only in continuous mode
    Return: (best_x_dict, best_pred, best_abs)
    """
    fixed = {k.upper(): float(v) for k, v in (fixed or {}).items()}

    # validate fixed bounds
    for k, v in fixed.items():
        if k not in FEATURES:
            raise ValueError(f"Unknown key: {k} (must be A/B/C/D)")
        lo, hi = BOUNDS[k]
        if not (lo <= v <= hi):
            raise ValueError(f"Fixed {k}={v} out of bounds [{lo},{hi}]")

    free = [k for k in FEATURES if k not in fixed]

    # candidate lists
    cand = {}
    if mode == "discrete":
        for k in free:
            cand[k] = LEVELS[k]
    elif mode == "continuous":
        for k in free:
            lo, hi = BOUNDS[k]
            cand[k] = np.linspace(lo, hi, int(grid_per_dim)).tolist()
    else:
        raise ValueError("mode must be 'continuous' or 'discrete'")

    # cartesian product
    lists = [cand[k] for k in free]
    if not lists:
        combos = [()]
    else:
        combos = np.array(np.meshgrid(*lists, indexing='ij')).reshape(len(lists), -1).T

    best_abs = float("inf")
    best_pred = None
    best_x = None

    base = dict(fixed)

    if len(free) == 0:
        x = dict(base)
        pred = svr_predict_sink(x)
        return x, pred, abs(pred)

    # iterate combos
    for row in combos:
        x = dict(base)
        for k, v in zip(free, row):
            x[k] = float(v)
        pred = svr_predict_sink(x)
        a = abs(pred)
        if a < best_abs:
            best_abs, best_pred, best_x = a, pred, x

    return best_x, best_pred, best_abs


# =========================
# GUI (Tkinter)
# =========================

class SinkOptimizerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SVR Sink Mark Optimizer (4 Sliders + Lock)")
        self.geometry("860x520")
        self.resizable(False, False)

        self._best_cache = None  # (x, pred, abs)

        # --- top controls ---
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Search Mode:").pack(side="left", padx=(0, 6))
        self.mode_var = tk.StringVar(value="continuous")
        self.mode_combo = ttk.Combobox(
            top, textvariable=self.mode_var,
            values=["continuous", "discrete"], state="readonly", width=12
        )
        self.mode_combo.pack(side="left", padx=(0, 20))

        ttk.Label(top, text="Grid (continuous only):").pack(side="left", padx=(0, 6))
        self.grid_var = tk.IntVar(value=81)
        self.grid_combo = ttk.Combobox(
            top, textvariable=self.grid_var,
            values=[41, 61, 81, 101, 121, 161, 201], state="readonly", width=6
        )
        self.grid_combo.pack(side="left", padx=(0, 20))

        self.btn_opt = ttk.Button(top, text="Optimize (sink → 0)", command=self.on_optimize)
        self.btn_opt.pack(side="left", padx=(0, 10))

        self.btn_apply = ttk.Button(top, text="Apply Best (to unlocked)", command=self.on_apply_best)
        self.btn_apply.pack(side="left")

        # --- sliders area ---
        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        self.vars = {}
        self.locks = {}
        self.readouts = {}

        # helper to build each row
        def add_row(r, key, label, from_, to_, resolution, init):
            ttk.Label(body, text=label, width=16).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=10)

            v = tk.DoubleVar(value=init)
            self.vars[key] = v

            s = ttk.Scale(body, from_=from_, to=to_, orient="horizontal", variable=v)
            s.grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=10)

            # small step buttons for finer control (ttk.Scale doesn't support step well)
            btn_frame = ttk.Frame(body)
            btn_frame.grid(row=r, column=2, sticky="w", pady=10)
            ttk.Button(btn_frame, text="−", width=3,
                       command=lambda k=key, step=resolution: self.nudge(k, -step)).pack(side="left")
            ttk.Button(btn_frame, text="+", width=3,
                       command=lambda k=key, step=resolution: self.nudge(k, +step)).pack(side="left", padx=(6, 0))

            ro = ttk.Label(body, text=f"{init:.3f}", width=10)
            ro.grid(row=r, column=3, sticky="w", padx=(0, 10), pady=10)
            self.readouts[key] = ro

            lock = tk.BooleanVar(value=False)
            self.locks[key] = lock
            ttk.Checkbutton(body, text=f"Lock {key}", variable=lock, command=self.update_status).grid(
                row=r, column=4, sticky="w", pady=10
            )

            # trace update
            def _on_change(*_):
                self.readouts[key].configure(text=f"{self.vars[key].get():.3f}")
                self.update_status()

            v.trace_add("write", _on_change)

        body.columnconfigure(1, weight=1)

        add_row(0, "A", "A Fill time (s)", *BOUNDS["A"], 0.01, 2.0)
        add_row(1, "B", "B Pack time (s)", *BOUNDS["B"], 0.01, 10.0)
        add_row(2, "C", "C Pack press (%)", *BOUNDS["C"], 0.1, 75.0)
        add_row(3, "D", "D Cool time (s)", *BOUNDS["D"], 0.01, 15.0)

        # --- status panel ---
        status = ttk.Labelframe(self, text="Status", padding=10)
        status.pack(fill="x", padx=10, pady=(0, 10))

        self.lbl_now = ttk.Label(status, text="", justify="left")
        self.lbl_now.pack(anchor="w")

        self.lbl_best = ttk.Label(status, text="", justify="left")
        self.lbl_best.pack(anchor="w", pady=(8, 0))

        self.progress = ttk.Progressbar(status, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.stop()
        self.progress.pack_forget()

        self.update_status()

    def nudge(self, key, delta):
        v = float(self.vars[key].get()) + float(delta)
        lo, hi = BOUNDS[key]
        v = max(lo, min(hi, v))
        self.vars[key].set(v)

    def current_x(self):
        return {k: float(self.vars[k].get()) for k in FEATURES}

    def current_fixed(self):
        fixed = {}
        for k in FEATURES:
            if bool(self.locks[k].get()):
                fixed[k] = float(self.vars[k].get())
        return fixed

    def update_status(self):
        x = self.current_x()
        pred = svr_predict_sink(x)
        self.lbl_now.configure(
            text=(
                f"Current:  A={x['A']:.3f}, B={x['B']:.3f}, C={x['C']:.3f}, D={x['D']:.3f}\n"
                f"Pred sink = {pred:.10f}   |pred|={abs(pred):.10f}"
            )
        )

        fixed = self.current_fixed()
        if fixed:
            lock_txt = ", ".join([f"{k}={v:g}" for k, v in fixed.items()])
        else:
            lock_txt = "(none locked)"

        hint = ""
        if (len(fixed) == 0) and (self.mode_var.get() == "continuous") and (int(self.grid_var.get()) > 61):
            hint = "  ⚠️ 0 locked + continuous + large grid may be slow; consider 'discrete' or smaller grid."

        self.lbl_best.configure(text=f"Locked: {lock_txt}{hint}")

    def set_busy(self, busy: bool):
        if busy:
            self.btn_opt.configure(state="disabled")
            self.btn_apply.configure(state="disabled")
            self.progress.pack(fill="x", pady=(10, 0))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.btn_opt.configure(state="normal")
            self.btn_apply.configure(state="normal")

    def on_optimize(self):
        # run optimization in a worker thread to avoid freezing UI
        def worker():
            try:
                fixed = self.current_fixed()
                mode = self.mode_var.get()
                grid = int(self.grid_var.get())

                # if none locked and continuous huge grid -> warn but allow
                if (len(fixed) == 0) and (mode == "continuous") and (grid > 101):
                    # safety: reduce automatically to avoid accidental long run
                    grid = 81

                best_x, best_pred, best_abs = optimize_sink(
                    fixed=fixed, mode=mode, grid_per_dim=grid
                )
                self._best_cache = (best_x, best_pred, best_abs)

                def ui_update():
                    self.set_busy(False)
                    bx, bp, ba = self._best_cache
                    self.lbl_best.configure(
                        text=(
                            self.lbl_best.cget("text") + "\n"
                            f"Best:     A={bx['A']:.6f}, B={bx['B']:.6f}, C={bx['C']:.6f}, D={bx['D']:.6f}\n"
                            f"Pred sink = {bp:.12f}   |pred|={ba:.12f}"
                        )
                    )

                self.after(0, ui_update)

            except Exception as e:
                def ui_err():
                    self.set_busy(False)
                    messagebox.showerror("Optimize Error", str(e))
                self.after(0, ui_err)

        self.set_busy(True)
        threading.Thread(target=worker, daemon=True).start()

    def on_apply_best(self):
        if not self._best_cache:
            messagebox.showinfo("Info", "Please click Optimize first.")
            return

        bx, bp, ba = self._best_cache

        # apply only to unlocked sliders
        for k in FEATURES:
            if not bool(self.locks[k].get()):
                self.vars[k].set(float(bx[k]))

        self.update_status()


if __name__ == "__main__":
    app = SinkOptimizerGUI()
    app.mainloop()