import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import os
import sys


def _is_correct(y, out):
    y = np.asarray(y).ravel()
    out = np.asarray(out).ravel()

    if len(out) > 1:
        # multi class classification
        pred = np.argmax(out)
        true = np.argmax(y) if len(y) > 1 else y[0]
    else:
        # one output neuron: 0.5 or above means class 1
        pred = 1 if out[0] >= 0.5 else 0
        true = y[0]

    return pred == true


_LIME, _RED, _GREEN = "#B6F500", "#FF4D5E", "#2BFF88"
_BG = "#0A0A0A"

_STYLE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": _BG,
    "savefig.facecolor": _BG,
    "axes.facecolor": _BG,
    "axes.edgecolor": "#2E2E2E",
    "axes.labelcolor": "#9CA3AF",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#1C1C1C",
    "grid.linewidth": 1,
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "savefig.bbox": "tight",
}


def _glow(color, lw=2.4):
    # soft halo behind the line
    return [pe.Stroke(linewidth=lw + 5, foreground=color, alpha=0.16), pe.Normal()]


def plot_history(history, log_loss=False, save_path=None):
    epochs = np.arange(1, len(history["loss"]) + 1)
    loss, acc = np.asarray(history["loss"]), np.asarray(history["accuracy"])

    with plt.rc_context(_STYLE_RC):
        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, constrained_layout=True)  # 16:9

        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "training"
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(script_name)

        # ---- loss (left axis)
        ax.plot(epochs, loss, color=_RED, lw=2.4, solid_capstyle="round",
                path_effects=_glow(_RED))
        ax.set_ylabel("Loss", color=_RED, labelpad=10)
        ax.tick_params(axis="y", colors=_RED)
        if log_loss:
            ax.set_yscale("log")
        else:
            # align loss 0 with accuracy 0 so both baselines match
            f = 0.03 / 1.08
            top = loss.max() * 1.08
            ax.set_ylim(-f * top / (1 - f), top)

        # ---- accuracy (right axis, same plane)
        ax2 = ax.twinx()
        ax2.plot(epochs, acc, color=_GREEN, lw=2.4, drawstyle="steps-post",
                 solid_capstyle="round", path_effects=_glow(_GREEN))
        ax2.set_ylim(-0.03, 1.05)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.set_ylabel("Accuracy", color=_GREEN, labelpad=10)
        ax2.tick_params(axis="y", colors=_GREEN)
        ax2.grid(False)

        # ---- clean frame
        for a in (ax, ax2):
            for s in ("top", "left", "right"):
                a.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#2E2E2E")
        ax2.spines["bottom"].set_visible(False)

        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_xlim(0, epochs[-1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

        # ---- direct end labels instead of a legend
        ax.annotate(f"Loss  {loss[-1]:.2g}", (epochs[-1], loss[-1]), xytext=(-6, 14),
                    textcoords="offset points", ha="right", color=_RED,
                    fontsize=11, fontweight="bold")
        ax2.annotate(f"Accuracy  {acc[-1]:.0%}", (epochs[-1], acc[-1]), xytext=(-6, -20),
                     textcoords="offset points", ha="right", color=_GREEN,
                     fontsize=11, fontweight="bold")

        # ---- title + subtitle
        ax.set_title("Training History", loc="left", pad=34, fontsize=20,
                     fontweight="bold", color=_LIME)
        ax.text(0, 1.03, f"{len(epochs):,} epochs", transform=ax.transAxes,
                color="#9CA3AF", fontsize=11, va="bottom")

        if save_path:
            fig.savefig(save_path, dpi=300)
        plt.show()