import torch
from diffusers import DDIMScheduler
import matplotlib.pyplot as plt
import numpy as np


def get_signal_curve(schedule_name, T, beta_start=0.0001, beta_end=0.02):

    scheduler = DDIMScheduler(
        num_train_timesteps=T,
        beta_schedule=schedule_name,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )

    alphas = scheduler.alphas_cumprod.numpy()
    # 核心公式: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
    signal_strength = np.sqrt(alphas)

    return signal_strength


configs = [
    ("scaled_linear", 100, 0.00085, 0.012, "Scaled Linear (T=100)", "red", "-"),
    ("squaredcos_cap_v2", 100, 0.0001, 0.02, "Squared Cosine (T=100)", "green", "-"),
    ("linear", 100, 0.0001, 0.02, "Linear (T=100)", "orange", "-"),
    ("scaled_linear", 1000, 0.00085, 0.012, "Standard SD: Scaled Linear (T=1000)", "blue", ":"),
]
plt.figure(figsize=(10, 6), dpi=120)

for name, T, b_start, b_end, label, color, style in configs:
    curve = get_signal_curve(name, T, beta_start=b_start, beta_end=b_end)

    x_axis = np.linspace(0, 1, len(curve))

    plt.plot(x_axis, curve, label=label, color=color, linestyle=style, linewidth=2.5)

    end_val = curve[-1]
    plt.scatter([1.0], [end_val], color=color, s=50, zorder=5)
    plt.text(1.02, end_val, f"{end_val:.2f}", color=color, va='center', fontweight='bold')

plt.title(r"Signal Retention Strength ($\sqrt{\bar{\alpha}_t}$) over Diffusion Steps", fontsize=14)
plt.xlabel("Diffusion Progress (t / T)", fontsize=12)
plt.ylabel(r"Signal Strength (Coefficient of $x_0$)", fontsize=12)

plt.axhline(y=0, color='black', linewidth=1)
plt.axhspan(0, 0.05, color='green', alpha=0.1, label="Target Zone (Pure Noise)")
plt.axhspan(0.05, 1.0, color='red', alpha=0.05, label="Signal Leakage Zone")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.xlim(0, 1.1)  # 留出右侧空间给文字
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.show()