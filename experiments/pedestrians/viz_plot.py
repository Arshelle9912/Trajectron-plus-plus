import matplotlib.pyplot as plt
import numpy as np
import os
def plot_trajectories(past, future_gt, samples, title="Scene"):
    past = np.asarray(past)
    future_gt = np.asarray(future_gt)
    samples = np.asarray(samples)

    plt.figure(figsize=(10, 4))

    # past
    plt.plot(past[:, 0], past[:, 1], 'k--', linewidth=2)
    plt.scatter(past[-1, 0], past[-1, 1], s=80, c='green', zorder=5)

    # GT future
    fut_x = np.concatenate((past[-1:, 0], future_gt[:, 0]))
    fut_y = np.concatenate((past[-1:, 1], future_gt[:, 1]))
    plt.plot(fut_x, fut_y, 'k--', linewidth=2)

    # samples
    for k in range(samples.shape[0]):
        s = samples[k]
        if s.ndim == 3:
            s = s[..., 0]   # (T_pred, 2, 1) -> (T_pred, 2)
        sx = np.concatenate((past[-1:, 0], s[:, 0]))
        sy = np.concatenate((past[-1:, 1], s[:, 1]))
        color = 'tab:orange' if k % 2 == 0 else 'tab:blue'
        plt.plot(sx, sy, color=color, linewidth=1, alpha=0.4)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()

    # 🔹 Save instead of show (no blocking, always creates a file)
    os.makedirs("results/viz", exist_ok=True)
    safe_title = title.replace(" ", "_").replace(":", "_")
    plt.savefig(f"results/viz/{safe_title}.png")
    plt.close()
