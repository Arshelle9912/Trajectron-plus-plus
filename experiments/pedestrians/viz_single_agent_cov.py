import os
import sys
import dill
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.append("../../trajectron")

from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories


# Model + env helpers
def load_model(model_dir, env, checkpoint, device="cpu"):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(checkpoint)

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        hyperparams = json.load(f)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


def build_scene_graphs(env, hyperparams):
    print("-- Preparing Node Graphs")
    for scene in tqdm(env.scenes):
        scene.calculate_scene_graph(
            env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )


def ensure_node_first(pred_dict, hist_dict, fut_dict):
    if not pred_dict:
        return pred_dict, hist_dict, fut_dict

    first_key = next(iter(pred_dict))
    if isinstance(first_key, (int, np.integer)):
        new_pred, new_hist, new_fut = {}, {}, {}
        for t_key, node_map in pred_dict.items():
            for node, arr in node_map.items():
                new_pred.setdefault(node, {})[t_key] = arr
        for t_key, node_map in hist_dict.items():
            for node, arr in node_map.items():
                new_hist.setdefault(node, {})[t_key] = arr
        for t_key, node_map in fut_dict.items():
            for node, arr in node_map.items():
                new_fut.setdefault(node, {})[t_key] = arr
        return new_pred, new_hist, new_fut

    return pred_dict, hist_dict, fut_dict


def pick_node_and_t(pred_dict_node_first, desired_t=None):
    nodes = list(pred_dict_node_first.keys())
    if not nodes:
        return None, None

    if desired_t is not None:
        for node in nodes:
            if desired_t in pred_dict_node_first[node]:
                return node, desired_t

    node = nodes[0]
    t_key = sorted(pred_dict_node_first[node].keys())[0]
    return node, t_key


# Shape normalization
def normalize_samples(samples):
    samples = np.asarray(samples)

    if samples.ndim == 4 and samples.shape[0] == 1 and samples.shape[-1] == 2:
        samples = samples[0]
    if samples.ndim == 4 and samples.shape[-1] == 1:
        samples = samples[..., 0]
    if samples.ndim == 2:
        samples = samples[None, :, :]
    if samples.ndim == 4:
        samples = np.squeeze(samples)
        if samples.ndim == 2:
            samples = samples[None, :, :]

    return samples


# Covariance math (velocity -> position)
def estimate_vel_samples_from_pos(samples_pos, p0, dt):
    samples_pos = np.asarray(samples_pos)
    p0 = np.asarray(p0)

    K, ph, _ = samples_pos.shape
    vel = np.zeros_like(samples_pos)

    for i in range(ph):
        prev = p0 if i == 0 else samples_pos[:, i - 1, :]
        vel[:, i, :] = (samples_pos[:, i, :] - prev) / max(1e-12, dt)

    return vel


def propagate_pos_cov_from_vel(vel_samples, dt):
    """
    Single-integrator uncertainty propagation:
        Σ_p^{t+1} = Σ_p^t + (dt)^2 Σ_u^t with Σ_p^0 = 0.
    This is the Trajectron++ pedestrian assumption.
    """
    vel_samples = np.asarray(vel_samples)
    K, ph, _ = vel_samples.shape

    cov_p_list = []
    Sigma_p = np.zeros((2, 2), dtype=float)

    for i in range(ph):
        v_pts = vel_samples[:, i, :]
        Sigma_u = np.cov(v_pts, rowvar=False)
        if Sigma_u.shape != (2, 2) or not np.all(np.isfinite(Sigma_u)):
            Sigma_u = np.zeros((2, 2), dtype=float)

        Sigma_p = Sigma_p + (dt ** 2) * Sigma_u
        cov_p_list.append(Sigma_p.copy())

    return cov_p_list


# 2D ellipse helper
def add_cov_ellipse(ax, mu, cov, conf=0.7, fill=False, facecolor=None, **kwargs):
    mu = np.asarray(mu)
    cov = np.asarray(cov)

    if mu.shape != (2,) or cov.shape != (2, 2):
        return
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    chi2 = -2.0 * np.log(max(1e-12, 1.0 - conf))
    width = 2.0 * np.sqrt(chi2 * vals[0])
    height = 2.0 * np.sqrt(chi2 * vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    e = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        fill=fill,
        facecolor=facecolor if fill else None,
        **kwargs
    )
    ax.add_patch(e)


# 2D plot: draw ONE node onto an existing axis
def draw_node_cov_chain_2d(
    ax,
    past,
    future_gt,
    samples,
    conf=0.7,
    draw_samples=True,
    time_colored=True,
    multi_rings=False,
    draw_mean_points=True,
    cov_source="vel",
    dt=0.4,
    fill_ellipses=True,
    label_once=None,
):
    """
    dict of booleans controlling whether to add labels
      keys: history, current, pred, conf, gt
    """
    if label_once is None:
        label_once = {"history": True, "current": True, "pred": True, "conf": True, "gt": True}

    past = np.asarray(past)
    future_gt = np.asarray(future_gt)
    samples = normalize_samples(samples)
    K, ph, _ = samples.shape

    # History
    ax.plot(
        past[:, 0], past[:, 1],
        "k--", linewidth=2,
        label="History" if label_once["history"] else None
    )

    # Current
    ax.scatter(
        past[-1, 0], past[-1, 1],
        s=80, c="green", zorder=5,
        label="Current" if label_once["current"] else None
    )

    # Ground truth
    if future_gt.size != 0:
        fut_x = np.concatenate((past[-1:, 0], future_gt[:, 0]))
        fut_y = np.concatenate((past[-1:, 1], future_gt[:, 1]))
        ax.plot(
            fut_x, fut_y,
            "k--", linewidth=2, alpha=0.7,
            label="Ground Truth" if label_once["gt"] else None
        )

    # Sampled futures
    if draw_samples:
        labeled = not label_once["pred"]
        for k in range(K):
            s = samples[k]
            sx = np.concatenate((past[-1:, 0], s[:, 0]))
            sy = np.concatenate((past[-1:, 1], s[:, 1]))
            if not labeled:
                ax.plot(sx, sy, linewidth=1, alpha=0.25, color="tab:blue", label="Prediction")
                labeled = True
            else:
                ax.plot(sx, sy, linewidth=1, alpha=0.25, color="tab:blue")

    # Confidence rings
    conf_levels = [conf] if not multi_rings else [0.5, 0.7, 0.9]

    # Precompute propagated covariances if needed
    cov_p_list = None
    if cov_source == "vel":
        p0 = past[-1]
        vel_samples = estimate_vel_samples_from_pos(samples, p0, dt)
        cov_p_list = propagate_pos_cov_from_vel(vel_samples, dt)

    # Ellipse chain
    for i in range(ph):
        pts = samples[:, i, :]
        mu = pts.mean(axis=0)
        cov = cov_p_list[i] if cov_source == "vel" else np.cov(pts, rowvar=False)

        col = plt.cm.viridis(i / max(1, ph - 1)) if time_colored else "tab:blue"

        for idx, c in enumerate(conf_levels):
            edge_a = 0.9 if idx == 0 else 0.5
            face_a = 0.18 if idx == 0 else 0.08

            add_cov_ellipse(
                ax, mu, cov, conf=c,
                fill=fill_ellipses,
                facecolor=col if fill_ellipses else None,
                edgecolor=col,
                linewidth=1.5,
                alpha=face_a if fill_ellipses else edge_a
            )

        if draw_mean_points:
            ax.scatter(mu[0], mu[1], s=10, color=col, alpha=0.7)

    # Add a single legend hint for conf
    if label_once["conf"]:
        dummy = Ellipse((0, 0), 0, 0, fill=False, edgecolor="tab:blue", linewidth=1.5)
        ax.add_patch(dummy)
        dummy.set_label(f"{int(conf * 100)}% conf")


# 2D plot: MULTI-NODE overlay
def plot_timestep_all_nodes_2d(
    pred_dict,
    hist_dict,
    fut_dict,
    t_key,
    title,
    conf=0.7,
    draw_samples=True,
    time_colored=True,
    multi_rings=False,
    draw_mean_points=True,
    cov_source="vel",
    dt=0.4,
    fill_ellipses=True,
):
    fig, ax = plt.subplots(figsize=(6.8, 7.2))

    nodes = [n for n in pred_dict.keys() if t_key in pred_dict[n]]
    if not nodes:
        print("No nodes had predictions at this timestep.")
        return

    label_once = {"history": True, "current": True, "pred": True, "conf": True, "gt": True}

    for idx, node in enumerate(nodes):
        samples = normalize_samples(pred_dict[node][t_key])
        past = hist_dict[node][t_key]
        future_gt = fut_dict[node][t_key]

        draw_node_cov_chain_2d(
            ax,
            past,
            future_gt,
            samples,
            conf=conf,
            draw_samples=draw_samples,
            time_colored=time_colored,
            multi_rings=multi_rings,
            draw_mean_points=draw_mean_points,
            cov_source=cov_source,
            dt=dt,
            fill_ellipses=fill_ellipses,
            label_once=label_once,
        )
        label_once = {"history": False, "current": False, "pred": False, "conf": False, "gt": False}

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.axis("equal")

    # Deduplicate legend robustly
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")

    plt.tight_layout()
    os.makedirs("results/viz", exist_ok=True)
    fname = title.replace(" ", "_").replace(":", "_") + ".png"
    out_path = os.path.join("results/viz", fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


# 3D helpers + plot 
def ellipse_outline_points(mu, cov, conf=0.7, num=80):
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    if mu.shape != (2,) or cov.shape != (2, 2):
        return None
    if not np.all(np.isfinite(cov)):
        return None

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    chi2 = -2.0 * np.log(max(1e-12, 1.0 - conf))
    scale = np.sqrt(chi2)

    theta = np.linspace(0, 2 * np.pi, num)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    L = vecs @ np.diag(np.sqrt(vals))
    pts = mu + scale * (circle @ L.T)
    return pts


def plot_cov_chain_3d(
    past,
    future_gt,
    samples,
    title="3D covariance chain",
    conf=0.7,
    dt=0.4,
    draw_samples=False,
    time_colored=True,
    max_ellipses=None,
    draw_mean_points=True,
):
    past = np.asarray(past)
    future_gt = np.asarray(future_gt)
    samples = normalize_samples(samples)
    K, ph, _ = samples.shape

    if max_ellipses is not None:
        ph = min(ph, max_ellipses)
        samples = samples[:, :ph, :]
        if future_gt.size != 0:
            future_gt = future_gt[:ph, :]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(past[:, 0], past[:, 1], np.zeros(len(past)),
            linestyle="--", linewidth=2, label="History")
    ax.scatter(past[-1, 0], past[-1, 1], 0, s=80, c="green", label="Current")

    if future_gt.size != 0:
        xs = np.concatenate([past[-1:, 0], future_gt[:, 0]])
        ys = np.concatenate([past[-1:, 1], future_gt[:, 1]])
        zs = np.concatenate([[0.0], (np.arange(1, len(xs)) * dt)])
        ax.plot(xs, ys, zs, linestyle="--", linewidth=2, alpha=0.8, label="Ground Truth")

    if draw_samples:
        labeled = False
        for k in range(K):
            s = samples[k]
            xs = np.concatenate([past[-1:, 0], s[:, 0]])
            ys = np.concatenate([past[-1:, 1], s[:, 1]])
            zs = np.concatenate([[0.0], (np.arange(1, len(xs)) * dt)])
            if not labeled:
                ax.plot(xs, ys, zs, alpha=0.15, label="Prediction samples")
                labeled = True
            else:
                ax.plot(xs, ys, zs, alpha=0.15)

    for i in range(ph):
        pts = samples[:, i, :]
        mu = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)

        col = plt.cm.viridis(i / max(1, ph - 1)) if time_colored else "tab:blue"
        ellipse_xy = ellipse_outline_points(mu, cov, conf=conf, num=100)
        if ellipse_xy is None:
            continue

        z = (i + 1) * dt
        ax.plot(ellipse_xy[:, 0], ellipse_xy[:, 1], np.full(len(ellipse_xy), z),
                color=col, linewidth=1.5, alpha=0.95)

        if draw_mean_points:
            ax.scatter(mu[0], mu[1], z, s=10, color=col, alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    os.makedirs("results/viz", exist_ok=True)
    fname = title.replace(" ", "_").replace(":", "_") + ".png"
    out_path = os.path.join("results/viz", fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=571)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.7)

    parser.add_argument("--mode", type=str, default="2d", choices=["2d", "3d"])

    parser.add_argument("--cov_source", type=str, default="vel",
                        choices=["position", "vel"])

    parser.add_argument("--no_sample_lines", action="store_true")
    parser.add_argument("--time_colored", action="store_true")
    parser.add_argument("--multi_rings", action="store_true")
    parser.add_argument("--no_mean_points", action="store_true")

    parser.add_argument("--fill_ellipses", action="store_true",
                        help="(2D) Fill ellipses (paper-style)")
    parser.add_argument("--all_nodes", action="store_true",
                        help="(2D) Overlay all nodes at this timestep")

    parser.add_argument("--max_ellipses", type=int, default=None)
    parser.add_argument("--draw_3d_samples", action="store_true")

    parser.add_argument("--title_suffix", type=str, default="",
                        help="Extra text appended to the title, e.g. '(DDPM, 100 steps, 20 samples, Uncalibrated)'")

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        env = dill.load(f, encoding="latin1")

    eval_stg, hyperparams = load_model(args.model, env, args.checkpoint, args.device)

    if "override_attention_radius" in hyperparams:
        for override in hyperparams["override_attention_radius"]:
            n1, n2, ar = override.split(" ")
            env.attention_radius[(n1, n2)] = float(ar)

    build_scene_graphs(env, hyperparams)

    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]

    scene = env.scenes[args.scene_index]
    t = args.timestep
    timesteps = np.array([t], dtype=int)

    print(f"-- Predicting scene {args.scene_index}, timestep {t} with {args.num_samples} samples")

    with torch.no_grad():
        predictions = eval_stg.predict(
            scene,
            timesteps,
            ph,
            num_samples=args.num_samples,
            min_history_timesteps=7,
            min_future_timesteps=12,
            z_mode=False,
            gmm_mode=False,
            full_dist=False,
        )

    if not predictions:
        print("No predictions returned.")
        return

    pred_dict, hist_dict, fut_dict = prediction_output_to_trajectories(
        predictions,
        scene.dt,
        max_hl,
        ph,
        map=None,
        prune_ph_to_future=True,
    )

    pred_dict, hist_dict, fut_dict = ensure_node_first(pred_dict, hist_dict, fut_dict)

    # 2D multi-node overlay 
    if args.mode == "2d" and args.all_nodes:
        title = f"Scene_{args.scene_index}_Timestep_{t} {args.title_suffix}".strip()
        plot_timestep_all_nodes_2d(
            pred_dict, hist_dict, fut_dict,
            t_key=t,
            title=title,
            conf=args.conf,
            draw_samples=(not args.no_sample_lines),
            time_colored=True if args.time_colored else True,  # default-on for screenshot-like style
            multi_rings=args.multi_rings,
            draw_mean_points=(not args.no_mean_points),
            cov_source=args.cov_source,
            dt=scene.dt,
            fill_ellipses=True if args.fill_ellipses else True,  # default-on
        )
        return

    # single node fallback 
    node, t_key = pick_node_and_t(pred_dict, desired_t=t)
    if node is None:
        print("Could not find any valid node/timestep in predictions.")
        return

    samples = pred_dict[node][t_key]
    past = hist_dict[node][t_key]
    future_gt = fut_dict[node][t_key]
    samples_arr = normalize_samples(samples)

    if args.mode == "2d":
        fig, ax = plt.subplots(figsize=(6.8, 7.2))
        title = f"Scene_{args.scene_index}_Timestep_{t_key} {args.title_suffix}".strip()

        draw_node_cov_chain_2d(
            ax,
            past, future_gt, samples_arr,
            conf=args.conf,
            draw_samples=(not args.no_sample_lines),
            time_colored=True if args.time_colored else True,
            multi_rings=args.multi_rings,
            draw_mean_points=(not args.no_mean_points),
            cov_source=args.cov_source,
            dt=scene.dt,
            fill_ellipses=True if args.fill_ellipses else True,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.axis("equal")

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")

        plt.tight_layout()
        os.makedirs("results/viz", exist_ok=True)
        fname = title.replace(" ", "_").replace(":", "_") + ".png"
        out_path = os.path.join("results/viz", fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure to {out_path}")

    else:
        title = f"Scene_{args.scene_index}_t_{t_key}_{args.num_samples}_cov_chain_3d"
        plot_cov_chain_3d(
            past,
            future_gt,
            samples_arr,
            title=title,
            conf=args.conf,
            dt=scene.dt,
            draw_samples=args.draw_3d_samples,
            time_colored=True if args.time_colored else True,
            max_ellipses=args.max_ellipses,
            draw_mean_points=(not args.no_mean_points),
        )


if __name__ == "__main__":
    main()
