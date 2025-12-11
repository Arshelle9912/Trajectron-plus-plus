import os
import sys
import dill
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

# Adjust this path if your folder structure differs
sys.path.append("../../trajectron")

from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories


def load_model(model_dir, env, checkpoint, device="cpu"):
    """Load Trajectron++ model and hyperparams."""
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
    """
    prediction_output_to_trajectories can return dicts in either of these shapes:

    A) node-first:
        pred_dict[node][t] -> (K, ph, 2)
    B) timestep-first:
        pred_dict[t][node] -> (K, ph, 2)

    This function converts everything to node-first.
    """
    if not pred_dict:
        return pred_dict, hist_dict, fut_dict

    first_key = next(iter(pred_dict))

    # Heuristic: if first key is int-like, assume timestep-first.
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

    # Already node-first
    return pred_dict, hist_dict, fut_dict


def pick_node_and_t(pred_dict_node_first, desired_t=None):
    """
    Picks a node and timestep that actually exist in the predictions.
    If desired_t is provided, prefer a node that has that timestep.
    """
    nodes = list(pred_dict_node_first.keys())
    if not nodes:
        return None, None

    if desired_t is not None:
        for node in nodes:
            if desired_t in pred_dict_node_first[node]:
                return node, desired_t

    # Fall back: first node, earliest timestep available
    node = nodes[0]
    t_key = sorted(pred_dict_node_first[node].keys())[0]
    return node, t_key


def normalize_samples(samples):
    """
    Force samples into (K, ph, 2).

    Handles:
      - (1, K, ph, 2)  -> squeeze to (K, ph, 2)
      - (K, ph, 2, 1)  -> squeeze last dim
      - (ph, 2)        -> expand to (1, ph, 2)
    """
    samples = np.asarray(samples)

    # Case: extra leading dim (e.g., timesteps dimension)
    # (1, K, ph, 2) -> (K, ph, 2)
    if samples.ndim == 4 and samples.shape[0] == 1 and samples.shape[-1] == 2:
        samples = samples[0]

    # Case: trailing singleton
    # (K, ph, 2, 1) -> (K, ph, 2)
    if samples.ndim == 4 and samples.shape[-1] == 1:
        samples = samples[..., 0]

    # Case: single trajectory
    # (ph, 2) -> (1, ph, 2)
    if samples.ndim == 2:
        samples = samples[None, :, :]

    # Safety: if still 4D in a weird way, try squeezing singletons
    if samples.ndim == 4:
        samples = np.squeeze(samples)
        if samples.ndim == 2:
            samples = samples[None, :, :]

    return samples


def plot_k_predictions(past, future_gt, samples, title="Single agent samples"):
    """
    past:      (T_obs, 2)
    future_gt: (T_pred, 2)
    samples:   (K, T_pred, 2) or variants handled by normalize_samples
    """
    past = np.asarray(past)
    future_gt = np.asarray(future_gt)
    samples = normalize_samples(samples)

    K = samples.shape[0]

    plt.figure(figsize=(10, 4))

    # Past trajectory
    plt.plot(past[:, 0], past[:, 1], "k--", linewidth=2, label="Past")

    # Current position
    plt.scatter(past[-1, 0], past[-1, 1], s=80, c="green", zorder=5, label="Current")

    # Ground-truth future (if available)
    if future_gt.size != 0:
        fut_x = np.concatenate((past[-1:, 0], future_gt[:, 0]))
        fut_y = np.concatenate((past[-1:, 1], future_gt[:, 1]))
        plt.plot(fut_x, fut_y, "k--", linewidth=2, label="GT future")

    # Sampled futures
    for k in range(K):
        s = samples[k]  # (T_pred, 2)
        sx = np.concatenate((past[-1:, 0], s[:, 0]))
        sy = np.concatenate((past[-1:, 1], s[:, 1]))
        plt.plot(sx, sy, linewidth=1, alpha=0.25, color="tab:orange")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs("results/viz", exist_ok=True)
    fname = title.replace(" ", "_").replace(":", "_") + ".png"
    out_path = os.path.join("results/viz", fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory (contains config.json)")
    parser.add_argument("--checkpoint", type=int, required=True,
                        help="Checkpoint number, e.g. 10")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to processed .pkl (e.g. processed/eth_test.pkl)")
    parser.add_argument("--scene_index", type=int, default=0,
                        help="Which scene to visualize (default 0)")
    parser.add_argument("--timestep", type=int, default=571,
                        help="Which timestep to visualize")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of sampled futures to plot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    args = parser.parse_args()

    device = args.device

    # --- load env ---
    with open(args.data, "rb") as f:
        env = dill.load(f, encoding="latin1")

    # --- load model ---
    eval_stg, hyperparams = load_model(args.model, env, args.checkpoint, device)

    # override attention radii if specified
    if "override_attention_radius" in hyperparams:
        for override in hyperparams["override_attention_radius"]:
            n1, n2, ar = override.split(" ")
            env.attention_radius[(n1, n2)] = float(ar)

    # --- build scene graphs ---
    build_scene_graphs(env, hyperparams)

    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]

    # --- pick scene & timestep ---
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
        print("No predictions returned. Possibly not enough history/future at this timestep.")
        print("Try a different --timestep or a different --scene_index.")
        return

    # Convert model outputs to ABSOLUTE positions (Trajectron handles normalization internally)
    pred_dict, hist_dict, fut_dict = prediction_output_to_trajectories(
        predictions,
        scene.dt,
        max_hl,
        ph,
        map=None,
        prune_ph_to_future=True,
    )

    # Normalize dict orientation to node-first
    pred_dict, hist_dict, fut_dict = ensure_node_first(pred_dict, hist_dict, fut_dict)

    # Pick a node that actually has predictions at timestep t
    node, t_key = pick_node_and_t(pred_dict, desired_t=t)
    if node is None:
        print("Could not find any valid node/timestep in predictions.")
        return

    samples = pred_dict[node][t_key]
    past = hist_dict[node][t_key]
    future_gt = fut_dict[node][t_key]

    # Debug shapes (useful if anything still looks off)
    samples_arr = normalize_samples(samples)
    print("-- Extracted:")
    print("   node:", node)
    print("   timestep:", t_key)
    print("   past shape:", np.asarray(past).shape)
    print("   future_gt shape:", np.asarray(future_gt).shape)
    print("   samples shape:", samples_arr.shape)

    title = f"Scene_eth_test_t_{t_key}_{args.num_samples}_samples"
    plot_k_predictions(past, future_gt, samples_arr, title=title)


if __name__ == "__main__":
    main()
