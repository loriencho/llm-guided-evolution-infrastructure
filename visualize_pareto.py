"""
Pareto front visualizer for surrogate evolution runs.

Reads global_gen_*.pkl files produced by run_improved.py and generates:
  - Per-generation scatter plots (kendall_tau vs mse) with Pareto front overlay
  - A summary plot of best fitness values over generations
  - An animated GIF across generations

Usage:
    uv run python visualize_pareto.py --run_dir <path/to/run>
    uv run python visualize_pareto.py --run_dir titanic_test --no_gif
"""

import argparse
import glob
import math
import os
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


INVALID_SENTINELS = {
    (-math.inf, math.inf, math.inf),
    (-9999999999, 9999999999, 9999999999),
}


def is_valid(fitness):
    if fitness is None:
        return False
    if len(fitness) < 2:
        return False
    if fitness in INVALID_SENTINELS:
        return False
    if any(math.isinf(v) or math.isnan(v) for v in fitness):
        return False
    return True


def extract_fitness(dataset):
    """Return list of valid (kendall_tau, mse, runtime) tuples from a GLOBAL_DATA dict."""
    result = []
    for attrs in dataset.values():
        f = attrs.get("fitness")
        if is_valid(f):
            result.append(tuple(float(v) for v in f))
    return result


def pareto_front_2d(points, idx_x=0, idx_y=1, maximize_x=True, maximize_y=False):
    """
    Compute the 2D Pareto front from a projection of n-dimensional points.
    maximize_x / maximize_y control objective directions.
    Returns the subset of points that are Pareto-optimal in this 2D projection.
    """
    if not points:
        return []

    sign_x = 1 if maximize_x else -1
    sign_y = 1 if maximize_y else -1

    sorted_pts = sorted(points, key=lambda p: (-sign_x * p[idx_x], sign_y * p[idx_y]))

    front = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        dominated = any(
            sign_x * q[idx_x] >= sign_x * p[idx_x] and sign_y * q[idx_y] <= sign_y * p[idx_y]
            for q in front
        )
        if not dominated:
            front.append(p)
    return front


def load_generation(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    current = extract_fitness(data.get("GLOBAL_DATA", {}))
    hist = extract_fitness(data.get("GLOBAL_DATA_HIST", {}))
    return current, hist


def discover_generations(run_dir):
    """Return sorted list of (gen_number, path) for all global_gen_*.pkl in run_dir."""
    pattern = os.path.join(run_dir, "global_gen_*.pkl")
    files = glob.glob(pattern)
    pairs = []
    for f in files:
        m = re.search(r"global_gen_(\d+)\.pkl$", f)
        if m:
            pairs.append((int(m.group(1)), f))
    return sorted(pairs, key=lambda x: x[0])


def plot_generation(gen, current_pts, hist_pts, output_path, cumulative_front=None):
    all_pts = current_pts + hist_pts
    if not all_pts:
        print(f"  Gen {gen}: no valid fitness values, skipping.")
        return

    front = pareto_front_2d(all_pts, idx_x=0, idx_y=1, maximize_x=True, maximize_y=False)
    front_sorted = sorted(front, key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(9, 5))

    if hist_pts:
        hx, hy = zip(*[(p[0], p[1]) for p in hist_pts])
        ax.scatter(hx, hy, s=12, alpha=0.35, color="steelblue", label=f"Historical ({len(hist_pts)})")

    if current_pts:
        cx, cy = zip(*[(p[0], p[1]) for p in current_pts])
        ax.scatter(cx, cy, s=18, alpha=0.7, color="darkorange", label=f"Current gen ({len(current_pts)})")

    if front_sorted:
        fx, fy = zip(*[(p[0], p[1]) for p in front_sorted])
        ax.scatter(fx, fy, s=30, color="crimson", zorder=5, label=f"Pareto front ({len(front_sorted)})")
        ax.step(
            [0] + list(fx) + [max(fx)],
            [max(fy) * 2] + list(fy) + [0],
            where="post", color="crimson", linewidth=1.2, alpha=0.6,
        )

    if cumulative_front:
        cfx, cfy = zip(*[(p[0], p[1]) for p in sorted(cumulative_front, key=lambda p: p[0])])
        ax.step(
            [0] + list(cfx) + [max(cfx)],
            [max(cfy) * 2] + list(cfy) + [0],
            where="post", color="gray", linewidth=1, linestyle="--", alpha=0.5,
            label="Cumulative front",
        )

    ax.set_xlabel("Kendall's Tau (higher is better)")
    ax.set_ylabel("MSE (lower is better)")
    ax.set_title(f"Pareto Front — Generation {gen}  ({len(all_pts)} individuals)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Gen {gen}: saved {output_path}  (front size={len(front_sorted)})")


def plot_summary(gen_stats, output_path):
    """Line plot of best kendall_tau and best mse per generation."""
    if not gen_stats:
        return

    gens = [s["gen"] for s in gen_stats]
    best_tau = [s["best_tau"] for s in gen_stats]
    best_mse = [s["best_mse"] for s in gen_stats]
    n_valid = [s["n_valid"] for s in gen_stats]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(gens, best_tau, marker="o", color="darkorange", linewidth=1.5)
    axes[0].set_ylabel("Best Kendall's Tau")
    axes[0].set_title("Best Fitness per Generation")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, best_mse, marker="o", color="steelblue", linewidth=1.5)
    axes[1].set_ylabel("Best MSE (log)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(gens, n_valid, color="gray", alpha=0.6)
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("# Valid Individuals")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Summary plot saved: {output_path}")


def make_gif(image_dir, output_path, duration=600):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not available — skipping GIF generation. Install with: uv add Pillow")
        return

    pattern = os.path.join(image_dir, "pareto_gen_*.png")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r"(\d+)", os.path.basename(f)).group(1)),
    )
    if not files:
        print("No PNG frames found for GIF.")
        return

    frames = [Image.open(f) for f in files]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved: {output_path}  ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Pareto fronts from surrogate evolution run.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing global_gen_*.pkl files (e.g. titanic_test/global_data)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for images (default: <run_dir>/pareto_fronts)")
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--gif_duration", type=int, default=600, help="Milliseconds per GIF frame")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "pareto_fronts")
    os.makedirs(out_dir, exist_ok=True)

    generations = discover_generations(run_dir)
    if not generations:
        print(f"No global_gen_*.pkl files found in: {run_dir}")
        return

    print(f"Found {len(generations)} generation(s) in {run_dir}")

    gen_stats = []
    cumulative_all = []

    for gen, path in generations:
        img_path = os.path.join(out_dir, f"pareto_gen_{gen:04d}.png")
        current_pts, hist_pts = load_generation(path)
        all_pts = current_pts + hist_pts
        cumulative_all.extend(all_pts)

        cumulative_front = pareto_front_2d(
            cumulative_all, idx_x=0, idx_y=1, maximize_x=True, maximize_y=False
        )

        plot_generation(gen, current_pts, hist_pts, img_path, cumulative_front=cumulative_front)

        if all_pts:
            gen_stats.append({
                "gen": gen,
                "best_tau": max(p[0] for p in all_pts),
                "best_mse": min(p[1] for p in all_pts),
                "n_valid": len(all_pts),
            })

    summary_path = os.path.join(out_dir, "fitness_summary.png")
    plot_summary(gen_stats, summary_path)

    if not args.no_gif:
        gif_path = os.path.join(out_dir, "pareto_evolution.gif")
        make_gif(out_dir, gif_path, duration=args.gif_duration)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
