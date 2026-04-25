import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = './sota/Surrogate/results/'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS = ['Kendall_Tau', 'MSE', 'Runtime']
KT_MAX = 1
MSE_MAX = .5
RUNTIME_MAX = 1000

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(directory):
    """
    This method takes in a directory where result .txt files are stored. For each result.txt file,
    kendall tau, mse, and runtime values are extracted from the pattern {final_kendall_tau},{final_mse},{runtime}

    Args:
        directory (str): A string representation of the path to the directory full of results
    """
    data = []
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            path = os.path.join(directory, fname)
            try:
                with open(path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        values = [float(x) for x in content.split(',')]
                        data.append(values)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return pd.DataFrame(data, columns=METRICS)

def get_pareto_front_2d(x_values, y_values, x_max=True, y_max=False):
    """
    Takes in 2 metrics that are objectives. This will then check each combination (x[i], y[i]) to see if it 
    is pareto optimal. If yes, is_optimal[i] = true, else is_optimal[i]

    Args:
        x_values (pd Dataframe): The values of the first objective metric
        y_values (pd Dataframe): The values of the second objective metric
        x_max (bool): The boolean that controls if we want to maximize the first metric
        y_max (bool): The boolean that controls if we want to maximize the second metric

    Returns:
        (list): a list of booleans that serves as a mask to tell other functions if a certain datapoint is pareto optimal
    """
    pts = np.vstack((x_values, y_values)).T
    n_points = pts.shape[0]
    is_optimal = np.zeros(n_points, dtype=bool)
    
    idx = np.lexsort((pts[:, 1] if y_max else -pts[:, 1], 
                      -pts[:, 0] if x_max else pts[:, 0]))
    sorted_pts = pts[idx]
    
    current_best_y = -np.inf if y_max else np.inf
    
    for i, (x, y) in enumerate(sorted_pts):
        if y_max:
            if y > current_best_y:
                is_optimal[idx[i]] = True
                current_best_y = y
        else:
            if y < current_best_y:
                is_optimal[idx[i]] = True
                current_best_y = y
                
    return is_optimal

def plot_pareto_kt_mse(df, output_dir, trivial_points=None):
    """
    Plots ALL individuals, not just optimal ones, but will plot pareto optimal points in a different color.
    This plots kendall tau vs mse specifically.

    Args:
        df (pd DataFrame): The dataframe full of all result values for all individuals
        output_dir (str): A string representation of a path to output the graphs
        trivial_points (list(tuple(float))): a list of 2 tuples of numeric value that represents 2 trivial points on the graph
    
    Returns:
        None
    """
    x_col = 'Kendall_Tau'
    y_col = 'MSE'
    
    # 1. Prepare data (Experimental + Trivial)
    # We use plot_df for Pareto calculation, but df for plotting "All points"
    plot_df = df[[x_col, y_col]].copy()
    if trivial_points:
        triv_df = pd.DataFrame(trivial_points, columns=[x_col, y_col])
        combined_df = pd.concat([plot_df, triv_df], ignore_index=True)
    else:
        combined_df = plot_df

    # 2. Get the Pareto Mask
    mask = get_pareto_front_2d(
        combined_df[x_col].values, 
        combined_df[y_col].values, 
        x_max=True, 
        y_max=False
    )
    
    # 3. Extract and Sort Pareto Points for the line
    pareto_df = combined_df[mask].sort_values(by=x_col)

    # 4. Plotting
    plt.figure(figsize=(10, 6))

    plt.xlim(0, 1)
    plt.ylim(0, .5)
    
    # --- ALL EXPERIMENTAL DATA ---
    # These are the "sub-optimal" points
    plt.scatter(
        df[x_col], df[y_col], 
        c='royalblue', 
        alpha=0.25,      # Faded so the front is visible
        s=30,            # Slightly smaller
        label='All Experimental Models', 
        zorder=1
    )
    
    # --- THE PARETO LINE ---
    # This draws the boundary through the optimal points
    plt.step(
        pareto_df[x_col], pareto_df[y_col], 
        where='pre', 
        color='crimson', 
        lw=2.5,          # Thicker line
        label='Pareto Front', 
        zorder=3
    )
    
    # --- PARETO POINTS ---
    # Distinctly marking the points that define the line
    plt.scatter(
        pareto_df[x_col], pareto_df[y_col], 
        color='crimson', 
        edgecolor='black', # Add outline for pop
        s=70, 
        label='Pareto Optimal Points',
        zorder=4
    )
    
    # --- TRIVIAL POINTS ---
    if trivial_points:
        t_x, t_y = zip(*trivial_points)
        plt.scatter(
            t_x, t_y, 
            color='black', 
            marker='x', 
            s=120, 
            linewidths=2,
            label='Trivial Anchors', 
            zorder=5
        )

    # 5. Formatting
    plt.xlabel('Kendall Tau (Higher is Better)')
    plt.ylabel('Mean Squared Error (Lower is Better)')
    plt.title('Pareto Efficiency: All Results vs. Optimal Front')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pareto_kt_vs_mse.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph generated with all points and front: {save_path}")

def plot_pareto_kt_runtime(df, output_dir, trivial_points=None, runtime_limit=500):
    """
    Plots ALL individuals, not just optimal ones, but will plot pareto optimal points in a different color.
    This plots kendall tau vs runtime specifically.

    Args:
        df (pd DataFrame): The dataframe full of all result values for all individuals
        output_dir (str): A string representation of a path to output the graphs
        trivial_points (list(tuple(float))): a list of 2 tuples of numeric value that represents 2 trivial points on the graph
    
    Returns:
        None
    """
    x_col = 'Kendall_Tau'
    y_col = 'Runtime'
    
    # 1. Get Pareto points from EXPERIMENTAL data
    # Kendall Tau: Max (True), Runtime: Min (False)
    mask = get_pareto_front_2d(df[x_col].values, df[y_col].values, x_max=True, y_max=False)
    pareto_df = df[mask].copy()

    # 2. Force-attach anchors to ensure the line connects to trivial baselines
    if trivial_points:
        triv_df = pd.DataFrame(trivial_points, columns=[x_col, y_col])
        pareto_df = pd.concat([pareto_df, triv_df], ignore_index=True)
    
    # 3. Sort by Kendall Tau for a continuous line
    pareto_df = pareto_df.sort_values(by=x_col)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Standardized Limits: KT (0-1), Runtime (User defined or data max)
    plt.xlim(0, 1)
    plt.ylim(0, runtime_limit)

    # Layer 1: All Models (Faded background)
    plt.scatter(
        df[x_col], df[y_col], 
        c='seagreen', 
        alpha=0.25, 
        s=30, 
        label='All Experimental Models', 
        zorder=1
    )
    
    # Layer 2: Corrected Pareto Line (Hugging bottom-right)
    plt.step(
        pareto_df[x_col], pareto_df[y_col], 
        where='pre', 
        color='darkorange', 
        lw=2.5, 
        label='Efficiency Frontier', 
        zorder=3
    )
    
    # Layer 3: Optimal Points
    plt.scatter(
        pareto_df[x_col], pareto_df[y_col], 
        color='darkorange', 
        edgecolor='black', 
        s=70, 
        zorder=4
    )
    
    # Layer 4: Trivial Anchors
    if trivial_points:
        t_x, t_y = zip(*trivial_points)
        plt.scatter(t_x, t_y, color='black', marker='x', s=120, linewidths=2, label='Anchors', zorder=5)

    # 5. Formatting
    plt.xlabel('Kendall Tau (Higher is Better)')
    plt.ylabel('Runtime (Seconds - Lower is Better)')
    plt.title('Pareto Front: Model Accuracy vs. Computational Cost')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pareto_kt_vs_runtime.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pareto_mse_runtime(df, output_dir, trivial_points=None, mse_limit=100, runtime_limit=500):
    """
    Plots ALL individuals, not just optimal ones, but will plot pareto optimal points in a different color.
    This plots mse vs runtime specifically.

    Args:
        df (pd DataFrame): The dataframe full of all result values for all individuals
        output_dir (str): A string representation of a path to output the graphs
        trivial_points (list(tuple(float))): a list of 2 tuples of numeric value that represents 2 trivial points on the graph
    
    Returns:
        None
    """
    x_col = 'MSE'
    y_col = 'Runtime'
    
    # 1. Get Pareto points (Both Min: x_max=False, y_max=False)
    mask = get_pareto_front_2d(df[x_col].values, df[y_col].values, x_max=False, y_max=False)
    pareto_df = df[mask].copy()

    # 2. Attach anchors (Trivial high-error/high-time baselines)
    if trivial_points:
        triv_df = pd.DataFrame(trivial_points, columns=[x_col, y_col])
        pareto_df = pd.concat([pareto_df, triv_df], ignore_index=True)
    
    # 3. Sort by MSE (X-axis)
    pareto_df = pareto_df.sort_values(by=x_col)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.xlim(0, mse_limit)
    plt.ylim(0, runtime_limit)

    # Layer 1: All Models
    plt.scatter(df[x_col], df[y_col], c='purple', alpha=0.2, s=30, label='All Models', zorder=1)
    
    # Layer 2: Pareto Line (where='post' to hug bottom-left)
    plt.step(pareto_df[x_col], pareto_df[y_col], where='post', color='darkviolet', lw=2.5, label='Front', zorder=3)
    
    # Layer 3: Optimal Points
    plt.scatter(pareto_df[x_col], pareto_df[y_col], color='darkviolet', edgecolor='black', s=70, zorder=4)
    
    if trivial_points:
        t_x, t_y = zip(*trivial_points)
        plt.scatter(t_x, t_y, color='black', marker='x', s=120, label='Anchors', zorder=5)

    plt.xlabel('MSE (Lower is Better)')
    plt.ylabel('Runtime (Lower is Better)')
    plt.title('Pareto Front: MSE vs. Runtime')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pareto_mse_vs_runtime.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    df_results = load_data(RESULTS_DIR)
    
    if not df_results.empty:
        # 1. KT (Max) vs MSE (Min)
        plot_pareto_kt_mse(
            df_results, OUTPUT_DIR, 
            trivial_points=[(0.0, 100), (0.1, 80)]
        )
        
        # 2. KT (Max) vs Runtime (Min)
        plot_pareto_kt_runtime(
            df_results, OUTPUT_DIR, 
            trivial_points=[(0.0, 0), (1, 500)],
            runtime_limit=600
        )
        
        # 3. MSE (Min) vs Runtime (Min)
        # Anchor example: high error (100) at high runtime (500)
        plot_pareto_mse_runtime(
            df_results, OUTPUT_DIR, 
            trivial_points=[(100, 500), (80, 450)],
            mse_limit=110,
            runtime_limit=600
        )
        
        print("All three Pareto fronts have been generated.")
    else:
        print("Data directory empty or inaccessible.")