import pandas as pd
import numpy as np
import os

def generate_leaderboard(csv_path, output_path="results/analysis/top_10_leaderboard.csv"):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter out rows where fitness is missing (NaN)
    df_finite = df.dropna(subset=['fitness_1', 'fitness_2']).copy()

    # Filter out rows where fitness is infinite (inf or -inf)
    df_finite = df_finite[~np.isinf(df_finite['fitness_1']) & ~np.isinf(df_finite['fitness_2'])]

    # Sort by fitness to find the strongest entities
    df_sorted = df_finite.sort_values(by=['fitness_1', 'fitness_2'], ascending=[False, False])

    # Select the requested columns and rename 'entity_status' to 'status'
    leaderboard = df_sorted[['entity_id', 'fitness_1', 'fitness_2', 'entity_status']].copy()
    leaderboard.rename(columns={'entity_status': 'status'}, inplace=True)

    # Grab the top 10 individuals
    top_10 = leaderboard.head(10)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the table to a CSV for the dashboard to use
    top_10.to_csv(output_path, index=False)
    print(f"Saved leaderboard CSV to {output_path}\n")
    
    # Print a preview to the terminal
    print("--- Top Individuals Leaderboard Preview ---")
    print(top_10.to_markdown(index=False))

if __name__ == '__main__':
    # Pointing directly to the entity metrics CSV that contains the parsed pickle data!
    generate_leaderboard('results/analysis/entity_metrics_summary.csv')