#!/usr/bin/env python3
"""
Extract K-Means cluster parameters from actual Bangalore traffic data.
This script reads the clustered CSV data and generates clusterParams.json
for use in the web simulation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# File paths
DATA_DIR = Path(__file__).parent.parent / "city-traffic-clustering" / "data" / "processed"
CSV_FILE = DATA_DIR / "bangalore_traffic_with_clusters_k4.csv"
SCALED_CSV = DATA_DIR / "bangalore_traffic_scaled.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "simulation" / "data"
OUTPUT_FILE = OUTPUT_DIR / "clusterParams.json"

# Feature columns (14 features total)
FEATURE_COLS = [
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Pedestrian and Cyclist Count",
    "IsWeekend",
    "Weather Conditions_Fog",
    "Weather Conditions_Overcast",
    "Weather Conditions_Rain",
    "Weather Conditions_Windy",
    "Roadwork and Construction Activity_Yes"
]

# Traffic state names for each cluster (from user's notebook)
TRAFFIC_STATES = {
    0: "Free Flow",
    1: "Normal",
    2: "Peak Congestion",
    3: "Incident / Disrupted"
}

# Colors for each cluster
COLORS = {
    0: "#10b981",  # Green - Free Flow
    1: "#3b82f6",  # Blue - Normal
    2: "#f59e0b",  # Orange - Peak Congestion
    3: "#ef4444"   # Red - Incident/Disrupted
}

# Signal timings (adaptive based on cluster)
SIGNAL_TIMINGS = {
    0: {"green_time": 45, "yellow_time": 3, "cycle_length": 100},  # Free Flow - longer green
    1: {"green_time": 30, "yellow_time": 3, "cycle_length": 75},   # Normal - standard
    2: {"green_time": 35, "yellow_time": 3, "cycle_length": 90},   # Peak - moderate
    3: {"green_time": 50, "yellow_time": 3, "cycle_length": 120}   # Incident - longest
}

def main():
    print("=" * 70)
    print("Extracting Cluster Parameters from Bangalore Traffic Data")
    print("=" * 70)
    
    # Load the clustered data
    print(f"\n1. Loading data from: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"   ✓ Loaded {len(df):,} observations")
        print(f"   ✓ Columns: {len(df.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading CSV: {e}")
        return
    
    # Load scaled data for centroids
    print(f"\n2. Loading scaled data from: {SCALED_CSV}")
    try:
        df_scaled = pd.read_csv(SCALED_CSV)
        print(f"   ✓ Loaded scaled data: {df_scaled.shape}")
    except Exception as e:
        print(f"   ✗ Error loading scaled CSV: {e}")
        return
    
    # Calculate cluster statistics
    print("\n3. Calculating cluster statistics...")
    clusters_data = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_scaled_df = df_scaled.iloc[cluster_df.index]
        
        print(f"\n   Cluster {cluster_id}: {TRAFFIC_STATES[cluster_id]}")
        print(f"   - Samples: {len(cluster_df):,} ({len(cluster_df)/len(df)*100:.1f}%)")
        
        # Calculate centroid (mean of scaled values)
        centroid_scaled = cluster_scaled_df[FEATURE_COLS].mean().tolist()
        
        # Calculate feature statistics from unscaled data
        features = {}
        for col in FEATURE_COLS:
            col_data = cluster_df[col]
            features[col.lower().replace(' ', '_').replace('__', '_')] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max())
            }
            
        # Print key metrics
        print(f"   - Avg Volume: {cluster_df['Traffic Volume'].mean():,.0f}")
        print(f"   - Avg Speed: {cluster_df['Average Speed'].mean():.1f} km/h")
        print(f"   - Avg Congestion: {cluster_df['Congestion Level'].mean():.1f}%")
        
        clusters_data.append({
            "id": int(cluster_id),
            "centroid_scaled": centroid_scaled,
            "features": features,
            "traffic_state": TRAFFIC_STATES[cluster_id],
            "description": get_cluster_description(cluster_id),
            "color": COLORS[cluster_id],
            "sample_count": int(len(cluster_df)),
            "percentage": float(len(cluster_df) / len(df) * 100)
        })
    
    # Calculate global scaler parameters (for normalization)
    print("\n4. Calculating scaler parameters...")
    scaler_params = {}
    for col in FEATURE_COLS:
        scaler_params[col.lower().replace(' ', '_').replace('__', '_')] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    
    # Create output JSON structure
    output_data = {
        "metadata": {
            "description": "K-Means cluster parameters extracted from Bangalore traffic data",
            "num_clusters": 4,
            "num_features": 14,
            "num_samples": len(df),
            "silhouette_score": 0.2305
        },
        "clusters": clusters_data,
        "signal_timings": SIGNAL_TIMINGS,
        "feature_names": [col.lower().replace(' ', '_').replace('__', '_') for col in FEATURE_COLS],
        "scaler_params": scaler_params
    }
    
    # Save to JSON file
    print(f"\n5. Saving to: {OUTPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✓ Successfully saved cluster parameters!")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Clusters extracted: 4")
    print(f"Features per cluster: 14")
    print(f"Total samples: {len(df):,}")
    print(f"\nCluster distribution:")
    for cluster in clusters_data:
        print(f"  {cluster['id']}: {cluster['traffic_state']:20s} - {cluster['sample_count']:,} samples ({cluster['percentage']:.1f}%)")
    print("\n✓ Ready to use in simulation!")
    print("=" * 70)

def get_cluster_description(cluster_id):
    """Get human-readable description for each cluster."""
    descriptions = {
        0: "Moderate traffic with some capacity usage",
        1: "Low traffic volume with good speeds",
        2: "High traffic with moderate congestion",
        3: "Very high traffic with incidents and disruptions"
    }
    return descriptions.get(cluster_id, "Unknown cluster")

if __name__ == "__main__":
    main()
