import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Safe import for OR-Tools; automatically falls back if not installed
ORTOOLS_AVAILABLE = False
try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_AVAILABLE = True
except (ImportError, Exception):
    ORTOOLS_AVAILABLE = False


# ==============================================================================
# 1. DATA GENERATOR (Simulating Telemetry & Orders)
# ==============================================================================
def generate_sample_delivery_data(num_orders: int = 150, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    depot_lat, depot_lon = 40.7128, -74.0060  # Hub Reference Coordinates
    
    lats = depot_lat + np.random.normal(0, 0.04, num_orders)
    lons = depot_lon + np.random.normal(0, 0.04, num_orders)
    package_weights = np.random.uniform(0.5, 18.0, num_orders).round(2)
    is_peak = np.random.choice([0, 1], size=num_orders, p=[0.4, 0.6])
    traffic_index = np.random.uniform(1.0, 4.5, num_orders).round(2)
    
    # Distance approximation from depot in km
    distances_km = np.sqrt((lats - depot_lat)**2 + (lons - depot_lon)**2) * 111.0
    density = np.random.uniform(15.0, 75.0, num_orders).round(1)
    
    # Delay probability calculation based on traffic, distance, and peak hours
    delay_prob = 1 / (1 + np.exp(-(0.15 * distances_km + 0.8 * is_peak + 0.5 * traffic_index - 3.0)))
    is_delayed = (np.random.uniform(0, 1, num_orders) < delay_prob).astype(int)

    return pd.DataFrame({
        "order_id": [f"ORD_{1000 + i}" for i in range(num_orders)],
        "latitude": lats,
        "longitude": lons,
        "package_weight_kg": package_weights,
        "distance_km": distances_km.round(2),
        "is_peak_hour": is_peak,
        "traffic_congestion_index": traffic_index,
        "delivery_density_per_sqkm": density,
        "is_delayed": is_delayed
    })


# ==============================================================================
# 2. SPATIAL CLUSTERING (Vehicle Zone Allocation)
# ==============================================================================
def assign_delivery_clusters(df_orders: pd.DataFrame, num_vans: int = 4) -> pd.DataFrame:
    coords = df_orders[["latitude", "longitude"]].to_numpy()
    kmeans = KMeans(n_clusters=num_vans, random_state=42, n_init="auto")
    df_orders["route_cluster_id"] = kmeans.fit_predict(coords)
    return df_orders


# ==============================================================================
# 3. PREDICTIVE DELAY MODELING
# ==============================================================================
def train_delay_prediction_model(df_features: pd.DataFrame):
    features = [
        "distance_km",
        "package_weight_kg",
        "is_peak_hour",
        "traffic_congestion_index",
        "delivery_density_per_sqkm",
    ]
    X = df_features[features]
    y = df_features["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    val_accuracy = model.score(X_test, y_test)
    print(f"[Model Training] Validation Accuracy: {val_accuracy * 100:.2f}%")
    return model, features


# ==============================================================================
# 4. ROUTE OPTIMIZATION SOLVER
# ==============================================================================
def solve_with_heuristic(coords: np.ndarray, depot_idx: int = 0):
    n = len(coords)
    unvisited = set(range(n))
    unvisited.remove(depot_idx)
    current = depot_idx
    route = [current]
    total_dist = 0.0

    while unvisited:
        next_node = min(unvisited, key=lambda node: np.linalg.norm(coords[current] - coords[node]))
        total_dist += np.linalg.norm(coords[current] - coords[next_node]) * 100000
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    total_dist += np.linalg.norm(coords[current] - coords[depot_idx]) * 100000
    route.append(depot_idx)
    return route, int(total_dist)


# ==============================================================================
# 5. ENTRY POINT EXECUTION (What prints the output)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LOGISTICS DATA EXPLORATION & OPTIMIZATION PIPELINE")
    print("=" * 60)

    # 1. Generate Data
    print("\n[Step 1] Ingesting delivery data...")
    orders_df = generate_sample_delivery_data(num_orders=150)
    print(orders_df[["order_id", "distance_km", "traffic_congestion_index", "is_delayed"]].head())

    # 2. Cluster Zones
    print("\n[Step 2] Clustering delivery zones into 4 van territories...")
    clustered_df = assign_delivery_clusters(orders_df, num_vans=4)
    print("Orders assigned per cluster:")
    print(clustered_df["route_cluster_id"].value_counts().sort_index())

    # 3. Train Model
    print("\n[Step 3] Training delay prediction model...")
    model, features = train_delay_prediction_model(clustered_df)
    clustered_df["delay_risk_score"] = model.predict_proba(clustered_df[features])[:, 1].round(3)

    # 4. Route Optimization
    print("\n[Step 4] Optimizing stop sequence for Cluster 0...")
    cluster_0 = clustered_df[clustered_df["route_cluster_id"] == 0].reset_index(drop=True)
    coords = cluster_0[["latitude", "longitude"]].to_numpy()

    route, cost = solve_with_heuristic(coords)
    print(f"Optimized Route Sequence (Stop Indices): {route}")
    print(f"Total Traversal Cost Metric: {cost:,}")
    print("\nPipeline execution complete.")