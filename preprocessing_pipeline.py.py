import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ==============================================================================
# STEP 1: SIMULATE DIRTY LOGISTICS TELEMETRY (n=2,500 Shipments)
# ==============================================================================
def generate_dirty_logistics_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    carriers = ['Apex Express', 'FreightLine', 'MetroVans', 'BlueDart']
    services = ['Same-Day', 'Express 24h', 'Standard Freight']
    
    # Base clean variables
    shipment_ids = [f"SHP_{10000 + i}" for i in range(n_samples)]
    carrier_col = np.random.choice(carriers, size=n_samples, p=[0.35, 0.25, 0.20, 0.20])
    service_col = np.random.choice(services, size=n_samples, p=[0.20, 0.45, 0.35])
    
    weights = np.random.lognormal(mean=2.8, sigma=1.1, size=n_samples).round(2)
    volumes = (weights * np.random.uniform(0.002, 0.006, size=n_samples)).round(3)
    
    origin_lat = 40.7128 + np.random.normal(0, 0.15, n_samples)
    origin_lon = -74.0060 + np.random.normal(0, 0.15, n_samples)
    
    planned_hours = np.random.uniform(4.0, 72.0, size=n_samples).round(1)
    actual_hours = planned_hours + np.random.normal(1.2, 4.0, size=n_samples).round(1)
    shipping_costs = (weights * 2.45 + planned_hours * 11.20 + np.random.uniform(15, 60, n_samples)).round(2)
    temperatures = np.random.normal(4.5, 3.2, size=n_samples).round(1)
    
    df = pd.DataFrame({
        'shipment_id': shipment_ids,
        'carrier': carrier_col,
        'service_level': service_col,
        'package_weight_kg': weights,
        'package_volume_cbm': volumes,
        'origin_lat': origin_lat,
        'origin_lon': origin_lon,
        'planned_transit_hours': planned_hours,
        'actual_transit_hours': actual_hours,
        'shipping_cost_usd': shipping_costs,
        'cold_chain_temp_c': temperatures
    })
    
    # Inject realistic data quality corruptions
    # A. Missing values: 7.2% missing temperatures, 5.4% missing volume
    df.loc[np.random.choice(n_samples, size=int(n_samples * 0.072), replace=False), 'cold_chain_temp_c'] = np.nan
    df.loc[np.random.choice(n_samples, size=int(n_samples * 0.054), replace=False), 'package_volume_cbm'] = np.nan
    
    # B. Outliers: Extreme sensor errors in actual transit hours and negative costs
    outlier_idx = np.random.choice(n_samples, size=25, replace=False)
    df.loc[outlier_idx[:15], 'actual_transit_hours'] = df.loc[outlier_idx[:15], 'actual_transit_hours'] * 4.5
    df.loc[outlier_idx[15:], 'shipping_cost_usd'] = -49.99
    
    # C. GPS glitches: (0.0, 0.0)
    gps_idx = np.random.choice(n_samples, size=18, replace=False)
    df.loc[gps_idx, ['origin_lat', 'origin_lon']] = 0.0
    
    # D. Duplicate rows
    duplicates = df.iloc[:20].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df

# ==============================================================================
# STEP 2: AUDIT & CLEANING PIPELINE
# ==============================================================================
def execute_cleaning_pipeline(df_raw: pd.DataFrame):
    print("=" * 65)
    print("LOGISTICS DATA PREPROCESSING PIPELINE AUDIT")
    print("=" * 65)
    
    print(f"\n[1] Initial Shape: {df_raw.shape}")
    print(f"Missing values by column:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
    
    # A. Deduplication
    df = df_raw.drop_duplicates(subset=['shipment_id']).copy()
    print(f"\n[2] Shape after Deduplication: {df.shape}")
    
    # B. Coordinate Glitch Sanitization
    depot_lat, depot_lon = 40.7128, -74.0060
    invalid_gps_mask = (df['origin_lat'] < 25.0) | (df['origin_lat'] > 50.0) | (df['origin_lon'] > -60.0)
    df.loc[invalid_gps_mask, 'origin_lat'] = depot_lat
    df.loc[invalid_gps_mask, 'origin_lon'] = depot_lon
    print(f"[3] Sanitized {invalid_gps_mask.sum()} corrupted GPS coordinates.")
    
    # C. Non-Negative Constraint Enforcement
    cost_mask = df['shipping_cost_usd'] < 0
    df.loc[cost_mask, 'shipping_cost_usd'] = np.nan
    print(f"[4] Flagged {cost_mask.sum()} negative costs for median imputation.")
    
    # D. Context-Aware Statistical Imputation
    temp_imputer = SimpleImputer(strategy='median')
    df['cold_chain_temp_c'] = temp_imputer.fit_predict(df[['cold_chain_temp_c']])
    
    vol_imputer = SimpleImputer(strategy='median')
    df['package_volume_cbm'] = vol_imputer.fit_predict(df[['package_volume_cbm']])
    
    cost_imputer = SimpleImputer(strategy='median')
    df['shipping_cost_usd'] = cost_imputer.fit_predict(df[['shipping_cost_usd']])
    print("[5] Median imputation complete for cold-chain, volume, and cost columns.")
    
    # ==============================================================================
    # STEP 3: IQR OUTLIER REMEDIATION (WINSORIZATION)
    # ==============================================================================
    def winsorize_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        return series.clip(lower=max(0, lower_bound), upper=upper_bound)

    raw_max_transit = df['actual_transit_hours'].max()
    df['actual_transit_hours'] = winsorize_iqr(df['actual_transit_hours'])
    print(f"[6] Winsorized transit hours: Max clipped from {raw_max_transit:.1f}h to {df['actual_transit_hours'].max():.1f}h.")

    # ==============================================================================
    # STEP 4: CATEGORICAL ENCODING & FEATURE SCALING
    # ==============================================================================
    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = ohe.fit_transform(df[['carrier', 'service_level']])
    encoded_feature_names = ohe.get_feature_names_out(['carrier', 'service_level'])
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
    
    # Feature Scaling: RobustScaler for heavy-tailed metrics
    numeric_features = ['package_weight_kg', 'package_volume_cbm', 'shipping_cost_usd', 'actual_transit_hours']
    scaler = RobustScaler()
    scaled_matrix = scaler.fit_transform(df[numeric_features])
    df_scaled_num = pd.DataFrame(scaled_matrix, columns=[f"{col}_scaled" for col in numeric_features], index=df.index)
    
    # Combine final feature matrix
    df_final = pd.concat([df[['shipment_id', 'cold_chain_temp_c']], df_scaled_num, df_encoded], axis=1)
    print(f"\n[7] Final Cleaned Feature Matrix Shape: {df_final.shape}")
    print(f"Total Missing Values in Final Dataset: {df_final.isnull().sum().sum()}")
    print("=" * 65)
    return df_final

if __name__ == "__main__":
    raw_telemetry = generate_dirty_logistics_data(n_samples=2500)
    clean_features = execute_cleaning_pipeline(raw_telemetry)
    print("\nSample Preprocessed Records:")
    print(clean_features.head(3))