import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set publication styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# ==============================================================================
# 1. LOGISTICS DATA SIMULATION (n=3,000 Consignments)
# ==============================================================================
np.random.seed(42)
n_samples = 3000

carriers = ['Apex Express', 'FreightLine', 'MetroVans', 'BlueDart']
service_tiers = ['Same-Day', 'Express 24h', 'Standard Freight']
regions = ['North Metro', 'South Metro', 'East Metro', 'West Metro']

carrier_choice = np.random.choice(carriers, size=n_samples, p=[0.30, 0.28, 0.22, 0.20])
service_choice = np.random.choice(service_tiers, size=n_samples, p=[0.20, 0.45, 0.35])
region_choice = np.random.choice(regions, size=n_samples, p=[0.35, 0.25, 0.20, 0.20])

distance_km = np.random.uniform(5.0, 75.0, size=n_samples).round(1)
weight_kg = np.random.lognormal(mean=2.8, sigma=0.9, size=n_samples).round(2)
volume_cbm = (weight_kg * np.random.uniform(0.002, 0.005, size=n_samples)).round(3)

speed_factor = {'Apex Express': 35.0, 'FreightLine': 28.0, 'MetroVans': 22.0, 'BlueDart': 30.0}
speeds = np.array([speed_factor[c] for c in carrier_choice])

base_transit = distance_km / speeds
traffic_noise = np.random.exponential(scale=0.8, size=n_samples)
weather_delay = np.random.choice([0, 1.5, 3.5], size=n_samples, p=[0.80, 0.15, 0.05])

actual_transit_hours = (base_transit + traffic_noise + weather_delay + 0.5).round(2)
planned_transit_hours = (base_transit + 0.8).round(2)
delay_hours = (actual_transit_hours - planned_transit_hours).round(2)
is_delayed = (delay_hours > 0.5).astype(int)

# Cost Model: $14.50 base + $1.85/km + $0.45/kg
shipping_cost_usd = (14.50 + (distance_km * 1.85) + (weight_kg * 0.45) + np.random.normal(0, 4.0, size=n_samples)).round(2)
shipping_cost_usd = np.clip(shipping_cost_usd, 15.0, None)

df = pd.DataFrame({
    'shipment_id': [f"SHP_{20000 + i}" for i in range(n_samples)],
    'carrier': carrier_choice,
    'service_level': service_choice,
    'region': region_choice,
    'distance_km': distance_km,
    'weight_kg': weight_kg,
    'volume_cbm': volume_cbm,
    'planned_transit_hours': planned_transit_hours,
    'actual_transit_hours': actual_transit_hours,
    'delay_hours': delay_hours,
    'is_delayed': is_delayed,
    'shipping_cost_usd': shipping_cost_usd
})

os.makedirs('figures', exist_ok=True)

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS SUMMARY
# ==============================================================================
print("=" * 65)
print("LOGISTICS EXPLORATORY DATA ANALYSIS SUMMARY (n=3000)")
print("=" * 65)
print(df[['distance_km', 'weight_kg', 'actual_transit_hours', 'delay_hours', 'shipping_cost_usd']].describe().round(2))
print("\nSLA Violation Rate (>0.5h delay):", f"{df['is_delayed'].mean() * 100:.2f}%")

# ==============================================================================
# 3. VISUALIZATION PIPELINE (4 Publication Figures)
# ==============================================================================

# Figure 1: Transit Duration KDE & Distribution
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
sns.histplot(data=df, x='actual_transit_hours', hue='service_level', kde=True, bins=35, palette='tab10', ax=ax, alpha=0.55)
ax.set_title('Figure 1: Distribution of Actual Transit Hours by Service Level', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Actual Transit Duration (Hours)')
ax.set_ylabel('Shipment Volume Frequency')
plt.tight_layout()
plt.savefig('figures/fig1_transit_distribution.png')
plt.close()

# Figure 2: Multivariate Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
corr_cols = ['distance_km', 'weight_kg', 'volume_cbm', 'planned_transit_hours', 'actual_transit_hours', 'delay_hours', 'shipping_cost_usd']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, annot_kws={'size': 9})
ax.set_title('Figure 2: Multivariate Correlation Matrix Across Logistics Drivers', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('figures/fig2_correlation_heatmap.png')
plt.close()

# Figure 3: Carrier Delay Boxplot with SLA Line
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
sns.boxplot(data=df, x='carrier', y='delay_hours', hue='service_level', palette='Set2', ax=ax, showfliers=False)
ax.axhline(0.5, color='crimson', linestyle='--', linewidth=1.5, label='SLA Violation Threshold (+0.5h)')
ax.set_title('Figure 3: Carrier Delay Variance Across Service Levels', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Carrier Entity')
ax.set_ylabel('Transit Delay (Hours vs Planned)')
ax.legend(title='Service Level', loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('figures/fig3_carrier_delay_boxplot.png')
plt.close()

# Figure 4: Cost vs Distance by Weight Tier
df['weight_tier'] = pd.qcut(df['weight_kg'], q=3, labels=['Light (<10kg)', 'Medium (10-30kg)', 'Heavy (>30kg)'])
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
sns.scatterplot(data=df, x='distance_km', y='shipping_cost_usd', hue='weight_tier', style='service_level', alpha=0.65, palette='viridis', ax=ax)
ax.set_title('Figure 4: Operating Cost vs. Route Distance Across Weight Tiers', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Route Distance (km)')
ax.set_ylabel('Total Shipping Cost (USD)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig('figures/fig4_cost_distance_scatter.png')
plt.close()

print("\nVisualizations generated and saved to 'figures/' folder successfully!")