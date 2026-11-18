import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import linprog

# ==============================================================================
# PART 1: LOGISTICS PROBLEM & DATA SIMULATION (n=3,500 Shipments)
# ==============================================================================
print("=" * 72)
print("PREDICTIVE MODELING & DISPATCH OPTIMIZATION IN LOGISTICS SYSTEMS")
print("=" * 72)

np.random.seed(42)
n_samples = 3500

distance_km = np.random.uniform(5.0, 120.0, size=n_samples).round(1)
package_weight_kg = np.random.lognormal(mean=2.6, sigma=0.85, size=n_samples).round(2)
traffic_index = np.random.uniform(1.0, 4.5, size=n_samples).round(2)          # 1.0=clear, 4.5=gridlock
weather_severity = np.random.choice([0, 1, 2], size=n_samples, p=[0.75, 0.18, 0.07])  # 0=Clear, 1=Rain, 2=Storm
stop_density = np.random.uniform(5.0, 50.0, size=n_samples).round(1)          # stops/km^2
is_peak_hour = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])

# Ground-truth non-linear delivery transit duration (hours)
base_time = (distance_km / 42.0) + 0.35
congestion_penalty = 0.28 * (traffic_index ** 1.45) * (1 + 0.35 * is_peak_hour)
weather_penalty = 0.65 * weather_severity + 0.12 * (weather_severity * distance_km / 50.0)
handling_time = 0.008 * package_weight_kg + (1.2 / np.sqrt(stop_density))
noise = np.random.normal(0, 0.22, size=n_samples)

actual_delivery_hours = np.clip(
    base_time + congestion_penalty + weather_penalty + handling_time + noise, 0.4, None
).round(3)

df = pd.DataFrame({
    'shipment_id': [f"SHP_{40000 + i}" for i in range(n_samples)],
    'distance_km': distance_km,
    'package_weight_kg': package_weight_kg,
    'traffic_index': traffic_index,
    'weather_severity': weather_severity,
    'stop_density': stop_density,
    'is_peak_hour': is_peak_hour,
    'actual_delivery_hours': actual_delivery_hours
})

features = ['distance_km', 'package_weight_kg', 'traffic_index', 'weather_severity', 'stop_density', 'is_peak_hour']
X = df[features]
y = df['actual_delivery_hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"\n[1] Dataset Prepared: {len(X_train)} training samples | {len(X_test)} test samples")

# ==============================================================================
# PART 2: MODEL SELECTION, 5-FOLD CROSS-VALIDATION & EVALUATION
# ==============================================================================
models = {
    'Ridge Regression (Baseline)': Ridge(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=200, learning_rate=0.07, max_depth=5, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
trained_models = {}

for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2').mean()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({
        'Model': name,
        'CV_R2': round(cv_r2, 4),
        'Test_RMSE': round(rmse, 4),
        'Test_MAE': round(mae, 4),
        'Test_R2': round(r2, 4)
    })
    trained_models[name] = (model, preds)

res_df = pd.DataFrame(results)
print("\n[2] Predictive Model Evaluation (5-Fold CV & Holdout Set):")
print(res_df.to_string(index=False))

best_model, best_preds = trained_models['Gradient Boosting Regressor']
importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
print("\n[3] Gradient Boosting Feature Importance Share:")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  * {feat:20s}: {imp * 100:.1f}%")

# ==============================================================================
# PART 3: LINEAR PROGRAMMING DISPATCH WAVE OPTIMIZATION
# ==============================================================================
# Allocate 1,200 daily orders across 3 dispatch waves:
# Wave 1: 06:00 Off-Peak  (Avg transit: 1.95h | Total cost + SLA risk: $5.25/parcel)
# Wave 2: 11:00 Midday    (Avg transit: 2.38h | Total cost + SLA risk: $6.20/parcel)
# Wave 3: 17:00 Peak      (Avg transit: 3.15h | Total cost + SLA risk: $9.35/parcel)

total_daily_orders = 1200
c_cost = np.array([5.25, 6.20, 9.35])
x_baseline = np.array([300, 420, 480])
baseline_daily_cost = np.dot(c_cost, x_baseline)

A_ub = [[1.95, 2.38, 3.15]]          # Network average transit <= 2.40 hours
b_ub = [2.40 * total_daily_orders]
A_eq = [[1, 1, 1]]                     # Conserve all 1,200 daily shipments
b_eq = [total_daily_orders]
bounds = [(0, 550), (0, 480), (220, 1200)]

opt_res = linprog(c_cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
x_opt = np.round(opt_res.x).astype(int)
opt_daily_cost = np.dot(c_cost, x_opt)

daily_savings = baseline_daily_cost - opt_daily_cost
monthly_savings = daily_savings * 26

print("\n[4] Linear Programming Dispatch Optimization (1,200 Daily Orders):")
print(f"  * Baseline Allocation [W1, W2, W3] : {x_baseline} -> Daily Cost: ${baseline_daily_cost:,.2f}")
print(f"  * Optimal Allocation  [W1, W2, W3] : {x_opt} -> Daily Cost: ${opt_daily_cost:,.2f}")
print(f"  * Net Daily Savings                : ${daily_savings:,.2f} (${monthly_savings:,.2f} / 26-day month)")
print(f"  * Network Avg Transit Time         : {np.dot([1.95, 2.38, 3.15], x_baseline)/1200:.2f}h -> {np.dot([1.95, 2.38, 3.15], x_opt)/1200:.2f}h")

# ==============================================================================
# PART 4: EXPORT DIAGNOSTIC CHARTS TO './figures/'
# ==============================================================================
os.makedirs('figures', exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
width = 0.35

# Figure 1: Model Comparison
fig, ax = plt.subplots(figsize=(7.5, 4.1), dpi=200)
x_idx = np.arange(len(res_df))
rects1 = ax.bar(x_idx - width/2, res_df['Test_RMSE'], width, label='Test RMSE (Hours)', color='#1B365D')
rects2 = ax.bar(x_idx + width/2, res_df['Test_MAE'], width, label='Test MAE (Hours)', color='#4B8BBE')
ax.set_ylabel('Prediction Error (Hours)')
ax.set_title('Figure 1: Predictive Model Error Comparison (5-Fold CV & Holdout Test)', fontsize=11.5, fontweight='bold')
ax.set_xticks(x_idx)
ax.set_xticklabels(['Ridge Regression', 'Random Forest', 'Gradient Boosting'])
ax.set_ylim(0, 0.36)
ax.legend(loc='upper right')
for rect in rects1:
    ax.annotate(f"{rect.get_height():.3f}h", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')
for rect in rects2:
    ax.annotate(f"{rect.get_height():.3f}h", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5)
plt.tight_layout()
plt.savefig('figures/fig1_model_comparison.png')
plt.close()

# Figure 2: Actual vs Predicted Fit
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
ax.scatter(y_test, best_preds, alpha=0.45, color='#2A6F97', edgecolors='none', s=18)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, 'r--', linewidth=1.8, label='Ideal Prediction Line (y = x)')
ax.set_xlabel('Actual Delivery Transit Time (Hours)')
ax.set_ylabel('Predicted Delivery Transit Time (Hours)')
ax.set_title(f'Figure 2: Gradient Boosting Actual vs. Predicted Transit Time (R² = {res_df.loc[2, "Test_R2"]:.3f})', fontsize=11.5, fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/fig2_actual_vs_predicted.png')
plt.close()

# Figure 3: Feature Importance
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
importances.plot(kind='barh', color='#1B365D', ax=ax)
ax.set_xlabel('Relative Feature Importance (Gain Contribution)')
ax.set_title('Figure 3: Key Predictive Drivers of Delivery Transit Duration', fontsize=11.5, fontweight='bold')
for i, v in enumerate(importances):
    ax.text(v + 0.008, i, f"{v*100:.1f}%", va='center', fontsize=8.5, fontweight='bold')
ax.set_xlim(0, importances.max() * 1.15)
plt.tight_layout()
plt.savefig('figures/fig3_feature_importance.png')
plt.close()

# Figure 4: Dispatch Wave Optimization
fig, ax = plt.subplots(figsize=(7.5, 4.1), dpi=200)
waves = ['Wave 1: 06:00\n(Off-Peak)', 'Wave 2: 11:00\n(Midday)', 'Wave 3: 17:00\n(Evening Peak)']
x_w = np.arange(len(waves))
b1 = ax.bar(x_w - width/2, x_baseline, width, label='Baseline Dispatch (Static)', color='#94A3B8')
b2 = ax.bar(x_w + width/2, x_opt, width, label='Optimized Dispatch (Linear Programming)', color='#10B981')
ax.set_ylabel('Allocated Daily Shipments (Parcels)')
ax.set_title('Figure 4: Dispatch Wave Optimization (Baseline vs. LP Optimal Allocation)', fontsize=11.5, fontweight='bold')
ax.set_xticks(x_w)
ax.set_xticklabels(waves)
ax.set_ylim(0, 680)
ax.legend(loc='upper right')
for rect in b1:
    ax.annotate(f"{int(rect.get_height())}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5)
for rect in b2:
    ax.annotate(f"{int(rect.get_height())}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_dispatch_optimization.png')
plt.close()

print("\n[5] All 4 figures exported to './figures/' successfully!")