import sys
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Blog 28: Surge and Overpressure Modeling - Visualization Generator
Generates surge curves and feature importance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS


logger.info("=" * 70)
logger.info("Blog 28: Surge and Overpressure Modeling - Visualizations")
logger.info("=" * 70)

plt.rcParams['font.family'] = 'serif'

# ============================================================================
# Generate Synthetic Surge Training Data
# ============================================================================
logger.info("\nGenerating synthetic transient scenarios...")

np.random.seed(2025)
N = 5000

linepack = np.random.uniform(0.6, 1.4, N)
closure_time = np.random.uniform(0.2, 10.0, N)
pump_trip = np.random.randint(0, 2, N)
velocity = np.random.uniform(0.5, 3.0, N)
elev = np.random.uniform(-80, 120, N)
temp = np.random.uniform(0, 35, N)

# Physics-based target
base = 35 * velocity / (1 + closure_time / 2.0)
head = 0.433 * (elev / 10.0)
trip = pump_trip * (12 + 6 * np.tanh(3 * (1.5 - velocity)))
pack = 8 * (linepack - 1.0)
noise = np.random.normal(0, 2.0, N)

peak_overpress = 200 + base + head + trip + pack + 0.2 * temp + noise

df = pd.DataFrame({
    'linepack': linepack,
    'closure_time_s': closure_time,
    'pump_trip': pump_trip,
    'velocity_ms': velocity,
    'elevation_drop_m': elev,
    'temperature_c': temp,
    'peak_overpress_psig': peak_overpress
})

logger.info(f"✓ Generated {len(df):,} scenarios")
logger.info(f"Peak overpressure range: {peak_overpress.min():.1f} - {peak_overpress.max():.1f} psig")

# ============================================================================
# Train Surrogate Model
# ============================================================================
logger.info("\nTraining surrogate model...")

X = df.drop(columns=['peak_overpress_psig'])
y = df['peak_overpress_psig']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, random_state=42))
])

model.fit(X_train, y_train)
logger.info("✓ Model trained")

# ============================================================================
# Visualization 1: Surge vs. Closure Time (Multi-velocity)
# ============================================================================
logger.info("\nGenerating surge vs. closure time curves...")

set_tufte_defaults()

fig, ax = plt.subplots(figsize=(10, 6))

velocities = [0.8, 1.5, 2.5]
# Use black/gray with different line styles to distinguish velocities
styles = ['-', '--', ':']
labels = ['Low (0.8 m/s)', 'Med (1.5 m/s)', 'High (2.5 m/s)']

closure_grid = np.linspace(0.2, 12, 120)

for v, style, label in zip(velocities, styles, labels):
    probe = pd.DataFrame({
        'linepack': 1.0,
        'closure_time_s': closure_grid,
        'pump_trip': 0,
        'velocity_ms': v,
        'elevation_drop_m': 0,
        'temperature_c': 15
    })
    
    predicted = model.predict(probe)
    ax.plot(closure_grid, predicted, color=COLORS['black'], 
            linestyle=style, linewidth=2, label=label, alpha=0.85)

# Add MAOP line (use accent color to call out the critical threshold)
ax.axhline(y=260, color=COLORS['accent_red'], linestyle='-', linewidth=2, 
           label='MAOP (260 psig)', alpha=0.8)

ax.set_xlabel('Valve Closure Time (seconds)')
ax.set_ylabel('Predicted Peak Overpressure (psig)')
ax.set_title('Surge vs. Closure Time by Flow Velocity', pad=15)
ax.legend(loc='upper right', frameon=False)
apply_tufte_style(ax, show_grid=False)

# Add annotation
ax.annotate('Safe Zone\n(Below MAOP)', 
            xy=(8, 230), fontsize=9, color=COLORS['gray'],
            ha='center', style='italic', alpha=0.7)

plt.tight_layout()
save_tufte_figure('28_surge_vs_closure_time.png')
plt.close()
logger.info("✓ Surge curves saved")

# ============================================================================
# Visualization 2: Feature Importance
# ============================================================================
logger.info("Generating feature importance plot...")

# Extract feature importance
feature_names = X.columns.tolist()
importances = model.named_steps['gbr'].feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(range(len(importances)), importances[indices],
              color=COLORS['white'], edgecolor=COLORS['black'], linewidth=1.5, alpha=0.9)

ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance for Surge Prediction', pad=15)

# Add percentage labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{importances[indices[i]]:.1%}',
            ha='center', va='bottom', fontsize=9)

apply_tufte_style(ax, show_grid=False)

plt.tight_layout()
save_tufte_figure('28_surge_feature_importance.png')
plt.close()
logger.info("✓ Feature importance saved")

# ============================================================================
# Summary Statistics
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("All visualizations generated successfully!")
logger.info("=" * 70)
logger.info("\nFiles created:")
logger.info("  - 28_surge_vs_closure_time.png")
logger.info("  - 28_surge_feature_importance.png")

logger.info("\nModel Performance:")
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"  Test MAE: {mae:.2f} psig")
logger.info(f"  Test R²: {r2:.4f}")

logger.info("\nFeature Importance Ranking:")
for i, idx in enumerate(indices):
    logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.1%}")

logger.info("\nSafe Closure Times (at MAOP=260 psig):")
for v in velocities:
    probe = pd.DataFrame({
        'linepack': 1.0,
        'closure_time_s': closure_grid,
        'pump_trip': 0,
        'velocity_ms': v,
        'elevation_drop_m': 0,
        'temperature_c': 15
    })
    peaks = model.predict(probe)
    safe_idx = np.where(peaks < 260)[0]
    if len(safe_idx) > 0:
        min_safe = closure_grid[safe_idx[0]]
        logger.info(f"  Velocity {v:.1f} m/s: ≥{min_safe:.1f} seconds")
    else:
        logger.info(f"  Velocity {v:.1f} m/s: No safe closure time")

