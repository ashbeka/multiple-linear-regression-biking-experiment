import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import FTestAnovaPower
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f as f_dist
import numpy as np
from statsmodels.graphics.gofplots import qqplot



# ==============================================================================
# SECTION 1: DATA LOADING AND PRE-PROCESSING
# ==============================================================================

# Load JSON data exported from Apple Health
with open('Workouts-2025-05-01 08_11_00-2025-07-04 08_11_40.json', 'r') as f:
    health_data = json.load(f)

# Extract workout data into a list
workouts = health_data['data']['workouts']
print(f"Step 0: Total workouts loaded from JSON: {len(workouts)}")


# Extract calorie data
def get_calories(workout):
    """
    This might cause a samll confusion, but basically, Apple Health JSON provides 'activeEnergy' as a list of kcal measurements and 'activeEnergyBurned' as a single kJ measurement. 
    We calculate total active calories in Kcal. Summing the 'activeEnergy' list is the more reliable method.
    """
    if 'activeEnergy' in workout and isinstance(workout['activeEnergy'], list):
        # list of kcal values
        total_kcal = sum(item.get('qty', 0) for item in workout['activeEnergy'])
        if total_kcal > 0:
            return total_kcal
            
    # convert from kJ.
    elif 'activeEnergyBurned' in workout:
        return workout.get('activeEnergyBurned', {}).get('qty', 0) / 4.184
        
    return 0

processed_data = []
for i, workout in enumerate(workouts):
    # Get Outdoor Cycling only
    if workout.get('name') == 'Outdoor Cycling':
        processed_data.append({
            'ID': i + 1,
            'Date': pd.to_datetime(workout.get('start')).date(),
            'Duration (min)': workout.get('duration', 0) / 60,
            'Distance (km)': workout.get('distance', {}).get('qty', 0),
            'Elevation Up (m)': workout.get('elevationUp', {}).get('qty', 0),
            'Active Calories (Kcal)': get_calories(workout) # Convert kJ to Kcal
        })

df = pd.DataFrame(processed_data)
print(f"Step 1: After filtering for 'Outdoor Cycling': {len(df)} workouts remaining")

# Apply filtering protocol from Section 2.4 step by tep
print("\n--- Applying Data Quality Filters ---")
df_step1 = df[df['Distance (km)'] > 0]
print(f"Step 2: After filtering for 'Distance > 0 km': {len(df_step1)} workouts remaining")

df_step2 = df_step1[df_step1['Elevation Up (m)'] > 0]
print(f"Step 3: After filtering for 'Elevation > 0 m': {len(df_step2)} workouts remaining")

df_step3 = df_step2[df_step2['Active Calories (Kcal)'] > 10]
print(f"Step 4: After filtering for 'Calories > 10 Kcal': {len(df_step3)} workouts remaining")

df_filtered = df_step3[df_step3['Duration (min)'] > 5].copy()
print(f"Step 5 (Final): After filtering for 'Duration > 5 min': {len(df_filtered)} workouts remaining")
print("--- Filtering Complete ---")


# Some cleaning for clarity
df_filtered.rename(columns={
    'Active Calories (Kcal)': 'Calories',
    'Distance (km)': 'Distance',
    'Elevation Up (m)': 'Elevation'
}, inplace=True)

print("--- Data Pre-processing Complete ---")
print(f"Initial workouts: {len(df)}")
print(f"Workouts after filtering: {len(df_filtered)}")
print("\nFirst 5 rows of filtered data:")
print(df_filtered.head())


# ==============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS & VISUALIZATION (for Section 4)
# ==============================================================================

print("\n--- Descriptive Statistics ---")
# The table for Section 4.1
print(df_filtered[['Calories', 'Distance', 'Elevation', 'Duration (min)']].describe())

# Figure 1: Scatter Plot Matrix (Pairs Plot)
print("\nGenerating Figure 1: Pairs Plot...")
sns.set_theme(style="ticks")

def corrfunc(x, y, **kws):
    """Adds Pearson correlation coefficient to a plot."""
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f"r = {r:.3f}\np = {p:.3f}",
                xy=(.05, .95), xycoords=ax.transAxes, 
                verticalalignment='top', fontsize=10)

# Create pairs plot with better labels
df_plot = df_filtered[['Calories', 'Distance', 'Elevation']].copy()
df_plot.columns = ['Calories (kcal)', 'Distance (km)', 'Elevation (m)']

pair_plot = sns.pairplot(df_plot, kind='reg', diag_kind='kde')
pair_plot.map_upper(corrfunc)
pair_plot.fig.suptitle("Scatter Plot Matrix of Key Workout Variables", y=1.02)
plt.tight_layout()
plt.savefig("figure1_pairs_plot.png", dpi=300, bbox_inches='tight')
print("Figure 1 saved as figure1_pairs_plot.png")
plt.close()
#plt.show() 


# ==============================================================================
# SECTION 3: LINEAR REGRESSION AND ASSUMPTION CHECKING (for Section 4)
# ==============================================================================

print("\n--- Regression Analysis ---")
# Defining and fiting the multiple linear regression model
# --- STAGE 1: MAIN EFFECTS MODEL ---
print("\n--- Main Effects Model (Calories ~ Distance + Elevation) ---")
main_effects_model_formula = 'Calories ~ Distance + Elevation'
main_effects_model = smf.ols(formula=main_effects_model_formula, data=df_filtered).fit()

# Get the model summary which contains coefficients, p-values, R-squared, and other values.
main_model_summary = main_effects_model.summary()
print("\n--- Main Effects Regression Model Summary ---")
print(main_model_summary)

# Test H0: β₁(Distance) = β₂(Elevation)
hypothesis = 'Distance = Elevation' 
f_test_main = main_effects_model.f_test(hypothesis)
print("\n--- Hypothesis Test on Main Effects Model (H0: β₁ = β₂) ---")
print(f_test_main)

# --- STAGE 2: INTERACTION MODEL ---
print("\n--- Interaction Model (Calories ~ Distance * Elevation) ---")
interaction_model_formula = 'Calories ~ Distance * Elevation'
interaction_model = smf.ols(formula=interaction_model_formula, data=df_filtered).fit()

# Get the model summary which contains coefficients, p-values, R-squared, and other values.
interaction_model_summary = interaction_model.summary()
print("\n--- Interaction Regression Model Summary ---")
print(interaction_model_summary)

# Test H0: β₁(Distance) = β₂(Elevation)
hypothesis = 'Distance = Elevation' 
f_test_main = interaction_model.f_test(hypothesis)
print("\n--- Hypothesis Test on Interaction Model (H0: β₁ = β₂) ---")
print(f_test_main)

# Assumption 1: Normality of Residuals
residuals = interaction_model.resid
shapiro_test = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test for Normality of Residuals: W-statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

# Figure 2: Q-Q Plot with Confidence Bands (Replicating R's qqPlot style)
print("Generating Figure 2: Q-Q Plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 7))
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", fit=True)
n = len(residuals)
prob_points = (np.arange(n) + 0.5) / n # Use midpoints to avoid exact 0 or 1
prob_points = np.clip(prob_points, 1e-6, 1 - 1e-6) # Clip for safety
ci = 0.95
z_critical = stats.norm.ppf(1 - (1 - ci) / 2)
se = (slope / stats.norm.pdf(stats.norm.ppf(prob_points))) * np.sqrt(prob_points * (1 - prob_points) / n)
upper_band = (intercept + slope * osm) + z_critical * se
lower_band = (intercept + slope * osm) - z_critical * se
ax.plot(osm, osr, 'o', markersize=8, label='Sample Residuals') # The data points
ax.plot(osm, intercept + slope * osm, 'r-', lw=2, label='Normal Line') # The regression line
ax.plot(osm, upper_band, 'r--', lw=2, label='95% Confidence')
ax.plot(osm, lower_band, 'r--', lw=2)
ax.set_title("Q-Q Plot of Model Residuals", fontsize=16)
ax.set_xlabel("Theoretical Quantiles", fontsize=12)
ax.set_ylabel("Sample Quantiles", fontsize=12)
ax.legend()
plt.savefig("figure2_qq_plot.png", dpi=300, bbox_inches='tight')
print("Figure 2 saved as figure2_qq_plot.png")
plt.close()

# Assumption 2: Homoscedasticity
# Figure 3: Residuals vs. Fitted Values Plot (Minimalist R Style)
print("Generating Figure 3: Residuals vs. Fitted Plot...")
plt.style.use('seaborn-v0_8-whitegrid')
# Use a simple scatter plot instead of residplot to have full control
fitted_vals = interaction_model.fittedvalues
plt.figure(figsize=(8, 6))
plt.scatter(fitted_vals, residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5) # 'k' for black edges
plt.grid(linestyle=':', linewidth=0.5) # Use a light dotted grid
plt.title('Flinger-Killeen', fontsize=16)
plt.xlabel('Fitted Values (Predicted Calories)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.savefig("figure3_residuals_plot.png", dpi=300, bbox_inches='tight')
print("Figure 3 saved as figure3_residuals_plot.png")
plt.close()
#plt.show()


# ==============================================================================
# SECTION 4: POST-HOC POWER ANALYSIS (for Section 4)
# ==============================================================================

print("\n--- Post-Hoc Power Analysis ---")
# Calculate the actual power of the experiment given the sample size
# For multiple regression, we need to calculate Cohen's f² effect size
# f² = R² / (1 - R²)
r_squared = main_effects_model.rsquared
cohens_f_squared = r_squared / (1 - r_squared)
cohens_f = cohens_f_squared ** 0.5

print(f"R-squared: {r_squared:.3f}")
print(f"Cohen's f²: {cohens_f_squared:.3f}")
print(f"Cohen's f: {cohens_f:.3f}")

k = interaction_model.df_model  # number of predictors
n = len(df_filtered)  # sample size
alpha = 0.05

# Critical F-value
f_critical = f_dist.ppf(1 - alpha, k, n - k - 1)

# Non-centrality parameter for observed effect
ncp = n * cohens_f_squared

# Power = P(F > f_critical | ncp)
actual_power = 1 - f_dist.cdf(f_critical, k, n - k - 1, ncp)

print(f"Actual Power of the test with n={n}: {actual_power:.4f}")
print(f"Critical F-value: {f_critical:.3f}")
print(f"Observed F-statistic: {main_effects_model.fvalue:.3f}")