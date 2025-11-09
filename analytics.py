import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ----------------------------------------
# Function: run_case_analysis
# ----------------------------------------
def run_case_analysis(validation_df, calibration_df, case_label="Case", true_mu=4.0, true_sigma=1.0):
    # Drop missing values
    validation_df = validation_df.copy()
    calibration_df = calibration_df.copy()

    # Mean and confidence intervals
    mu_values = calibration_df["Calibrated Mu"].dropna()
    sigma_values = calibration_df["Calibrated Sigma"].dropna()

    mu_mean = mu_values.mean()
    mu_sem = stats.sem(mu_values)
    mu_ci = stats.t.interval(0.95, len(mu_values)-1, loc=mu_mean, scale=mu_sem)

    sigma_mean = sigma_values.mean()
    sigma_sem = stats.sem(sigma_values)
    sigma_ci = stats.t.interval(0.95, len(sigma_values)-1, loc=sigma_mean, scale=sigma_sem)

    x_vals = pd.to_numeric(calibration_df["Window Start"], errors='coerce').to_numpy()
    mu_lower = np.full_like(x_vals, mu_ci[0], dtype=np.float64)
    mu_upper = np.full_like(x_vals, mu_ci[1], dtype=np.float64)
    sigma_lower = np.full_like(x_vals, sigma_ci[0], dtype=np.float64)
    sigma_upper = np.full_like(x_vals, sigma_ci[1], dtype=np.float64)

    # --- Validation Indicator Plots ---

    plt.figure(figsize=(8, 4))
    plt.plot(validation_df["N_parts"], validation_df["DTW indicator"], color='blue')
    plt.scatter(validation_df["N_parts"], validation_df["DTW indicator"], color='red')
    plt.axhline(0.98, color='gray', linestyle='--', label="Threshold")
    plt.title(f"DTW Indicator - {case_label}")
    plt.xlabel("n parts")
    plt.ylabel("ind = DTW")
    plt.ylim(0.90, 1.01)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(validation_df["N_parts"], validation_df["mLCSS indicator"], color='green')
    plt.scatter(validation_df["N_parts"], validation_df["mLCSS indicator"], color='red')
    plt.axhline(0.95, color='gray', linestyle='--', label="Threshold")
    plt.title(f"mLCSS Indicator - {case_label}")
    plt.xlabel("n parts")
    plt.ylabel("ind = mLCSS")
    plt.ylim(0.90, 1.01)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(validation_df["N_parts"], validation_df["TIC"], color='orange')
    plt.scatter(validation_df["N_parts"], validation_df["TIC"], color='red')
    plt.axhline(0.05, color='gray', linestyle='--', label="Threshold")
    plt.title(f"TIC - {case_label}")
    plt.xlabel("n parts")
    plt.ylabel("ind = TIC")
    plt.ylim(0.00, 0.15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Calibration Performance Plots ---

    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, calibration_df["Calibrated Mu"], marker='o', label="Calibrated μ")
    plt.axhline(true_mu, color='red', linestyle='--', label="True μ = 4.0")
    plt.axhline(mu_mean, color='blue', linestyle='-', label=f"Avg μ̂ = {mu_mean:.3f}")
    plt.fill_between(x_vals, mu_lower, mu_upper, color='blue', alpha=0.2, label="95% CI")
    plt.title(f"Calibrated Mean (μ̂) with CI - {case_label}")
    plt.xlabel("Window Start Index")
    plt.ylabel("Calibrated μ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, calibration_df["Calibrated Sigma"], marker='x', color='orange', label="Calibrated σ")
    plt.axhline(true_sigma, color='green', linestyle='--', label="True σ = 1.0")
    plt.axhline(sigma_mean, color='orange', linestyle='-', label=f"Avg σ̂ = {sigma_mean:.3f}")
    plt.fill_between(x_vals, sigma_lower, sigma_upper, color='orange', alpha=0.2, label="95% CI")
    plt.title(f"Calibrated Std Dev (σ̂) with CI - {case_label}")
    plt.xlabel("Window Start Index")
    plt.ylabel("Calibrated σ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# Main Script: Load Data & Call Function
# ----------------------------------------
val_path = r"C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\validation\Intial exprements\digital_twin_validation\validation campaigns.xlsx"
cal_path = r"C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\validation\Intial exprements\digital_twin_validation\calibration_summary.csv"

# Choose correct sheet
validation_df = pd.read_excel(val_path, sheet_name="Large deviation")
calibration_df = pd.read_csv(cal_path)

# Run analysis
run_case_analysis(validation_df, calibration_df, case_label="Large Deviation")
