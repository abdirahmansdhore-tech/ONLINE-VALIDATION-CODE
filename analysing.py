import pandas as pd
import matplotlib.pyplot as plt

# Paths
summary_path = r"C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\validation\Intial exprements\digital_twin_validation\calibration_summary.csv"
real_data_path = r"C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\validation\Intial exprements\digital_twin_validation\procees_time_real .txt"
validation_path = r"C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\validation\Intial exprements\digital_twin_validation\validation campaigns.xlsx"

# Load data
df = pd.read_csv(summary_path)
real_proc = pd.read_csv(real_data_path, sep=None, engine="python", header=None)[3]
real_mu = real_proc.mean()
real_sigma = real_proc.std()
avg_calibrated_mu = df["Calibrated Mu"].mean()
avg_calibrated_sigma = df["Calibrated Sigma"].mean()

# --- Calibration Plot ---
plt.figure(figsize=(12, 5))

# Mean μ plot
plt.subplot(1, 2, 1)
plt.plot(df["Window Start"], df["Calibrated Mu"], marker='o', label="Calibrated μ")
plt.axhline(real_mu, color='red', linestyle='--', label=f"Real μ = {real_mu:.2f}")
plt.axhline(avg_calibrated_mu, color='blue', linestyle='--', label=f"Avg Calibrated μ = {avg_calibrated_mu:.2f}")
plt.title("Mean Processing Time Across Windows")
plt.xlabel("Window Start Index")
plt.ylabel("Mean (μ)")
plt.legend()
plt.grid(True)

# Standard deviation σ plot
plt.subplot(1, 2, 2)
plt.plot(df["Window Start"], df["Calibrated Sigma"], marker='o', label="Calibrated σ")
plt.axhline(real_sigma, color='red', linestyle='--', label=f"Real σ = {real_sigma:.2f}")
plt.axhline(avg_calibrated_sigma, color='blue', linestyle='--', label=f"Avg Calibrated σ = {avg_calibrated_sigma:.2f}")
plt.title("Standard Deviation Across Windows")
plt.xlabel("Window Start Index")
plt.ylabel("Standard Deviation (σ)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Validation Indicators ---
val_df = pd.read_excel(validation_path)
val_df["DTW norm"] = val_df["DTW indicator"].round(2)
val_df["mLCSS norm"] = val_df["mLCSS indicator"].round(1)
val_df["TIC norm"] = val_df["TIC"].round(3)

# DTW
plt.figure(figsize=(8, 5))
plt.plot(val_df["N_parts"], val_df["DTW norm"], color='blue')
plt.scatter(val_df["N_parts"], val_df["DTW norm"], color='red')
plt.title("mean=4 std=1")
plt.xlabel("n parts")
plt.ylabel("ind = DTW")
plt.ylim(0.90, 1.01)
plt.grid(True)
plt.tight_layout()
plt.show()

# mLCSS
plt.figure(figsize=(8, 5))
plt.plot(val_df["N_parts"], val_df["mLCSS norm"], color='green')
plt.scatter(val_df["N_parts"], val_df["mLCSS norm"], color='red')
plt.title("mean=4 std=1")
plt.xlabel("n parts")
plt.ylabel("ind = mLCSS")
plt.ylim(0.90, 1.01)
plt.grid(True)
plt.tight_layout()
plt.show()

# TIC
plt.figure(figsize=(8, 5))
plt.plot(val_df["N_parts"], val_df["TIC norm"], color='orange')
plt.scatter(val_df["N_parts"], val_df["TIC norm"], color='red')
plt.title("mean=4 std=1")
plt.xlabel("n parts")
plt.ylabel("ind = TIC")
plt.ylim(0.00, 0.15)
plt.grid(True)
plt.tight_layout()
plt.show()
