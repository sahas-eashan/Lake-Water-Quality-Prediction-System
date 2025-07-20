import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV
file_path = "C:/Users/Cyborg/Documents/GitHub/Water-Quality-MEasuring-ML/location_10_cleaned.csv"

# Load the CSV
df = pd.read_csv(file_path)

# Check for missing or extreme values
print(df[["Tur", "pH", "Cond"]].describe())

# Plot the values
plt.figure(figsize=(15, 5))

# Turbidity
plt.subplot(1, 3, 1)
plt.plot(df["Tur"], marker="o")
plt.title("Turbidity")
plt.xlabel("Sample Index")
plt.ylabel("NTU")

# pH
plt.subplot(1, 3, 2)
plt.plot(df["pH"], marker="o", color="orange")
plt.title("pH")
plt.xlabel("Sample Index")
plt.ylabel("pH")

# Solids (Conductivity)
plt.subplot(1, 3, 3)
plt.plot(df["Cond"], marker="o", color="green")
plt.title("Conductivity (Solids)")
plt.xlabel("Sample Index")
plt.ylabel("Cond")

plt.tight_layout()
plt.show()
