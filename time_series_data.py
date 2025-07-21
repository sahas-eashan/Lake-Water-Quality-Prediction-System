import pandas as pd

# Load cleaned dataset
df = pd.read_csv("location_10_cleaned.csv")

# Sort by year and month to ensure correct ordering
df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

# Feature columns (current month readings)
features = ["pH", "Tur", "Cond"]

# Targets for next month's prediction
df["pH_next"] = df["pH"].shift(-1)
df["Tur_next"] = df["Tur"].shift(-1)
df["Cond_next"] = df["Cond"].shift(-1)

# Select required columns
df_model = df[features + ["pH_next", "Tur_next", "Cond_next"]]

# Drop the last row (NaNs from shift)
df_model = df_model.dropna().reset_index(drop=True)

# Save to file
df_model.to_csv("location_10_realtime_model_data.csv", index=False)

print("✅ Real-time friendly dataset saved as 'location_10_realtime_model_data.csv'")
print(df_model.head())
