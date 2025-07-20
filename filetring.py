import pandas as pd
import matplotlib.pyplot as plt

file_path = "C:/Users/Cyborg/Documents/GitHub/Water-Quality-MEasuring-ML/location_10_combined.csv"


# Simple IQR method to remove outliers
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]


df = pd.read_csv(file_path)
# Apply cleaning
df_cleaned = df.copy()
for col in ["Tur", "pH", "Cond"]:
    df_cleaned = remove_outliers_iqr(df_cleaned, col)

print("After cleaning:", df_cleaned.shape)

# Save cleaned file
df_cleaned.to_csv("location_10_cleaned.csv", index=False)
