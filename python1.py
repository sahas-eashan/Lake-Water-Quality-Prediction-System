import pandas as pd
import os

folder_path = os.path.dirname(__file__)
target_location_id = 10
filtered_data = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".csv") and filename.startswith("lake_"):
        filepath = os.path.join(folder_path, filename)
        print(f"Reading: {filename}")

        name_parts = filename.replace(".csv", "").split("_")
        year = int(name_parts[1])
        month = int(name_parts[2])

        df = pd.read_csv(filepath)
        print(f"  → Rows in file: {len(df)}")
        if "ID" not in df.columns:
            print(f"  ⚠️ Skipping {filename}: 'ID' column not found.")
            continue

        df_filtered = df[df["ID"] == target_location_id]

        if not df_filtered.empty:
            print(f"  ✅ Found data for ID {target_location_id}")
            df_filtered["Year"] = year
            df_filtered["Month"] = month
            filtered_data.append(df_filtered)
        else:
            print(f"  ❌ No data for ID {target_location_id} in {filename}")

# Final result
if filtered_data:
    result_df = pd.concat(filtered_data, ignore_index=True)
    output_filename = f"location_{target_location_id}_combined.csv"
    result_df.to_csv(output_filename, index=False)
    print(f"\n✅ Data saved to {output_filename}")
else:
    print("\n⚠️ No data found for the specified ID.")
