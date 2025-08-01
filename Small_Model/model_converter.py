import joblib
import m2cgen as m2c

model_files = [
    ("models/xgboost_ph.pkl", "model_xgb_ph.h"),
    ("models/xgboost_tur.pkl", "model_xgb_tur.h"),
    ("models/xgboost_cond.pkl", "model_xgb_cond.h"),
]

for pkl_path, h_path in model_files:
    model = joblib.load(pkl_path)
    c_code = m2c.export_to_c(model)
    with open(h_path, "w") as f:
        f.write(c_code)
    print(f"Exported {pkl_path} → {h_path}")
