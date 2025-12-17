"""Ad-hoc runner to validate classification + regression training on a small sample."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.utils import extract_advanced_features, log_experiment_result, save_regressor

LIMIT = 8  # small sample for quick verification


def main() -> None:
    meta_path = Path("HAM10000_metadata.csv")
    if not meta_path.exists():
        raise FileNotFoundError("Metadata file not found: HAM10000_metadata.csv")

    meta = pd.read_csv(meta_path)

    image_dirs = [Path("HAM10000_images_part_1"), Path("HAM10000_images_part_2")]
    image_map = {}
    for d in image_dirs:
        if d.exists():
            for img_file in d.glob("*.jpg"):
                image_map[img_file.stem] = img_file

    X_list: list[np.ndarray] = []
    labels: list[str] = []
    ages: list[float] = []

    for _, row in meta.iterrows():
        if len(X_list) >= LIMIT:
            break
        img_id = row.get("image_id")
        if img_id not in image_map:
            continue
        feats = extract_advanced_features(str(image_map[img_id]))
        if feats is None:
            continue
        X_list.append(feats)
        labels.append(row.get("dx"))
        age_val = float(row.get("age")) if pd.notna(row.get("age")) else np.nan
        ages.append(age_val)

    if not X_list:
        raise RuntimeError("No images processed; ensure HAM10000_images_part_* are present.")

    X = np.array(X_list)
    labels_arr = np.array(labels)
    ages_arr = np.array(ages)

    mean_age = float(np.nanmean(ages_arr)) if not np.all(np.isnan(ages_arr)) else 50.0
    ages_arr = np.where(np.isnan(ages_arr), mean_age, ages_arr)

    # Classification
    le = LabelEncoder()
    y_cls = le.fit_transform(labels_arr)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall_mel = 0.0
    if "mel" in le.classes_:
        mel_idx = list(le.classes_).index("mel")
        recall_mel = recall_score(y_test, y_pred, labels=[mel_idx], average="macro", zero_division=0)

    # Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, ages_arr, test_size=0.25, random_state=42)
    regr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regr.fit(X_train_r, y_train_r)
    y_pred_r = regr.predict(X_test_r)
    rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))

    reg_path = save_regressor(regr)

    log_experiment_result("RandomForest", "n_estimators=200,max_depth=20", {"accuracy": acc, "f1_score": f1, "recall_melanoma": recall_mel})
    log_experiment_result("RandomForestRegressor", "n_estimators=100", {"rmse": rmse})

    print({
        "samples_used": len(X_list),
        "mean_age_used": mean_age,
        "accuracy": acc,
        "f1": f1,
        "recall_mel": recall_mel,
        "rmse": rmse,
        "regressor_saved": reg_path,
    })


if __name__ == "__main__":
    main()
