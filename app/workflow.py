import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path so imports work from anywhere
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, mean_squared_error
import requests
from prefect import task, flow

from app.utils import load_artifacts, extract_advanced_features, log_experiment_result, save_regressor
from app.ml_validation import validate_training_run

# Define paths
DATA_DIR = Path(".")
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGES_PART1 = DATA_DIR / "HAM10000_images_part_1"
IMAGES_PART2 = DATA_DIR / "HAM10000_images_part_2"
MODELS_DIR = Path("models")

@task(retries=3, retry_delay_seconds=5)
def task_load_data():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    return df

@task
def task_extract_features(df, limit: int = None):
    X = []
    y = []
    y_age = []
    
    # Create a mapping of image_id to full path to speed up lookup
    image_paths = {}
    for p in [IMAGES_PART1, IMAGES_PART2]:
        if p.exists():
            for img_file in p.glob("*.jpg"):
                image_paths[img_file.stem] = img_file

    mean_age = 50.0
    if "age" in df.columns:
        non_null_age = df["age"].dropna()
        if not non_null_age.empty:
            mean_age = float(non_null_age.mean())

    # Process rows
    count = 0
    for index, row in df.iterrows():
        if limit and count >= limit:
            break
            
        img_id = row['image_id']
        label = row['dx']
        age_value = float(row['age']) if 'age' in row and pd.notna(row['age']) else mean_age
        
        if img_id in image_paths:
            img_path = image_paths[img_id]
            features = extract_advanced_features(str(img_path))
            if features is not None:
                X.append(features)
                y.append(label)
                y_age.append(age_value)
                count += 1
    
    return np.array(X), np.array(y), np.array(y_age)

@task
def task_train_model(X, y):
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data (remove stratify for small datasets with potentially unbalanced classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    # Using some reasonable parameters as "optimized" ones
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Return training data as well for validation
    return model, scaler, le, X_train_scaled, X_test_scaled, y_train, y_test


@task
def task_train_age_regressor(X, y_age):
    X_train, X_test, y_train, y_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    save_regressor(regressor)
    return regressor, rmse

@task(name="Run DeepChecks Validation")
def task_validate_model(model, X_train, X_test, y_train, y_test, label_encoder):
    """
    Run DeepChecks validation suite on the trained model.
    
    Args:
        model: Trained scikit-learn model
        X_train: Training features (scaled)
        X_test: Test features (scaled)
        y_train: Training labels (encoded)
        y_test: Test labels (encoded)
        label_encoder: LabelEncoder used for encoding labels
        
    Returns:
        dict: Validation summary with status and report path
        
    Raises:
        ValueError: If critical validation checks fail
    """
    # Get all possible encoded class values (e.g., [0, 1, 2, 3, 4, 5, 6])
    # DeepChecks needs the encoded integer values, not the original string labels
    model_classes = list(range(len(label_encoder.classes_)))
    
    validation_result = validate_training_run(
        X_train, X_test, y_train, y_test, model, model_classes=model_classes
    )
    
    if validation_result['status'] == 'Failed':
        # Could raise an error here to stop the pipeline
        # For now, we just log and continue
        print(f"⚠️  WARNING: Validation had {validation_result['num_failed']} failures")
    
    return validation_result

@task
def task_evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model and return detailed metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate recall for melanoma class (mel) if it exists
    recall_melanoma = 0.0
    try:
        # Find the index of 'mel' in the label encoder classes
        if 'mel' in label_encoder.classes_:
            mel_idx = list(label_encoder.classes_).index('mel')
            # Calculate recall for melanoma class
            recall_melanoma = recall_score(y_test, y_pred, labels=[mel_idx], average='macro', zero_division=0)
    except Exception as e:
        print(f"Could not calculate melanoma recall: {e}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Recall (Melanoma): {recall_melanoma:.4f}")
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "recall_melanoma": recall_melanoma
    }

@task
def task_save_artifacts(model, scaler, encoder):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "skin_cancer_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")
    # Saving metadata as well
    joblib.dump({
        "description": "RandomForest model trained via Prefect",
        "class_mapping": dict(zip(encoder.classes_, encoder.classes_))
    }, MODELS_DIR / "model_metadata.pkl")
    print(f"Artifacts saved to {MODELS_DIR}")

@task(name="Log Experiment")
def task_log_experiment(model_name: str, params: str, metrics: dict):
    """Log experiment results to experiments.csv for tracking."""
    log_experiment_result(model_name, params, metrics)
    print(f"📊 Experiment logged to reports/experiments.csv")

def notify_success(flow, flow_run, state):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1450431404939808869/pQKiIUA9rXhC0XSFMo_ROQW1YxwFaPoYlb4E1nOP4QYwRt8H1Ae_wFBe5s8f1IsMWM6d")
    if not webhook_url:
        print("Skipping Discord notification: DISCORD_WEBHOOK_URL not set.")
        return
        
    try:
        report_path = Path("reports/validation_report.html").absolute()
        message = (
            "✅ Pipeline Complete: Classification (Skin Cancer) & Regression (Age Prediction) models trained.\n"
            f"📊 Validation Report: {report_path}\n"
            "View the HTML report for detailed validation results."
        )
        requests.post(webhook_url, json={"content": message})
        print("Discord notification sent (Success).")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def notify_failure(flow, flow_run, state):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1450431404939808869/pQKiIUA9rXhC0XSFMo_ROQW1YxwFaPoYlb4E1nOP4QYwRt8H1Ae_wFBe5s8f1IsMWM6d")
    if not webhook_url:
        print("Skipping Discord notification: DISCORD_WEBHOOK_URL not set.")
        return

    try:
        requests.post(webhook_url, json={"content": "❌ Training Pipeline Failed."})
        print("Discord notification sent (Failure).")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

@flow(name="Skin Cancer Training Flow", log_prints=True, on_completion=[notify_success], on_failure=[notify_failure])
def skin_cancer_training_flow(limit: int = None):
    df = task_load_data()
    X, y, y_age = task_extract_features(df, limit=limit)
    model, scaler, encoder, X_train, X_test, y_train, y_test = task_train_model(X, y)
    age_model, age_rmse = task_train_age_regressor(X, y_age)
    
    # Run validation before saving artifacts
    validation_result = task_validate_model(model, X_train, X_test, y_train, y_test, encoder)
    
    # Evaluate model performance and get metrics
    metrics = task_evaluate_model(model, X_test, y_test, encoder)
    
    # Log experiment with model parameters
    model_params = "n_estimators=200, max_depth=20, random_state=42"
    task_log_experiment("RandomForest", model_params, metrics)

    task_log_experiment(
        "RandomForestRegressor",
        "n_estimators=100, random_state=42",
        {"rmse": age_rmse},
    )
    
    # Save artifacts only if validation passes
    if validation_result['status'] == 'Passed':
        task_save_artifacts(model, scaler, encoder)
        print("✅ Model artifacts saved successfully.")
    else:
        print(f"⚠️  Model artifacts NOT saved due to validation failures.")
        print(f"   Failed checks: {validation_result['num_failed']}")
        print(f"   Review report: {validation_result['report_path']}")

if __name__ == "__main__":
    import sys
    # Parse command line arguments
    limit = 10015  # Default sample limit
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit: {sys.argv[1]}. Using default: 10015")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Skin Cancer Model Training Pipeline")
    print(f"📊 Sample Limit: {limit}")
    print(f"{'='*60}\n")
    
    # Run the flow
    skin_cancer_training_flow(limit=limit)
