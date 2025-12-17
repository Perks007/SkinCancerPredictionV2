"""Utility helpers for inference-time feature extraction and artifact loading."""

from __future__ import annotations

import csv
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

PathOrArray = Union[str, os.PathLike, np.ndarray]


def setup_logger() -> None:
    """Configure rotating file handler and console logging for the application."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Set up rotating file handler (10MB max size, 5 backup files)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, "api.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def log_experiment_result(
    model_name: str,
    params: str,
    metrics: Dict[str, float],
    report_path: str = "reports/experiments.csv"
) -> None:
    """Log ML experiment results to a CSV file for tracking and comparison.
    
    Args:
        model_name: Name of the model (e.g., 'RandomForest', 'SVM')
        params: String describing model parameters (e.g., 'n_estimators=200, max_depth=20')
        metrics: Dictionary containing metrics like accuracy, f1_score, recall_melanoma
        report_path: Path to the CSV file where experiments are logged
    """
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(report_path)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the row data
    row_data = {
        "Timestamp": timestamp,
        "Model_Name": model_name,
        "Parameters": params,
        "Accuracy": metrics.get("accuracy", 0.0),
        "F1_Score": metrics.get("f1_score", 0.0),
        "Recall_Melanoma": metrics.get("recall_melanoma", 0.0),
        "RMSE": metrics.get("rmse", 0.0),
    }
    
    # Write to CSV
    with open(report_path, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Timestamp", "Model_Name", "Parameters", "Accuracy", "F1_Score", "Recall_Melanoma", "RMSE"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the experiment data
        writer.writerow(row_data)
    
    logging.getLogger(__name__).info(f"Experiment logged: {model_name} - Accuracy: {row_data['Accuracy']:.4f}")


def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Load the trained model, scaler, encoder, and metadata from disk."""
    model_path = os.path.join(models_dir, "skin_cancer_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.pkl")
    age_regressor_path = os.path.join(models_dir, "age_regressor.pkl")

    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "encoder": joblib.load(encoder_path),
        "metadata": joblib.load(metadata_path),
        "age_regressor": joblib.load(age_regressor_path),
    }


def extract_advanced_features(image_input: Union[PathOrArray, pd.DataFrame], img_size: tuple[int, int] = (128, 128)) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Extract color/texture features and optionally return regression targets.

    - If ``image_input`` is a pandas DataFrame (training), it must contain an ``age``
      column. Missing ages are imputed with the column mean (fallback 50.0) and the
      function returns ``(X_features, y_reg)``.
    - If ``image_input`` is a path or ndarray, returns a single feature vector.
    """
    if isinstance(image_input, pd.DataFrame):
        if "age" not in image_input.columns:
            raise ValueError("DataFrame input must include an 'age' column for regression targets.")

        age_series = image_input["age"]
        mean_age = float(age_series.dropna().mean()) if not age_series.dropna().empty else 50.0
        if np.isnan(mean_age):
            mean_age = 50.0

        # Determine the column that holds image sources
        image_col = None
        for candidate in ("image_path", "path", "image", "img"):
            if candidate in image_input.columns:
                image_col = candidate
                break

        if image_col is None:
            raise ValueError("DataFrame input must include an image column (image_path/path/image/img).")

        features: list[np.ndarray] = []
        y_reg: list[float] = []

        for _, row in image_input.iterrows():
            age_value = float(row["age"]) if pd.notna(row["age"]) else mean_age
            feats = extract_advanced_features(row[image_col], img_size=img_size)
            if feats is None:
                continue
            features.append(feats)
            y_reg.append(age_value)

        return np.array(features), np.array(y_reg)

    if isinstance(image_input, (str, os.PathLike)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input

    if img is None:
        return None

    img = cv2.resize(img, img_size)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_img], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_img], [2], None, [8], [0, 256]).flatten()

    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = np.mean(graycoprops(glcm, "contrast"))
    energy = np.mean(graycoprops(glcm, "energy"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
    correlation = np.mean(graycoprops(glcm, "correlation"))

    features = np.concatenate(
        [
            hist_h,
            hist_s,
            hist_v,
            [contrast, energy, homogeneity, correlation],
        ]
    )

    return features


def save_regressor(model: Any, models_dir: str = "models") -> str:
    """Persist the regression model used for age prediction."""
    os.makedirs(models_dir, exist_ok=True)
    output_path = os.path.join(models_dir, "age_regressor.pkl")
    joblib.dump(model, output_path)
    return output_path


def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes into an OpenCV BGR image."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def extract_features_from_bytes(image_bytes: bytes, img_size: tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
    """Convenience wrapper to decode bytes then run the feature extractor."""
    img = preprocess_image(image_bytes)
    if img is None:
        return None
    return extract_advanced_features(img, img_size=img_size)


__all__ = [
    "setup_logger",
    "log_experiment_result",
    "load_artifacts",
    "extract_advanced_features",
    "save_regressor",
    "preprocess_image",
    "extract_features_from_bytes",
]
