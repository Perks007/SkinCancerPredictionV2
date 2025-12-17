<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge" alt="Status">
</p>

<h1 align="center">🔬 Skin Cancer Detection System</h1>

<p align="center">
  <strong>An end-to-end machine learning system for dermatoscopic image classification with real-time inference API</strong>
</p>

<p align="center">
  <a href="https://skincancerpred-qm3zp.ondigitalocean.app/">🌐 Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-model-architecture">Model</a>
</p>

---

## 🎯 Overview

This project implements a **production-grade skin lesion classification system** trained on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — one of the largest publicly available collections of dermatoscopic images. The system classifies skin lesions into **7 diagnostic categories** and provides confidence scores to assist medical professionals in early detection of skin cancer.

> ⚠️ **Medical Disclaimer**: This tool is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

### 🏥 Supported Lesion Types

| Code | Diagnosis | Description |
|------|-----------|-------------|
| `mel` | **Melanoma** | Malignant skin cancer — early detection critical |
| `nv` | Melanocytic Nevus | Benign mole |
| `bcc` | Basal Cell Carcinoma | Common skin cancer, rarely metastasizes |
| `akiec` | Actinic Keratosis | Pre-cancerous lesion |
| `bkl` | Benign Keratosis | Non-cancerous growth |
| `df` | Dermatofibroma | Benign fibrous nodule |
| `vasc` | Vascular Lesion | Blood vessel-related lesion |

---

## ✨ Features

### 🚀 Production-Ready API
- **Real-time inference** via RESTful endpoints
- **Image upload support** with automatic feature extraction
- **Confidence scoring** with safety-net thresholds (flags predictions < 50% confidence)
- **Age prediction** auxiliary model for enhanced diagnostics

### 🧠 Machine Learning Pipeline
- **Automated training workflow** using [Prefect](https://www.prefect.io/)
- **Advanced feature extraction**: HSV color histograms + GLCM texture analysis
- **Model validation** with [DeepChecks](https://deepchecks.com/) quality assurance
- **Drift detection** monitors feature and prediction distributions
- **Experiment tracking** logs all training runs to CSV

### 📊 Quality Assurance
- **Comprehensive validation suite** detects overfitting, data drift, and feature issues
- **HTML reports** with interactive visualizations
- **Discord notifications** on pipeline completion/failure

### 🐳 Deployment
- **Docker-ready** with optimized multi-stage builds
- **DigitalOcean App Platform** live deployment
- **Health checks** and graceful shutdown handling

---

## 🌐 Live Demo

**Try the live application:** [https://skincancerpred-qm3zp.ondigitalocean.app/](https://skincancerpred-qm3zp.ondigitalocean.app/)

Upload a dermatoscopic image and get instant classification results with confidence scores.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Perks007/SkinCancerPredictionV2.git
   cd SkinCancerPredictionV2
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the API Server

**Option 1: Direct execution**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Option 2: Docker**
```bash
docker build -t skin-cancer-api .
docker run -p 8000:8000 skin-cancer-api
```

**Option 3: Docker Compose**
```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`

---

## 📡 API Reference

### `GET /`
Returns the web interface for image upload and classification.

### `POST /predict/image`
Upload an image for classification.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@skin_lesion.jpg"
```

**Response:**
```json
{
  "class_id": 0,
  "class_code": "mel",
  "class_name": "Melanoma",
  "confidence": 0.847,
  "predicted_age": 52.3
}
```

### `POST /predict/features`
Submit pre-extracted features for classification.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/features" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.5]}'  # 36 features
```

**Response:** Same format as `/predict/image`

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `class_id` | int | Numeric class identifier (-1 if inconclusive) |
| `class_code` | string | Short diagnostic code (e.g., "mel", "nv") |
| `class_name` | string | Full diagnosis name |
| `confidence` | float | Model confidence (0.0 - 1.0) |
| `predicted_age` | float | Estimated patient age (auxiliary model) |

> **Safety Net**: Predictions with confidence < 50% return `"Inconclusive"` to prevent false positives.

---

## 🧠 Model Architecture

### Feature Extraction Pipeline

The system extracts **36 engineered features** from each image:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (128×128)                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   HSV Color Analysis    │     │    GLCM Texture Analysis    │
│  ─────────────────────  │     │  ─────────────────────────  │
│  • Hue histogram (16)   │     │  • Contrast                 │
│  • Saturation hist (8)  │     │  • Energy                   │
│  • Value histogram (8)  │     │  • Homogeneity              │
│                         │     │  • Correlation              │
│  Total: 32 features     │     │  Total: 4 features          │
└─────────────────────────┘     └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Vector (36 dimensions)                 │
│                    StandardScaler                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              RandomForestClassifier                         │
│  ─────────────────────────────────────────────────────────  │
│  • n_estimators: 200                                        │
│  • max_depth: 20                                            │
│  • Probability calibration enabled                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           7-Class Prediction + Confidence Score             │
└─────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Description | File |
|-----------|-------------|------|
| Classifier | RandomForestClassifier (200 trees) | `skin_cancer_model.pkl` |
| Scaler | StandardScaler for feature normalization | `scaler.pkl` |
| Encoder | LabelEncoder for class mapping | `label_encoder.pkl` |
| Age Regressor | RandomForestRegressor for age prediction | `age_regressor.pkl` |
| Metadata | Class mappings and model info | `model_metadata.pkl` |

---

## 🔄 Training Pipeline

### Running the Training Workflow

```bash
# Train with default settings (full dataset)
python app/workflow.py

# Train with sample limit
python app/workflow.py 1000

# Using convenience scripts
.\run.ps1        # PowerShell
.\run.bat        # Command Prompt
```

### Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PREFECT WORKFLOW                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA LOADING                                                     │
│     └─► Load HAM10000 metadata CSV                                   │
│                                                                      │
│  2. FEATURE EXTRACTION                                               │
│     └─► Process images → 36-dim feature vectors                      │
│                                                                      │
│  3. MODEL TRAINING                                                   │
│     ├─► Train RandomForestClassifier (skin lesion)                   │
│     └─► Train RandomForestRegressor (age prediction)                 │
│                                                                      │
│  4. DEEPCHECKS VALIDATION                                            │
│     ├─► Train/Test feature drift analysis                            │
│     ├─► Prediction drift detection                                   │
│     ├─► Overfitting detection                                        │
│     ├─► Feature correlation analysis                                 │
│     └─► Generate HTML validation report                              │
│                                                                      │
│  5. EXPERIMENT LOGGING                                               │
│     └─► Log metrics to reports/experiments.csv                       │
│                                                                      │
│  6. ARTIFACT SAVING (if validation passes)                           │
│     └─► Save models to models/                                       │
│                                                                      │
│  7. DISCORD NOTIFICATION                                             │
│     └─► Send success/failure webhook                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Validation Report

After training, view the comprehensive validation report:
```
reports/validation_report.html
```

The report includes:
- 📈 Feature distribution graphs
- 🎯 Train vs Test performance comparison
- 🔄 Drift detection analysis
- 🎭 Confusion matrix visualization
- 📉 Performance metrics (Accuracy, F1, Recall)

---

## 📁 Project Structure

```
SkinCancerPredictionV2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── utils.py             # Feature extraction & utilities
│   ├── workflow.py          # Prefect training pipeline
│   ├── ml_validation.py     # DeepChecks validation suite
│   └── static/
│       └── index.html       # Web interface
├── models/
│   ├── skin_cancer_model.pkl
│   ├── age_regressor.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_metadata.pkl
├── tests/
│   ├── test_api.py          # API endpoint tests
│   └── test_main.py         # Unit tests
├── reports/
│   └── experiments.csv      # Experiment tracking
├── logs/
│   └── api.log              # Application logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t skin-cancer-api .

# Run container
docker run -d -p 8000:8000 --name skin-api skin-cancer-api

# View logs
docker logs -f skin-api
```

### Docker Compose

```yaml
services:
  skin-cancer-api:
    build: .
    ports:
      - "8000:8000"
    restart: always
```

```bash
docker compose up -d
```

---

## 📊 Dataset

This project uses the **HAM10000** ("Human Against Machine with 10000 training images") dataset:

- **Source**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **Images**: 10,015 dermatoscopic images
- **Classes**: 7 diagnostic categories
- **Resolution**: Various (resized to 128×128 for processing)

### Citation

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic 
         images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018},
  publisher={Nature Publishing Group}
}
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/AI** | scikit-learn, NumPy, Pandas, OpenCV, scikit-image |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Orchestration** | Prefect |
| **Validation** | DeepChecks |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Docker, DigitalOcean App Platform |
| **Testing** | pytest |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) creators
- [DeepChecks](https://deepchecks.com/) for ML validation tools
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Prefect](https://www.prefect.io/) for workflow orchestration

---

<p align="center">
  <strong>Built with ❤️ for early skin cancer detection</strong>
</p>

<p align="center">
  <a href="https://skincancerpred-qm3zp.ondigitalocean.app/">🌐 Try Live Demo</a>
</p>
