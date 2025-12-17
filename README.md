# 🔬 Skin Cancer Detection ML Pipeline

A production-ready machine learning pipeline for skin cancer classification with automated quality assurance using DeepChecks.

## 🌟 Features

- **Automated ML Pipeline**: Complete workflow from data loading to model deployment
- **Quality Assurance**: DeepChecks integration for comprehensive model validation
- **Rich Visualizations**: Detailed HTML reports with graphs and analysis
- **Drift Detection**: Monitors feature and prediction distributions
- **Overfitting Detection**: Automatically compares train vs test performance
- **Confusion Matrix**: Shows exactly where your model is confused
- **Discord Notifications**: Get notified when training completes

## 🚀 Quick Start

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "New Project"
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment:**
   - **Windows PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt:**
     ```cmd
     .venv\Scripts\activate.bat
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

**Option 1: Using the run script (Recommended)**
```powershell
# PowerShell
.\run.ps1

# Or with custom sample limit
.\run.ps1 1000
```

```cmd
# Command Prompt
run.bat

# Or with custom sample limit
run.bat 1000
```

**Option 2: Manual execution**
```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run the workflow
python app/workflow.py

# Or with custom sample limit
python app/workflow.py 1000
```

### What Happens Next

1. **Data Loading**: Loads skin cancer image metadata and histograms
2. **Feature Extraction**: Processes images into ML-ready features
3. **Model Training**: Trains RandomForest classifier
4. **🔍 DeepChecks Validation** (1-2 minutes):
   - Analyzes feature distributions
   - Detects data drift
   - Checks for overfitting
   - Generates confusion matrix
   - Compares model performance

5. **Results**: 
   - ✅ If validation passes → Model saved to `models/`
   - ⚠️ If validation fails → Review issues in report

## 📊 Viewing Validation Results

Open `reports/validation_report.html` in your browser to see:

- **📈 Feature Graphs**: Visualizations of all features
- **🎯 Overfitting Analysis**: Train vs Test performance comparison
- **🔄 Drift Detection**: Changes in data distributions
- **🎭 Confusion Matrix**: Exactly where the model is confused
- **📉 Performance Metrics**: Accuracy, precision, recall, F1-score
- **🔍 Feature Importance**: Which features matter most

## 📁 Project Structure

```
New Project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI deployment
│   ├── utils.py             # Utility functions
│   ├── workflow.py          # Main training pipeline ⭐
│   └── ml_validation.py     # DeepChecks validation
├── models/                  # Saved models (gitignored)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── reports/                 # Validation reports
│   └── validation_report.html
├── HAM10000_metadata.csv    # Skin lesion metadata
├── hmnist_*.csv            # Image histograms
├── requirements.txt         # Python dependencies
├── run.ps1                  # PowerShell run script
└── run.bat                  # Batch run script
```

## 🔧 Configuration

### Sample Limit

Control how many samples to use for training:

```bash
python app/workflow.py 500   # Use 500 samples
python app/workflow.py 1000  # Use 1000 samples
```

### Discord Notifications

Add your Discord webhook URL in `app/workflow.py`:

```python
DISCORD_WEBHOOK_URL = "your-webhook-url-here"
```

## 🧪 What DeepChecks Validates

| Check | Description |
|-------|-------------|
| **Model Info** | Basic model metadata |
| **Train-Test Performance** | Detects overfitting by comparing metrics |
| **Confusion Matrix** | Shows classification errors |
| **Feature Drift** | Detects changes in feature distributions |
| **Prediction Drift** | Detects changes in label distributions |
| **Boosting Overfit** | Specific overfitting check for tree models |
| **Unused Features** | Identifies features that don't help |
| **Simple Model Comparison** | Compares against baseline model |

## 📈 Understanding the Report

### ✅ Green Indicators
- All checks passed
- Model is ready for deployment
- No significant issues detected

### ⚠️ Yellow Warnings
- Minor issues detected
- Review recommended but not critical
- Consider investigating before deployment

### ❌ Red Failures
- Critical issues found
- Do NOT deploy the model
- Review and fix before proceeding

## 🐛 Troubleshooting

**Problem**: `ModuleNotFoundError`
- **Solution**: Make sure virtual environment is activated

**Problem**: Validation takes too long
- **Solution**: Reduce sample limit: `python app/workflow.py 200`

**Problem**: Validation report shows errors
- **Solution**: Check console output for specific issues, review dataset quality

**Problem**: Exit code 1
- **Solution**: Check Python version (requires 3.8+), verify all dependencies installed

## 📚 Learn More

- [DeepChecks Documentation](https://docs.deepchecks.com/)
- [Prefect Workflows](https://docs.prefect.io/)
- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

## 🎯 Next Steps

1. **✅ Run the pipeline and review the validation report**
2. **🔧 Tune hyperparameters** if validation shows issues
3. **📊 Analyze confusion matrix** to improve problem areas
4. **🚀 Deploy the model** using `uvicorn app.main:app`
5. **📈 Monitor in production** using the same DeepChecks suite

## 💡 Tips

- Start with small sample limits (200-500) for faster iteration
- Always check the validation report before deploying
- Pay special attention to the confusion matrix - it tells you WHERE your model struggles
- Monitor feature drift over time to detect data quality issues
- Use the Discord notifications to track training progress

---

**Made with ❤️ for ML Project. Dr sandhu you are Champion**
