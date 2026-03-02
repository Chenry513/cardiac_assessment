# CardioScan Clinical — Cardiac Risk Assessment Dashboard

**Live Site:** [cardiac-assessment.onrender.com](https://cardiac-assessment.onrender.com)

A clinical web application that screens patients for cardiac disease risk using machine learning. Built on all four UCI Heart Disease datasets combined (n=920 patients), it gives clinicians an interactive dashboard to enter patient measurements and receive an instant risk assessment with explainable AI-driven feature contributions.

> **Note:** The site runs on a free hosting tier, so the first visit after 15 minutes of inactivity takes about 30–60 seconds to load. After that it's fast!

---

## What This Project Does

CardioScan helps clinical staff assess cardiac risk by providing:

- **Patient Registry**: A searchable table of all patients sorted by risk probability, with risk badges (HIGH / MODERATE / LOW) and quick vitals at a glance
- **New Patient Intake**: A structured clinical form to enter 13 measurements — vitals, lab values, ECG results — and receive an instant ML risk score
- **Explainable AI (SHAP-style)**: Per-feature contribution analysis showing exactly *which* clinical factors are driving each patient's risk up or down
- **Population Analytics**: Data-driven insights from the full 920-patient training cohort — disease rates by age group, sex disparity, chest pain distribution, and source provenance
- **Model Performance Dashboard**: ROC curve, AUC, accuracy, precision, recall, F1, cross-validation scores, and feature importances

---

## How It Works

### For Clinical Staff
1. **Browse Registry**: See all patients ranked by cardiac risk probability
2. **View Patient Detail**: Click any patient to open a side panel with their full clinical values and feature contribution breakdown
3. **Add New Patient**: Fill in the intake form with the patient's measurements — the model scores them instantly and saves to the database
4. **Explore Analytics**: Switch to the Population Analytics tab to understand risk patterns across the full dataset

### For the Technical Pipeline
```
4 UCI CSV Files → ETL (clean + impute) → Gradient Boosting Model → REST API → Clinical UI
                                                      ↓
                                              PostgreSQL Database
                                           (persists patient records)
```

---

## Tech Stack

**Machine Learning & ETL:**
- scikit-learn — Gradient Boosting Classifier (XGBoost-equivalent)
- pandas + numpy — ETL pipeline, missing value imputation, feature engineering
- 5-fold stratified cross-validation, ROC/AUC evaluation

**Backend:**
- Flask (Python web framework)
- PostgreSQL — production database (auto-detected via `DATABASE_URL`)
- SQLite — local development fallback
- gunicorn — WSGI server for production

**Frontend:**
- Vanilla JS + Chart.js — interactive charts (ROC curve, feature importances, population analytics)
- IBM Plex Sans / IBM Plex Mono — clinical typography
- Single-file HTML template — no build step required

**Infrastructure:**
- Hosted on Render.com (free tier)
- PostgreSQL hosted on Render
- Auto-deploy on GitHub push

---

## The Dataset

Combines all four UCI Heart Disease collection sites into one cohort:

| Source | Institution | Patients | Disease Rate |
|--------|-------------|----------|--------------|
| Cleveland | Cleveland Clinic Foundation | 303 | 54.1% |
| Hungarian | Hungarian Institute of Cardiology | 294 | 36.4% |
| Switzerland | University Hospital Zurich | 123 | 87.8% |
| VA | VA Medical Center, Long Beach | 200 | 67.0% |
| **Combined** | | **920** | **55.3%** |

### ETL Pipeline

The raw data has significant missing values across non-Cleveland sites (up to 66% missing for some features). The pipeline handles this systematically:

1. **Extract** — Load all 4 CSV files with standardized column names
2. **Transform**:
   - Binarize target (0 = no disease, 1 = disease present)
   - Remap `cp` encoding to 0-based (1–4 → 0–3)
   - Remap `thal` values (3→0 normal, 6→1 fixed defect, 7→2 reversible)
   - Remap `slope` to 0-based (1–3 → 0–2)
   - Impute missing values with per-column median (robust to outliers)
3. **Load** — Score all 920 patients with the trained model and insert into PostgreSQL

```
Missing values handled:
  slope:  309 missing (33.6%) → median imputed
  ca:     611 missing (66.4%) → median imputed
  thal:   486 missing (52.8%) → median imputed
  + 7 other features with smaller gaps
```

---

## The Model

**Algorithm:** Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`) — the same algorithm as XGBoost, using sklearn's native implementation.

**Hyperparameters:**
```
n_estimators=300, learning_rate=0.05, max_depth=4,
min_samples_leaf=5, subsample=0.8, random_state=42
```

**Performance:**

| Metric | Value |
|--------|-------|
| Test AUC | **0.892** |
| CV AUC (5-fold) | **0.872 ± 0.022** |
| Test Accuracy | 84.2% |
| Test Precision | 84.1% |
| Test Recall | 88.2% |
| Test F1 | 86.1% |

**Top predictive features** (by mean decrease in impurity):
1. Chest Pain Type
2. Max Heart Rate Achieved
3. ST Depression (Oldpeak)
4. Major Vessels Colored (Fluoroscopy)
5. Age

### Explainability

Each patient gets a per-feature contribution score computed via permutation — replacing one feature at a time with the median value and measuring the change in predicted probability. This gives clinicians an interpretable breakdown of *why* the model flagged a patient as high risk, similar to SHAP values.

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/cardiac-assessment.git
cd cardiac-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model + load patients into SQLite
python train_model.py

# 4. Run the app (uses SQLite automatically if no DATABASE_URL set)
python app.py
```

Visit `http://localhost:5000`

### Using PostgreSQL locally

```bash
# Set your PostgreSQL connection string
$env:DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/cardioscan"

# Train + load into PostgreSQL
python train_model.py

# Run app (auto-detects PostgreSQL via DATABASE_URL)
python app.py
```

The app automatically detects which database to use based on whether `DATABASE_URL` is set — no code changes needed between local and production.

---

## Deployment (Render)

The app is deployed on Render.com with a managed PostgreSQL database.

**Build command:**
```
pip install -r requirements.txt && python train_model.py
```

**Start command:**
```
gunicorn app:app
```

**Environment variable:**
```
DATABASE_URL = <External Database URL from Render PostgreSQL>
```

On each deploy, `train_model.py` re-runs the full ETL pipeline, retrains the model, and reloads all 920 patients into the database.

---

## Project Structure

```
cardiac-assessment/
├── app.py                        # Flask backend + REST API
├── train_model.py                # ETL pipeline + model training
├── templates/
│   └── index.html                # Full clinical UI (single file)
├── processed.cleveland.data      # UCI Cleveland dataset
├── processed.hungarian.data      # UCI Hungarian dataset
├── processed.switzerland.data    # UCI Switzerland dataset
├── processed.va.data             # UCI VA dataset
├── model.pkl                     # Trained GBM model
├── metadata.json                 # Analytics + metrics + ROC data
├── patients.db                   # SQLite (local dev only)
├── requirements.txt
├── Procfile                      # gunicorn start command
├── render.yaml                   # Render deploy config
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the clinical dashboard |
| GET | `/api/patients` | All patients ordered by risk |
| POST | `/api/patients` | Add new patient + score |
| DELETE | `/api/patients/<id>` | Remove patient |
| POST | `/api/predict` | Score without saving |
| GET | `/api/metadata` | Model metrics + analytics |
| GET | `/api/stats` | Risk level counts |

---

## Key Design Decisions

**Why Gradient Boosting over Random Forest?**
The UCI model comparison benchmarks (shown in the repository) show XGBoost-class models outperforming Random Forest, SVM, and Logistic Regression on this dataset in both accuracy and AUC. Gradient Boosting handles the class imbalance and mixed feature types well without requiring extensive preprocessing.

**Why combine all 4 sites?**
Using only Cleveland (303 patients) limits generalizability. Combining all four sites triples the dataset size and introduces real-world variation across institutions and geographies, producing a more robust model.

**Why SQLite → PostgreSQL fallback?**
SQLite is perfectly adequate for local development and single-user use, but Render's filesystem is ephemeral — it resets on every deploy, wiping any SQLite file. PostgreSQL persists independently of the app container, so patient records survive redeployments.

---

## Future Improvements

- Tableau Public dashboard embedded for richer population analytics
- Apache Airflow DAG to schedule monthly model retraining
- User authentication for multi-clinician environments
- Export patient records to PDF clinical reports
- Confidence intervals on risk predictions
- SHAP library integration for more accurate feature attributions

---

## Data Attribution

> Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64, 304–310.

UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

---

## Disclaimer

This tool is for **educational and research purposes only**. It is not validated for clinical use and should not be used as the sole basis for medical decisions.

---

## License

MIT License
