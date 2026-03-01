# CardioScan Clinical — Cardiac Risk Assessment

## Local Setup (SQLite → PostgreSQL)

### 1. Install PostgreSQL locally
Download from https://www.postgresql.org/download/ and create a database:
```sql
CREATE DATABASE cardioscan;
```

### 2. Set environment variable
**Windows:**
```
set DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/cardioscan
```
**Mac/Linux:**
```
export DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/cardioscan
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Train model + load 920 patients
```
python train_model.py
```

### 5. Run
```
python app.py
```
Open http://localhost:5000

---

## Deploy to Render (Free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Blueprint
3. Connect your repo — Render reads `render.yaml` automatically
4. It will create a PostgreSQL database + web service, run `train_model.py`, and deploy

Your live URL will be: `https://cardioscan.onrender.com`

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting (XGBoost-equivalent) |
| Dataset | UCI Heart Disease — 4 sites, n=920 |
| Test AUC | 0.892 |
| CV AUC (5-fold) | 0.872 ± 0.022 |
| Accuracy | 84.2% |
| Precision | 84.1% |
| Recall | 88.2% |
