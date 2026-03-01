"""
ETL + Model Training — UCI Heart Disease (All 4 Sites)
======================================================
Combines Cleveland, Hungarian, Switzerland, VA datasets.
Uses GradientBoostingClassifier (sklearn's XGBoost equivalent).
Run once:  python train_model.py
Outputs:  model.pkl, metadata.json
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

BASE = os.path.dirname(os.path.abspath(__file__))

COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

FEATURE_LABELS = {
    'age':      'Age',
    'sex':      'Sex',
    'cp':       'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol':     'Serum Cholesterol',
    'fbs':      'Fasting Blood Sugar >120',
    'restecg':  'Resting ECG Results',
    'thalach':  'Max Heart Rate',
    'exang':    'Exercise Induced Angina',
    'oldpeak':  'ST Depression (Oldpeak)',
    'slope':    'ST Slope',
    'ca':       'Major Vessels Colored',
    'thal':     'Thalassemia'
}

SOURCES = {
    'cleveland':   'processed.cleveland.data',
    'hungarian':   'processed.hungarian.data',
    'switzerland': 'processed.switzerland.data',
    'va':          'processed.va.data',
}

def load_source(name, filename):
    path = os.path.join(BASE, filename)
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values='?')
    df['source'] = name
    print(f"  [{name}] {len(df)} rows, {df.isnull().sum().sum()} missing values")
    return df

def etl():
    print("\n=== ETL: LOADING ALL 4 SOURCES ===")
    frames = [load_source(k, v) for k, v in SOURCES.items()]
    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(df)} total rows")

    df['target'] = (df['target'] > 0).astype(int)
    df['cp'] = df['cp'].clip(1, 4) - 1

    def remap_thal(v):
        if pd.isna(v): return np.nan
        if v == 3: return 0
        if v == 6: return 1
        if v == 7: return 2
        return np.nan
    df['thal'] = df['thal'].apply(remap_thal)
    df['slope'] = df['slope'].apply(lambda v: v - 1 if not pd.isna(v) else np.nan)

    print("\n  Missing values per feature:")
    missing = df[FEATURES].isnull().sum()
    for col, cnt in missing[missing > 0].items():
        pct = cnt / len(df) * 100
        print(f"    {col}: {cnt} ({pct:.1f}%)")
        df[col] = df[col].fillna(df[col].median())

    print(f"\n  After cleaning: {len(df)} rows, {df[FEATURES].isnull().sum().sum()} missing")
    print(f"  Disease prevalence: {df['target'].mean()*100:.1f}% ({df['target'].sum()}/{len(df)})")
    return df

def compute_analytics(df):
    a = {}
    a['n_total']    = int(len(df))
    a['n_disease']  = int(df['target'].sum())
    a['n_healthy']  = int((df['target'] == 0).sum())
    a['prevalence'] = round(df['target'].mean() * 100, 1)

    a['by_source'] = {
        src: {
            'n': int((df['source'] == src).sum()),
            'disease_pct': round(df[df['source'] == src]['target'].mean() * 100, 1)
        }
        for src in SOURCES.keys()
    }

    a['age_mean']         = round(df['age'].mean(), 1)
    a['age_mean_disease'] = round(df[df['target'] == 1]['age'].mean(), 1)
    a['age_mean_healthy'] = round(df[df['target'] == 0]['age'].mean(), 1)

    a['disease_by_sex'] = {
        'Male':   round(df[df['sex'] == 1]['target'].mean() * 100, 1),
        'Female': round(df[df['sex'] == 0]['target'].mean() * 100, 1),
    }

    bins = [(0,40,'<40'),(40,50,'40-49'),(50,60,'50-59'),(60,70,'60-69'),(70,120,'70+')]
    a['disease_by_age'] = {}
    a['count_by_age']   = {}
    for lo, hi, label in bins:
        mask = (df['age'] >= lo) & (df['age'] < hi)
        sub  = df[mask]
        a['count_by_age'][label]   = int(mask.sum())
        a['disease_by_age'][label] = round(sub['target'].mean() * 100, 1) if len(sub) > 0 else 0

    cp_labels = {0: 'Typical Angina', 1: 'Atypical', 2: 'Non-anginal', 3: 'Asymptomatic'}
    a['cp_dist'] = {cp_labels[int(k)]: int(v) for k, v in df['cp'].value_counts().sort_index().items()}

    a['chol_mean_disease']    = round(df[df['target'] == 1]['chol'].mean(), 1)
    a['chol_mean_healthy']    = round(df[df['target'] == 0]['chol'].mean(), 1)
    a['thalach_mean_disease'] = round(df[df['target'] == 1]['thalach'].mean(), 1)
    a['thalach_mean_healthy'] = round(df[df['target'] == 0]['thalach'].mean(), 1)
    a['oldpeak_mean_disease'] = round(df[df['target'] == 1]['oldpeak'].mean(), 2)
    a['oldpeak_mean_healthy'] = round(df[df['target'] == 0]['oldpeak'].mean(), 2)

    thal_labels  = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect'}
    thal_disease = df[df['target'] == 1]['thal'].value_counts()
    a['thal_disease'] = {thal_labels.get(int(k), str(k)): int(v) for k, v in thal_disease.items()}

    return a

def train(df):
    print("\n=== TRAINING: GRADIENT BOOSTING (XGBoost-equivalent) ===")
    X = df[FEATURES].values
    y = df['target'].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    model.fit(X_tr, y_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"  CV AUC:      {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"  CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    print(f"\n  Test AUC:       {roc_auc_score(y_te, y_prob):.3f}")
    print(f"  Test Accuracy:  {accuracy_score(y_te, y_pred):.3f}")
    print(f"  Test Precision: {precision_score(y_te, y_pred):.3f}")
    print(f"  Test Recall:    {recall_score(y_te, y_pred):.3f}")
    print(f"  Test F1:        {f1_score(y_te, y_pred):.3f}")

    fpr, tpr, _ = roc_curve(y_te, y_prob)

    return model, {
        'cv_auc_mean':    float(cv_auc.mean()),
        'cv_auc_std':     float(cv_auc.std()),
        'cv_acc_mean':    float(cv_acc.mean()),
        'test_auc':       float(roc_auc_score(y_te, y_prob)),
        'test_acc':       float(accuracy_score(y_te, y_pred)),
        'test_precision': float(precision_score(y_te, y_pred)),
        'test_recall':    float(recall_score(y_te, y_pred)),
        'test_f1':        float(f1_score(y_te, y_pred)),
        'roc':            {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        'feature_importances': dict(zip(FEATURES, model.feature_importances_.tolist())),
    }

def run():
    df = etl()

    print("\n=== ANALYTICS ===")
    analytics = compute_analytics(df)
    print(f"  Sources: {list(analytics['by_source'].keys())}")
    print(f"  Total patients: {analytics['n_total']}")

    model, metrics = train(df)

    print("\n=== SAVING ===")
    with open(os.path.join(BASE, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'model_name':     'Gradient Boosting Classifier (XGBoost-equivalent)',
        'feature_names':  FEATURES,
        'feature_labels': FEATURE_LABELS,
        'metrics':        metrics,
        'analytics':      analytics,
    }
    with open(os.path.join(BASE, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("  Saved: model.pkl, metadata.json")
    load_to_postgres(df, model)

if __name__ == '__main__':
    run()


# ─── 5. LOAD PATIENTS INTO POSTGRESQL (if DATABASE_URL set) ──────────────────

def load_to_postgres(df, model):
    import psycopg2
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    if not db_url:
        print("\n  Skipping PostgreSQL load (DATABASE_URL not set)")
        return

    print("\n=== LOADING 920 PATIENTS INTO POSTGRESQL ===")
    conn = psycopg2.connect(db_url)
    cur  = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id SERIAL PRIMARY KEY, name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL,
            fbs REAL, restecg REAL, thalach REAL, exang REAL,
            oldpeak REAL, slope REAL, ca REAL, thal REAL,
            probability REAL, risk_level TEXT, contributions TEXT
        )
    ''')
    cur.execute('DELETE FROM patients')

    for i, row in df.iterrows():
        feats = {f: float(row[f]) for f in FEATURES}
        x     = np.array([[feats[f] for f in FEATURES]])
        prob  = float(model.predict_proba(x)[0][1])
        contribs = {}
        for j, fname in enumerate(FEATURES):
            xp = x.copy()
            xp[0, j] = float(np.median(list(feats.values())))
            contribs[fname] = round(prob - float(model.predict_proba(xp)[0][1]), 4)
        risk = 'HIGH' if prob >= 0.6 else 'MODERATE' if prob >= 0.3 else 'LOW'
        cur.execute('''
            INSERT INTO patients
            (name,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,
             oldpeak,slope,ca,thal,probability,risk_level,contributions)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ''', (
            f"{row['source'].capitalize()} Patient {i+1}",
            *[feats[f] for f in FEATURES],
            round(prob*100,1), risk, json.dumps(contribs)
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Loaded {len(df)} patients into PostgreSQL")
