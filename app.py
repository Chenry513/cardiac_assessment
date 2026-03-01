"""
CardioScan — Flask Backend (PostgreSQL)
Run locally:  python app.py  →  http://localhost:5000
Deploy:       Render picks up DATABASE_URL automatically
"""

from flask import Flask, render_template, request, jsonify
import pickle, json, os, numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load model & metadata ─────────────────────────────────────────────────────
with open(os.path.join(BASE, 'model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
with open(os.path.join(BASE, 'metadata.json')) as f:
    META = json.load(f)

FEATURES = META['feature_names']

# ── DB connection ─────────────────────────────────────────────────────────────
# Set DATABASE_URL env var — Render injects this automatically.
# Local example: postgresql://postgres:password@localhost:5432/cardioscan
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id           SERIAL PRIMARY KEY,
            name         TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age          REAL, sex REAL, cp REAL, trestbps REAL, chol REAL,
            fbs          REAL, restecg REAL, thalach REAL, exang REAL,
            oldpeak      REAL, slope REAL, ca REAL, thal REAL,
            probability  REAL, risk_level TEXT, contributions TEXT
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(features: dict):
    x = np.array([[features[f] for f in FEATURES]])
    prob = float(MODEL.predict_proba(x)[0][1])
    contribs = {}
    for i, fname in enumerate(FEATURES):
        xp = x.copy()
        xp[0, i] = float(np.median(list(features.values())))
        contribs[fname] = round(prob - float(MODEL.predict_proba(xp)[0][1]), 4)
    risk = 'HIGH' if prob >= 0.6 else 'MODERATE' if prob >= 0.3 else 'LOW'
    return {'probability': round(prob * 100, 1), 'risk_level': risk, 'contributions': contribs}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata')
def metadata():
    return jsonify(META)

@app.route('/api/patients', methods=['GET'])
def get_patients():
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT * FROM patients ORDER BY probability DESC')
    rows = cur.fetchall()
    cur.close(); conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get('contributions'): d['contributions'] = json.loads(d['contributions'])
        result.append(d)
    return jsonify(result)

@app.route('/api/patients', methods=['POST'])
def add_patient():
    data  = request.json
    feats = {f: float(data[f]) for f in FEATURES}
    res   = predict(feats)
    conn  = get_db(); cur = conn.cursor()
    cur.execute('''
        INSERT INTO patients
        (name,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,
         oldpeak,slope,ca,thal,probability,risk_level,contributions)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING *
    ''', (
        data.get('name', 'New Patient'),
        *[feats[f] for f in FEATURES],
        res['probability'], res['risk_level'], json.dumps(res['contributions'])
    ))
    row = dict(cur.fetchone())
    if row.get('contributions'): row['contributions'] = json.loads(row['contributions'])
    conn.commit(); cur.close(); conn.close()
    return jsonify(row), 201

@app.route('/api/predict', methods=['POST'])
def predict_only():
    data  = request.json
    feats = {f: float(data[f]) for f in FEATURES}
    return jsonify(predict(feats))

@app.route('/api/patients/<int:pid>', methods=['DELETE'])
def delete_patient(pid):
    conn = get_db(); cur = conn.cursor()
    cur.execute('DELETE FROM patients WHERE id = %s', (pid,))
    conn.commit(); cur.close(); conn.close()
    return jsonify({'deleted': pid})

@app.route('/api/stats')
def stats():
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT risk_level FROM patients')
    levels = [r['risk_level'] for r in cur.fetchall()]
    cur.close(); conn.close()
    return jsonify({'total': len(levels), 'high': levels.count('HIGH'),
                    'moderate': levels.count('MODERATE'), 'low': levels.count('LOW')})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
