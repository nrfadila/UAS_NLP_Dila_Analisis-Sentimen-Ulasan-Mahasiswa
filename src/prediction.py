import sqlite3
import os
from src.preprocessing import preprocess_pipeline
from src.training import load_model

DB_PATH = os.path.join('dataset', 'history.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            clean_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_positif REAL NOT NULL,
            prob_negatif REAL NOT NULL,
            prob_netral REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def predict_sentiment(text):
    model, vectorizer = load_model()
    if not model or not vectorizer:
        return None
    
    clean = preprocess_pipeline(text)
    if not clean:
        clean = text.lower()
        
    vec = vectorizer.transform([clean])
    probabilities = model.predict_proba(vec)[0]
    classes = model.classes_
    
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    
    prob_pos = prob_dict.get('positif', 0.0)
    prob_neg = prob_dict.get('negatif', 0.0)
    prob_net = prob_dict.get('netral', 0.0)
    
    pred_label = max(prob_dict, key=prob_dict.get)
    confidence = float(prob_dict[pred_label])
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (text, clean_text, sentiment, confidence, prob_positif, prob_negatif, prob_netral)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (text, clean, pred_label, confidence, prob_pos, prob_neg, prob_net))
    conn.commit()
    conn.close()
    
    return {
        'text': text,
        'clean_text': clean,
        'sentiment': pred_label,
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            'positif': round(prob_pos * 100, 2),
            'negatif': round(prob_neg * 100, 2),
            'netral': round(prob_net * 100, 2)
        }
    }

def get_prediction_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY id DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_prediction_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history')
    conn.commit()
    conn.close()
