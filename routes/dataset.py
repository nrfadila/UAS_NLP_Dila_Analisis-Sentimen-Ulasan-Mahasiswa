import os
import pandas as pd
from flask import render_template, request, jsonify, Response
from routes import dataset_bp
from src.training import load_dataset, train_model, DATASET_PATH

@dataset_bp.route('/dataset')
def dataset_page():
    df = load_dataset()
    records = df.to_dict(orient='records')
    total_data = len(records)
    return render_template('dataset.html', dataset=records, total_data=total_data, active_page='dataset')

@dataset_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Nama file kosong'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Format file harus CSV'}), 400

    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            return jsonify({'status': 'error', 'message': 'CSV harus memiliki kolom text dan sentiment'}), 400
        
        df = df[['text', 'sentiment']].dropna()
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
        
        valid_sentiments = {'positif', 'negatif', 'netral'}
        df = df[df['sentiment'].isin(valid_sentiments)]
        
        df.to_csv(DATASET_PATH, index=False)
        train_model(df)
        
        return jsonify({
            'status': 'success',
            'message': f'Berhasil mengimpor {len(df)} data dan melatih ulang model!',
            'total': len(df)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal memproses file CSV: {str(e)}'}), 500

@dataset_bp.route('/export')
def export_dataset():
    df = load_dataset()
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=mahasiswa_dataset_export.csv"}
    )
