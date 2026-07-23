import pandas as pd
from flask import render_template, request, jsonify, Response
from routes import prediction_bp
from src.prediction import predict_sentiment, get_prediction_history, clear_prediction_history

@prediction_bp.route('/prediction')
def prediction_page():
    history = get_prediction_history()
    return render_template('predict.html', history=history, active_page='predict')

@prediction_bp.route('/predict', methods=['POST'])
def handle_prediction():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'status': 'error', 'message': 'Teks tanggapan tidak boleh kosong!'}), 400
        
    result = predict_sentiment(text)
    if not result:
        return jsonify({'status': 'error', 'message': 'Model belum siap atau belum dilatih.'}), 500
        
    return jsonify({
        'status': 'success',
        'data': result
    })

@prediction_bp.route('/clear-history', methods=['POST'])
def clear_history():
    clear_prediction_history()
    return jsonify({'status': 'success', 'message': 'Riwayat prediksi berhasil dibersihkan.'})

@prediction_bp.route('/export-history')
def export_history():
    history = get_prediction_history()
    if not history:
        df = pd.DataFrame(columns=['id', 'text', 'clean_text', 'sentiment', 'confidence', 'prob_positif', 'prob_negatif', 'prob_netral', 'created_at'])
    else:
        df = pd.DataFrame(history)
    
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=riwayat_prediksi_export.csv"}
    )
