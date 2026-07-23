import os
import joblib
from flask import render_template, jsonify, send_file
from routes import training_bp
from src.training import get_training_info, train_model, MODEL_PATH, VECTORIZER_PATH

@training_bp.route('/training')
def training_page():
    info = get_training_info()
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return render_template('train.html', info=info, model_exists=model_exists, active_page='train')

@training_bp.route('/train', methods=['POST'])
def run_train():
    try:
        model, vectorizer, X_train, X_test, y_train, y_test, elapsed = train_model()
        info = get_training_info()
        return jsonify({
            'status': 'success',
            'message': 'Pelatihan model Naive Bayes berhasil diselesaikan!',
            'elapsed_time': elapsed,
            'vocab_size': info['vocab_size'],
            'train_size': info['train_size'],
            'test_size': info['test_size']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal melatih model: {str(e)}'}), 500

@training_bp.route('/reset', methods=['POST'])
def reset_model():
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(VECTORIZER_PATH):
            os.remove(VECTORIZER_PATH)
            
        train_model()
        info = get_training_info()
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil di-reset dan dilatih ulang dengan parameter standar.',
            'vocab_size': info['vocab_size']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal mereset model: {str(e)}'}), 500

@training_bp.route('/download-model')
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True, download_name='naive_bayes.pkl')
    return jsonify({'status': 'error', 'message': 'File model belum ada'}), 444
