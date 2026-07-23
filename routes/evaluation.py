import os
from flask import render_template, jsonify, send_file, Response
from routes import evaluation_bp
from src.evaluation import get_evaluation_metrics

@evaluation_bp.route('/evaluation')
def evaluation_page():
    metrics = get_evaluation_metrics()
    return render_template('evaluation.html', metrics=metrics, active_page='evaluation')

@evaluation_bp.route('/download-report')
def download_report():
    metrics = get_evaluation_metrics()
    if not metrics:
        return jsonify({'status': 'error', 'message': 'Evaluasi tidak tersedia'}), 500
        
    content = f"""=== LAPORAN EVALUASI MODEL SENTIMEN NAIVE BAYES ===

Akurasi    : {metrics['accuracy']}%
Presisi    : {metrics['precision']}%
Recall     : {metrics['recall']}%
F1-Score   : {metrics['f1_score']}%

--- CLASSIFICATION REPORT ---
{metrics['report_text']}

--- CONFUSION MATRIX ---
Labels: {metrics['labels']}
Matrix:
{metrics['confusion_matrix']}
"""
    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=evaluation_report.txt"}
    )

@evaluation_bp.route('/download-confusion-matrix')
def download_confusion_matrix():
    path = os.path.join('static', 'img', 'confusion_matrix.png')
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='confusion_matrix.png')
    return jsonify({'status': 'error', 'message': 'Gambar confusion matrix tidak ditemukan'}), 404

@evaluation_bp.route('/download-classification-report')
def download_classification_report():
    metrics = get_evaluation_metrics()
    if not metrics:
        return jsonify({'status': 'error', 'message': 'Evaluasi tidak tersedia'}), 500
    
    return Response(
        metrics['report_text'],
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=classification_report.txt"}
    )
