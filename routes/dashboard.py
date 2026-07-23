from flask import render_template, jsonify
from routes import dashboard_bp
from src.training import load_dataset, get_training_info
from src.evaluation import get_evaluation_metrics

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
def dashboard():
    df = load_dataset()
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    
    total_data = len(df)
    pos_count = sentiment_counts.get('positif', 0)
    neg_count = sentiment_counts.get('negatif', 0)
    net_count = sentiment_counts.get('netral', 0)
    
    total_words = sum(df['text'].apply(lambda x: len(str(x).split())))
    
    metrics = get_evaluation_metrics()
    accuracy = metrics['accuracy'] if metrics else 0.0
    
    stats = {
        'total_dataset': total_data,
        'positif': pos_count,
        'negatif': neg_count,
        'netral': net_count,
        'total_words': total_words,
        'model_name': 'Multinomial Naive Bayes',
        'accuracy': accuracy
    }
    
    return render_template('dashboard.html', stats=stats, active_page='dashboard')

@dashboard_bp.route('/api/dashboard-stats')
def dashboard_stats_api():
    df = load_dataset()
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    metrics = get_evaluation_metrics()
    
    total_words = sum(df['text'].apply(lambda x: len(str(x).split())))
    
    return jsonify({
        'total_dataset': len(df),
        'positif': sentiment_counts.get('positif', 0),
        'negatif': sentiment_counts.get('negatif', 0),
        'netral': sentiment_counts.get('netral', 0),
        'total_words': total_words,
        'model_name': 'Multinomial Naive Bayes',
        'accuracy': metrics['accuracy'] if metrics else 0.0
    })
