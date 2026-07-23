import os
from flask import Flask, render_template
from src.training import MODEL_PATH, VECTORIZER_PATH, train_model
from routes.dashboard import dashboard_bp
from routes.dataset import dataset_bp
from routes.training import training_bp
from routes.prediction import prediction_bp
from routes.evaluation import evaluation_bp
from routes.about import about_bp

app = Flask(__name__)

app.register_blueprint(dashboard_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(training_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(evaluation_bp)
app.register_blueprint(about_bp)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

def init_app():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_model()

init_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
