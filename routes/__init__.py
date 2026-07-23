from flask import Blueprint

dashboard_bp = Blueprint('dashboard', __name__)
dataset_bp = Blueprint('dataset', __name__)
training_bp = Blueprint('training', __name__)
prediction_bp = Blueprint('prediction', __name__)
evaluation_bp = Blueprint('evaluation', __name__)
about_bp = Blueprint('about', __name__)
