import sys
import flask
import pandas as pd
import numpy as np
import sklearn
import joblib
import nltk
from flask import render_template
from routes import about_bp

@about_bp.route('/about')
def about_page():
    versions = {
        'Python': sys.version.split()[0],
        'Flask': flask.__version__,
        'Pandas': pd.__version__,
        'NumPy': np.__version__,
        'Scikit-Learn': sklearn.__version__,
        'Joblib': joblib.__version__,
        'NLTK': nltk.__version__
    }
    return render_template('about.html', versions=versions, active_page='about')
