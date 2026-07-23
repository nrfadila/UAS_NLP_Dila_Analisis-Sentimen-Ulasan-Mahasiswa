import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_pipeline
from src.training import load_dataset, load_model

def get_evaluation_metrics():
    df = load_dataset()
    model, vectorizer = load_model()
    
    if not model or not vectorizer:
        return None

    if 'processed' not in df.columns:
        df['processed'] = df['text'].apply(preprocess_pipeline)
        
    X = df['processed']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    
    labels = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5), facecolor='#FFFDF8')
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    
    ax.set_facecolor('#FFFDF8')
    plt.title('Confusion Matrix', fontsize=14, color='#4A403A', pad=15, fontweight='bold')
    plt.xlabel('Predicted Sentiment', fontsize=11, color='#4A403A')
    plt.ylabel('Actual Sentiment', fontsize=11, color='#4A403A')
    plt.xticks(color='#4A403A')
    plt.yticks(color='#4A403A')
    plt.tight_layout()
    
    os.makedirs(os.path.join('static', 'img'), exist_ok=True)
    img_path = os.path.join('static', 'img', 'confusion_matrix.png')
    plt.savefig(img_path, dpi=200, bbox_inches='tight', transparent=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'report_dict': report_dict,
        'report_text': report_text,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'matrix_image': f'data:image/png;base64,{img_base64}',
        'matrix_image_url': '/static/img/confusion_matrix.png'
    }
