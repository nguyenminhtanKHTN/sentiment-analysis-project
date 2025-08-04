from flask import Flask, request, jsonify
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình và vectorizer
model = joblib.load(f'{os.path.dirname(CURRENT_DIR)}/models/sentiment_model.pkl')
vectorizer = joblib.load(f'{os.path.dirname(CURRENT_DIR)}/models/tfidf_vectorizer.pkl')

# Tải tài nguyên NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Xóa HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Xóa ký tự đặc biệt
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Route cho API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        review = data.get('review', '')
        model_name = data.get('model', 'logistic_regression')
        model_path = {
            'logistic_regression': f'{os.path.dirname(CURRENT_DIR)}/models/logistic_regression_model.pkl',
            'naive_bayes': f'{os.path.dirname(CURRENT_DIR)}/models/naive_bayes_model.pkl',
            'linear_svc': f'{os.path.dirname(CURRENT_DIR)}/models/linear_svc_model.pkl',
        }
        
        if model_name not in model_path:
            return jsonify({'error': 'Invalid model specified'}), 400
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400
        
        model = joblib.load(model_path[model_name])
        
        # Tiền xử lý và vector hóa
        cleaned_review = preprocess_text(review)
        X_new = vectorizer.transform([cleaned_review])
        
        # Dự đoán
        prediction = model.predict(X_new)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        # Trả về kết quả
        return jsonify({
            'review': review,
            'cleaned_review': cleaned_review,
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        model_name = data.get('model', 'logistic_regression')
        
        model_path = {
            'logistic_regression': f'{os.path.dirname(CURRENT_DIR)}/models/logistic_regression_model.pkl',
            'naive_bayes': f'{os.path.dirname(CURRENT_DIR)}/models/naive_bayes_model.pkl',
            'linear_svc': f'{os.path.dirname(CURRENT_DIR)}/models/linear_svc_model.pkl',
        }
        
        if model_name not in model_path:
            return jsonify({'error': 'Invalid model specified'}), 400
        
        model = joblib.load(model_path[model_name])
        
        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400
        
        cleaned_reviews = [preprocess_text(review) for review in reviews]
        X_new = vectorizer.transform(cleaned_reviews)
        
        predictions = model.predict(X_new)
        sentiments = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
        
        results = [{'review': review, 'cleaned_review': cleaned_review, 'sentiment': sentiment}
                   for review, cleaned_review, sentiment in zip(reviews, cleaned_reviews, sentiments)]
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)