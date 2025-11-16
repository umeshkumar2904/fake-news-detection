# Web-based Fake News Detection Application
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import warnings
import logging
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.accuracy = 0.0
        self.model_info = {
            'trained': False,
            'accuracy': 0.0,
            'training_samples': 0,
            'test_samples': 0,
            'last_trained': None
        }
        
    def load_and_prepare_data(self, sample_size=8000):
        """Load and prepare the training data"""
        try:
            logger.info("Loading datasets...")
            
            # Check if files exist
            if not os.path.exists("True.csv") or not os.path.exists("Fake.csv"):
                raise FileNotFoundError("Dataset files not found. Please ensure True.csv and Fake.csv are in the current directory.")
            
            # Load datasets
            true_df = pd.read_csv("True.csv")
            fake_df = pd.read_csv("Fake.csv")
            
            logger.info(f"Loaded {len(true_df)} real articles and {len(fake_df)} fake articles")
            
            # Add labels
            true_df['label'] = 'REAL'
            fake_df['label'] = 'FAKE'
            
            # Sample data for balanced training
            true_sample_size = min(sample_size//2, len(true_df))
            fake_sample_size = min(sample_size//2, len(fake_df))
            
            true_sample = true_df.sample(true_sample_size, random_state=42)
            fake_sample = fake_df.sample(fake_sample_size, random_state=42)
            
            # Combine and shuffle
            data = pd.concat([true_sample, fake_sample], ignore_index=True)
            data = data.sample(frac=1).reset_index(drop=True)
            
            logger.info(f"Using {len(true_sample)} real and {len(fake_sample)} fake articles for training")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_model(self):
        """Train the fake news detection model"""
        try:
            # Load and prepare data
            sample_data = self.load_and_prepare_data(sample_size=8000)
            X = sample_data['text']
            y = sample_data['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Vectorize text with enhanced parameters
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_df=0.7,
                min_df=2,
                max_features=5000,
                ngram_range=(1, 2)  # Include bigrams for better context
            )
            tfidf_train = self.vectorizer.fit_transform(X_train)
            tfidf_test = self.vectorizer.transform(X_test)
            
            # Train model with regularization
            self.model = PassiveAggressiveClassifier(max_iter=100, random_state=42, C=0.5)
            self.model.fit(tfidf_train, y_train)
            
            # Test accuracy
            y_pred = self.model.predict(tfidf_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Update model info
            self.model_info = {
                'trained': True,
                'accuracy': self.accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Model training completed! Accuracy: {self.accuracy:.3f}")
            
            return True, f"Model trained successfully! Accuracy: {self.accuracy:.1%}"
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False, f"Training failed: {str(e)}"
    
    def predict_news(self, news_text):
        """Predict if news is real or fake"""
        if not self.is_trained:
            return None, "Model not trained yet!"
        
        if not news_text or len(news_text.strip()) < 20:
            return None, "Please enter a longer news text (at least 20 characters)."
        
        try:
            logger.info(f"Analyzing text of length: {len(news_text)}")
            
            # Vectorize input
            vectorized_input = self.vectorizer.transform([news_text])
            prediction = self.model.predict(vectorized_input)[0]
            
            # Get confidence score
            try:
                decision_scores = self.model.decision_function(vectorized_input)
                confidence = abs(decision_scores[0])
            except:
                confidence = 0.0
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None, f"Analysis failed: {str(e)}"

# Initialize detector
detector = FakeNewsDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Train the model"""
    success, message = detector.train_model()
    return jsonify({'success': success, 'message': message, 'model_info': detector.model_info})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict news"""
    try:
        data = request.get_json()
        
        if not data or 'news_text' not in data:
            return jsonify({'success': False, 'error': 'No text provided. Please provide news_text in the request body.'}), 400
        
        news_text = data['news_text'].strip()
        
        if not news_text:
            return jsonify({'success': False, 'error': 'Empty text provided. Please provide valid news text.'}), 400
        
        prediction, confidence = detector.predict_news(news_text)
        
        if prediction is None:
            return jsonify({'success': False, 'error': confidence}), 400
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 3),
            'news_length': len(news_text),
            'word_count': len(news_text.split())
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'success': False, 'error': f'Request failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get model status"""
    return jsonify({
        'model_info': detector.model_info,
        'status': 'ready' if detector.is_trained else 'not_trained'
    })

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Fake News Detection API',
        'version': '2.0',
        'endpoints': {
            'predict': '/predict (POST) - Analyze news text',
            'train': '/train (POST) - Retrain the model',
            'status': '/status (GET) - Get model status'
        },
        'model_info': detector.model_info
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Train model on startup
        logger.info("Starting Fake News Detection Web Application")
        print("🤖 Training model on startup...")
        success, message = detector.train_model()
        print(f"✅ {message}")
        
        # Run app
        logger.info("🌐 Starting web server on http://0.0.0.0:5000")
        print(f"📊 Model accuracy: {detector.accuracy:.1%}")
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise