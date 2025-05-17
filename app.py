# Spam Detection Application
# Complete, optimized version with enhanced error handling and functionality

import os
import sys
import pandas as pd
import numpy as np
import joblib
import string
import nltk
import warnings
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, jsonify, render_template_string
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class SpamDetector:
    def __init__(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.stopwords = None
        
    def download_nltk_resources(self):
        """Download necessary NLTK resources"""
        try:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {str(e)}")
            # Create a fallback list of common stopwords if download fails
            self.stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                                 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                                 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                                 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                                 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                                 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                                 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                                 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                                 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                                 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                                 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                                 'don', 'should', 'now'])
            
    def load_dataset(self):
        """Load the SMS spam dataset"""
        try:
            # First, try to load from the current directory
            if os.path.exists("spam.csv"):
                df = pd.read_csv("spam.csv", encoding='latin-1')
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                df = df[['label', 'text']]
                logger.info("Dataset loaded from spam.csv")
                return df
            
            # Second, try to load from SMSSpamCollection format
            elif os.path.exists("SMSSpamCollection"):
                df = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'text'])
                logger.info("Dataset loaded from SMSSpamCollection")
                return df
            
            # If neither exists, download from a URL
            else:
                logger.info("Downloading dataset from URL...")
                url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
                df = pd.read_csv(url, encoding='latin-1')
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                df = df[['label', 'text']]
                # Save it locally for future use
                df.to_csv("spam.csv", index=False)
                logger.info("Dataset downloaded and saved as spam.csv")
                return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            sys.exit(1)
            
    def preprocess_text(self, text):
        """Preprocess a text message"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Remove stopwords
        words = text.split()
        text = ' '.join([word for word in words if word not in self.stopwords])
        
        return text
    
    def preprocess_dataset(self, df):
        """Preprocess the entire dataset"""
        # Map labels to binary values
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        return df
    
    def train_model(self, df):
        """Train the Naive Bayes model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Create and fit the vectorizer
        vectorizer = CountVectorizer(min_df=2, max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train the model
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix:")
        logger.info(f"True Negative: {cm[0][0]}, False Positive: {cm[0][1]}")
        logger.info(f"False Negative: {cm[1][0]}, True Positive: {cm[1][1]}")
        
        # Save the model and vectorizer
        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Vectorizer saved to {self.vectorizer_path}")
        
        return model, vectorizer
    
    def load_or_train_model(self):
        """Load existing model or train a new one if not available"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info("Model and vectorizer loaded successfully")
                return self.model, self.vectorizer
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}. Training a new model...")
        
        # If model doesn't exist or failed to load, train a new one
        logger.info("Training a new model...")
        df = self.load_dataset()
        df = self.preprocess_dataset(df)
        self.model, self.vectorizer = self.train_model(df)
        return self.model, self.vectorizer
    
    def predict(self, message):
        """Predict if a message is spam or ham"""
        if not self.model or not self.vectorizer:
            logger.error("Model or vectorizer not loaded. Cannot make predictions.")
            return None
        
        processed_message = self.preprocess_text(message)
        message_vec = self.vectorizer.transform([processed_message])
        prediction = self.model.predict(message_vec)[0]
        probability = self.model.predict_proba(message_vec)[0][1]  # Probability of being spam
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'label': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': f"{probability*100:.2f}%" if prediction == 1 else f"{(1-probability)*100:.2f}%"
        }
    
    def create_flask_app(self):
        """Create a Flask app for spam detection"""
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Spam Message Detector</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f5f5f5;
                    }
                    .container {
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                        margin-bottom: 30px;
                    }
                    textarea {
                        width: 100%;
                        height: 150px;
                        padding: 10px;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        resize: vertical;
                        font-size: 16px;
                    }
                    button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 12px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 16px;
                        margin-top: 10px;
                        width: 100%;
                    }
                    button:hover {
                        background-color: #45a049;
                    }
                    .result-container {
                        margin-top: 20px;
                        padding: 20px;
                        border-radius: 4px;
                        display: none;
                    }
                    .spam {
                        background-color: #ffe6e6;
                        border: 1px solid #ff8080;
                    }
                    .ham {
                        background-color: #e6ffe6;
                        border: 1px solid #80ff80;
                    }
                    .result-text {
                        font-size: 24px;
                        font-weight: bold;
                        text-align: center;
                        margin-bottom: 10px;
                    }
                    .confidence {
                        text-align: center;
                        font-size: 16px;
                    }
                    .examples {
                        margin-top: 30px;
                        padding: 15px;
                        background-color: #e6f2ff;
                        border-radius: 4px;
                    }
                    .examples h2 {
                        margin-top: 0;
                        font-size: 18px;
                    }
                    .examples ul {
                        padding-left: 20px;
                    }
                    .footer {
                        margin-top: 30px;
                        text-align: center;
                        color: #666;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>EELU Project Spam Detector</h1>
                    <textarea id="message" placeholder="Enter your message here to check if it's spam..."></textarea>
                    <button onclick="checkSpam()">Check Message</button>
                    
                    <div id="result" class="result-container">
                        <div class="result-text" id="result-text"></div>
                        <div class="confidence" id="confidence"></div>
                    </div>
                    
                    <div class="examples">
                        <h2>Examples to try:</h2>
                        <ul>
                            <li>Congratulations! You've won a free iPhone. Click here to claim now!</li>
                            <li>Hey, are we still meeting for lunch at 12:30 today?</li>
                            <li>URGENT: Your bank account has been compromised. Call this number immediately.</li>
                            <li>Don't forget to pick up milk on your way home.</li>
                        </ul>
                    </div>
                    
                    <div class="footer">
                        This spam detector uses machine learning to classify messages.
                    </div>
                </div>
                
                <script>
                    function checkSpam() {
                        const message = document.getElementById('message').value;
                        if (!message.trim()) {
                            alert('Please enter a message to check');
                            return;
                        }
                        
                        fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: message })
                        })
                        .then(response => response.json())
                        .then(data => {
                            const result = document.getElementById('result');
                            const resultText = document.getElementById('result-text');
                            const confidence = document.getElementById('confidence');
                            
                            result.style.display = 'block';
                            if (data.prediction === 1) {
                                result.className = 'result-container spam';
                                resultText.textContent = '⚠️ SPAM DETECTED';
                                confidence.textContent = `Confidence: ${data.confidence}`;
                            } else {
                                result.className = 'result-container ham';
                                resultText.textContent = '✅ NOT SPAM';
                                confidence.textContent = `Confidence: ${data.confidence}`;
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('An error occurred while checking the message');
                        });
                    }
                </script>
            </body>
            </html>
            ''')

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.json
            message = data.get('message', '')
            result = self.predict(message)
            return jsonify(result)

        return app

# Main function
def main():
    # Initialize the spam detector
    spam_detector = SpamDetector()
    spam_detector.download_nltk_resources()
    spam_detector.load_or_train_model()
    
    # Create and run the Flask app
    app = spam_detector.create_flask_app()
    app.run(host='0.0.0.0', port=5000)

# Run the application
if __name__ == '__main__':
    main()
