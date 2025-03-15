# Spam Message Detector

A simple web application that detects whether a message is spam or legitimate using machine learning.

## Overview

The Spam Message Detector is a Flask-based web application with an intuitive HTML interface that allows users to input text messages and receive instant feedback on whether the message is likely to be spam or not. The application uses a pre-trained machine learning model to analyze and classify messages.

## Features

- Clean, user-friendly interface
- Real-time spam detection
- Visual feedback with color-coded results
- Responsive design for both desktop and mobile devices

## Requirements

- Python 3.6+
- Flask
- NLTK
- scikit-learn
- joblib

## Installation

1. Clone this repository or download the source code.
2. Install the required dependencies:
   ```
   pip install flask nltk scikit-learn joblib
   ```
3. Ensure you have the pre-trained model files in the same directory:
   - `spam_model.pkl`: The trained spam detection model
   - `vectorizer.pkl`: The fitted text vectorizer

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000/`
3. Enter a message in the text area and click "Check Message"
4. View the result: green for legitimate messages, red for spam

## Technical Details

- **Backend**: Flask server that processes requests and serves the application
- **Text Processing**: Uses NLTK for preprocessing (removing stopwords and punctuation)
- **Classification**: Pre-trained machine learning model for spam detection
- **Frontend**: HTML, CSS, and JavaScript for a responsive user interface

## Example

1. Enter a message like "Congratulations! You've won a free iPhone. Click here to claim your prize!" 
2. The application will likely classify this as spam
3. Enter a message like "Can we meet tomorrow at 3 PM to discuss the project?" 
4. The application will likely classify this as legitimate

## Copyright

© 2025 Nour Sallam. All rights reserved.

This software is provided for educational and demonstrational purposes. Unauthorized reproduction or distribution of this software, or any portion of it, may result in civil and criminal penalties.
