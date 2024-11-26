from flask import Flask, request, jsonify, render_template
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if not already present
nltk.download('stopwords')

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('model.pkl')  # Your pre-trained model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Vectorizer for text preprocessing

# Initialize the stemmer
port_stem = PorterStemmer()

# Define the stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    stemmed_content = stemmed_content.lower()            # Convert to lowercase
    stemmed_content = stemmed_content.split()            # Split into words
    # Remove stopwords and apply stemming
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    data = request.json
    content = data.get('content', '')

    # Preprocess and vectorize input
    stemmed_content = stemming(content)
    vectorized_content = vectorizer.transform([stemmed_content])

    # Predict using the pre-trained model
    prediction = model.predict(vectorized_content)[0]  # Single prediction value
    confidence_scores = model.predict_proba(vectorized_content)  # Get confidence scores
    confidence = confidence_scores[0][1] if prediction == 1 else confidence_scores[0][0]

    result = 'Fake' if prediction == 1 else 'Real'

    return jsonify({'prediction': result, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
