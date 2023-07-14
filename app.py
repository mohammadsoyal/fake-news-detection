from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='./templates', static_folder='./static')

train_df = pd.read_csv(r'C:\Users\AITHA\OneDrive\Desktop\fake_news\train.csv')
label_train = train_df['Label']
loaded_model = pickle.load(open('final.pkl', 'rb'))
tfidf_v = pickle.load(open('vectorized.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def predict_news_label(news, tfidf_v, loaded_model, stop_words, lemmatizer):
    corpus = []
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for word in review:
        if word not in stop_words:
            corpus.append(lemmatizer.lemmatize(word))
    input_data = [' '.join(corpus)]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    if prediction[0] == 0:
        return "Prediction of the News: Looking Fake⚠ News📰"
    else:
        return "Prediction of the News: Looking Real News📰"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['text']
        prediction = predict_news_label(message, tfidf_v, loaded_model, stop_words, lemmatizer)
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('prediction.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
