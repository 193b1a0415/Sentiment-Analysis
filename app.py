from flask import Flask, request, jsonify, render_template, redirect, url_for
import mysql.connector
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    punctuations_to_remove = "!\"#$%&'*+,./:;<=>?@\\^_`{|}~"
    remove_table = str.maketrans('', '', punctuations_to_remove)
    tweet = tweet.translate(remove_table)
    tweet = tweet.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_tweet)

# Database connection
con=mysql.connector.connect(user="root",password="",database="sentiment")
c=con.cursor()


@app.route("/")
def root():
    return render_template('index.html')

@app.route("/index")
def home():
    return render_template("index.html")

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    processed_tweet = preprocess_tweet(tweet)
    tfidf_tweet = tfidf_vectorizer.transform([processed_tweet])
    prediction = knn_model.predict(tfidf_tweet)[0]
    
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[prediction]
    
@app.route("/tweetsDB", methods=['POST'])
def tweetsDB():
    tweet = request.form['tweet']
    processed_tweet = preprocess_tweet(tweet)
    tfidf_tweet = tfidf_vectorizer.transform([processed_tweet])
    prediction = knn_model.predict(tfidf_tweet)[0]
    
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[prediction]

    # Insert into the database
    query = "INSERT INTO tweets (tweet, sentiment) VALUES (%s, %s)"
    values = (tweet, sentiment)
    c.execute(query, values)
    con.commit()
    return render_template("index.html", sentiment=sentiment) 

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
