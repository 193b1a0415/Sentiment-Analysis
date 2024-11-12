import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

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

# Load dataset
df = pd.read_csv('d:/twitter_training.csv')

# Drop rows with missing tweets or sentiments
df = df.dropna(subset=['tweet', 'sentiment'])

# Preprocess tweets
df['cleaned_tweet'] = df['tweet'].apply(preprocess_tweet)

# Remove rows where the cleaned tweet is empty
df = df[df['cleaned_tweet'].str.strip() != '']

# Map sentiments to numerical values
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Drop rows with unmapped sentiments (if any)
df = df.dropna(subset=['sentiment'])
df['sentiment'] = df['sentiment'].astype(int)

# Transform tweets into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000)
X = tfidf_vectorizer.fit_transform(df['cleaned_tweet'])
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Save the trained model and TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)
