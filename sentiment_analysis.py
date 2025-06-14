# COVID-19 Tweet Sentiment Analysis with LinearSVC (Fast & Accurate)
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data with efficient chunking for large files
def load_data(path):
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, usecols=['OriginalTweet', 'Sentiment'])
        except (UnicodeDecodeError, KeyError):
            continue
    raise ValueError(f"Failed to load {path}")

train_df = load_data('Corona_NLP_train.csv')
test_df = load_data('Corona_NLP_test.csv')

# Optimized text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Punctuation
    words = [lemmatizer.lemmatize(word) for word in text.split() 
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)

print("Preprocessing tweets...")
train_df['CleanedTweet'] = train_df['OriginalTweet'].apply(preprocess_text)
test_df['CleanedTweet'] = test_df['OriginalTweet'].apply(preprocess_text)

# Sentiment mapping
sentiment_mapping = {
    'Extremely Negative': 0,
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3,
    'Extremely Positive': 4
}
y_train = train_df['Sentiment'].map(sentiment_mapping)
y_test = test_df['Sentiment'].map(sentiment_mapping)

# Efficient TF-IDF with feature hashing
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True  # Use 1+log(tf)
)

X_train = tfidf.fit_transform(train_df['CleanedTweet'])
X_test = tfidf.transform(test_df['CleanedTweet'])

# LinearSVC model - Fast and accurate for text
model = LinearSVC(
    C=0.5,  # Regularization
    class_weight='balanced',  # Handle class imbalance
    max_iter=2000,  # Ensure convergence
    dual=False  # Faster when n_samples > n_features
)
print("Training LinearSVC model...")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=sentiment_mapping.keys()))

# Save model (smaller files than XGBoost)
joblib.dump(model, 'linearsvc_sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nModel saved (3-5x smaller than XGBoost)!")

# Sample predictions
sample_tweets = [
    "Vaccines are saving lives during this pandemic!",
    "I lost my job because of COVID lockdowns",
    "Just another day working from home",
    "The government failed us in pandemic response",
    "COVID test came back positive, feeling awful"
]

sample_features = tfidf.transform(sample_tweets)
sample_preds = model.predict(sample_features)
print("\nSample Predictions:")
for tweet, pred in zip(sample_tweets, sample_preds):
    print(f"{tweet[:70]}... -> {list(sentiment_mapping.keys())[pred]}")