import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


df = pd.read_csv('reviews.csv', on_bad_lines='skip', sep=',', quotechar='"')


print("Dosyadaki sütunlar:", df.columns.tolist())

if 'review_text' not in df.columns:
    df.columns = ['id', 'review_text']

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]

    return " ".join(cleaned_words)


def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


df['cleaned_review'] = df['review_text'].apply(clean_text)
df['sentiment'] = df['cleaned_review'].apply(get_sentiment)


print("Sentiment Statistics")
print(df['sentiment'].value_counts())

print("\nPercentage Distribution (%)")
print((df['sentiment'].value_counts(normalize=True) * 100).round(2))