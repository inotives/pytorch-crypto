import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# custom methods
import utils.text_preprocessing as pretext

def perform_sa_v1(data, content_col):
    """Ver:1 sentiment analysis with unsupervised model with just VADER"""

    # Drop rows with missing content
    data = data.dropna(subset=[content_col])

    # Apply preprocessing to content
    data['processed_content'] = data[content_col].apply(pretext.text_preprocessing)

    # Initialize VADER SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Function to apply VADER and extract sentiment scores
    def get_vader_sentiment(text):
        return sid.polarity_scores(text)
    
    # Apply sentiment analysis
    data['sentiment_scores'] = data['processed_content'].apply(get_vader_sentiment)

    # Separate compound scores for easier interpretation
    data['compound_score'] = data['sentiment_scores'].apply(lambda x: x['compound'])

    # Define sentiment based on the compound score
    # Compound score ranges from -1 (most negative) to 1 (most positive)
    def categorize_sentiment(compound_score):
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    # Apply categorization
    data['predicted_sentiment'] = data['compound_score'].apply(categorize_sentiment)

    # Display the result
    print(data[['processed_content', 'compound_score', 'predicted_sentiment']])
    
    return data[['uniq_key', 'compound_score', 'predicted_sentiment']]



def perform_sa_v2(data, content_col):
    """Ver:2 unsupervised sentiment analysis with TFIDF and VADER"""

    data = data.dropna(subset=[content_col])

    # Apply preprocessing to content
    data['processed_content'] = data[content_col].apply(pretext.text_preprocessing)

    
    sia = SentimentIntensityAnalyzer() # Initialize VADER sentiment analyzer
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3)) # Vectorize using TfidfVectorizer
    X = tfidf.fit_transform(data['processed_content'])

    # Convert TF-IDF matrix to DataFrame
    X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

    # Apply VADER sentiment analysis
    data['vader_sentiment_score'] = data['processed_content'].apply(lambda x: sia.polarity_scores(x)['compound'])

    def categorize_sentiment(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    data['vader_sentiment_label'] = data['vader_sentiment_score'].apply(categorize_sentiment)

    final_df = data[['uniq_key', 'processed_content', 'vader_sentiment_score', 'vader_sentiment_label']]
    print (final_df)

    return final_df

    