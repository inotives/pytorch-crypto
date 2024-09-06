import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def text_preprocessing(text):
    '''Preprocessing for text related dataset'''
    
    # Initialize the stemmer, lemmatizer, and stopwords list
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Convert text to lowercase
    processed_text = text.lower()
    
    # Remove digits and special characters
    processed_text = re.sub(r'\d+', '', processed_text)
    processed_text = re.sub(r'\W+', ' ', processed_text)

    # Tokenize the text into words
    words = processed_text.split()

    # Remove stopwords, apply stemming, and lemmatization
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]

    # Rejoin the words into a single string
    processed_text = ' '.join(words)
    
    return processed_text
