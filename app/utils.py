import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    """
    Tokenize the input text by converting to lowercase, removing punctuation, 
    tokenizing the text, and removing stopwords.

    Parameters:
    text (str): The input text to tokenize.

    Returns:
    list: A list of tokens with punctuation removed and stopwords filtered out.
    """
    # convert the input text to lowercase
    text = text.lower()

    # clear the punctuation
    text = re.sub(r"[^0-9a-zA-Z]", " ", text)

    # tokenize text
    words = word_tokenize(text)

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Apply lemmatization to each word (token) in the tokenized list, ignoring stopwords
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words("english")]

    return tokens
