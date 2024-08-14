import pandas as pd
import string
import re
import pickle
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Function to clean and preprocess the data
def preprocess_data(df):
    df = df[['text', 'sentiment']]
    df['text'] = df['text'].fillna(' ')
    df['sentiment'].replace(['negative', 'neutral', 'positive'], [-1, 0, 1], inplace=True)

    punctuations = string.punctuation

    def clean_punctuations(tweet):
        translator = str.maketrans('', '', punctuations)
        return tweet.translate(translator)

    df['text'] = df['text'].apply(lambda i: clean_punctuations(i))

    def remove_stop_word(text):
        words = remove_stopwords(text)
        return words

    df['text'] = df['text'].apply(lambda i: remove_stop_word(i))

    df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df['text']]
    porter_stemmer = PorterStemmer()
    df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in df['tokenized_text']]
    df['stemmed_tokens'] = [','.join(map(str, i)) for i in df['stemmed_tokens']]

    def remove_comma(text):
        return re.sub(',', ' ', text)
    
    df['stemmed_tokens'] = df['stemmed_tokens'].apply(lambda i: remove_comma(i))
    return df

# Load and preprocess the data
df = pd.read_csv(r"C:\Users\91986\Desktop\sentiment\train.csv (1).zip", encoding='unicode_escape')
df = preprocess_data(df)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df['stemmed_tokens'], df['sentiment'], test_size=0.3, random_state=0)

# Vectorize the text
cv = CountVectorizer()
cv.fit(x_train)
x_train = cv.transform(x_train)
x_test = cv.transform(x_test)

# Train the Logistic Regression model
LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LRmodel.fit(x_train, y_train)

# Save the trained model and vectorizer using pickle
with open('LRmodel.pkl', 'wb') as model_file:
    pickle.dump(LRmodel, model_file)

with open('CountVectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model and vectorizer have been saved as pickle files.")
