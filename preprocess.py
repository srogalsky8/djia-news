import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import string
import numpy as np
from pywsd.utils import lemmatize_sentence
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
news = pd.read_csv("./data/djia_news.csv")
translator = str.maketrans('', '', string.punctuation)
additional_stopwords = ['’', '–', '‘', '“', '”',
'u', 'v', 'f', '—', '…', 'w', 'x', 'c', 'g', 'q', '•', '»', 'b', 'n']
combined = []


X_tr = news['Headline'].to_numpy()
y_tr = news['Label'].to_numpy()

print(np.size(X_tr))

# TODO: remove duplicates

transformed = []
for i in range(0,len(news)): 
    headline = news['Headline'][i]
    no_punc = headline.translate(translator)
    # words = [word.lower() for word in no_punc.split()] # get the lowercase version of words
    words = lemmatize_sentence(no_punc) # lemmatize with context
    words = [word for word in words if word not in stopwords.words('english') and word not in additional_stopwords] # remove stopwords
    # words = [lemmatizer.lemmatize(word) for word in words] # no context lemmatization
    # words = [ps.stem(word) for word in words] # no context stemming
    transformed.append(' '.join(words))
    combined = combined + words

# my_list_count = Counter(combined)
# unique_list = [i for i in my_list_count]
# unique_dict = {word: i for (i, word) in enumerate(unique_list)}

# data = np.zeros((len(news), len(unique_list)))
# for (i, headline) in enumerate(transformed):
#     for word in headline:
#         data[i, unique_dict[word]] += 1
# X = data
# y = np.array(news['Label'])

X_raw = np.array(transformed)
y = news['Label'].to_numpy()

order = np.random.permutation(len(X_raw))
X_rand = X_raw[order]
y_rand = y[order]
cutoff = len(X_rand) * 4 // 5
X_tr = X_rand[:cutoff]
y_tr = y_rand[:cutoff]
X_te = X_rand[cutoff:]
y_te = y_rand[cutoff:]

vectorizer = TfidfVectorizer()
# create tf-idf matrix
counts_tr = vectorizer.fit_transform(X_tr)
counts_te = vectorizer.transform(X_te)

X_tr = counts_tr.toarray()
X_te = counts_te.toarray()