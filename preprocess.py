import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MinMaxScaler #fixed import
import string
import numpy as np
from pywsd.utils import lemmatize_sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

np.random.seed(10)

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
news = pd.read_csv("../data/djia_news.csv")
translator = str.maketrans('', '', string.punctuation)
additional_stopwords = ['’', '–', '‘', '“', '”',
'u', 'v', 'f', '—', '…', 'w', 'x', 'c', 'g', 'q', '•', '»', 'b', 'n']
combined = []


X_tr = news['Headline'].to_numpy()
y_tr = news['Label'].to_numpy()

sentiment = SentimentIntensityAnalyzer()

# TODO: remove duplicates
transformed = []
neg = np.zeros(len(news))
neu = np.zeros(len(news))
pos = np.zeros(len(news))
compound = np.zeros(len(news))
for i in range(0,len(news)): 
    headline = news['Headline'][i]
    sent = sentiment.polarity_scores(headline)
    neg[i] = sent['neg']
    neu[i] = sent['neu']
    pos[i] = sent['pos']
    compound[i] = sent['compound']
    no_punc = headline.translate(translator)
    # words = [word.lower() for word in no_punc.split()] # get the lowercase version of words
    words = lemmatize_sentence(no_punc) # lemmatize with context
    words = [word for word in words if word not in stopwords.words('english') and word not in additional_stopwords] # remove stopwords
    # words = [lemmatizer.lemmatize(word) for word in words] # no context lemmatization
    # words = [ps.stem(word) for word in words] # no context stemming
    transformed.append(' '.join(words))
    combined = combined + words

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


scaler = MinMaxScaler()

neg = neg.reshape((len(neg),1))
neu = neu.reshape((len(neu),1))
pos = pos.reshape((len(pos),1))
compound = compound.reshape((len(compound),1))

X_tr = counts_tr.toarray()
X_tr = np.append(X_tr, neg[0:len(X_tr), :], axis=1)
X_tr = np.append(X_tr, neu[0:len(X_tr), :], axis=1)
X_tr = np.append(X_tr, pos[0:len(X_tr), :], axis=1)
X_tr = np.append(X_tr, compound[0:len(X_tr), :], axis=1)
X_te = counts_te.toarray()
X_te = np.append(X_te, neg[len(X_tr):, :], axis=1)
X_te = np.append(X_te, neu[len(X_tr):, :], axis=1)
X_te = np.append(X_te, pos[len(X_tr):, :], axis=1)
X_te = np.append(X_te, compound[len(X_tr):, :], axis=1)

# X_tr = np.append(neg[0:len(X_tr), :], neu[0:len(X_tr), :], axis=1)
# X_tr = np.append(X_tr, pos[0:len(X_tr), :], axis=1)
# X_tr = np.append(X_tr, compound[0:len(X_tr), :], axis=1)
# X_te = np.append(neg[len(X_tr):, :], neu[len(X_tr):, :], axis=1)
# X_te = np.append(X_te, pos[len(X_tr):, :], axis=1)
# X_te = np.append(X_te, compound[len(X_tr):, :], axis=1)