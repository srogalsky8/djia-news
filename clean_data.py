import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import string
import numpy as np
from pywsd.utils import lemmatize_sentence

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
news = pd.read_csv("./data/djia_news.csv")
translator = str.maketrans('', '', string.punctuation)
combined = []

transformed = []
for i in range(0,len(news)): 
    headline = news['Headline'][i]
    no_punc = headline.translate(translator)
    # words = [word.lower() for word in no_punc.split()] # get the lowercase version of words
    words = lemmatize_sentence(no_punc) # lemmatize with context
    words = [word for word in words if word not in stopwords.words('english')] # remove stopwords
    # words = [lemmatizer.lemmatize(word) for word in words] # no context lemmatization
    # words = [ps.stem(word) for word in words] # no context stemming
    transformed.append(words)
    combined = combined + words

my_list_count = Counter(combined)
unique_list = [i for i in my_list_count]
unique_dict = {word: i for (i, word) in enumerate(unique_list)}

data = np.zeros((len(news), len(unique_list)))
for (i, headline) in enumerate(transformed):
    for word in headline:
        data[i, unique_dict[word]] += 1

X = data
y = np.array(news['Label'])