import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import string
import numpy as np

news = pd.read_csv("./data/djia_news.csv")
translator = str.maketrans('', '', string.punctuation)
combined = []
for i in range(0,len(news)): 
    my_headline = news['Headline'][i]
    no_punc = my_headline.translate(translator)
    combined = combined + no_punc.split()

filtered_words = [word for word in combined if word not in stopwords.words('english')]
my_list_count = Counter(filtered_words)
unique_list = [i for i in my_list_count]
unique_dict = {word: i for (i, word) in enumerate(unique_list)}

data = np.zeros((len(news), len(unique_list)))
for (i, headline) in enumerate(news['Headline']):
    words = headline.translate(translator).split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    for word in filtered_words:
        data[i, unique_dict[word]] += 1

print(data)