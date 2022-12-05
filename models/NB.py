import sys,os
sys.path.append(os.path.realpath('..'))
from preprocess import X_tr, y_tr, X_te, y_te
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

mnb = MultinomialNB()
mnb.fit(X_tr, y_tr)
pred = mnb.predict(X_te)
print(sum(pred == 1))
print(sum(pred == 0))

accuracy = 1 - sum(y_te != pred)/len(y_te)
f1 = metrics.f1_score(y_te, pred, average='macro')

print(f'accuracy: {accuracy}')
print(f'f1 score: {f1}')