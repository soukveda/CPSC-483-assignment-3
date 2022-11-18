import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

    
df = pd.read_csv('emails.csv')


X = df["text"]
y = df["spam"]

count_vectorizer = CountVectorizer(ngram_range = (1, 1), stop_words = 'english', max_df = 0.7, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = 0.2)
count_vectorizer.fit(X)
X_train_count = count_vectorizer.transform(X_train)
X_test_count =  count_vectorizer.transform(X_test)

for n in range(1, 7, 2):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train_count, y_train)
    y_predictions = knn.predict(X_test_count)

    print(f'Accuracy Score: {metrics.accuracy_score(y_test,y_predictions) * 100}')
    print(f'Confusion Matrix: {metrics.confusion_matrix(y_test,y_predictions)}')