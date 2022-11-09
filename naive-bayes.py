import re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

'''
Sklearn Multinomial Naive Bayes
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
'''

'''
Remove any unnecessary punctuation, extra spacing, and converting all words to lowercasing
'''
def message_preprocessing(data_set):
    data_set['text'] = data_set['text'].str.replace(r'\W', ' ', regex=True)
    data_set['text'] = data_set['text'].str.lower()
    removed_extra_whitespace = []

    for message in data_set['text']:
        cleaned_up = re.sub(' +', ' ', message)
        removed_extra_whitespace.append(cleaned_up)

    return pd.Series(removed_extra_whitespace)

email_spam = pd.read_csv('./emails.csv')

df_text = message_preprocessing(email_spam)
df_spam = email_spam['spam']

count_vectorizer = CountVectorizer()
df_text_count = count_vectorizer.fit_transform(df_text)

x_training, x_testing, y_training, y_testing = train_test_split(df_text_count, df_spam, test_size=0.2, random_state=42)

classify = MultinomialNB()
classify.fit(x_training, y_training)

result = classify.score(x_testing, y_testing)
