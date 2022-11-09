import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

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


# confusion matrix
predicted = classify.predict(x_testing)
confusion_result = confusion_matrix(y_testing, predicted)
print(f'Confusion matrix: \n{confusion_result}\n')

# comparison based on % accuracy
test_result = classify.score(x_testing, y_testing)
print(f'Comparison based on % accuracy: {test_result}\n')

# comparison based on sensitivity, specificity, and precision
true_positives = confusion_result[0][0]
false_positives = confusion_result[0][1]
true_negatives = confusion_result[1][1]
false_negatives = confusion_result[1][0]

# sensitivity
sensitivity = true_positives / (true_positives + false_negatives)
# specificity
specificity = true_negatives / (true_negatives + false_positives)
# precision
precision = true_positives / (true_positives + false_positives)

print(f'Sensitivity: {sensitivity}, Specificity: {specificity}, Precision: {precision}\n')

# calculate roc curve
y_score = classify.predict_proba(x_testing)[::,1] # same thing as calculating the probability estimates ie y_prob

fpr, tpr, thresholds = roc_curve(y_testing, y_score, pos_label=1)
print(f'ROC=> false positive rate: \n{fpr}\n, true positive rate: \n{tpr}\n, thresholds: \n{thresholds}\n')

# plot ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="red", lw=lw, label="ROC curve")
plt.plot([0,1], [0,1], color="black", lw=lw, linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title('Testing actual data vs testing predicted data ROC')
plt.legend(loc="lower right")
plt.show()
