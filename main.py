#%% md
# # Naive Bayes classifer
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fontTools.varLib import load_designspace
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#%%
df = pd.read_csv('emails.csv')
#%%
df.head()
#%%
len(df.columns)
#%%
df.shape[0]
#%%
target = df.iloc[:,-1]
target.head() # so the name for tha target column is Predictin 1 for spam and 0 for not spam
#%%
df['Prediction'].value_counts() # 0 is not spam and 1 is spam
#%%
total_spam = df['Prediction'].value_counts()[1]
total_ham = df['Prediction'].value_counts()[0]
#%%
print(f"Total spam emails: {total_spam}")
print(f"Total ham emails: {total_ham}")
#%%
df.columns
#%%
tot_emails = len(df.index)
tot_emails
#%%
p_of_spam = round(df['Prediction'].value_counts()[1] / tot_emails,2)
p_of_ham = round(df['Prediction'].value_counts()[0] / tot_emails,2)
#%%
print(f"Probability of spam: {p_of_spam}")
print(f"Probability of ham: {p_of_ham}")
#%%
spam_row = df[df['Prediction'] == 1]
spam_row.head()

#%%
X = df.drop(columns=['Email No.','Prediction'])
X
#%%
y = df['Prediction']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
X_train

#%%
from sklearn.naive_bayes import MultinomialNB
classifer = MultinomialNB()
classifer.fit(X_train, y_train)
#%%
y_pred = classifer.predict(X_test)
#%%
print(classification_report(y_test, y_pred))
#%%
print(accuracy_score(y_test, y_pred))
#%%
import joblib
from sklearn.feature_extraction.text import CountVectorizer
#%%
vocab = list(X.columns)
vocab
#%%
from sklearn.base import TransformerMixin,BaseEstimator
class ToDataFrameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X.toarray(), columns=self.columns)
#%%
vectorizer = CountVectorizer(vocabulary=vocab)
to_df = ToDataFrameTransformer(columns=vocab)
#%%
X_new = vectorizer.transform([sample_email])
X_new
#%%
X_new_df = pd.DataFrame(X_new.toarray(), columns=vocab)
X_new_df
#%%
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(vectorizer,to_df,classifer)
joblib.dump(pipeline, 'pipeline.pkl')
#%%
loaded_pipeline = joblib.load('pipeline.pkl')
#%%
def predict_email(email):
    prediction = loaded_pipeline.predict([email])
    if prediction[0] == 1:
        return "The email is spam."
    else:
        return "The email is not spam."
#%%
spam_sample_email = """ Subject: 🎁 Congratulations! You've won a $1000 Gift Card!

Dear User,

You have been selected as the winner of a $1000 Walmart gift card!
To claim your prize, simply click the link below and complete the short survey.

👉 [Click here to claim your reward](https://giveawaywinner.com)

Hurry! This offer is valid for the next 24 hours only.

Best regards,
Rewards Team
 """
#%%
predict_email(spam_sample_email)
#%%
non_spam_sample_email="""Subject: Meeting Reminder

Hi Nadin,

This is a friendly reminder about our scheduled meeting tomorrow at 10:00 AM in the conference room. Please let me know if you have any questions or need to reschedule.

Looking forward to seeing you there!

Best regards,
[Your Name]"""
#%%
predict_email(non_spam_sample_email)