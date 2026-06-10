import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("mail_data.csv")

df=pd.DataFrame(data)

"""Label Encoding"""

df.loc[df['Category'] == 'spam', 'Category',] = 0

df.loc[df['Category'] == 'ham', 'Category',] = 1

#separating data as text and label
x = df['Message']
y = df['Category']

"""Splitting the data into training data and test data"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


"""Feature Extraction"""

#transform the test data into feature vectors that can be used as input to the Logistic Regression

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

Y_train = y_train.astype('int')
Y_test  = y_test.astype('int')


"""Training Model"""

model = LogisticRegression()

model.fit(x_train_feature, Y_train)

"""Evaluating the trained model"""

predict_on_train_data = model.predict(x_train_feature)
acc1=accuracy_score(Y_train, predict_on_train_data)

print(acc1)

predict_on_test_data = model.predict(x_test_feature)
acc2=accuracy_score(Y_test, predict_on_test_data)

print(acc2)

joblib.dump(model, 'model.joblib')