import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


df = pd.read_csv('glass.csv')

x = df.drop('Type', axis=1)
y = df['Type']

print(y.value_counts())

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.1, random_state=70)

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred = nb.predict(X_test)

acc_NB = round(nb.score(X_test, Y_test) * 100, 2)
print("nb accuracy is:", acc_NB)

report = classification_report(Y_test, Y_pred)
print(report)

