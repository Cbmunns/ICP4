import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Read the file in
df = pd.read_csv('glass.csv')
# Place all the data points in one df and all the Types in another
x = df.drop('Type', axis=1)
y = df['Type']
# This is to see the amount of different values
print(y.value_counts())
# Split sets up for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=.25, random_state=0)
# Start, Fit, and Predict the data
svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
# Tally up the score on the test data using the model
acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
print("svm accuracy is:", acc_svc)
# Bring up the report based on the predictions and true results
report = classification_report(Y_test, Y_pred)
print(report)

