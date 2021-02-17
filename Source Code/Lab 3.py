import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('glass.csv')



x = df.drop('Type', axis=1)
y = df['Type']

print(y.value_counts())

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=.25, random_state=0)


svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
print("svm accuracy is:", acc_svc)

report = classification_report(Y_test, Y_pred)
print(report)

