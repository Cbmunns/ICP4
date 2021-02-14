import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis=1).copy()
print(train_df[train_df.isnull().any(axis=1)])

##KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:",acc_knn)

print("\n")

## Create train and test data just off of sex
X_train = train_df.drop(["Survived","Pclass", "Age", "Fare", "Embarked"],axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop(["PassengerId","Pclass","Age","Fare","Embarked"],axis=1).copy()

# Pivot the data around sex to find correlation
print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
.sort_values(by='Sex', ascending=False))

##KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:",acc_knn)
