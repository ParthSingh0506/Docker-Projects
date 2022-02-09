import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


train_data = pd.read_csv("train.csv", usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                                               'Survived'])

survived = train_data.pop('Survived')

train_data.insert(5, 'Survived', survived)
test_data = pd.read_csv("test.csv", usecols=[
                        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])

final_data = pd.read_csv("gender_submission.csv", usecols=['Survived'])
true_pred = final_data

# Getting different classes
train_data = pd.get_dummies(data=train_data, columns=['Sex'], drop_first=True)

# Rename the Sex_male Column to Sex that has been created by pd.get_dummies
train_data.rename(columns={'Sex_male': 'Sex'}, inplace=True)

# 0 is female
# 1 is male

# Finding Missing Values of The Age column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(train_data.iloc[:, 1:2])
train_data.iloc[:, 1:2] = imputer.transform(train_data.iloc[:, 1:2])

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

classfier = SVC(kernel='linear', degree=3)
classfier.fit(X_train, y_train)
y_pred = classfier.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)

# Now Predicting test data

test_data = pd.get_dummies(test_data, columns=['Sex'], drop_first=True)
test_data.rename(columns={'Sex_male': 'Sex'}, inplace=True)

test_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
test_imputer.fit(test_data.iloc[:, 1:2])
test_data.iloc[:, 1:2] = test_imputer.transform(test_data.iloc[:, 1:2])

test_pred = classfier.predict(test_data)
final_score = accuracy_score(test_pred, true_pred)

print("Enter the following details to make the prediction:-\n")

pclass = int(input("Enter The Pclass:- "))
Age = int(input("Enter The Age:- "))
SibSp = int(input("Enter The SibSp:- "))
Parch = float(input("Enter The Parch:- "))
Sex = int(input("Enter The Sex:- "))


user_prediction = classfier.predict([[pclass,Age,SibSp,Parch,Sex]])

if user_prediction == 0 :
    print("Not Survived.")
else :
    print("Survived")