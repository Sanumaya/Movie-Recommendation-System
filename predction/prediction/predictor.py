import pandas as pd
import joblib

import numpy as np
#reading the csv file
df = pd.read_csv('diabetes.csv')
df.info()
df.describe()

#
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size] #all rows & testsetsizecolumn
    train_indices = shuffled[test_set_size:] #first
    return data.iloc[train_indices], data.iloc[test_indices]

#training 80% of our data and testing 20%
train, test = data_split(df, 0.2)
print(train)
print(test)

#converting to 2d array
X_train = train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']].to_numpy()
X_test = test[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']].to_numpy()
print('2D array is -')
print(X_train)

Y_train = train[['Outcome']].to_numpy().reshape(615,)
Y_test = test[['Outcome']].to_numpy().reshape(153,)
print('1d array is-')
print(Y_train)


#modeling with the help of logistic regression and kneighbourclassifier


#from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, Y_train)
inputFeatures = [10,15,8,3,0,30,0.7,5]
infProb = clf.predict_proba([inputFeatures])[0][1]
print(infProb)
print('Accuray is' ,clf.score(X_test,Y_test)*100, "%")

#pickeling the above model
filename = 'model.sav'
joblib.dump(clf,filename)
