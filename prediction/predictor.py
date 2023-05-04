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

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.classes = None
        self.class_counts = None

    def fit(self, X, y):
        self.classes, self.class_counts = np.unique(y, return_counts=True)
        if len(self.classes) == 1:
            # If only one class is present, make this node a leaf
            return self
        if len(y) < self.min_samples_split or (self.max_depth is not None and self.max_depth <= 1):
            # If we've reached the minimum number of samples or maximum depth, make this node a leaf
            self.class_counts = self.class_counts / len(y)
            return self
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            # If we can't find a good split, make this node a leaf
            self.class_counts = self.class_counts / len(y)
            return self
        self.feature = best_feature
        self.threshold = best_threshold
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        self.left = DecisionTreeClassifier(max_depth=self.max_depth-1, min_samples_split=self.min_samples_split)
        self.right = DecisionTreeClassifier(max_depth=self.max_depth-1, min_samples_split=self.min_samples_split)
        self.left.fit(X[left_mask], y[left_mask])
        self.right.fit(X[right_mask], y[right_mask])
        return self

    def predict(self, X):
        X = np.array(X)
        if self.feature is None:
            # If this node is a leaf, return the most common class
            return np.array([self.classes[np.argmax(self.class_counts)]] * len(X))
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = X[:, self.feature] > self.threshold
        y_left = self.left.predict(X[left_mask])
        y_right = self.right.predict(X[right_mask])
        y_pred = np.empty(len(X), dtype=int)
        y_pred[left_mask] = y_left
        y_pred[right_mask] = y_right
        return y_pred

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = -np.inf
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                score = self._split_criterion(y[left_mask], y[right_mask])
                if score > best_score:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score
        if best_feature is None:
            return None
        return best_feature, best_threshold

    def _split_criterion(self, y_left, y_right):
        # Calculate the Gini impurity of the split
        n = len(y_left) + len(y_right)
        p_left = len(y_left) / n
        p_right = len(y_right) / n
        impurity = p_left * self._gini(y_left) + p_right * self._gini(y_right)
        return -impurity

    def _gini(self, y):
        # Calculate the Gini impurity of a node
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        impurity = 1 - np.sum(p ** 2)
        return impurity

from sklearn.metrics import accuracy_score
clf= DecisionTreeClassifier(max_depth=3, min_samples_split=10)
clf.fit(X_train, Y_train)
givenInput = np.array([1,85,66,29,0,26.6,0.351,33]).reshape(1, -1)
outputToDisplay = clf.predict(givenInput)
print(outputToDisplay)

# from sklearn.linear_model import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier

# clf= DecisionTreeClassifier(criterion='entropy', random_state=0)
# clf.fit(X_train, Y_train)
# inputFeatures = [10,15,8,3,0,30,0.7,5]
# infProb = clf.predict_proba([inputFeatures])[0][1]
# print(infProb)
# print('Accuray is' ,clf.score(X_test,Y_test)*100, "%")

#pickeling the above model
filename = 'model.sav'
joblib.dump(clf,filename)
