from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss



iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=1000, dual=False, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
result = lr.predict(X_test)


rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)


print(y_test)
print(result)
#print(precision_score(y_test,result))