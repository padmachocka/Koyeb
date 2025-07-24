import joblib
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from fastapi import FastAPI
from sklearn.datasets import load_iris 

iris = load_iris()
#X = iris.data
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
print (X.head())
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
joblib.dump(rf, './api/mlmodel.joblib')   