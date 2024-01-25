from sklearn.linear_model import RidgeCV
import pandas as pd
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,r2_score

def score_ridge(X, y, model: RidgeCV):
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
    y_pred = model.predict(X_test)
    print(r2_score(y_test,y_pred))

def main(X, y):    
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    model = RidgeCV(alphas=arange(0.5, 2.0, 0.1), cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    model.fit(X, y)
    
    score_ridge()