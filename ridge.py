from sklearn.linear_model import RidgeCV
import pandas as pd
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

def ridge(X, y):    
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    model = RidgeCV(alphas=arange(0.1, 1, 0.1), cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)
    print('RMSE: %f' % model.score(X, y))


    # model = Ridge(alpha=1.0)
    # scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    # # force scores to be positive
    # scores = absolute(scores)
    # print('Mean Ridge RMSE: %.3f (%.3f)' % (mean(scores), std(scores)))