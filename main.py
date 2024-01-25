import pandas as pd
import data_prep
import ridge_model
import keras_model


df = data_prep.get_data()
X, y = df.drop(columns=[df.columns[0]], inplace=False), df.iloc[:, 0]
print('RIDGE:\n')
ridge_model.main(X, y)
print('\n\nKERAS:\n')
keras_model.main(X, y)
