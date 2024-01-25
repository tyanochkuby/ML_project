import pandas as pd
import data_prep
# import ridge
import keras


df = data_prep.get_data()
X, y = df.drop(columns=[df.columns[0]], inplace=False), df.iloc[:, 0]
# print('RIDGE:')
# ridge.ridge(X, y)
print('\n\nKERAS:')
keras.keras(X, y)
