from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import data

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(6420, input_shape=(6420,), activation='relu'))
    model.add(Dense(1604, actication='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def keras(X, y):
    estimator = KerasClassifier(model=create_baseline, epochs=500, batch_size=16, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))