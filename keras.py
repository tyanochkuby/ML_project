from tf.keras.layers import Sequential
from tf.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import data

def score_keras(X, y, model: Sequential):
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
    score = model.evaluate(X_test, y_test, verbose=0)


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(6420, input_shape=(6420,), activation='relu'))
    model.add(Dense(1604, actication    ='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metrics.mean_absolute_error()])
    return model

def keras(X, y):
    model = create_baseline()
    estimator = KerasClassifier(model=model, epochs=500, batch_size=16, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    score_keras(X, y, model)
