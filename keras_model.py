import keras
from keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

def score_keras(X_test, y_test, model: keras.Sequential):
    score = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    print(f'score1: {score[0]}, score2: {r2_score(y_test,y_pred)}')


def create_baseline():
    model = keras.Sequential()
    model.add(layers.Dense(6419, input_shape=(6419,), activation='relu'))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

def main(X, y):
    model = create_baseline()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=0)
    score_keras(X_test, y_test, model)
