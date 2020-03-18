import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def model01():
    np.random.seed(1211)
    data_directory = 'D:\\CODE\\Python\\Go\\code\\generated_games\\'
    features = 'features-40k.npy'
    labels = 'labels-40k.npy'
    X = np.load(data_directory + features)
    Y = np.load(data_directory + labels)

    samples = X.shape[0]
    board_size = 9 * 9

    X = X.reshape(samples, board_size)
    Y = Y.reshape(samples, board_size)

    train_samples = int(0.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]

    model = Sequential()
    model.add(Dense(1000, activation='sigmoid', input_shape=(board_size, )))
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(board_size, activation='sigmoid'))

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=64,
              epochs=15,
              verbose=1,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    return score


if __name__ == '__main__':
    score = model01()
    print('Test loss score[0]: {0}\nTest accuracy score[1]: {1}'.format(score[0], score[1]))
