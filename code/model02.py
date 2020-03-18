import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def model02():
    np.random.seed(1211)
    data_directory = 'D:\\CODE\\Python\\Go\\code\\generated_games\\'
    features = 'features-40k.npy'
    labels = 'labels-40k.npy'
    X = np.load(data_directory + features)
    Y = np.load(data_directory + labels)

    samples = X.shape[0]
    size = 9
    input_shape = (size, size, 1)
    board_size = 9 * 9

    X = X.reshape(samples, size, size, 1)

    train_samples = int(0.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]

    model = Sequential()
    model.add(Conv2D(filters=48,
                     kernel_size=(3, 3),
                     activation='sigmoid',
                     padding='same',
                     input_shape=input_shape))
    model.add(Conv2D(filters=48,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(size*size, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=64,
              epochs=100,
              verbose=1,
              validation_data=(X_test, Y_test))

    score1 = model.evaluate(X_test, Y_test, verbose=0)
    return score1


if __name__ == '__main__':
    score = model02()
    print('Test loss score[0]: {0}\nTest accuracy score[1]: {1}'.format(score[0], score[1]))
