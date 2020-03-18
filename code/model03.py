from datetime import datetime
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.encoders.simple import SimpleEncoder

from dlgo.networks import small, large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adagrad, Adadelta, SGD


def model03():
    mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Start model03(): ' + mainmodel_start_time)

    optimizer = Adagrad()
    # optimizer = Adadelta()
    # optimizer = SGD(lr=0.01, momentum=0.9, decay=0.001)

    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    one_plane_encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    seven_plane_encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    simple_encoder = SimpleEncoder((go_board_rows, go_board_cols))

    encoder = seven_plane_encoder

    processor = GoDataProcessor(encoder=encoder.name())

    train_generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator  = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)

    network_large = large
    network_small = small

    network = network_small
    network_layers = network.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)

    model.add(Dense(num_classes, activation='relu'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    epochs = 5
    batch_size = 128
    model.fit_generator(
        generator=train_generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=train_generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[
            ModelCheckpoint(
                filepath='D:\\CODE\\Python\\Go\\code\\dlgo\\data\\checkpoints\\small_epoch_{epoch:02d}'
                         '-acc-{accuracy:.4f}-val_acc_{'
                         'val_accuracy:.4f}f.h5',
                monitor='accuracy'
            )
        ]
    )
    model.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size
    )


if __name__ == '__main__':
    model03()
