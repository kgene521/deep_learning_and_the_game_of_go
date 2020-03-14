from datetime import datetime
from dlgo.data.parallel_processor import GoDataProcessor
# from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
# from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks import small, large
# import os
# import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint


def mainmodel():
    mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Inside mainmodel(): ' + mainmodel_start_time)

    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    # encoder = OnePlaneEncoder((go_board_rows, go_board_cols))       # 1 plane
    # encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))   # 7 planes
    encoder = SimpleEncoder((go_board_rows, go_board_cols))       # 11 planes

    processor = GoDataProcessor(encoder=encoder.name())

    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    # generator = processor.load_go_data('train', num_games, use_generator=True)
    # test_generator = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape)
    model = Sequential()
    mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Inside mainmodel(): before adding all layers: ' + mainmodel_start_time)

    for layer in network_layers:
        mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('Before model.add(' + layer.name + '): ' + mainmodel_start_time)
        model.add(layer)
        mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('After model.add(' + layer.name + '): ' + mainmodel_start_time)

    mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Inside mainmodel(): after adding all layers: ' + mainmodel_start_time)

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    mainmodel_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Inside mainmodel(): after compiling the model: ' + mainmodel_start_time)

    # For more information:
    # https://keras.io/callbacks/
    epochs = 2
    batch_size = 128
    model.fit_generator(
        # model.fit(
        generator=generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[
            ModelCheckpoint('data\\checkpoints\\small_model_epoch_{epoch:02d}-{val_loss:.2f}.h5'),
            # EarlyStopping(monitor='accuracy'),
            # ProgbarLogger(),
            # CSVLogger('data\\logs\\training.log'),
            # TensorBoard(log_dir='data\\logs', batch_size=128, write_images=True)
        ]
    )
    model.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size
    )


if __name__ == '__main__':  # Start: 2020-03-06 22:08:54
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Before mainmodel(): ' + start_time_str)
    mainmodel()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n============== Finished Modeling ===========================\n')

    print('Start Time: ' + start_time_str + ', End Time: ' + end_time_str)
