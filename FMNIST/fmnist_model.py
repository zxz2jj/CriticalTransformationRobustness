import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense


class FMnistModel(object):
    def __init__(self, train_class_number, train_data, train_label, test_data, test_label, model_save_path,
                 validation_data=None, validation_label=None,):
        self.train_class_number = train_class_number
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.model_save_path = model_save_path
        self.validation_data = validation_data
        self.validation_label = validation_label

    def create_model(self):
        model = tf.keras.Sequential()

        model.add(Conv2D(filters=40, kernel_size=(5, 5), strides=(1, 1), input_shape=self.train_data.shape[1:], padding='same',
                         data_format='channels_last', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten(data_format='channels_last'))
        model.add(Dense(320, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(160, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation=None))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        model = self.create_model()
        # model.fit(self.train_data, self.train_label, epochs=10, verbose=2, shuffle=True, validation_split=0.2)
        model.fit(self.train_data, self.train_label, epochs=1, verbose=2,
                  validation_data=(self.test_data, self.test_label))
        model.save(self.model_save_path)
        print("save path:", self.model_save_path)

    def show_model(self):
        model = tf.keras.models.load_model(self.model_save_path)
        model.summary()
        print("train dataset:")
        print(self.train_data.shape, self.train_label.shape)
        model.evaluate(self.train_data, self.train_label, verbose=2)

        print("test dataset:")
        print(self.test_data.shape, self.test_label.shape)
        model.evaluate(self.test_data, self.test_label, verbose=2)


def train_models():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    model_save_path = r'model/'
    model_name = 'fmnist_model.h5'

    fmnist_model = FMnistModel(train_class_number=10, train_data=x_train, train_label=y_train,
                               test_data=x_test, test_label=y_test, model_save_path=model_save_path+model_name,
                               validation_data=x_test, validation_label=y_test)

    if os.path.exists(model_save_path+model_name):
        print('{} is existed!'.format(model_name))
        fmnist_model.show_model()
    else:
        fmnist_model.train()


if __name__ == "__main__":
    train_models()
