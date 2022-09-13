import tensorflow as tf
import os
from tensorflow.keras.layers import Activation, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class MLPModel(object):
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

    def create_mlp_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(128, activation='relu', input_shape=self.train_data.shape[1:],))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation=None))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        model = self.create_mlp_model()
        model.fit(self.train_data, self.train_label, epochs=20, verbose=2,
                  validation_data=(self.test_data, self.test_label))
        model.save(self.model_save_path)
        print("save path:", self.model_save_path)

    def show_model(self):
        model = tf.keras.models.load_model(self.model_save_path)
        _, train_acc = model.evaluate(self.train_data, self.train_label, verbose=2)
        _, test_acc = model.evaluate(self.test_data, self.test_label, verbose=2)

        return train_acc, test_acc


def train_mlp_models(train_x, train_y, test_x, test_y, model_save_path, result_file):
    print('-------------MLP--------------')
    train_y = tf.one_hot(train_y, 2)
    test_y = tf.one_hot(test_y, 2)
    model = MLPModel(train_class_number=2, train_data=train_x, train_label=train_y,
                     test_data=test_x, test_label=test_y, model_save_path=model_save_path,
                     validation_data=test_x, validation_label=test_y)

    if os.path.exists(model_save_path):
        print('{} is existed!'.format(model_save_path))
        train_acc, test_acc = model.show_model()
        print('-------------MLP--------------', file=result_file)
        print('train acc: {}'.format(train_acc), file=result_file)
        print('test acc: {}'.format(test_acc), file=result_file)
        return test_acc
    else:
        model.train()
        train_acc, test_acc = model.show_model()
        print('-------------MLP--------------', file=result_file)
        print('train acc: {}'.format(train_acc), file=result_file)
        print('test acc: {}'.format(test_acc), file=result_file)
        return test_acc


def train_random_frost(train_x, train_y, test_x, test_y, result_file):
    print('-------------RF--------------')
    classifier = RandomForestClassifier()
    classifier.fit(train_x, train_y)
    test_prediction = classifier.predict(test_x)
    test_acc = accuracy_score(test_prediction, test_y)
    print(test_acc)
    print('-------------RF--------------', file=result_file)
    print('test acc: {}'.format(test_acc), file=result_file)
    return test_acc


def train_svm(train_x, train_y, test_x, test_y, result_file):
    print('-------------SVM--------------')
    classifier = SVC(kernel='rbf')
    classifier.fit(train_x, train_y)
    test_prediction = classifier.predict(test_x)
    test_acc = accuracy_score(test_prediction, test_y)
    print(test_acc)
    print('-------------SVM--------------', file=result_file)
    print('test acc: {}'.format(test_acc), file=result_file)
    return test_acc


