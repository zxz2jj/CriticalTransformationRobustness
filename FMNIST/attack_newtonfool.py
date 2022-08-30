import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, \
    AutoAttack, BrendelBethgeAttack, ElasticNet, HopSkipJump, NewtonFool, PixelAttack, SaliencyMapMethod, ShadowAttack, \
    SimBA, SpatialTransformation, SquareAttack, Wasserstein, ZooAttack
from art.estimators.classification import TensorFlowV2Classifier


class NewtonFoolAttack(object):

    def __init__(self, model_save_path='model/fmnist_model.h5'):
        self.model_save_path = model_save_path
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = tf.one_hot(y_train, 10)
        y_test = tf.one_hot(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def get_softmax_classifier(self):
        model = tf.keras.models.load_model(self.model_save_path)
        classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=10,
            input_shape=(28, 28, 1),
            clip_values=(0, 1),
            loss_object=tf.keras.losses.categorical_crossentropy
        )
        return classifier

    def get_logits_classifier(self):
        model = tf.keras.models.load_model(self.model_save_path)
        model = Model(inputs=model.input, outputs=model.layers[-2].output)
        classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=10,
            input_shape=(28, 28, 1),
            clip_values=(0, 1),
            loss_object=tf.keras.losses.categorical_crossentropy
        )
        return classifier

    def generate(self, max_iter=100):
        save_path = 'data/NewtonFool/'
        log = open(save_path+'log.txt', 'w')
        print("NewtonFool Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = NewtonFool(classifier=classifier, max_iter=max_iter)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Prediction changes: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(save_path+'adv_test.npy', adversarial_test)
        np.save(save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + save_path)
        return


if __name__ == "__main__":
    adv_generation = NewtonFoolAttack()
    adv_generation.generate()

