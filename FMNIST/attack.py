import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import os

from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, \
    BrendelBethgeAttack, ElasticNet, HopSkipJump, NewtonFool, PixelAttack, SaliencyMapMethod, ShadowAttack, \
    SpatialTransformation, SquareAttack, Wasserstein, ZooAttack, BasicIterativeMethod,\
    ProjectedGradientDescentTensorFlowV2
from art.estimators.classification import TensorFlowV2Classifier


class ModelAndData(object):
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


class Attacks(ModelAndData):
    def __init__(self):
        ModelAndData.__init__(self)
        self.save_path = None

    def generate(self):
        pass

    def is_finished(self):
        if os.path.exists(self.save_path + 'adv_test.npy') and os.path.exists(self.save_path + 'adv_test_targets.npy') \
                and os.path.exists(self.save_path + 'adv_test_ground_truth.npy') \
                and os.path.exists(self.save_path + 'ori_test_successful_attack.npy'):
            attack_name = self.save_path.split('/')[-2]
            print(f"{attack_name} is finished！")
            return True
        else:
            return False


class BA(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/BA/'

    def generate(self, targeted=False):
        self.save_path = 'data/BA/'
        log = open(self.save_path + 'log.txt', 'w')
        print("BA Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = BoundaryAttack(estimator=classifier, targeted=targeted)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path + "adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path + 'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path + 'adv_test.npy', adversarial_test)
        np.save(self.save_path + 'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path + 'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path + 'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class BrBeAttack(Attacks):
    def __init__(self, norm):
        Attacks.__init__(self)
        self.norm = norm
        if norm == 2:
            self.save_path = 'data/BrBeAttack-L2/'
        elif norm == np.inf:
            self.save_path = 'data/BrBeAttack-Inf/'
        else:
            pass

    def generate(self,
                 overshot=1.01,
                 targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("BrendelBethge Attack Information:", file=log)
        classifier = self.get_logits_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100))
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = BrendelBethgeAttack(estimator=classifier, norm=self.norm, overshoot=overshot, targeted=targeted)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class CWInf(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/CW-Inf/'

    def generate(self,
                 confidence=0.0,
                 targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("CW-Inf Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = CarliniLInfMethod(classifier=classifier, confidence=confidence, targeted=targeted)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class CWL2(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/CW-L2/'

    def generate(self,
                 confidence=0.0,
                 targeted=False):
        log = open(self.save_path + 'log.txt', 'w')
        print("CW-L2 Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = CarliniL2Method(classifier=classifier, confidence=confidence, targeted=targeted)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path + "adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path + 'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path + 'adv_test.npy', adversarial_test)
        np.save(self.save_path + 'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path + 'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path + 'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class DeepFoolAttack(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/DeepFool/'

    def generate(self,
                 max_iter=100,
                 epsilon=1e-6):
        log = open(self.save_path+'log.txt', 'w')
        print("DeepFool Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = DeepFool(classifier=classifier, max_iter=max_iter, epsilon=epsilon)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class EAD(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/EAD/'

    def generate(self,
                 confidence=0.0,
                 targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("EAD Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on clean test examples: {}%".format(accuracy * 100))
        print("Accuracy on clean test examples: {}%".format(accuracy * 100), file=log)

        attack = ElasticNet(classifier=classifier, confidence=confidence, targeted=targeted)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print("Prediction changes: {}".format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print("Successful Attack: {}".format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class FGSM(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/FGSM/'

    def generate(self,
                 eps=0.3,
                 eps_step=0.1):
        log = open(self.save_path + 'log.txt', 'w')
        print('FGSM Attack Information:', file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100), file=log)

        attack = FastGradientMethod(estimator=classifier, eps=eps, eps_step=eps_step)
        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print('Prediction changes: {}'.format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print('Successful Attack: {}'.format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+'adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg'.
                        format(i+1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i]*255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i+1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i]*255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print(f'result is saved in path: {self.save_path}')
        return


class HopSkipJumpAttack(Attacks):
    def __init__(self, norm):
        Attacks.__init__(self)
        self.norm = norm
        if norm == 2:
            self.save_path = 'data/HopSkipJumpAttack-L2/'
        elif norm == np.inf:
            self.save_path = 'data/HopSkipJumpAttack-Inf/'
        else:
            pass

    def generate(self,
                 targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print(f"{self.save_path.split('/')[-2]} Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = HopSkipJump(classifier=classifier, norm=self.norm, targeted=targeted)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class IFGSM(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/IFGSM/'

    def generate(self,
                 eps=0.3,
                 eps_step=0.1):
        log = open(self.save_path + 'log.txt', 'w')
        print('IFGSM Attack Information:', file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100), file=log)

        attack = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=eps_step)

        x_test_adv = attack.generate(x=self.x_test)
        adv_predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100), file=log)

        correct_prediction_index = np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)
        prediction_change_index = np.argmax(adv_predictions, axis=1) != np.argmax(ori_predictions, axis=1)
        print('Prediction changes: {}'.format(np.sum(prediction_change_index)), file=log)
        successful_attack_index = correct_prediction_index & prediction_change_index
        print('Successful Attack: {}'.format(np.sum(successful_attack_index)), file=log)

        original_test_successful_attack = self.x_test[successful_attack_index]
        original_test_successful_attack_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        adversarial_test = x_test_adv[successful_attack_index]
        adversarial_test_target = np.argmax(adv_predictions, axis=1)[successful_attack_index]
        adversarial_test_ground_truth = np.argmax(self.y_test, axis=1)[successful_attack_index]
        for i in range(len(adversarial_test[:50])):
            cv2.imwrite(self.save_path+'adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg'.
                        format(i+1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i]*255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i+1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i]*255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print(f'result is saved in path: {self.save_path}')
        return


class JSMA(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/JSMA/'

    def generate(self,
                 theta=0.1,
                 gamma=0.5,):
        log = open(self.save_path+'log.txt', 'w')
        print("JSMA Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class NewtonFoolAttack(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/NewtonFool/'

    def generate(self, max_iter=100):
        log = open(self.save_path+'log.txt', 'w')
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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class PGD(Attacks):
    def __init__(self, norm):
        Attacks.__init__(self)
        self.norm = norm
        if norm == 2:
            self.save_path = 'data/PGD-L2/'
        elif norm == np.inf:
            self.save_path = 'data/PGD-Inf/'
        else:
            pass

    def generate(self):
        log = open(self.save_path+'log.txt', 'w')
        print("PGD-Inf Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = ProjectedGradientDescentTensorFlowV2(estimator=classifier, norm=self.norm)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class PixelsAttack(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/PixelAttack/'

    def generate(self, targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("PixelAttack Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = PixelAttack(classifier=classifier, targeted=targeted, verbose=True)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class ShadowAttacks(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/ShadowAttack/'

    def generate(self, targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("ShadowAttack Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = ShadowAttack(estimator=classifier, targeted=targeted, nb_steps=30)

        x_test_adv = []
        width = self.x_test[0].shape[0]
        height = self.x_test[0].shape[1]
        channel = self.x_test[0].shape[2]
        for i in range(self.x_test.shape[0]):
            print("Attack process: {} / {}".format(i, self.x_test.shape[0]))
            adv_sample = attack.generate(x=self.x_test[i].reshape([-1, width, height, channel]))
            x_test_adv.append(adv_sample)
        x_test_adv = np.concatenate(x_test_adv)
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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class SpatialTransformationAttack(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/SpatialTransformationAttack/'

    def generate(self,
                 max_translation=10,
                 max_rotation=45,):
        log = open(self.save_path+'log.txt', 'w')
        print("SpatialTransformationAttack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = SpatialTransformation(classifier=classifier, max_translation=max_translation, max_rotation=max_rotation)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class SquareAttacks(Attacks):
    def __init__(self, norm):
        Attacks.__init__(self)
        self.norm = norm
        if norm == 2:
            self.save_path = 'data/SquareAttack-L2/'
        elif norm == np.inf:
            self.save_path = 'data/SquareAttack-Inf/'
        else:
            pass

    def generate(self,
                 max_iter=100,
                 eps=0.3,):
        log = open(self.save_path+'log.txt', 'w')
        print("SquareAttack_Inf Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = SquareAttack(estimator=classifier, norm=self.norm, max_iter=max_iter, eps=eps)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class WassersteinAttack(Attacks):
    def __init__(self, norm):
        Attacks.__init__(self)
        self.norm = norm
        if norm == '2':
            self.save_path = 'data/WassersteinAttack/'
        else:
            pass

    def generate(self,
                 targeted=False):
        log = open(self.save_path+'log.txt', 'w')
        print("WassersteinAttack Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = Wasserstein(estimator=classifier, norm=self.norm, targeted=targeted)

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
            cv2.imwrite(self.save_path+"adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path+'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path+'adv_test.npy', adversarial_test)
        np.save(self.save_path+'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path+'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path+'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


class ZOO(Attacks):
    def __init__(self):
        Attacks.__init__(self)
        self.save_path = 'data/ZOO/'

    def generate(self,
                 confidence=0.0,
                 targeted=False):
        log = open(self.save_path + 'log.txt', 'w')
        print("ZOO Attack Information:", file=log)
        classifier = self.get_softmax_classifier()
        ori_predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(ori_predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), file=log)

        attack = ZooAttack(classifier=classifier, confidence=confidence, targeted=targeted, max_iter=30)

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
            cv2.imwrite(self.save_path + "adv_successful_attack/picture_{}_ori_{}_adv_{}.jpg".
                        format(i + 1, adversarial_test_ground_truth[i], adversarial_test_target[i]),
                        np.clip(adversarial_test[i] * 255, 0, 255))
        for i in range(len(original_test_successful_attack[:50])):
            cv2.imwrite(self.save_path + 'ori_successful_attack/picture_{}_ori_{}.jpg'.
                        format(i + 1, original_test_successful_attack_ground_truth[i]),
                        np.clip(original_test_successful_attack[i] * 255, 0, 255))

        np.save(self.save_path + 'adv_test.npy', adversarial_test)
        np.save(self.save_path + 'adv_test_targets.npy', adversarial_test_target)
        np.save(self.save_path + 'adv_test_ground_truth.npy', adversarial_test_ground_truth)
        np.save(self.save_path + 'ori_test_successful_attack.npy', original_test_successful_attack)

        print("result is saved in path: " + self.save_path)
        return


if __name__ == "__main__":
    adv_generation = BA()
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = BrBeAttack(norm=np.inf)
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = BrBeAttack(norm=2)
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = CWInf()
    if not adv_generation.is_finished():
        adv_generation.generate(confidence=0.0, targeted=False)

    adv_generation = CWL2()
    if not adv_generation.is_finished():
        adv_generation.generate(confidence=0.0, targeted=False)

    adv_generation = DeepFoolAttack()
    if not adv_generation.is_finished():
        adv_generation.generate(max_iter=100, epsilon=1e-6)

    adv_generation = EAD()
    if not adv_generation.is_finished():
        adv_generation.generate(confidence=0.0, targeted=False)

    adv_generation = FGSM()
    if not adv_generation.is_finished():
        adv_generation.generate(eps=0.3, eps_step=0.1)

    adv_generation = HopSkipJumpAttack(norm=np.inf)
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = HopSkipJumpAttack(norm=2)
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = IFGSM()
    if not adv_generation.is_finished():
        adv_generation.generate(eps=0.3, eps_step=0.1)

    adv_generation = JSMA()
    if not adv_generation.is_finished():
        adv_generation.generate()

    adv_generation = NewtonFoolAttack()
    if not adv_generation.is_finished():
        adv_generation.generate()

    adv_generation = PGD(norm=np.inf)
    if not adv_generation.is_finished():
        adv_generation.generate()

    adv_generation = PGD(norm=2)
    if not adv_generation.is_finished():
        adv_generation.generate()

    adv_generation = PixelsAttack()
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = ShadowAttacks()
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False)

    adv_generation = SpatialTransformationAttack()
    if not adv_generation.is_finished():
        adv_generation.generate(max_translation=5, max_rotation=10)

    adv_generation = SquareAttacks(norm=np.inf)
    if not adv_generation.is_finished():
        adv_generation.generate(max_iter=100, eps=0.3)

    adv_generation = SquareAttacks(norm=2)
    if not adv_generation.is_finished():
        adv_generation.generate(max_iter=100, eps=0.3)

    adv_generation = WassersteinAttack(norm='2')
    if not adv_generation.is_finished():
        adv_generation.generate(targeted=False,)

    adv_generation = ZOO()
    if not adv_generation.is_finished():
        adv_generation.generate(confidence=0, targeted=False)



