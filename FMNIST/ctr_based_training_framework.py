import numpy as np
import random
import tensorflow as tf
import os
import pickle
from calculate_ctr import Transformation
from metrics import calculate_auroc, calculate_crossentropy, calculate_l1_distance, calculate_kl, \
    calculate_prediction_changes
from classifiers import train_mlp_models, train_random_frost, train_svm


class TrainingFramework(Transformation):
    def __init__(self,):
        self.model_save_path = 'model/fmnist_model.h5'
        self.attack_method = ['BA', 'BrBeAttack_Inf', 'BrBeAttack_L2', 'CW_Inf', 'CW_L2', 'DeepFool', 'EAD', 'FGSM',
                              'IFGSM', 'HopSkipJumpAttack_Inf', 'HopSkipJumpAttack_L2', 'JSMA', 'NewtonFool', 'PGD_Inf',
                              'PGD_L2', 'PixelAttack', 'ShadowAttack', 'SpatialTransformationAttack',
                              'SquareAttack_Inf', 'SquareAttack_L2', 'WassersteinAttack', 'ZOO']
        self.transformations = [
                                'brightness',
                                'darkness',
                                'rotation_left',
                                'rotation_right',
                                'blur_mean',
                                'shift_up',
                                'shift_down',
                                'shift_left',
                                'shift_right',
                                'horizontal_shear_right',
                                'horizontal_shear_left',
                                'scale_big',
                                'scale_small',
                                'contrast_big',
                                'contrast_small',
                                'jpeg_compression',
                                'webp_compression'
                               ]

        self.transformation_parameter = {
                               'brightness': {'start': 1, 'end': 255, 'inc': 1, 'n': 0},
                               'darkness': {'start': 1, 'end': 255, 'inc': 1, 'n': 0},
                               'rotation_left': {'start': 1, 'end': 90, 'inc': 1, 'n': 0},
                               'rotation_right': {'start': 1, 'end': 90, 'inc': 1, 'n': 0},
                               'blur_mean': {'start': 1, 'end': 10, 'inc': 1, 'n': 0},
                               'shift_up': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_down': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_left': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_right': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'horizontal_shear_right': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'horizontal_shear_left': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'scale_big': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'scale_small': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'contrast_big': {'start': 0.01, 'end': 1.0, 'inc': 0.01, 'n': 2},
                               'contrast_small': {'start': 0.01, 'end': 1.0, 'inc': 0.01, 'n': 2},
                               'jpeg_compression': {'start': 1, 'end': 100, 'inc': 1, 'n': 0},
                               'webp_compression': {'start': 1, 'end': 100, 'inc': 1, 'n': 0}
                               }
        self.transformation_function = {'flip_up2bottom': self.flip_up2bottom,
                                        'flip_left2right': self.flip_left2right,
                                        'brightness': self.brightness,
                                        'darkness': self.darkness,
                                        'rotation_left': self.rotation_left,
                                        'rotation_right': self.rotation_right,
                                        'blur_mean': self.mean_blur,
                                        'shift_up': self.shift_up,
                                        'shift_down': self.shift_down,
                                        'shift_left': self.shift_left,
                                        'shift_right': self.shift_right,
                                        'horizontal_shear_right': self.horizontal_shear_right,
                                        'horizontal_shear_left': self.horizontal_shear_left,
                                        'scale_big': self.scale_big,
                                        'scale_small': self.scale_small,
                                        'contrast_big': self.contrast_big,
                                        'contrast_small': self.contrast_small,
                                        'jpeg_compression': self.jpeg_compression,
                                        'webp_compression': self.webp_compression
                                        }
        self.data_path = {'BA': 'data/BA/',
                          'BrBeAttack_Inf': 'data/BrBeAttack-Inf/',
                          'BrBeAttack_L2': 'data/BrBeAttack-L2/',
                          'CW_Inf': 'data/CW-Inf/',
                          'CW_L2': 'data/CW-L2/',
                          'DeepFool': 'data/DeepFool/',
                          'EAD': 'data/EAD/',
                          'FGSM': 'data/FGSM/',
                          'IFGSM': 'data/IFGSM/',
                          'HopSkipJumpAttack_Inf': 'data/HopSkipJumpAttack-Inf/',
                          'HopSkipJumpAttack_L2': 'data/HopSkipJumpAttack-L2/',
                          'JSMA': 'data/JSMA/',
                          'NewtonFool': 'data/NewtonFool/',
                          'PGD_Inf': 'data/PGD-Inf/',
                          'PGD_L2': 'data/PGD-L2/',
                          'PixelAttack': 'data/PixelAttack/',
                          'ShadowAttack': 'data/ShadowAttack/',
                          'SpatialTransformationAttack': 'data/SpatialTransformationAttack/',
                          'SquareAttack_Inf': 'data/SquareAttack-Inf/',
                          'SquareAttack_L2': 'data/SquareAttack-L2/',
                          'WassersteinAttack': 'data/WassersteinAttack/',
                          'ZOO': 'data/ZOO/'}
        self.save_path = 'ctr_based_detector/'

    def data_sampling_random(self, train_data_size=300, test_data_size=100, refresh=False):
        if not refresh:
            if os.path.exists(self.save_path + 'random_sampling/train_ori_data.npy') and \
                os.path.exists(self.save_path + 'random_sampling/train_ori_ground_truth.npy') and \
                os.path.exists(self.save_path + 'random_sampling/train_adv_data.npy') and \
                os.path.exists(self.save_path + 'random_sampling/train_adv_target.npy') and \
                os.path.exists(self.save_path + 'random_sampling/test_ori_data.npy') and \
                os.path.exists(self.save_path + 'random_sampling/test_ori_ground_truth.npy') and \
                os.path.exists(self.save_path + 'random_sampling/test_adv_data.npy') and \
                    os.path.exists(self.save_path + 'random_sampling/test_adv_target.npy'):
                return

        train_example_number = train_data_size * self.attack_method.__len__()
        test_example_number = test_data_size * self.attack_method.__len__()
        print("Data Sampling ...")
        data_length = 0
        temp_ori_data = []
        temp_ori_ground_truth = []
        temp_adv_data = []
        temp_adv_target = []
        for attack in self.attack_method:
            ori_data = np.load(self.data_path[attack] + 'ori_test_successful_attack.npy')
            ori_ground_truth = np.load(self.data_path[attack] + 'adv_test_ground_truth.npy')
            adv_data = np.load(self.data_path[attack] + 'adv_test.npy')
            adv_target = np.load(self.data_path[attack] + 'adv_test_targets.npy')

            temp_ori_data.append(ori_data)
            temp_ori_ground_truth.append(ori_ground_truth)
            temp_adv_data.append(adv_data)
            temp_adv_target.append(adv_target)

            data_length += ori_data.shape[0]

        temp_ori_data = np.concatenate(temp_ori_data)
        temp_ori_ground_truth = np.concatenate(temp_ori_ground_truth)
        temp_adv_data = np.concatenate(temp_adv_data)
        temp_adv_target = np.concatenate(temp_adv_target)

        sampling_index = random.sample(range(data_length), train_example_number + test_example_number)

        start_index = 0
        split_index = train_example_number
        end_index = train_example_number + test_example_number
        train_ori_data = temp_ori_data[sampling_index[start_index: split_index]]
        test_ori_data = temp_ori_data[sampling_index[split_index: end_index]]
        train_ori_ground_truth = temp_ori_ground_truth[sampling_index[start_index: split_index]]
        test_ori_ground_truth = temp_ori_ground_truth[sampling_index[split_index: end_index]]
        train_adv_data = temp_adv_data[sampling_index[start_index: split_index]]
        test_adv_data = temp_adv_data[sampling_index[split_index: end_index]]
        train_adv_target = temp_adv_target[sampling_index[start_index: split_index]]
        test_adv_target = temp_adv_target[sampling_index[split_index: end_index]]

        np.save(self.save_path + 'random_sampling/train_ori_data.npy', train_ori_data)
        np.save(self.save_path + 'random_sampling/train_ori_ground_truth.npy', train_ori_ground_truth)
        np.save(self.save_path + 'random_sampling/train_adv_data.npy', train_adv_data)
        np.save(self.save_path + 'random_sampling/train_adv_target.npy', train_adv_target)
        np.save(self.save_path + 'random_sampling/test_ori_data.npy', test_ori_data)
        np.save(self.save_path + 'random_sampling/test_ori_ground_truth.npy', test_ori_ground_truth)
        np.save(self.save_path + 'random_sampling/test_adv_data.npy', test_adv_data)
        np.save(self.save_path + 'random_sampling/test_adv_target.npy', test_adv_target)

    def calculate_ctr(self,):
        model = tf.keras.models.load_model(self.model_save_path)
        for t in self.transformations:
            print("------------Transformation: {}-------------".format(t))
            print('parameter: {}'.format(self.transformation_parameter[t]))

            ori_robustness_result_file = self.save_path + 'ctr/{}_ori_test_robustness.npy'.format(t)
            adv_robustness_result_file = self.save_path + 'ctr/{}_adv_test_robustness.npy'.format(t)

            if os.path.exists(ori_robustness_result_file) and os.path.exists(adv_robustness_result_file):
                print("{} is finished!".format(t))
                continue

            ori_picture = np.load(self.save_path + 'test_ori_data.npy')
            ori_label = np.load(self.save_path + 'test_ori_ground_truth.npy')
            adv_picture = np.load(self.save_path + 'test_adv_data.npy')
            adv_target = np.load(self.save_path + 'test_adv_target.npy')

            transformation_parameter = []
            parameter = self.transformation_parameter[t]['start']
            while parameter < self.transformation_parameter[t]['end']:
                parameter += self.transformation_parameter[t]['inc']
                transformation_parameter.append(parameter)

            ori_ctr = []
            print("Clean Pictures:")
            picture_number = ori_picture.shape[0]
            for theta in transformation_parameter:
                print('{}  {}/{} : '.format(t, theta, transformation_parameter[-1]), end=' ')
                transformed_pictures = []
                for i in range(picture_number):
                    picture = ori_picture[i]
                    transformed_pictures.append(self.transformation_function[t](picture, theta))
                transformed_pictures = np.array(transformed_pictures)
                prediction = np.argmax(model.predict(transformed_pictures), axis=1)
                equals = prediction == ori_label
                changes = picture_number - np.sum(equals)
                for i in range(changes):
                    ori_ctr.append(theta - self.transformation_parameter[t]['inc'])
                print("Clean Pictures: {}, Changes: {}, Not Changes: {}".
                      format(picture_number, changes, np.sum(equals)))
                ori_picture = ori_picture[equals]
                ori_label = ori_label[equals]
                picture_number = ori_picture.shape[0]
                if picture_number == 0:
                    break
            for i in range(picture_number):
                ori_ctr.append(transformation_parameter[-1])
            ori_ctr = np.array(ori_ctr)
            print("\rClean Examples CTR: Num-{}, Avg-{:5}, Max-{:5}, Min-{:5}".
                  format(ori_ctr.shape, np.average(ori_ctr), np.max(ori_ctr), np.min(ori_ctr)))
            np.save(ori_robustness_result_file, ori_ctr)

            adv_ctr = []
            print("Adversarial Pictures:")
            picture_number = adv_picture.shape[0]
            for theta in transformation_parameter:
                print('{}  {}/{} : '.format(t, theta, transformation_parameter[-1]), end=' ')
                transformed_pictures = []
                for i in range(picture_number):
                    picture = adv_picture[i]
                    transformed_pictures.append(self.transformation_function[t](picture, theta))
                transformed_pictures = np.array(transformed_pictures)
                prediction = np.argmax(model.predict(transformed_pictures), axis=1)
                equals = prediction == adv_target
                changes = picture_number - np.sum(equals)
                for i in range(changes):
                    adv_ctr.append(theta - self.transformation_parameter[t]['inc'])
                print("Adversarial Pictures: {}, Changes: {}, Not Changes: {}".
                      format(picture_number, changes, np.sum(equals)))
                adv_picture = adv_picture[equals]
                adv_target = adv_target[equals]
                picture_number = adv_picture.shape[0]
                if picture_number == 0:
                    break
            for i in range(picture_number):
                adv_ctr.append(transformation_parameter[-1])
            adv_ctr = np.array(adv_ctr)
            print("\rAdversarial Examples, Robustness: Num-{}, Avg-{:5}, Max-{:5}, Min-{:5}".
                  format(adv_ctr.shape, np.average(adv_ctr), np.max(adv_ctr), np.min(adv_ctr)))
            np.save(adv_robustness_result_file, adv_ctr)

        return

    def calculate_baseline_detector_performance_on_mixed_attacks(self,):
        model = tf.keras.models.load_model(self.model_save_path)
        for t in self.transformations:
            print("------------Transformation: {}-------------".format(t))
            print('parameter: {}'.format(self.transformation_parameter[t]))

            if os.path.exists(self.save_path + 'baseline_detector_analyse/{}_crossentropy.pkl'.format(t)) and \
                    os.path.exists(self.save_path + 'baseline_detector_analyse/{}_prediction.pkl'.format(t)) and \
                    os.path.exists(self.save_path + 'baseline_detector_analyse/{}_AUROC_crossentropy.pkl'.format(t)):
                print("{} is finished!".format(t))
                continue

            ori_picture_path = self.save_path + 'test_ori_data.npy'
            adv_picture_path = self.save_path + 'test_adv_data.npy'
            ori_picture = np.load(ori_picture_path)
            adv_picture = np.load(adv_picture_path)

            picture_number = adv_picture.shape[0]

            ori_prediction = model.predict(ori_picture)
            adv_prediction = model.predict(adv_picture)

            transformation_parameter = []
            parameter = self.transformation_parameter[t]['start']
            while parameter < self.transformation_parameter[t]['end']:
                parameter += self.transformation_parameter[t]['inc']
                transformation_parameter.append(parameter)

            auroc_of_crossentropy = {}
            crossentropy = {}
            prediction = {}
            for para in transformation_parameter:
                crossentropy_dict = {'ori': None, 'adv': None}
                prediction_dict = {'ori': None, 'adv': None}
                print("transformation parameter: {} / {}".format(para, transformation_parameter[-1]))
                ori_transformed_pictures = []
                for i in range(picture_number):
                    picture = ori_picture[i]
                    ori_transformed_pictures.append(self.transformation_function[t](picture, para))
                ori_transformed_pictures = np.array(ori_transformed_pictures)

                adv_transformed_pictures = []
                for i in range(picture_number):
                    picture = adv_picture[i]
                    adv_transformed_pictures.append(self.transformation_function[t](picture, para))
                adv_transformed_pictures = np.array(adv_transformed_pictures)

                ori_transformed_prediction = model.predict(ori_transformed_pictures)
                adv_transformed_prediction = model.predict(adv_transformed_pictures)

                prediction_dict['ori'] = ori_transformed_prediction
                prediction_dict['adv'] = adv_transformed_prediction
                prediction[para] = prediction_dict
                crossentropy_dict['ori'] = calculate_crossentropy(ori_prediction, ori_transformed_prediction)
                crossentropy_dict['adv'] = calculate_crossentropy(adv_prediction, adv_transformed_prediction)
                crossentropy[para] = crossentropy_dict
                auroc_crossentropy = calculate_auroc(crossentropy_dict, picture_number)
                auroc_of_crossentropy[para] = auroc_crossentropy

            file1 = open(self.save_path + 'baseline_detector_analyse/{}_AUROC_crossentropy.pkl'.format(t), 'wb')
            pickle.dump(auroc_of_crossentropy, file1)
            file1.close()
            file2 = open(self.save_path + 'baseline_detector_analyse/{}_crossentropy.pkl'.format(t), 'wb')
            pickle.dump(crossentropy, file2)
            file2.close()
            file3 = open(self.save_path + 'baseline_detector_analyse{}_prediction.pkl'.format(t), 'wb')
            pickle.dump(prediction, file3)
            file3.close()

        return

    def transformation_and_parameter_selection(self, num):
        robustness_difference = {}
        robustness_adversarial = {}
        transformation_exist = []
        for t in self.transformations:
            if self.transformation_parameter[t] is None:
                continue
            ori_robustness_path = self.save_path + 'ctr/{}_ori_test_robustness.npy'.format(t)
            adv_robustness_path = self.save_path + 'ctr/{}_adv_test_robustness.npy'.format(t)
            if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                transformation_exist.append(t)
                ori_robustness = np.load(ori_robustness_path)
                adv_robustness = np.load(adv_robustness_path)
                robustness_adversarial[t] = np.average(adv_robustness)
                robustness_difference[t] = (np.average(ori_robustness) - np.average(adv_robustness)) / \
                                           (self.transformation_parameter[t]['end'] -
                                            self.transformation_parameter[t]['start'])

        selected_trans = []
        robustness_difference = sorted(robustness_difference.items(), key=lambda k: k[1], reverse=True)
        if num > robustness_difference.__len__():
            return None
        else:
            for i in range(num):
                selected_trans.append((robustness_difference[i][0],
                                       np.round(robustness_adversarial[robustness_difference[i][0]],
                                                self.transformation_parameter[robustness_difference[i][0]]['n']) +
                                       self.transformation_parameter[robustness_difference[i][0]]['inc']))
        return selected_trans

    def feature_construction_vector_concatenation(self):
        model = tf.keras.models.load_model(self.model_save_path)
        train_ori_picture = np.load(self.save_path + 'random_sampling/train_ori_data.npy')
        train_adv_picture = np.load(self.save_path + 'random_sampling/train_adv_data.npy')
        test_ori_picture = np.load(self.save_path + 'random_sampling/test_ori_data.npy')
        test_adv_picture = np.load(self.save_path + 'random_sampling/test_adv_data.npy')
        train_picture_number = train_ori_picture.shape[0]
        test_picture_number = test_ori_picture.shape[0]

        train_ori_softmax = model.predict(train_ori_picture)
        train_adv_softmax = model.predict(train_adv_picture)
        test_ori_softmax = model.predict(test_ori_picture)
        test_adv_softmax = model.predict(test_adv_picture)

        transformations = self.transformation_and_parameter_selection(num=17)
        num = 1
        for transformation, para in transformations:
            if os.path.exists(self.save_path + f'feature_construction/vector_concatenation/train_data_{num}.npy') and \
                os.path.exists(self.save_path + f'feature_construction/vector_concatenation/train_label_{num}.npy') and\
                os.path.exists(self.save_path + f'feature_construction/vector_concatenation/test_data_{num}.npy') and \
                    os.path.exists(self.save_path + f'feature_construction/vector_concatenation/test_label_{num}.npy'):
                continue

            print('\n', transformation, para)
            train_ori_transformed_pictures = []
            train_adv_transformed_pictures = []
            test_ori_transformed_pictures = []
            test_adv_transformed_pictures = []
            for i in range(train_picture_number):
                print('\r{} / {}'.format(i, train_picture_number), end='')
                ori_picture = train_ori_picture[i]
                adv_picture = train_adv_picture[i]
                train_ori_transformed_pictures.append(self.transformation_function[transformation](ori_picture, para))
                train_adv_transformed_pictures.append(self.transformation_function[transformation](adv_picture, para))
            train_ori_transformed_pictures = np.array(train_ori_transformed_pictures)
            train_adv_transformed_pictures = np.array(train_adv_transformed_pictures)
            train_ori_transformed_softmax = model.predict(train_ori_transformed_pictures)
            train_adv_transformed_softmax = model.predict(train_adv_transformed_pictures)
            train_ori_softmax = np.concatenate((train_ori_softmax, train_ori_transformed_softmax), axis=1)
            train_adv_softmax = np.concatenate((train_adv_softmax, train_adv_transformed_softmax), axis=1)

            for i in range(test_picture_number):
                print('\r{} / {}'.format(i, test_picture_number), end='')
                ori_picture = test_ori_picture[i]
                adv_picture = test_adv_picture[i]
                test_ori_transformed_pictures.append(self.transformation_function[transformation](ori_picture, para))
                test_adv_transformed_pictures.append(self.transformation_function[transformation](adv_picture, para))
            test_ori_transformed_pictures = np.array(test_ori_transformed_pictures)
            test_adv_transformed_pictures = np.array(test_adv_transformed_pictures)
            test_ori_transformed_softmax = model.predict(test_ori_transformed_pictures)
            test_adv_transformed_softmax = model.predict(test_adv_transformed_pictures)
            test_ori_softmax = np.concatenate((test_ori_softmax, test_ori_transformed_softmax), axis=1)
            test_adv_softmax = np.concatenate((test_adv_softmax, test_adv_transformed_softmax), axis=1)

            train_data = np.concatenate((train_ori_softmax, train_adv_softmax), axis=0)
            train_label = np.array([0 for _ in range(train_picture_number)] + [1 for _ in range(train_picture_number)])
            test_data = np.concatenate((test_ori_softmax, test_adv_softmax), axis=0)
            test_label = np.array([0 for _ in range(test_picture_number)] + [1 for _ in range(test_picture_number)])
            train_index = [i for i in range(2*train_picture_number)]
            test_index = [i for i in range(2*test_picture_number)]
            random.shuffle(train_index)
            random.shuffle(test_index)
            train_data = train_data[train_index]
            train_label = train_label[train_index]
            test_data = test_data[test_index]
            test_label = test_label[test_index]

            np.save(self.save_path + f'feature_construction/vector_concatenation/train_data_{num}.npy', train_data)
            np.save(self.save_path + f'feature_construction/vector_concatenation/train_label_{num}.npy', train_label)
            np.save(self.save_path + f'feature_construction/vector_concatenation/test_data_{num}.npy', test_data)
            np.save(self.save_path + f'feature_construction/vector_concatenation/test_label_{num}.npy', test_label)

            num += 1

        return

    def feature_construction_vector_difference(self):
        model = tf.keras.models.load_model(self.model_save_path)
        train_ori_picture = np.load(self.save_path + 'train_ori_data.npy')
        train_adv_picture = np.load(self.save_path + 'train_adv_data.npy')
        test_ori_picture = np.load(self.save_path + 'test_ori_data.npy')
        test_adv_picture = np.load(self.save_path + 'test_adv_data.npy')
        train_picture_number = train_ori_picture.shape[0]
        test_picture_number = test_ori_picture.shape[0]

        train_ori_softmax = model.predict(train_ori_picture)
        train_adv_softmax = model.predict(train_adv_picture)
        test_ori_softmax = model.predict(test_ori_picture)
        test_adv_softmax = model.predict(test_adv_picture)

        train_ori_feature_vector = np.array([[] for _ in range(train_picture_number)])
        train_adv_feature_vector = np.array([[] for _ in range(train_picture_number)])
        test_ori_feature_vector = np.array([[] for _ in range(test_picture_number)])
        test_adv_feature_vector = np.array([[] for _ in range(test_picture_number)])

        transformations = self.transformation_and_parameter_selection(num=17)

        num = 1
        for t, para in transformations:
            if os.path.exists(self.save_path + f'feature_construction/vector_difference/train_data_{num}.npy') and \
                    os.path.exists(self.save_path + f'feature_construction/vector_difference/train_label_{num}.npy')and\
                    os.path.exists(self.save_path + f'feature_construction/vector_difference/test_data_{num}.npy') and \
                    os.path.exists(self.save_path + f'feature_construction/vector_difference/test_label_{num}.npy'):
                continue

            train_ori_transformed_pictures = []
            train_adv_transformed_pictures = []
            test_ori_transformed_pictures = []
            test_adv_transformed_pictures = []

            print(t, para)
            for i in range(train_picture_number):
                print('\r{} / {}'.format(i, train_picture_number), end='')
                ori_picture = train_ori_picture[i]
                adv_picture = train_adv_picture[i]
                train_ori_transformed_pictures.append(self.transformation_function[t](ori_picture, para))
                train_adv_transformed_pictures.append(self.transformation_function[t](adv_picture, para))
            train_ori_transformed_pictures = np.array(train_ori_transformed_pictures)
            train_adv_transformed_pictures = np.array(train_adv_transformed_pictures)
            train_ori_transformed_softmax = model.predict(train_ori_transformed_pictures)
            train_adv_transformed_softmax = model.predict(train_adv_transformed_pictures)

            train_ori_kl_divergence = np.array(calculate_kl(
                train_ori_softmax, train_ori_transformed_softmax)).reshape([-1, 1])
            train_ori_crossentropy = np.array(calculate_crossentropy(
                train_ori_softmax, train_ori_transformed_softmax)).reshape([-1, 1])
            train_ori_l1_distance = np.array(calculate_l1_distance(
                train_ori_softmax, train_ori_transformed_softmax)).reshape([-1, 1])
            train_ori_prediction_change = np.array(int(calculate_prediction_changes(
                train_ori_softmax, train_ori_transformed_softmax))).reshape([-1, 1])
            train_ori_feature_vector = np.concatenate(
                (train_ori_feature_vector, train_ori_kl_divergence, train_ori_crossentropy,
                 train_ori_l1_distance, train_ori_prediction_change), axis=1)

            train_adv_kl_divergence = np.array(calculate_kl(
                train_adv_softmax, train_adv_transformed_softmax)).reshape([-1, 1])
            train_adv_crossentropy = np.array(calculate_crossentropy(
                train_adv_softmax, train_adv_transformed_softmax)).reshape([-1, 1])
            train_adv_l1_distance = np.array(calculate_l1_distance(
                train_adv_softmax, train_adv_transformed_softmax)).reshape([-1, 1])
            train_adv_prediction_change = np.array(int(calculate_prediction_changes(
                train_adv_softmax, train_adv_transformed_softmax))).reshape([-1, 1])
            train_adv_feature_vector = np.concatenate(
                (train_adv_feature_vector, train_adv_kl_divergence,
                 train_adv_crossentropy, train_adv_l1_distance, train_adv_prediction_change), axis=1)

            for i in range(test_picture_number):
                print('\r{} / {}'.format(i, test_picture_number), end='')
                ori_picture = test_ori_picture[i]
                adv_picture = test_adv_picture[i]
                test_ori_transformed_pictures.append(self.transformation_function[t](ori_picture, para))
                test_adv_transformed_pictures.append(self.transformation_function[t](adv_picture, para))
            test_ori_transformed_pictures = np.array(test_ori_transformed_pictures)
            test_adv_transformed_pictures = np.array(test_adv_transformed_pictures)
            test_ori_transformed_softmax = model.predict(test_ori_transformed_pictures)
            test_adv_transformed_softmax = model.predict(test_adv_transformed_pictures)

            test_ori_kl_divergence = np.array(calculate_kl(
                test_ori_softmax, test_ori_transformed_softmax)).reshape([-1, 1])
            test_ori_crossentropy = np.array(calculate_crossentropy(
                test_ori_softmax, test_ori_transformed_softmax)).reshape([-1, 1])
            test_ori_l1_distance = np.array(calculate_l1_distance(
                test_ori_softmax, test_ori_transformed_softmax)).reshape([-1, 1])
            test_ori_prediction_change = np.array(int(calculate_prediction_changes(
                test_ori_softmax, test_ori_transformed_softmax))).reshape([-1, 1])
            test_ori_feature_vector = np.concatenate(
                (test_ori_feature_vector, test_ori_kl_divergence, test_ori_crossentropy,
                 test_ori_l1_distance, test_ori_prediction_change), axis=1)

            test_adv_kl_divergence = np.array(calculate_kl(
                test_adv_softmax, test_adv_transformed_softmax)).reshape([-1, 1])
            test_adv_crossentropy = np.array(calculate_crossentropy(
                test_adv_softmax, test_adv_transformed_softmax)).reshape([-1, 1])
            test_adv_l1_distance = np.array(calculate_l1_distance(
                test_adv_softmax, test_adv_transformed_softmax)).reshape([-1, 1])
            test_adv_prediction_change = np.array(int(calculate_prediction_changes(
                test_adv_softmax, test_adv_transformed_softmax))).reshape([-1, 1])
            test_adv_feature_vector = np.concatenate(
                (test_adv_feature_vector, test_adv_kl_divergence,
                 test_adv_crossentropy, test_adv_l1_distance, test_adv_prediction_change), axis=1)

            train_data = np.concatenate((train_ori_feature_vector, train_adv_feature_vector), axis=0)
            train_label = np.array([0 for _ in range(train_picture_number)] + [1 for _ in range(train_picture_number)])
            test_data = np.concatenate((test_ori_feature_vector, test_adv_feature_vector), axis=0)
            test_label = np.array([0 for _ in range(test_picture_number)] + [1 for _ in range(test_picture_number)])
            train_index = [i for i in range(2*train_picture_number)]
            test_index = [i for i in range(2*test_picture_number)]
            random.shuffle(train_index)
            random.shuffle(test_index)
            train_data = train_data[train_index]
            print(train_data.shape)
            train_label = train_label[train_index]
            test_data = test_data[test_index]
            test_label = test_label[test_index]

            np.save(self.save_path + f'feature_construction/vector_difference/train_data_{num}.npy', train_data)
            np.save(self.save_path + f'feature_construction/vector_difference/train_label_{num}.npy', train_label)
            np.save(self.save_path + f'feature_construction/vector_difference/test_data_{num}.npy', test_data)
            np.save(self.save_path + f'feature_construction/vector_difference/test_label_{num}.npy', test_label)

            num += 1

        return

    def train_classifier(self, feature_path):

        data_path = self.save_path + 'feature_construction/' + feature_path
        result_save_file = open(data_path + 'detector_result.txt', 'w')
        mlp_acc_list = []
        svm_acc_list = []
        rf_acc_list = []
        for i in range(1, 18):
            print('*************transformation: {}*******************************'.format(i))
            print('*************transformation: {}*******************************'.format(i), file=result_save_file)
            x_train = np.load(data_path + 'train_data_{}.npy'.format(i))
            y_train = np.load(data_path + 'train_label_{}.npy'.format(i))
            x_test = np.load(data_path + 'test_data_{}.npy'.format(i))
            y_test = np.load(data_path + 'test_label_{}.npy'.format(i))

            model_path = data_path + 'mlp_{}.h5'.format(i)
            mlp_acc = train_mlp_models(x_train, y_train, x_test, y_test, model_path, result_save_file)

            svm_acc = train_svm(x_train, y_train, x_test, y_test, result_save_file)

            rf_acc = train_random_frost(x_train, y_train, x_test, y_test, result_save_file)

            mlp_acc_list.append(mlp_acc)
            svm_acc_list.append(svm_acc)
            rf_acc_list.append(rf_acc)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', file=result_save_file)
        print('MLP ACC:', mlp_acc_list, file=result_save_file)
        print('SVM ACC:', svm_acc_list, file=result_save_file)
        print('RF ACC:', rf_acc_list, file=result_save_file)
        result_save_file.close()


if __name__ == '__main__':
    training_framework = TrainingFramework()
    # training_framework.data_sampling_random(refresh=False)
    # training_framework.calculate_ctr()
    # training_framework.calculate_baseline_detector_performance_on_mixed_attacks()
    # training_framework.feature_construction_vector_concatenation()
    # training_framework.feature_construction_vector_difference()
    training_framework.train_classifier(feature_path='vector_concatenation/')