import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
from metrics import calculate_crossentropy, calculate_kl, calculate_l1_distance, calculate_auroc


class Transformation(object):

    @staticmethod
    def brightness(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            return np.clip((ori_picture + float(theta / 255.)), 0, 1)
        else:
            return np.clip((ori_picture + float(theta)), 0, 255) / 255.

    @staticmethod
    def darkness(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            return np.clip((ori_picture - float(theta / 255.)), 0, 1)
        else:
            return np.clip((ori_picture - float(theta)), 0, 255) / 255.

    @staticmethod
    def rotation_left(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        rotation_center = (width / 2.0, height / 2.0)
        affine_matrix = cv2.getRotationMatrix2D(rotation_center, theta, 1.0)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def rotation_right(ori_picture, theta):
        theta = - abs(theta)
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        rotation_center = (width / 2.0, height / 2.0)
        affine_matrix = cv2.getRotationMatrix2D(rotation_center, theta, 1.0)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def shift_up(ori_picture, y_shift):
        y_shift = - abs(y_shift)
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, 0, 0], [0, 1, y_shift]], dtype=np.float32)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def shift_down(ori_picture, y_shift):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, 0, 0], [0, 1, y_shift]], dtype=np.float32)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def shift_left(ori_picture, x_shift):
        x_shift = -abs(x_shift)
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, 0, x_shift], [0, 1, 0]], dtype=np.float32)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def shift_right(ori_picture, x_shift):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, 0, x_shift], [0, 1, 0]], dtype=np.float32)
        transformed_picture = cv2.warpAffine(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def mean_blur(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        transformed_picture = cv2.blur(ori_picture, (theta, theta))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def horizontal_shear_left(ori_picture, theta):
        theta = - abs(theta)
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, theta, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        transformed_picture = cv2.warpPerspective(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def horizontal_shear_right(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1, theta, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        transformed_picture = cv2.warpPerspective(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def scale_small(ori_picture, theta):
        theta = - abs(theta)
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1+theta, 0, 0], [0, 1+theta, 0], [0, 0, 1]], dtype=np.float32)
        transformed_picture = cv2.warpPerspective(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def scale_big(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        affine_matrix = np.array([[1+theta, 0, 0], [0, 1+theta, 0], [0, 0, 1]], dtype=np.float32)
        transformed_picture = cv2.warpPerspective(ori_picture, affine_matrix, (width, height))
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture

    @staticmethod
    def contrast_small(ori_picture, theta):
        theta = - abs(theta)
        ori_picture = np.array(ori_picture)
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            transformed_picture = np.clip(ori_picture * (1 + theta), 0, 1)
            return transformed_picture
        else:
            transformed_picture = np.clip((ori_picture * (1 + theta)).astype('int'), 0, 255)
            return transformed_picture

    @staticmethod
    def contrast_big(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            transformed_picture = np.clip(ori_picture * (1 + theta), 0, 1)
            return transformed_picture
        else:
            transformed_picture = np.clip((ori_picture * (1 + theta)).astype('int'), 0, 255)
            return transformed_picture

    @staticmethod
    def jpeg_compression(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            ori_picture = np.clip((ori_picture * 255), 0, 255)
        cv2.imwrite('JPEG_compression.jpeg', ori_picture, [cv2.IMWRITE_JPEG_QUALITY, 100-theta])
        if channel == 1:
            transformed_picture = cv2.imread('JPEG_compression.jpeg', cv2.IMREAD_GRAYSCALE)
        else:
            transformed_picture = cv2.imread('JPEG_compression.jpeg')
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture / 255

    @staticmethod
    def webp_compression(ori_picture, theta):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        max_pixel = np.max(ori_picture)
        if max_pixel <= 1:
            ori_picture = np.clip((ori_picture * 255), 0, 255)
        cv2.imwrite('WebP_compression.webp', ori_picture, [cv2.IMWRITE_WEBP_QUALITY, 100-theta])
        if channel == 1:
            transformed_picture = cv2.imread('WebP_compression.webp', cv2.IMREAD_GRAYSCALE)
        else:
            transformed_picture = cv2.imread('WebP_compression.webp')
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture / 255

    @staticmethod
    def bit_depth_reduction(ori_picture, theta):
        pass

    @staticmethod
    def flip_up2bottom(ori_picture):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        transformed_picture = cv2.flip(ori_picture, 1)
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])
        return transformed_picture

    @staticmethod
    def flip_left2right(ori_picture):
        ori_picture = np.array(ori_picture)
        width = ori_picture.shape[0]
        height = ori_picture.shape[1]
        channel = ori_picture.shape[2]
        transformed_picture = cv2.flip(ori_picture, 0)
        transformed_picture = np.reshape(transformed_picture, [width, height, channel])

        return transformed_picture


class CriticalTransformationRobustness(Transformation):
    def __init__(self, attack_methods):
        self.model_save_path = 'model/fmnist_model.h5'
        self.attack_methods = attack_methods
        self.transformation = [
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
                               # 'webp_compression'
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

    def ctr_for_each_attack(self,):
        print('********************** Calculate CTR ********************************')
        model = tf.keras.models.load_model(self.model_save_path)
        for t in self.transformation:
            print("------------Transformation: {}-------------".format(t))
            print('parameter: {}'.format(self.transformation_parameter[t]))
            for attack in self.attack_methods:
                ori_picture_path = self.data_path[attack] + 'ori_test_successful_attack.npy'
                ori_label_path = self.data_path[attack] + 'adv_test_ground_truth.npy'
                adv_picture_path = self.data_path[attack] + 'adv_test.npy'
                adv_target_path = self.data_path[attack] + 'adv_test_targets.npy'
                if not os.path.exists(ori_picture_path) or not os.path.exists(ori_label_path) \
                        or not os.path.exists(adv_picture_path) or not os.path.exists(adv_target_path):
                    print("There is no data in: {}.".format(self.data_path[attack]))
                    continue
                ori_ctr_file = "ctr_results/{}/{}_{}_ori_test_robustness.npy".format(t, t, attack)
                adv_ctr_file = "ctr_results/{}/{}_{}_adv_test_robustness.npy".format(t, t, attack)
                if os.path.exists(ori_ctr_file) and os.path.exists(adv_ctr_file):
                    print("{} on {} is finished!".format(t, attack))
                    continue

                ori_picture = np.load(ori_picture_path)
                ori_label = np.load(ori_label_path)
                adv_picture = np.load(adv_picture_path)
                adv_target = np.load(adv_target_path)
                picture_number = adv_picture.shape[0]

                print("Adversarial Examples' Number: {}".format(picture_number))

                transformation_parameter = []
                parameter = self.transformation_parameter[t]['start']
                while parameter < self.transformation_parameter[t]['end']:
                    parameter += self.transformation_parameter[t]['inc']
                    transformation_parameter.append(parameter)

                ori_ctr = []
                picture_number = ori_picture.shape[0]
                for theta in transformation_parameter:
                    print('{}  {}/{} : '.format(t, theta, transformation_parameter[-1]), end=' ')
                    transformed_pictures = []
                    for i in range(picture_number):
                        picture = ori_picture[i]
                        transformed_pictures.append(self.transformation_function[t](picture, theta))
                    transformed_pictures = np.array(transformed_pictures)
                    predictions = np.argmax(model.predict(transformed_pictures), axis=1)
                    equals = predictions == ori_label
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
                print("\rClean Pictures Robustness: Num-{}, Avg-{:5}, Max-{:5}, Min-{:5}".
                      format(ori_ctr.shape, np.average(ori_ctr), np.max(ori_ctr), np.min(ori_ctr)))
                np.save(ori_ctr_file, ori_ctr)

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
                print("\rAdversarial test example, Robustness: Num-{}, Avg-{:5}, Max-{:5}, Min-{:5}".
                      format(adv_ctr.shape, np.average(adv_ctr), np.max(adv_ctr), np.min(adv_ctr)))
                np.save(adv_ctr_file, adv_ctr)

        return

    def analyze_ctr_on_baseline_detector(self,):
        print('********************** Analyse CTR on Baseline Detector********************************')
        model = tf.keras.models.load_model(self.model_save_path)
        for t in self.transformation:
            print("------------Transformation: {}-------------".format(t))
            print('parameter: {}'.format(self.transformation_parameter[t]))
            for attack in self.attack_methods:
                if os.path.exists('ctr_analyse_results/CrossEntropy/{}_{}.pkl'.format(t, attack)) and \
                        os.path.exists('ctr_analyse_results/KL/{}_{}.pkl'.format(t, attack)) and \
                        os.path.exists('ctr_analyse_results/L1_Distance/{}_{}.pkl'.format(t, attack)):
                    print(f"{t}-{attack} is finished!")
                    continue

                ori_picture_path = self.data_path[attack] + 'ori_test_successful_attack.npy'
                adv_picture_path = self.data_path[attack] + 'adv_test.npy'
                if not os.path.exists(ori_picture_path) or not os.path.exists(adv_picture_path):
                    print("There is no data in: {}.".format(self.data_path[attack]))
                    continue

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
                auroc_of_kl = {}
                auroc_of_l0_distance = {}
                for para in transformation_parameter:
                    crossentropy_dict = {'ori': None, 'adv': None}
                    kl_dict = {'ori': None, 'adv': None}
                    l0_dict = {'ori': None, 'adv': None}
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

                    crossentropy_dict['ori'] = calculate_crossentropy(ori_prediction, ori_transformed_prediction)
                    crossentropy_dict['adv'] = calculate_crossentropy(adv_prediction, adv_transformed_prediction)
                    kl_dict['ori'] = calculate_kl(ori_prediction, ori_transformed_prediction)
                    kl_dict['adv'] = calculate_kl(adv_prediction, adv_transformed_prediction)
                    l0_dict['ori'] = calculate_l1_distance(ori_prediction, ori_transformed_prediction)
                    l0_dict['adv'] = calculate_l1_distance(adv_prediction, adv_transformed_prediction)

                    print('CrossEntropy ', end='')
                    auroc_crossentropy = calculate_auroc(crossentropy_dict, picture_number)
                    print('KL Divergence ', end='')
                    auroc_kl = calculate_auroc(kl_dict, picture_number)
                    print('L0 Distance ', end='')
                    auroc_l0 = calculate_auroc(l0_dict, picture_number)
                    auroc_of_crossentropy[para] = auroc_crossentropy
                    auroc_of_kl[para] = auroc_kl
                    auroc_of_l0_distance[para] = auroc_l0

                file1 = open('ctr_analyse_results/CrossEntropy/{}_{}.pkl'.format(t, attack), 'wb')
                file2 = open('ctr_analyse_results/KL/{}_{}.pkl'.format(t, attack), 'wb')
                file3 = open('ctr_analyse_results/L1_Distance/{}_{}.pkl'.format(t, attack), 'wb')
                pickle.dump(auroc_of_crossentropy, file1)
                pickle.dump(auroc_of_kl, file2)
                pickle.dump(auroc_of_l0_distance, file3)
                file1.close()
                file2.close()
                file3.close()
        return


if __name__ == '__main__':
    attacks = ['BA', 'BrBeAttack_Inf', 'BrBeAttack_L2', 'CW_Inf', 'CW_L2', 'DeepFool', 'EAD', 'FGSM',
               'IFGSM', 'HopSkipJumpAttack_Inf', 'HopSkipJumpAttack_L2', 'JSMA', 'NewtonFool', 'PGD_Inf',
               'PGD_L2', 'PixelAttack', 'ShadowAttack', 'SpatialTransformationAttack',
               'SquareAttack_Inf', 'SquareAttack_L2', 'WassersteinAttack', 'ZOO']
    ctr = CriticalTransformationRobustness(attacks)
    ctr.ctr_for_each_attack()
    ctr.analyze_ctr_on_baseline_detector()

