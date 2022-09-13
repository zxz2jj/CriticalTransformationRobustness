import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import seaborn as sns
import os
import pickle


class Draw(object):
    def __init__(self,):
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
                                # 'webp_compression'
                                ]
        self.transformation_parameter = {
                               'brightness': {'start': 1, 'end': 255, 'inc': 1, 'n': 0},
                               'darkness': {'start': 1, 'end': 255, 'inc': 1, 'n': 0},
                               'rotation_left': {'start': 1, 'end': 90, 'inc': 1, 'n': 0},
                               'rotation_right': {'start': 1, 'end': 90, 'inc': 1, 'n': 0},
                               'blur_mean': {'start': 2, 'end': 10, 'inc': 1, 'n': 0},
                               'shift_up': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_down': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_left': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'shift_right': {'start': 1, 'end': 14, 'inc': 1, 'n': 0},
                               'horizontal_shear_right': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'horizontal_shear_left': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'scale_big': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'scale_small': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'contrast_big': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'contrast_small': {'start': 0.01, 'end': 0.5, 'inc': 0.01, 'n': 2},
                               'jpeg_compression': {'start': 1, 'end': 100, 'inc': 1, 'n': 0},
                               'webp_compression': {'start': 1, 'end': 100, 'inc': 1, 'n': 0}
                               }
        self.attacks = ['BA', 'BrBeAttack_Inf', 'BrBeAttack_L2', 'CW_Inf', 'CW_L2', 'DeepFool', 'EAD', 'FGSM',
                        'IFGSM', 'HopSkipJumpAttack_Inf', 'HopSkipJumpAttack_L2', 'JSMA', 'NewtonFool', 'PGD_Inf',
                        'PGD_L2', 'PixelAttack', 'ShadowAttack', 'SpatialTransformationAttack',
                        'SquareAttack_Inf', 'SquareAttack_L2', 'WassersteinAttack', 'ZOO']
        self.attacks_abbreviation = {'BA': 'BA',
                                     'BrBeAttack_Inf': 'BBA_Inf',
                                     'BrBeAttack_L2': 'BBA_L2',
                                     'CW_Inf': "CW_Inf",
                                     'CW_L2': 'CW_L2',
                                     'DeepFool': 'DeepFool',
                                     'EAD': 'EAD',
                                     'FGSM': 'FGSM',
                                     'IFGSM':  'BIM',
                                     'HopSkipJumpAttack_Inf': 'HSJA_Inf',
                                     'HopSkipJumpAttack_L2': 'HSJA_L2',
                                     'JSMA': 'JSMA',
                                     'NewtonFool': 'NewtonFool',
                                     'PGD_Inf':  'PGD_Inf',
                                     'PGD_L2': 'PGD_L2',
                                     'PixelAttack': 'PixelAttack',
                                     'ShadowAttack': 'ShadowAttack',
                                     'SpatialTransformationAttack': 'STA',
                                     'SquareAttack_Inf': 'SA_Inf',
                                     'SquareAttack_L2': 'SA_L2',
                                     'WassersteinAttack': 'WA',
                                     'ZOO': 'ZOO'}
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

    def show_adversarial_label_distribution(self):
        for attack in self.attacks:
            ori_ground_truth_file = self.data_path[attack] + 'adv_test_ground_truth.npy'
            adv_target_file = self.data_path[attack] + 'adv_test_targets.npy'
            ori_ground_truth = np.load(ori_ground_truth_file)
            adv_target = np.load(adv_target_file)
            ori_ground_truth_distribution = []
            adv_target_distribution = []

            for c in range(10):
                ori_ground_truth_distribution.append(np.sum(ori_ground_truth == c) / ori_ground_truth.shape[0])
                adv_target_distribution.append(np.sum(adv_target == c) / adv_target.shape[0])

            print(attack, '\n', ori_ground_truth_distribution, '\n', adv_target_distribution)

            width = 0.25
            x = np.array([_ for _ in range(10)])
            plt.bar(x - width*0.5, ori_ground_truth_distribution, width, color='green', label='GroundTruth')
            plt.bar(x + width*0.5, adv_target_distribution, width, color='red', label='Target')

            plt.ylim(0.0, 1.0)
            plt.xticks(x, x)
            plt.legend()
            plt.show()

    def calculate_avg_ctr_for_each_transformation(self):
        for t in self.transformations:
            file = open(f'ctr_results/{t}/{t}_CTR_robustness.txt', 'w')
            for attack in self.attacks:
                print("-----------------------{}-------------------------".format(attack), file=file)
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                    ori_robustness = np.load(ori_robustness_path)
                    adv_robustness = np.load(adv_robustness_path)
                    print('Example Number: {}'.format(ori_robustness.shape), file=file)
                    print('Ori CTR robustness:{}'.format(np.average(ori_robustness)), file=file)
                    print('Adv CTR robustness:{}'.format(np.average(adv_robustness)), file=file)
            file.close()

        return

    def show_box_ctr_for_each_transformation(self):
        for t in self.transformations:
            print("draw CTR box picture of transformation: {}".format(t))
            box_data = pd.DataFrame()
            for attack in self.attacks:
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                ori_robustness = np.load(ori_robustness_path)
                adv_robustness = np.load(adv_robustness_path)
                box_data[attack+'_ori'] = pd.Series(ori_robustness)
                box_data[attack+'_adv'] = pd.Series(adv_robustness)

            box_data = pd.DataFrame(box_data)
            sns.boxplot(data=box_data)
            plt.xticks(rotation=-90)
            plt.tick_params(labelsize=10)
            ax = plt.gca()
            y_major_locator = MultipleLocator(10)
            ax.yaxis.set_major_locator(y_major_locator)
            plt.show()
            plt.close()

        return

    def show_avg_ctr_for_each_transformation(self):
        for t in self.transformations:
            print("draw picture of transformation: {}".format(t))
            ori_avg_robustness = []
            adv_avg_robustness = []
            attack_exist = []
            for attack in self.attacks:
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                    attack_exist.append(self.attacks_abbreviation[attack])
                    ori_robustness = np.load(ori_robustness_path)
                    adv_robustness = np.load(adv_robustness_path)
                    ori_avg_robustness.append(np.average(ori_robustness))
                    adv_avg_robustness.append(np.average(adv_robustness))

            x = [_ for _ in range(len(attack_exist))]
            plt.plot(x, ori_avg_robustness, color='blue', marker='o',
                     linestyle='-', label='ori_avg_robustness')
            for xy in zip(x, ori_avg_robustness):
                if xy[1] < 1:
                    plt.annotate('%.2f' % xy[1], xy=xy, xytext=(-10, 7), textcoords='offset points')
                else:
                    plt.annotate('%.1f' % xy[1], xy=xy, xytext=(-10, 7), textcoords='offset points')
            plt.plot(x, adv_avg_robustness, color='red', marker='*',
                     linestyle='-', label='adv_avg_robustness')
            for xy in zip(x, adv_avg_robustness):
                if xy[1] < 1:
                    plt.annotate('%.2f' % xy[1], xy=xy, xytext=(-10, 7), textcoords='offset points')
                else:
                    plt.annotate('%.1f' % xy[1], xy=xy, xytext=(-10, 7), textcoords='offset points')
            plt.xticks(x, attack_exist, rotation=-90)
            plt.legend(loc='lower right')
            plt.tick_params(labelsize=15)
            plt.show()
            plt.close()

        return

    def show_ctr_difference_for_each_transformation(self):
        for t in self.transformations:
            print("draw picture of transformation: {}".format(t))
            robustness_difference = {}
            for attack in self.attacks:
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                    attack_name_change = self.attacks_abbreviation[attack]
                    ori_robustness = np.load(ori_robustness_path)
                    adv_robustness = np.load(adv_robustness_path)
                    robustness_difference[attack_name_change] = np.average(ori_robustness) - np.average(adv_robustness)

            robustness_difference = sorted(robustness_difference.items(), key=lambda k: k[1], reverse=True)

            x = []
            y = []
            for item in robustness_difference:
                x.append(item[0])
                y.append(item[1])
            plt.bar(x=x, height=y, width=0.8)
            plt.axhline(0, color='k')
            plt.xticks(rotation=-90)
            plt.tick_params(labelsize=15)
            plt.show()

        return

    def show_baseline_detector_performance_for_each_transformation(self):
        detector_methods = ['CrossEntropy', 'KL', 'L1_Distance']
        for detector in detector_methods:
            print("****************{}****************".format(detector))
            for t in self.transformations:
                if self.transformation_parameter[t] is None:
                    continue
                print("draw picture of transformation: {}".format(t))
                transformation_parameter = []
                parameter = self.transformation_parameter[t]['start']
                while parameter < self.transformation_parameter[t]['end']:
                    parameter += self.transformation_parameter[t]['inc']
                    transformation_parameter.append(parameter)

                for attack in self.attacks:
                    performance_path = 'ctr_analyse_results/{}/{}_{}.pkl'.format(detector, t, attack)
                    if not os.path.exists(performance_path):
                        continue
                    performance_file = open(performance_path, 'rb')
                    performance = pickle.load(performance_file)
                    performance_list = []
                    for parameter in transformation_parameter:
                        performance_list.append(performance[parameter])
                    plt.plot(transformation_parameter, performance_list,
                             label="{}: ({:.2f}, {:.3f})".format(self.attacks_abbreviation[attack],
                                                                 transformation_parameter[performance_list.index(
                                                                    max(performance_list))],
                                                                 max(performance_list)))
                    # plt.plot(transformation_parameter, performance_list,
                    #          label="{}".format(self.attacks_abbreviation[attack]))

                plt.xlim(self.transformation_parameter[t]['start'],
                         self.transformation_parameter[t]['end'])
                plt.ylim(0, 1)
                # plt.legend(loc='lower left', bbox_to_anchor=(0, 0), ncol=3, prop={'size': 12})
                plt.legend(loc='upper right')
                plt.hlines(0.5, self.transformation_parameter[t]['start'],
                           self.transformation_parameter[t]['end'], colors='red', linestyles='--')
                plt.ylabel("AUROC", fontdict={'size': 15})
                plt.tick_params(labelsize=15)
                plt.show()

    def show_best_baseline_detector_performance(self):
        for t in self.transformations:
            print("draw picture of transformation: {}".format(t))
            robustness_difference = {}
            for attack in self.attacks:
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                    ori_robustness = np.load(ori_robustness_path)
                    adv_robustness = np.load(adv_robustness_path)
                    robustness_difference[attack] = np.average(ori_robustness) - np.average(adv_robustness)

            robustness_difference = sorted(robustness_difference.items(), key=lambda k: k[1], reverse=True)

            best_performance_crossentropy = []
            best_performance_kl = []
            best_performance_l1 = []
            for attack, _ in robustness_difference:
                crossentropy_path = f'ctr_analyse_results/CrossEntropy/{t}_{attack}.pkl'
                kl_path = f'ctr_analyse_results/KL/{t}_{attack}.pkl'
                l1_path = f'ctr_analyse_results/L1_Distance/{t}_{attack}.pkl'
                if os.path.exists(crossentropy_path) and os.path.exists(kl_path) and os.path.exists(l1_path):
                    crossentropy_performance = pickle.load(open(crossentropy_path, 'rb'))
                    kl_performance = pickle.load(open(kl_path, 'rb'))
                    l1_performance = pickle.load(open(l1_path, 'rb'))
                    crossentropy_performance = sorted(crossentropy_performance.items(), key=lambda k: k[1], reverse=True)
                    kl_performance = sorted(kl_performance.items(), key=lambda k: k[1], reverse=True)
                    l1_performance = sorted(l1_performance.items(), key=lambda k: k[1], reverse=True)
                    best_performance_crossentropy.append(crossentropy_performance[0][1])
                    best_performance_kl.append(kl_performance[0][1])
                    best_performance_l1.append(l1_performance[0][1])

            attack_list = []
            robustness_difference_list = []
            for item in robustness_difference:
                attack_list.append(self.attacks_abbreviation[item[0]])
                robustness_difference_list.append(item[1])

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            x = range(len(attack_list))
            line1 = ax1.plot(x, best_performance_crossentropy, label='CrossEntropy', marker='o')
            line2 = ax1.plot(x, best_performance_kl, label='KL', marker='o')
            line3 = ax1.plot(x, best_performance_l1, label='L1-Distance', marker='o')
            ax1.set_ylabel('AUROC', fontdict={'size': 12})
            ax1.set_ylim(0, 1)
            ax1.set_xlim(x[0], x[-1])
            for label in ax1.get_xticklabels():
                label.set_rotation(-90)
            ax1.hlines(0.5, 0, x[-1], colors='blue', linestyles='--')
            ax1.tick_params(labelsize=12)

            ax2 = ax1.twinx()
            line4 = ax2.plot(x, robustness_difference_list, label='CTR Difference', color='red', marker='o')
            ax2.set_ylabel('CTR Difference', fontdict={'size': 12})
            ax2.hlines(0, 0, x[-1], colors='red', linestyles='--')

            lines = line1 + line2 + line3 + line4
            labels = [line.get_label() for line in lines]
            plt.legend(lines, labels, loc='lower left')
            plt.xticks(x, attack_list)

            plt.show()

    def calculate_corrcoef_between_best_performance_and_ctr_difference(self):
        corrcoef_file = open('ctr_analyse_results/corrcoef.txt', 'a')
        for t in self.transformations:
            robustness_difference = {}
            for attack in self.attacks:
                ori_robustness_path = f'ctr_results/{t}/{t}_{attack}_ori_test_robustness.npy'
                adv_robustness_path = f'ctr_results/{t}/{t}_{attack}_adv_test_robustness.npy'
                if os.path.exists(ori_robustness_path) and os.path.exists(adv_robustness_path):
                    ori_robustness = np.load(ori_robustness_path)
                    adv_robustness = np.load(adv_robustness_path)
                    robustness_difference[attack] = np.average(ori_robustness) - np.average(adv_robustness)

            robustness_difference = sorted(robustness_difference.items(), key=lambda k: k[1], reverse=True)

            best_performance_crossentropy = []
            best_performance_kl = []
            best_performance_l1 = []
            for attack, _ in robustness_difference:
                crossentropy_path = f'ctr_analyse_results/CrossEntropy/{t}_{attack}.pkl'
                kl_path = f'ctr_analyse_results/KL/{t}_{attack}.pkl'
                l1_path = f'ctr_analyse_results/L1_Distance/{t}_{attack}.pkl'
                if os.path.exists(crossentropy_path) and os.path.exists(kl_path) and os.path.exists(l1_path):
                    crossentropy_performance = pickle.load(open(crossentropy_path, 'rb'))
                    kl_performance = pickle.load(open(kl_path, 'rb'))
                    l1_performance = pickle.load(open(l1_path, 'rb'))
                    crossentropy_performance = sorted(crossentropy_performance.items(), key=lambda k: k[1], reverse=True)
                    kl_performance = sorted(kl_performance.items(), key=lambda k: k[1], reverse=True)
                    l1_performance = sorted(l1_performance.items(), key=lambda k: k[1], reverse=True)
                    best_performance_crossentropy.append(crossentropy_performance[0][1])
                    best_performance_kl.append(kl_performance[0][1])
                    best_performance_l1.append(l1_performance[0][1])

            attack_list = []
            robustness_difference_list = []
            for item in robustness_difference:
                attack_list.append(self.attacks_abbreviation[item[0]])
                robustness_difference_list.append(item[1])

            corrcoef_crossentropy = np.corrcoef(robustness_difference_list, best_performance_crossentropy)
            corrcoef_kl = np.corrcoef(robustness_difference_list, best_performance_kl)
            corrcoef_l1 = np.corrcoef(robustness_difference_list, best_performance_l1)
            print("----------------{}-------------------".format(t), file=corrcoef_file)
            print('CrossEntropy: {}'.format(corrcoef_crossentropy[0][1]), file=corrcoef_file)
            print('KL: {}'.format(corrcoef_kl[0][1]), file=corrcoef_file)
            print('l1: {}'.format(corrcoef_l1[0][1]), file=corrcoef_file)

        corrcoef_file.close()
        print('result save path: ctr_analyse_results/corrcoef.txt')

        return

    def show_baseline_detector_performance_for_mixed_attacks(self):
        data_path = 'ctr_based_detector/'

        ori_label = np.load(data_path + 'random_sampling/test_ori_ground_truth.npy')
        adv_target = np.load(data_path + 'random_sampling/test_adv_target.npy')

        for t in self.transformations:
            print("draw picture of transformation: {}".format(t))
            transformation_parameter = []
            parameter = self.transformation_parameter[t]['start']
            while parameter < self.transformation_parameter[t]['end']:
                parameter += self.transformation_parameter[t]['inc']
                transformation_parameter.append(parameter)

            performance_path = data_path + 'baseline_detector_analyse/{}_AUROC_crossentropy.pkl'.format(t)
            if not os.path.exists(performance_path):
                continue
            performance_file = open(performance_path, 'rb')
            performance = pickle.load(performance_file)
            performance_list = []
            for parameter in transformation_parameter:
                performance_list.append(performance[parameter])

            crossentropy_path = data_path + 'baseline_detector_analyse/{}_crossentropy.pkl'.format(t)
            crossentropy_file = open(crossentropy_path, 'rb')
            crossentropy = pickle.load(crossentropy_file)
            ori_crossentropy_list = []
            adv_crossentropy_list = []
            difference_crossentropy = []
            for parameter in transformation_parameter:
                ori_crossentropy_list.append(np.average(crossentropy[parameter]['ori']))
                adv_crossentropy_list.append(np.average(crossentropy[parameter]['adv']))
                difference_crossentropy.append(np.average(crossentropy[parameter]['adv']) -
                                               np.average(crossentropy[parameter]['ori']))

            prediction_path = data_path + 'baseline_detector_analyse/{}_prediction.pkl'.format(t)
            prediction_file = open(prediction_path, 'rb')
            prediction = pickle.load(prediction_file)
            ori_prediction_list = []
            adv_prediction_list = []
            difference_prediction = []
            for parameter in transformation_parameter:
                ori_prediction = []
                adv_prediction = []
                for i in range(ori_label.shape[0]):
                    ori_prediction.append(prediction[parameter]['ori'][i][ori_label[i]])
                    adv_prediction.append(prediction[parameter]['adv'][i][adv_target[i]])
                ori_prediction_list.append(np.average(ori_prediction))
                adv_prediction_list.append(np.average(adv_prediction))
                difference_prediction.append(np.average(ori_prediction) - np.average(adv_prediction))

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            transformation_parameter = np.array(transformation_parameter) / \
                (self.transformation_parameter[t]['end'] -
                 self.transformation_parameter[t]['start'])
            ax1.plot(transformation_parameter, performance_list, color='blue',
                     label="{}".format(t))

            ax1.plot(transformation_parameter, adv_prediction_list, color='fuchsia',
                     label="Adv_Probability", linestyle='-')
            ax1.plot(transformation_parameter, ori_prediction_list, color='fuchsia',
                     label="Ori_Probability", linestyle='-.')

            ax1.set_ylabel('AUROC, Probability', fontdict={'size': 15})
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, 1)
            ax1.hlines(0.5, 0, 1, colors='red', linestyles='--')

            ori_robustness_path = data_path + 'ctr/{}_ori_test_robustness.npy'.format(t)
            adv_robustness_path = data_path + 'ctr/{}_adv_test_robustness.npy'.format(t)
            ori_robustness = np.load(ori_robustness_path)
            adv_robustness = np.load(adv_robustness_path)
            avg_ori_robustness = np.average(ori_robustness) / \
                (self.transformation_parameter[t]['end'] - self.transformation_parameter[t]['start'])
            avg_adv_robustness = np.average(adv_robustness) / \
                (self.transformation_parameter[t]['end'] - self.transformation_parameter[t]['start'])
            ax1.vlines(avg_adv_robustness, 0.0, 1.0, colors='red', label='CTR_Adv')
            ax1.vlines(avg_ori_robustness, 0.0, 1.0, colors='green', label='CTR_Ori')
            ax1.tick_params(labelsize=15)

            ax2 = ax1.twinx()
            ax2.plot(transformation_parameter, adv_crossentropy_list, color='lime',
                     label="Adv_CrossEntropy", linestyle='-')
            ax2.plot(transformation_parameter, ori_crossentropy_list, color='lime',
                     label="Ori_CrossEntropy", linestyle='-.')
            ax2.set_ylabel('CrossEntropy', fontdict={'size': 15})
            ax2.tick_params(labelsize=15)

            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, prop={'size': 10})
            plt.tick_params(labelsize=15)
            plt.show()

        return


if __name__ == "__main__":
    draw = Draw()
    # draw.show_adversarial_label_distribution()
    # draw.calculate_avg_ctr_for_each_transformation()
    # draw.show_box_ctr_for_each_transformation()
    # draw.show_avg_ctr_for_each_transformation()
    # draw.show_ctr_difference_for_each_transformation()
    # draw.show_baseline_detector_performance_for_each_transformation()
    # draw.show_best_baseline_detector_performance()
    # draw.calculate_corrcoef_between_best_performance_and_ctr_difference()
    draw.show_baseline_detector_performance_for_mixed_attacks()
