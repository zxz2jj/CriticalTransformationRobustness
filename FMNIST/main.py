import numpy as np
import sys

from fmnist_model import train_models
from attack import BA, BrBeAttack, CWInf, CWL2, DeepFoolAttack, EAD, FGSM, HopSkipJumpAttack, IFGSM, JSMA, \
    NewtonFoolAttack, PGD, PixelsAttack, ShadowAttacks, SpatialTransformationAttack, SquareAttacks, WassersteinAttack, ZOO
from calculate_ctr import CriticalTransformationRobustness
from draw_picture import Draw
from ctr_based_training_framework import TrainingFramework


if __name__ == '__main__':
    # train deep learning models
    train_models()

    # generate adversarial attacks
    attack_names = ['BA', 'BrBeAttack-Inf', 'BrBeAttack-L2', 'CW-Inf', 'CW-L2', 'DeepFool', 'EAD', 'FGSM',
                    'HopSkipJumpAttack-Inf', 'HopSkipJumpAttack-L2', 'IFGSM', 'JSMA', 'NewtonFool', 'PGD-Inf', 'PGD-L2',
                    'PixelAttack', 'ShadowAttack', 'SpatialTransformationAttack', 'SquareAttack-Inf', 'SquareAttacks-L2',
                    'WassersteinAttack', 'ZOO']
    ba = BA()
    bebe_inf = BrBeAttack(norm=np.inf)
    bebe_l2 = BrBeAttack(norm=2)
    cw_inf = CWInf()
    cw_l2 = CWL2()
    deepfool = DeepFoolAttack()
    ead = EAD()
    fgsm = FGSM()
    hsja_inf = HopSkipJumpAttack(norm=np.inf)
    hsja_l2 = HopSkipJumpAttack(norm=2)
    ifgsm = IFGSM()
    jsma = JSMA()
    newtonfool = NewtonFoolAttack()
    pgd_inf = PGD(norm=np.inf)
    pgd_l2 = PGD(norm=2)
    pixelattack = PixelsAttack()
    shadowattack = ShadowAttacks()
    sta = SpatialTransformationAttack()
    sa_inf = SquareAttacks(norm=np.inf)
    sa_l2 = SquareAttacks(norm=2)
    wa = WassersteinAttack(norm='2')
    zoo = ZOO()

    attack_methods = [ba, bebe_inf, bebe_l2, cw_inf, cw_l2, deepfool, ead, fgsm, hsja_inf, hsja_l2, ifgsm, jsma,
                      newtonfool, pgd_inf, pgd_l2, pixelattack, shadowattack, sta, sa_inf, sa_l2, wa, zoo]
    for attack in attack_methods:
        name = attack.save_path.split('/')[-2]
        if not attack.is_finished():
            attack_names.remove(name)

    if not attack_names:
        print("No adversarial data, please run 'attack.py' to generate！")
        sys.exit(0)

    ctr = CriticalTransformationRobustness(attack_names)
    ctr.ctr_for_each_attack()
    ctr.analyze_ctr_on_baseline_detector()

    draw = Draw()

    flag = True
    if flag:
        draw.show_adversarial_label_distribution()
    flag = False
    if flag:
        draw.calculate_avg_ctr_for_each_transformation()
    flag = False
    if flag:
        draw.show_box_ctr_for_each_transformation()
    flag = True
    if flag:
        draw.show_avg_ctr_for_each_transformation()
    flag = False
    if flag:
        draw.show_ctr_difference_for_each_transformation()
    flag = True
    if flag:
        draw.show_baseline_detector_performance_for_each_transformation()
    flag = True
    if flag:
        draw.show_best_baseline_detector_performance()
    flag = False
    if flag:
        draw.calculate_corrcoef_between_best_performance_and_ctr_difference()

    training_framework = TrainingFramework()
    training_framework.data_sampling_random(refresh=False)
    training_framework.calculate_ctr()
    training_framework.calculate_baseline_detector_performance_on_mixed_attacks()
    flag = True
    if flag:
        draw.show_baseline_detector_performance_for_mixed_attacks()

    training_framework.feature_construction_vector_concatenation()
    training_framework.train_classifier(feature_path='vector_concatenation/')

    training_framework.feature_construction_vector_difference()
    training_framework.train_classifier(feature_path='vector_difference/')


