from core.task.label_permuted_cifar10 import LabelPermutedCIFAR10
from core.task.utility_task import UtilityTask

from core.network.fcn_relu import ConvolutionalNetworkReLU, ConvolutionalNetworkReLUWithHooks

from core.learner.sgd import SGDLearner, SGDLearnerWithHesScale
from core.learner.adam import AdamLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.ewc import EWCLearner
from core.learner.scaled_noise_upgd import UPGDScaledWeightNormNoiseLearner,UPGDScaledGradNormNoiseLearner,UPGDScaledAdativeNormNoiseDLearner
from core.learner.upgd_sgd import UPGD_SGD_Learner, UPGD_DynamicclippedGradient_Learner

from core.learner.weight_upgd import FirstOrderLocalUPGDLearner, SecondOrderLocalUPGDLearner, FirstOrderGlobalUPGDLearner, SecondOrderGlobalUPGDLearner
from core.learner.kernel_avg import UPGD_KernelLearner
from core.learner.column_kernel_avg import UPGD_ColumnKernelLearner
from core.utilities.weight.fo_utility import FirstOrderUtility
from core.utilities.weight.so_utility import SecondOrderUtility
from core.utilities.weight.weight_utility import WeightUtility
from core.utilities.weight.oracle_utility import OracleUtility
from core.utilities.weight.grad2_utility import SquaredGradUtility

import torch
import numpy as np


tasks = {
    "weight_utils": UtilityTask,
    "feature_utils": UtilityTask,
    "label_permuted_cifar10" : LabelPermutedCIFAR10,
    "label_permuted_cifar10_stats" : LabelPermutedCIFAR10,
}

networks = {
    "convolutional_network_relu": ConvolutionalNetworkReLU,
    "convolutional_network_relu_with_hooks": ConvolutionalNetworkReLUWithHooks,
}

learners = {
    "sgd": SGDLearner,
    "sgd_with_hesscale": SGDLearnerWithHesScale,
    "adam": AdamLearner,
    "shrink_and_perturb": ShrinkandPerturbLearner,
    "ewc": EWCLearner,
    "upgd_fo_local": FirstOrderLocalUPGDLearner,
    "UPGDScaledWeightNormNoise":UPGDScaledWeightNormNoiseLearner,
    "UPGDScaledGradNormNoiseLearner":UPGDScaledGradNormNoiseLearner,
    "UPGDScaledAdativeNormNoiseDLearner":UPGDScaledAdativeNormNoiseDLearner,
    "upgd_sgd": UPGD_SGD_Learner,
    "upgd_dynamicclippedgradient": UPGD_DynamicclippedGradient_Learner,
    "upgd_kernel":UPGD_KernelLearner,
    "upgd_column_kernel": UPGD_ColumnKernelLearner,
    "upgd_so_local": SecondOrderLocalUPGDLearner,
    "upgd_fo_global": FirstOrderGlobalUPGDLearner,
    "upgd_so_global": SecondOrderGlobalUPGDLearner,
}

criterions = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
}

utility_factory = {
    "first_order": FirstOrderUtility,
    "second_order": SecondOrderUtility,
    "weight": WeightUtility,
    "g2": SquaredGradUtility,
    "oracle": OracleUtility,
}

#keep
def compute_spearman_rank_coefficient(approx_utility, oracle_utility):
    approx_list = []
    oracle_list = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        oracle_list += list(oracle.ravel().numpy())
        approx_list += list(fo.ravel().numpy())

    overall_count = len(approx_list)
    approx_list = np.argsort(np.asarray(approx_list))
    oracle_list = np.argsort(np.asarray(oracle_list))

    difference = np.sum((approx_list - oracle_list) ** 2)
    coeff = 1 - 6.0 * difference / (overall_count * (overall_count**2-1))
    return coeff

#keep    
def compute_spearman_rank_coefficient_layerwise(approx_utility, oracle_utility):
    coeffs = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        overall_count = len(list(oracle.ravel().numpy()))
        if overall_count == 1:
            continue
        oracle_list = np.argsort(list(oracle.ravel().numpy()))
        approx_list = np.argsort(list(fo.ravel().numpy()))
        difference = np.sum((approx_list - oracle_list) ** 2)
        coeff = 1 - 6.0 * difference / (overall_count * (overall_count**2-1))
        coeffs.append(coeff)
    coeff_average = np.mean(np.array(coeffs))
    return coeff_average
