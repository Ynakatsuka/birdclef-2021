from .arcface import ArcFaceLoss
from .combo_loss import SegmentationWithClassificationHeadLoss
from .dice_loss import DiceLoss
from .focal_loss import BinaryFocalLoss, FocalLoss
from .lovasz_loss import LovaszHingeLoss, LovaszSoftmaxLoss
from .noisy_loss import (
    IterativeSelfLearningLoss,
    JointOptimizationLoss,
    LabelSmoothingCrossEntropy,
    OUSMLoss,
    SymmetricBCELoss,
    SymmetricBinaryFocalLoss,
    SymmetricCrossEntropy,
)
from .ohem_loss import OHEMLoss
from .vat import VATLoss
