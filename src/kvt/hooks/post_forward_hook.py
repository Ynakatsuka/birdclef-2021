from __future__ import absolute_import, division, print_function

import abc

import torch


class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs):
        return outputs


class SigmoidPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs):
        return torch.sigmoid(outputs)
