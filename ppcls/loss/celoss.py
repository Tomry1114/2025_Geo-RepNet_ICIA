# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils import logger


class CELoss(nn.Layer):
    """
    Cross entropy loss
    """

    def __init__(self, reduction="mean", epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def _labelsmoothing(self, target, class_num):
        # case 1: label 是一维整型标签，进行 one-hot 编码
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            if not target.dtype in [paddle.int64, paddle.int32]:
                # 需要转换为整型才能使用 F.one_hot
                target = paddle.cast(target, 'int64')
            one_hot_target = F.one_hot(target, class_num)
        else:
            # case 2: 已是 one-hot 编码或 soft label，直接用
            one_hot_target = target

        # label_smooth 要求 float32
        one_hot_target = paddle.cast(one_hot_target, 'float32')

        # 应用 label smoothing
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)

        # reshape 成统一形状 [N, class_num]
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target


    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(
                x,
                label=label,
                soft_label=soft_label,
                reduction=self.reduction)
        return {"CELoss": loss}


class MixCELoss(object):
    def __init__(self, *args, **kwargs):
        msg = "\"MixCELos\" is deprecated, please use \"CELoss\" instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)
