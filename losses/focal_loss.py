# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

from monai_legacy_aspire.utils import LossReduction


class FocalLoss(_WeightedLoss):
    """
    Reimplementation of the Focal Loss described in:

        - "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole‐volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight (tensor): weights to apply to the voxels of each class. If None no weights are applied.
                This corresponds to the weights `\alpha` in [1].
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Example:
            .. code-block:: python

                import torch
                from monai.losses import FocalLoss

                pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
                grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
                fl = FocalLoss()
                fl(pred, grnd)

        """
        super(FocalLoss, self).__init__(weight=weight, reduction=LossReduction(reduction))
        self.gamma = gamma

    def forward(self, input, target):
        """
        Args:
            input: (tensor): the shape should be BCH[WD].
                where C is the number of classes.
            target: (tensor): the shape should be B1H[WD] or BCH[WD].
                If the target's shape is B1H[WD], the target that this loss expects should be a class index
                in the range [0, C-1] where C is the number of classes.

        Raises:
            ValueError: input and target must have the same number of dimensions, got {i.ndim} and {t.ndim}
            ValueError: target must have one channel or have the same shape as the input.
                If it has one channel, it should be a class index in the range [0, C-1]
                where C is the number of classes inferred from 'input': C={i.shape[1]}.
            ValueError: reduction={self.reduction} is invalid.

        """
        i = input
        t = target

        if i.ndim != t.ndim:
            raise ValueError(f"input and target must have the same number of dimensions, got {i.ndim} and {t.ndim}")

        if target.shape[1] != 1 and target.shape[1] != i.shape[1]:
            raise ValueError(
                "target must have one channel or have the same shape as the input. "
                "If it has one channel, it should be a class index in the range [0, C-1] "
                f"where C is the number of classes inferred from 'input': C={i.shape[1]}. "
            )
        # Change the shape of input and target to
        # num_batch x num_class x num_voxels.
        if input.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.view(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W or N,C,H*W
        else:  # Compatibility with classification.
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1 or N,C,1

        # Compute the log proba (more stable numerically than softmax).
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel.
        if target.shape[1] == 1:
            logpt = logpt.gather(1, t.long())  # N,C,H*W => N,1,H*W
            logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W or N,C,H*W

        if self.weight is not None:
            self.weight = self.weight.to(i)
            # Convert the weight to a map in which each voxel
            # has the weight associated with the ground-truth label
            # associated with this voxel in target.
            at = self.weight[None, :, None]  # C => 1,C,1
            at = at.expand((t.size(0), -1, t.size(2)))  # 1,C,1 => N,C,H*W
            if target.shape[1] == 1:
                at = at.gather(1, t.long())  # selection of the weights  => N,1,H*W
                at = torch.squeeze(at, dim=1)  # N,1,H*W => N,H*W
            # Multiply the log proba by their weights.
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        if target.shape[1] == 1:
            loss = torch.mean(-weight * logpt, dim=1)  # N
        else:
            loss = torch.mean(-weight * t * logpt, dim=-1)  # N,C

        if self.reduction == LossReduction.SUM:
            return loss.sum()
        if self.reduction == LossReduction.NONE:
            return loss
        if self.reduction == LossReduction.MEAN:
            return loss.mean()
        raise ValueError(f"reduction={self.reduction} is invalid.")
