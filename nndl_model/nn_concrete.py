import torch
import torch.nn as nn

from .mixin.base_nn import BaseNN
from .mixin.eval_mixin import ClassificationEvalMixin, RegressionEvalMixin
from .mixin.improvement_float import CheckImprovementFloatMixin
from .mixin.prior_mixin import PriorMixin
from .mixin.train_mixin import SingleVariateMixin


class RegressionNN(SingleVariateMixin, RegressionEvalMixin, CheckImprovementFloatMixin, PriorMixin, BaseNN):
    _criterion = nn.MSELoss()


class ClassificationNN(SingleVariateMixin, ClassificationEvalMixin, CheckImprovementFloatMixin, PriorMixin, BaseNN):
    _criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiLabelClassificationNN(
    SingleVariateMixin, ClassificationEvalMixin, CheckImprovementFloatMixin, PriorMixin, BaseNN
):
    _criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)
