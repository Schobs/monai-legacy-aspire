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

import logging
from typing import Callable, Optional

from monai_legacy_aspire.utils import ensure_tuple, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")
Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")


class LrScheduleHandler:
    """
    Ignite handler to update the Learning Rate based on PyTorch LR scheduler.
    """

    def __init__(
        self,
        lr_scheduler,
        print_lr: bool = True,
        name: Optional[str] = None,
        epoch_level: bool = True,
        step_transform: Callable = lambda engine: (),
    ):
        """
        Args:
            lr_scheduler (torch.optim.lr_scheduler): typically, lr_scheduler should be PyTorch
                lr_scheduler object. If customized version, must have `step` and `get_last_lr` methods.
            print_lr: whether to print out the latest learning rate with logging.
            name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
            epoch_level: execute lr_scheduler.step() after every epoch or every iteration.
                `True` is epoch level, `False` is iteration level.
            step_transform: a callable that is used to transform the information from `engine`
                to expected input data of lr_scheduler.step() function if necessary.

        Raises:
            ValueError: argument `step_transform` must be a callable.

        """
        self.lr_scheduler = lr_scheduler
        self.print_lr = print_lr
        self.logger = logging.getLogger(name)
        self.epoch_level = epoch_level
        if not callable(step_transform):
            raise ValueError("argument `step_transform` must be a callable.")
        self.step_transform = step_transform

        self._name = name

    def attach(self, engine: Engine):
        if self._name is None:
            self.logger = engine.logger
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine):
        args = ensure_tuple(self.step_transform(engine))
        self.lr_scheduler.step(*args)
        if self.print_lr:
            self.logger.info(f"Current learning rate: {self.lr_scheduler._last_lr[0]}")
