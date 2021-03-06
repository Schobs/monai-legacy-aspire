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

from typing import Optional, Callable

import numpy as np
from torch.utils.data import Dataset
from monai_legacy_aspire.transforms import LoadNifti, Randomizable, apply_transform
from monai_legacy_aspire.utils import get_seed


class NiftiDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of Nifti files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    """

    def __init__(
        self,
        image_files,
        seg_files=None,
        labels=None,
        as_closest_canonical: bool = False,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        image_only: bool = True,
        dtype: Optional[np.dtype] = np.float32,
    ):
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files (list of str): list of image filenames
            seg_files (list of str): if in segmentation task, list of segmentation filenames
            labels (list or array): if in classification task, list of classification labels
            as_closest_canonical: if True, load the image as closest to canonical orientation
            transform: transform to apply to image arrays
            seg_transform: transform to apply to segmentation arrays
            image_only: if True return only the image volume, other return image volume and header dict
            dtype (np.dtype, optional): if not None convert the loaded image to this data type

        Raises:
            ValueError: Must have same number of image and segmentation files

        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError("Must have same number of image and segmentation files")

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.as_closest_canonical = as_closest_canonical
        self.transform = transform
        self.seg_transform = seg_transform
        self.image_only = image_only
        self.dtype = dtype
        self.set_random_state(seed=get_seed())

        self._seed = 0  # transform synchronization seed

    def __len__(self):
        return len(self.image_files)

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._seed = self.R.randint(np.iinfo(np.int32).max)

    def __getitem__(self, index: int):
        self.randomize()
        meta_data = None
        img_loader = LoadNifti(
            as_closest_canonical=self.as_closest_canonical, image_only=self.image_only, dtype=self.dtype
        )
        if self.image_only:
            img = img_loader(self.image_files[index])
        else:
            img, meta_data = img_loader(self.image_files[index])
        seg = None
        if self.seg_files is not None:
            seg_loader = LoadNifti(image_only=True)
            seg = seg_loader(self.seg_files[index])
        label = None
        if self.labels is not None:
            label = self.labels[index]

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            img = apply_transform(self.transform, img)

        data = [img]

        if self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)
            seg = apply_transform(self.seg_transform, seg)

        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if not self.image_only and meta_data is not None:
            data.append(meta_data)
        if len(data) == 1:
            return data[0]
        return data
