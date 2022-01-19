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
"""
A collection of "vanilla" transforms for IO functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional

import numpy as np


from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai_legacy_aspire.data.utils import correct_nifti_header_if_necessary
from monai_legacy_aspire.transforms.compose import Transform
from monai_legacy_aspire.utils import optional_import, ensure_tuple

# @lawrence schobs changes
import matplotlib.pyplot as plt
from skimage import io


nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")


class LoadNifti(Transform):
    """
    Load Nifti format file or files from provided path. If loading a list of
    files, stack them together and add a new dimension as first dimension, and
    use the meta data of the first image to represent the stacked result. Note
    that the affine transform of all the images should be same if ``image_only=False``.
    """

    def __init__(
        self, as_closest_canonical: bool = False, image_only: bool = False, dtype: Optional[np.dtype] = np.float32
    ):
        """
        Args:
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.

        Note:
            The transform returns image data array if `image_only` is True,
            or a tuple of two elements containing the data array, and the Nifti
            header in a dict format otherwise.
            if a dictionary header is returned:

            - header['affine'] stores the affine of the image.
            - header['original_affine'] will be additionally created to store the original affine.
        """
        self.as_closest_canonical = as_closest_canonical
        self.image_only = image_only
        self.dtype = dtype

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        compatible_meta = dict()
        for name in filename:
            img = nib.load(name)
            img = correct_nifti_header_if_necessary(img)
            header = dict(img.header)
            header["filename_or_obj"] = name
            header["affine"] = img.affine
            header["original_affine"] = img.affine.copy()
            header["as_closest_canonical"] = self.as_closest_canonical
            ndim = img.header["dim"][0]
            spatial_rank = min(ndim, 3)
            header["spatial_shape"] = img.header["dim"][1 : spatial_rank + 1]

            if self.as_closest_canonical:
                img = nib.as_closest_canonical(img)
                header["affine"] = img.affine

            img_array.append(np.array(img.get_fdata(dtype=self.dtype)))
            img.uncache()

            if self.image_only:
                continue

            if not compatible_meta:
                for meta_key in header:
                    meta_datum = header[meta_key]
                    # pytype: disable=attribute-error
                    if (
                        type(meta_datum).__name__ == "ndarray"
                        and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None
                    ):
                        continue
                    # pytype: enable=attribute-error
                    compatible_meta[meta_key] = meta_datum
            else:
                assert np.allclose(
                    header["affine"], compatible_meta["affine"]
                ), "affine data of all images should be same."

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        if self.image_only:
            return img_array
        return img_array, compatible_meta


class LoadPNG(Transform):
    """
    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    It's based on the Image module in PIL library:
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """
   
    def __init__(self, image_only: bool = False, dtype: Optional[np.dtype] = np.float32):
        """
        Args:
            image_only: if True return only the image volume, otherwise return image data array and metadata.
            dtype: if not None convert the loaded image to this data type.
        """
        self.image_only = image_only
        self.dtype = dtype

    def centre_crop(self, img):
        return img
    # def centre_crop(self, img, size=256 ):
    #     # print(" cc ", img.shape)
    #     if img.ndim > 2:
    #         new_array = np.zeros((img.shape[0], size, size))
    #         # print("new array shape: ", new_array.shape)
    #         counter = 0
    #         for heatmap in img:
    #             res = heatmap.shape[0]
    #             half = size//2
    #             centre = res//2
    #             max_y = centre + half
    #             min_y = centre - half
    #             max_x = centre + half
    #             min_x = centre - half



    #             cropped_im = heatmap[min_y:max_y, min_x:max_x]
             
    #             new_array[counter] = cropped_im

            
    #             counter += 1

    #         # print("new array aftwewards shape: ", new_array.shape)
    #         return new_array

    #     else:
    #         res = img.shape[0]
    #         half = size//2
    #         centre = res//2
    #         max_y = centre + half
    #         min_y = centre - half
    #         max_x = centre + half
    #         min_x = centre - half

    #         cropped_im = img[min_y:max_y, min_x:max_x]
    #         return cropped_im

    #lets load the image, then we need to center it. 
    # then we should change the pixel range to be between 0 and 1
    def load_image(self, img_path):

        im = io.imread(img_path, as_gray=True)

    
        cropped_im = self.centre_crop(im)
        # cropped_im =
        oldRange = cropped_im.max() - cropped_im.min()
        newRange = 1 - 0

        im2 = ((cropped_im - cropped_im.min()) * newRange / oldRange) + 0

        return (im2)

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        compatible_meta = None
        for name in filename:
            if name.endswith('.npz'):
                load_array = (np.array(np.load(name)['arr_0'], dtype="float"))
                # print("array shape@ ", load_array.shape)
                # new_arr = np.zeros()
                # for a in load_array:
                img = self.centre_crop(load_array)

                # print("after c crop", img.shape)

                data = np.asarray(img)

                if self.dtype:
                    data = data.astype(self.dtype)
                img_array.append(data)
                meta = dict()
                meta["filename_or_obj"] = name
                meta["spatial_shape"] = data.shape[:2]
                # meta["format"] = img.format
                # meta["mode"] = img.mode
                # meta["width"] = img.width
                # meta["height"] = img.height
                # meta["info"] = img.info

            else:
                img = Image.open(name)
                data =  self.centre_crop(np.asarray(img))
                data = np.expand_dims(np.asarray(data),axis=2)
                # data = (np.asarray(img))
                data = (data - np.min(data))/np.ptp(data)
                # print("data shapoe: ", data.shape, data.min(), data.max())
                
                # img = Image.fromarray((img))

                # print("the image shapeL ", img.width,img.height )
               
                if self.dtype:
                    data = data.astype(self.dtype)
                img_array.append(data)
                meta = dict()
                meta["filename_or_obj"] = name
                meta["spatial_shape"] = data.shape[:2]
                meta["format"] = img.format
                meta["mode"] = img.mode
                meta["width"] = img.width
                meta["height"] = img.height
                meta["info"] = img.info

            if self.image_only:
                continue

            if not compatible_meta:
                compatible_meta = meta
            else:
                assert np.allclose(
                    meta["spatial_shape"], compatible_meta["spatial_shape"]
                ), "all the images in the list should have same spatial shape."

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]

        return img_array if self.image_only else (img_array, compatible_meta)


