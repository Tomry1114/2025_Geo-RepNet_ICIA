#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import os

from .common_dataset import CommonDataset
from ppcls.data.preprocess import transform
from ppcls.utils import logger
from PIL import Image
class ImageNetDataset(CommonDataset):
    """ImageNetDataset

    Args:
        image_root (str): image root, path to `ILSVRC2012`
        cls_label_path (str): path to annotation file `train_list.txt` or `val_list.txt`
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
    """

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 depth_root=None,
                 delimiter=None,
                 relabel=False):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        self.depth_root = depth_root
        super(ImageNetDataset, self).__init__(image_root, cls_label_path,
                                              transform_ops)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []
        self.depth_images = []
        #self.depth_labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if self.relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split(self.delimiter)
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(label_set)
                }

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."
                #加入深度图像路径
                if self.depth_root is not None:
                    depth_name = line[0].replace(".png", "_disp.png")
                    depth_path = os.path.join(self.depth_root, depth_name)
                    assert os.path.exists(depth_path), f"depth path {depth_path} does not exist."
                    #print(f"Depth path: {depth_path}") 
                    
                    with Image.open(depth_path) as depth_img:
                    # print(f"[Depth] Path: {depth_path}")
                    # print(f"[Depth] Size (W x H): {depth_img.size}")
                    # print(f"[Depth] Mode (channels): {depth_img.mode}")
                    # print(f"[Depth] Numpy shape: {np.array(depth_img).shape}")
                    # print(f"[Depth] Dtype: {np.array(depth_img).dtype}")
    
                        self.depth_images.append(depth_path)
                        

    def __getitem__(self, idx):
        max_retries = 5
        for _ in range(max_retries):
            try:
                with open(self.images[idx], 'rb') as f:
                    img = f.read()
                if self._transform_ops:
                    img = transform(img, self._transform_ops)
                    img = img.transpose(( 2, 0,1))
                    #print(f"Image shape after transform: {img.shape}")

                depth_img = None
                if hasattr(self, 'depth_images') and self.depth_images:
                    depth_path = self.depth_images[idx]
                    with open(depth_path, 'rb') as f:
                        depth_img = f.read()
                    if self._transform_ops:
                        #print(f'depth_img:{depth_img.shape}')
                        depth_img = transform(depth_img, self._transform_ops)
                        depth_img = depth_img.transpose((2, 0, 1))
                        #print(f"Depth image shape after transform: {depth_img.shape}")
    
                
                return img, depth_img, self.labels[idx]

            except Exception as ex:
                print(f"[ERROR] Failed to load sample at idx {idx}: {ex}")
                idx = np.random.randint(self.__len__())  # 换一个样本继续尝试

        raise RuntimeError("Failed to load data after 5 retries")

        




