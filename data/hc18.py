import os
import numpy as np
import cv2
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class hc18_2d(Dataset):
    def __init__(self, config, train=True, rotate=False):
        self.rotate = rotate
        self.fold = "False"
        self.train = train
        self.data_path = "/mnt/storage/fangyijie/HC18/"
        self.source = config.source
        self.bbox_shift = 20
        self.folder = 'img'

        print(f"folder: {self.folder}") #, slice_range: {self.range}")
        self.load_files(self.data_path)
     
    def img_transform(self, img):
        if self.rotate:
            # img = np.rot90(img, axes=(1, 2))
            img = np.rot90(img)
        return img

    def get_bounding_box(self, ground_truth_map):
        non_zero_pixels = np.where(ground_truth_map)
        if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
            x_min = np.min(non_zero_pixels[1])
            y_min = np.min(non_zero_pixels[0])
            x_max = np.max(non_zero_pixels[1])
            y_max = np.max(non_zero_pixels[0])

            bbox = np.array([x_min, y_min, x_max, y_max])

        return bbox
    
    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.image_dir, self.images[idx])
        mask_name = self.volume_files[idx].replace(".png", "_Annotation.png")

        img_path = os.path.join(self.data_path, self.folder, self.volume_files[idx])
        mask_path = os.path.join(self.data_path, self.folder, mask_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32')
        labels[labels == 255.0] = 1.0

        img = cv2.resize(img, (256, 256))
        labels = cv2.resize(labels, (256, 256))

        data = self.img_transform(img)
        labels = self.img_transform(labels)

        data = data.copy()
        #labels = torch.from_numpy(labels.copy())
        bounding_boxes = self.get_bounding_box(labels)

        return data, labels, self.volume_files[idx] #bounding_boxes, #, bboxes, voxel_dim
            