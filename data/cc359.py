# import os
# import numpy as np
# import cv2
# import pandas as pd
# import nibabel as nib

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# from IPython import embed

# class cc359_3d_volume(Dataset):
    
#     def __init__(self, config, site, train = True, rotate=True ):
#         self.rotate = rotate
#         self.fold = "False"
#         self.train = train
#         self.site = site
#         self.data_path = "/home/sidra/Documents/Domain_Apatation/UDAS/src/Data/CC359/Reconstructed/"
#         self.source = config.source
#         self.bbox_shift = 20
    
#         if self.site == 1:
#             self.folder = 'GE_15'
#             # self.range = (60,195)
#         elif self.site == 2:
#             self.folder = 'GE_3'
#             self.range = (25,175)
#         elif self.site == 3:
#             self.folder = 'Philips_15'
#             self.range = (10,150)
#         elif self.site == 4:
#             self.folder = 'Philips_3'
#             self.range = (20,155)
#         elif self.site == 5:
#             self.folder = 'Siemens_15'
#             self.range = (25,165)
#         elif self.site == 6:
#             self.folder = 'Siemens_3'
#             self.range = (60,165)
#         else:
#             self.folder = 'GE_3'

#         print(f"folder: {self.folder}")#, slice_range: {self.range}")
#         self.load_files(self.data_path)
    
#     def pad_image(self, img):
#         s, h, w = img.shape
#         if h < w:
#             b = (w - h) // 2
#             a = w - (b + h)
#             return np.pad(img, ((0, 0), (b, a), (0, 0)), mode='edge')
#         elif w < h:
#             b = (h - w) // 2
#             a = h - (b + w)
#             return np.pad(img, ((0, 0), (0, 0), (b, a)), mode='edge')
#         else:
#             return img


#     def load_files(self, data_path):
#         self.sagittal = True
        
#         if  self.source == "True" and self.train:
#             self.images_path = os.path.join(data_path, 'Original', self.folder, "train.csv")
#             print("train_path ", self.images_path )

#         # if self.stage == "refine" and not self.train:
#         elif self.source == "True" and not self.train:
#             self.images_path = os.path.join(data_path, 'Original', self.folder, "val.csv")
#             print("val_path ",self.images_path)

        
#         else:
#             print(self.source, self.train)
#             self.images_path = os.path.join(data_path, 'Original', self.folder, "test.csv")   # replace it for rest of domains
#             print("test_path ", self.images_path)
        
#         self.volume_files = pd.read_csv(self.images_path, header=None).values.ravel().tolist()

#     def img_transform(self, img):
        
        
#         self.sagittal = True
            
#         if not self.sagittal:
#             img = np.moveaxis(img, -1, 0)
            
#         if self.rotate:
#             img = np.rot90(img, axes=(1, 2))

#         return img

#     def pad_image_w_size(self, data_array, max_size):
#         current_size = data_array.shape[-1]
#         b = (max_size - current_size) // 2
#         a = max_size - (b + current_size)
#         return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode='edge')

#     def unify_sizes(self, input_images, input_labels):
#         sizes = np.zeros(len(input_images), int)
#         for i in range(len(input_images)):
#             sizes[i] = input_images[i].shape[-1]
#         max_size = np.max(sizes)
#         for i in range(len(input_images)):
#             if sizes[i] != max_size:
#                 input_images[i] = self.pad_image_w_size(input_images[i], max_size)
#                 input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
#         return input_images, input_labels

#     def get_bounding_box(self, ground_truth_map):
    
#         bounding_boxes = []

#         for slice_idx in range(ground_truth_map.shape[0]):
#             slice_data = ground_truth_map[slice_idx, :, :]

#             non_zero_pixels = np.where(slice_data)
#             if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
#                 x_min = np.min(non_zero_pixels[1])
#                 y_min = np.min(non_zero_pixels[0])
#                 x_max = np.max(non_zero_pixels[1])
#                 y_max = np.max(non_zero_pixels[0])

#                 bbox = np.array([x_min, y_min, x_max, y_max])
#                 bounding_boxes.append(bbox)

#         return bounding_boxes
    

#     def __len__(self):
#         return len(self.volume_files)

#     def __getitem__(self, idx):
        
#         # Load the volume data from the NIfTI file using nibabel
#         # print("file_name", self.volume_files[idx])
#         img = nib.load(self.volume_files[idx]).get_fdata('unchanged', dtype=np.float32)       
#         nib_file = nib.load(self.volume_files[idx])
#         # slice_range = self.range
#         spacing = [nib_file.header.get_zooms()] * nib_file.shape[0]
#         self.voxel_dim = np.array(spacing)
    
#         # img = img[slice_range[0]:slice_range[1]+1, :, :]
        
#         lbl = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, self.volume_files[idx][:-7].split("/")[-1] + '_ss.nii.gz')).get_fdata('unchanged', dtype=np.float32)
 
#         # removing black slices
#         non_zero_slices = [i for i, slice_ in enumerate(lbl) if not np.all(slice_ == 0)]
        
#         img = img[non_zero_slices]
#         img = img[20:-20, :, :]
#         img = self.img_transform(img)

#         lbl = lbl[non_zero_slices]
#         lbl = lbl[20:-20, :, :]
#         lbl = self.img_transform(lbl)
#         bboxes = torch.from_numpy(np.array(self.get_bounding_box(lbl)))

#         data = np.expand_dims(img.copy(), axis=1)
#         labels =  torch.from_numpy(np.expand_dims(lbl.copy(), axis=1))
#         voxel_dim = torch.from_numpy(self.voxel_dim)

#         return   data, labels , bboxes, voxel_dim
import os
import numpy as np
import cv2
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class cc359_3d_volume(Dataset):
    def __init__(self, config, site, train = True, rotate=True ):
        self.rotate = rotate
        self.fold = "False"
        self.train = train
        self.site = site
        self.data_path = "/home/sidra/Documents/Domain_Apatation/UDAS/src/Data/CC359/Reconstructed/"
        self.source = config.source
        self.bbox_shift = 20
    
        if self.site == 1:
            self.folder = 'GE_15'
            # self.range = (60,195)
        elif self.site == 2:
            self.folder = 'GE_3'
            self.range = (25,175)
        elif self.site == 3:
            self.folder = 'Philips_15'
            self.range = (10,150)
        elif self.site == 4:
            self.folder = 'Philips_3'
            self.range = (20,155)
        elif self.site == 5:
            self.folder = 'Siemens_15'
            self.range = (25,165)
        elif self.site == 6:
            self.folder = 'Siemens_3'
            self.range = (60,165)
        else:
            self.folder = 'GE_3'

        print(f"folder: {self.folder}")#, slice_range: {self.range}")
        self.load_files(self.data_path)
    
    def pad_image(self, img):
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode='edge')
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode='edge')
        else:
            return img


    def load_files(self, data_path):
        self.sagittal = True
        
        if  self.source == "True" and self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "train.csv")
            print("train_path ", self.images_path )

        # if self.stage == "refine" and not self.train:
        elif self.source == "True" and not self.train:
            self.images_path = os.path.join(data_path, 'Original', self.folder, "val.csv")
            print("val_path ",self.images_path)


        else:
            print(self.source, self.train)
            self.images_path = os.path.join(data_path, 'Original', self.folder, "test.csv")   # replace it for rest of domains
            print("test_path ", self.images_path)
        
        self.volume_files = pd.read_csv(self.images_path, header=None).values.ravel().tolist()
     
    def img_transform(self, img):
        
        
        self.sagittal = True
            
        if not self.sagittal:
            img = np.moveaxis(img, -1, 0)
            
        if self.rotate:
            img = np.rot90(img, axes=(1, 2))
        # if img.shape[1] != img.shape[2]:
        #     img = self.pad_image(img)

        return img

    def pad_image_w_size(self, data_array, max_size):
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode='edge')

    def unify_sizes(self, input_images, input_labels):
        sizes = np.zeros(len(input_images), int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)
        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels

    def get_bounding_box(self, ground_truth_map):
    
#       # get bounding box from mask
#         non_zero_pixels = np.where(gt)
#         x_min = np.min(non_zero_pixels[1])
#         y_min = np.min(non_zero_pixels[0])
#         x_max = np.max(non_zero_pixels[1])
#         y_max = np.max(non_zero_pixels[0])
#         x_min, y_min, x_max, y_max

#         bbox = np.array([x_min, y_min, x_max, y_max])

#         return bbox
        bounding_boxes = []

        for slice_idx in range(ground_truth_map.shape[0]):
            slice_data = ground_truth_map[slice_idx, :, :]

            non_zero_pixels = np.where(slice_data)
            if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
                x_min = np.min(non_zero_pixels[1])
                y_min = np.min(non_zero_pixels[0])
                x_max = np.max(non_zero_pixels[1])
                y_max = np.max(non_zero_pixels[0])

                bbox = np.array([x_min, y_min, x_max, y_max])
                bounding_boxes.append(bbox)

        return bounding_boxes
    

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        
        # Load the volume data from the NIfTI file using nibabel
        # print("file_name", self.volume_files[idx])
        img = nib.load(self.volume_files[idx]).get_fdata('unchanged', dtype=np.float32)       
        nib_file = nib.load(self.volume_files[idx])
        # slice_range = self.range
        spacing = [nib_file.header.get_zooms()] * nib_file.shape[0]
        self.voxel_dim = np.array(spacing)
    
        # img = img[slice_range[0]:slice_range[1]+1, :, :]
        
        lbl = nib.load(os.path.join(self.data_path, 'Silver-standard', self.folder, self.volume_files[idx][:-7].split("/")[-1] + '_ss.nii.gz')).get_fdata('unchanged', dtype=np.float32)
        # lbl = lbl[slice_range[0]:slice_range[1]+1, :, :]
 
        # removing black slices
        non_zero_slices = [i for i, slice_ in enumerate(lbl) if not np.all(slice_ == 0)]
        
        img = img[non_zero_slices]
        img = img[20:-20, :, :]
        img = self.img_transform(img)

        lbl = lbl[non_zero_slices]
        lbl = lbl[20:-20, :, :]
        lbl = self.img_transform(lbl)
        # bboxes = torch.from_numpy(np.array(self.get_bounding_box(lbl)))

        # img , lbl = self.unify_sizes(img, lbl)
        data = np.expand_dims(img.copy(), axis=1)

        labels =  torch.from_numpy(np.expand_dims(lbl.copy(), axis=1))

       
        voxel_dim = torch.from_numpy(self.voxel_dim)

        return   data, labels #, bboxes, voxel_dim