import os
import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from IPython import embed

class CXR(Dataset):

    def __init__(self, config):
        
        self.folder_path = config.data_path
        self.image_list = self._load_images()


    def _load_images(self):
        image_list = []

        for filename in sorted(os.listdir(self.folder_path)):
    
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.folder_path, filename)

                ground_truth_path = self.folder_path.replace("CXR_png", "masks")
                ground_truth_file_path = os.path.join(ground_truth_path, filename.replace(".png", "_mask.png"))
            
                if os.path.exists(ground_truth_file_path):

                    image_list.append(image_path)
                                
        return image_list


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image_path = self.image_list[index]
        file_path = image_path.split("/")[-1]



        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))

        file_name = os.path.basename(image_path)
        file_name = file_name.replace(".png", "_mask.png")

        ground_truth_path = self.folder_path.replace("CXR_png", "masks")

        ground_truth_file_path = os.path.join(ground_truth_path, file_name)
        ground_truth = cv2.imread(ground_truth_file_path)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

        ground_truth = cv2.resize(ground_truth, (256, 256))
        contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        

        # Plot each contour as individual images
        contour_images = []
        bounding_boxes = []
        for i, contour in enumerate(sorted_contours):
            # Create a blank image of the same size as the gt image
            if cv2.contourArea(contour) > 0: 
                contour_img = np.zeros_like(ground_truth)
                # Draw the contour on the blank image
                cv2.drawContours(contour_img, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
                contour_img = contour_img / 255.0
                # Append the contour image to the list
                contour_images.append(contour_img)

                # Get the bounding box of the contour for prompted SAM
                non_zero_pixels = np.where(contour_img)
                if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
                    x_min = np.min(non_zero_pixels[1])
                    y_min = np.min(non_zero_pixels[0])
                    x_max = np.max(non_zero_pixels[1])
                    y_max = np.max(non_zero_pixels[0])

                    bbox = np.array([x_min, y_min, x_max, y_max])
                    bounding_boxes.append(bbox)
        bounding_boxes = np.array(bounding_boxes)
        gray_img_normalized = ground_truth / 255.0
        threshold = 0.5
        ground_truth = np.where(gray_img_normalized > threshold, 1, 0)

        # return image, ground_truth, contour_images[0], contour_images[1]
        # return image, ground_truth
        return image, ground_truth, contour_images, bounding_boxes, file_path
        

    