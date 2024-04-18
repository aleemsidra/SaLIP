import cv2 
import json
import random
import numpy as np
from torch import Tensor
# from utils.utils import log_images
import wandb
import clip
from IPython import embed

from segment_anything import  SamAutomaticMaskGenerator
from sam_clip import retrieve_relevant_crop, get_crops
import torch

from typing import List  
from PIL import Image  
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from datetime import datetime
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import log_images

def sam_mask_generator_params(sam, model, preprocess, resize_transform, dataset, config):
   
    num_slices = 5
    num_iterations = 5

    points_per_side_range = [16, 32, 64]
    pred_iou_thresh_range = [0.2, 0.5, 0.8, 0.9]
    stability_score_thresh_range = [0.2, 0.5, 0.8, 0.95]

    random_indices = random.sample(range(len(dataset)), 2)
    print("Random Indices:", random_indices)

    best_score = -1
    best_params = None

    for _ in tqdm(range(num_iterations), desc="Iteration: Hyper-params tuning", unit="iteration"):
        points_per_side = random.choice(points_per_side_range)
        stability_score_thresh = random.choice(stability_score_thresh_range)
        pred_iou_thresh = random.choice(pred_iou_thresh_range)

        print(f"hyper-params: points_per_side: {points_per_side}, stability_score_thresh: {stability_score_thresh},pred_iou_thresh: {pred_iou_thresh} ")
        avg_dice_coeff = 0

        for idx in random_indices:
            print(f"image id: {idx}")
            input_samples, gt_samples = dataset[idx]
            spacing = input_samples.shape[0] // num_slices
            slices_dice_coeff = []

            for slice_idx in range(spacing, input_samples.shape[0], spacing):

                img_slice = input_samples[slice_idx, :, :]
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=points_per_side,
                    points_per_batch=256,
                   
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=200,
                )


                bbox = get_bbox(img_slice, model, preprocess, mask_generator, config) # bbox via SAM+CLIP
                # Format required by SAM predictor
                batched_input = [{
                    'image': prepare_image(img_slice, resize_transform, "cuda"),
                    'boxes': resize_transform.apply_boxes_torch(torch.from_numpy(bbox), img_slice.shape[:2]).to("cuda"),
                    'original_size': img_slice[0,:,:].shape
                }]

                preds = sam(batched_input, multimask_output=False) 
     
                mask = torch.sigmoid(preds.squeeze()).cpu().detach().numpy() > 0.5
                slices_dice_coeff.append(dice_coeff(torch.tensor(mask, dtype=torch.float32), gt_samples[slice_idx].squeeze()))

                del preds
                del mask

            avg_dice_coeff += np.mean(slices_dice_coeff)

        avg_dice_coeff /= len(random_indices)

        print("avg_dice_coeff", avg_dice_coeff)
        if avg_dice_coeff > best_score:
            best_score = avg_dice_coeff
            print(f"best dice: {best_score}")
            best_params = {
                'points_per_side': points_per_side,
                'pred_iou_thresh': pred_iou_thresh,
                'stability_score_thresh': stability_score_thresh,
            }

    print("Best Parameters:", best_params)

    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side= best_params["points_per_side"],
    points_per_batch= 128,
    pred_iou_thresh= best_params["pred_iou_thresh"],
    stability_score_thresh= best_params["stability_score_thresh"],
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200,  # Requires open-cv to run post-processing
    )

    return mask_generator

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def prepare_image(image, transform, device):

    bgr_img = cv2.cvtColor( image[0,:,:], cv2.COLOR_GRAY2BGR)
    scaled_tensor = 255 * (bgr_img - bgr_img.min()) / (bgr_img.max() - bgr_img.min())
    image = scaled_tensor.astype(np.uint8)
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    image = image.to(device)
    return image.permute(2, 0, 1).contiguous()

def get_bbox(image, model, preprocess, mask_generator, config):
   
    rgb_slice = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)  
    image = cv2.normalize(rgb_slice, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    masks = mask_generator.generate(image)
    img_crops = get_crops(image, masks, "crops")

    # ----- CLIP Prompts -----
    
    with open(config.clip_prompts, "r") as file:
        prompts = json.load(file)
   
    scores, _ = retrieve_relevant_crop(img_crops,prompts, model, preprocess, config)

    # ------  bbox prompts cordinates relevant to ROI for SAM------
    max_index = list(scores.values())[0]
    bbox = masks[max_index]["bbox"]    
    bbox = np.array([bbox[0] , bbox[1], bbox[0] + bbox[2],  bbox[1] + bbox[3]])
    
    return bbox


def get_eval_3d(dataset, sam, config, suffix, wandb_mode):

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    wandb_run = wandb.init( project='SAM', entity='sidra-aleem2', name = config['model_name'] + "_" + suffix +"_"+ folder_time, mode = wandb_mode)

    # ----- loading the models  ----- 
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.to("cuda")

    mask_generator = sam_mask_generator_params(sam, clip_model, preprocess, resize_transform, dataset, config)
    avg_dice = []

    # ----- Inference -----
    with torch.no_grad():
            
        for idx in tqdm(range(len(dataset)), desc= f"Processing  images", unit= "image"):          
            
            input_samples, gt_samples = dataset[idx]
            slices = []
           
            for _, img_slice in tqdm(enumerate(input_samples), total=len(input_samples), desc="Processing slices"): # looping over single img    
        
                bbox = get_bbox(img_slice, clip_model, preprocess, mask_generator, config )  # bbox from SAM+CLIP
                # Format required by SAM predictor
                batched_input = [
                    {
                        'image': prepare_image(img_slice, resize_transform, "cuda"),
                        'boxes': resize_transform.apply_boxes_torch(torch.from_numpy(bbox), img_slice.shape[:2]).to("cuda"),
                        'original_size':img_slice[0,:,:].shape
                    }]

                preds = sam(batched_input, multimask_output=False)
                slices.append(preds.squeeze().detach().cpu())
                del preds

            segmented_volume = torch.stack(slices, axis=0)
            slices.clear()
            mask = torch.zeros(segmented_volume.shape) 
            mask[torch.sigmoid(segmented_volume) > 0.5] = 1
            del segmented_volume
 
            test_dice = dice_coeff(mask, gt_samples.squeeze())

            print("test dice",test_dice )
            avg_dice.append(test_dice)
            mask = mask.unsqueeze(1)

            # logging images to wandb
            log_images(input_samples, mask, gt_samples, "10" , "test", idx) 
           
        final_avg_dice = np.mean(avg_dice)
        return final_avg_dice
                        
            

           

                