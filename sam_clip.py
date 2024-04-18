import cv2
import random
import json
import time

import torch
import numpy as np

import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide 

from PIL import Image  
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import wandb
from IPython import embed
from tqdm import tqdm
from datetime import datetime
from utils.utils import log_images_to_wandb, log_images_to_wandb_batch, dice_coeff


def hyper_params_tuning(sam, dataset,prompt_mode,resize_transform, prompts,clip_model, preprocess, config, mode):

    # ----- Hyper-params tuning -----

    random.seed(time.time())
    random_indices = random.sample(range(len(dataset)), 5)
    print("Random Indices:", random_indices)

    points_per_side_range = [4, 8, 16]
    pred_iou_thresh_range = [0.2, 0.5, 0.8]
    stability_score_thresh_range = [0.2, 0.5, 0.8]
    crop_nms_thresh_range = [0.7, 0.8,0.9]
    box_nms_thresh_range = [0.7, 0.8, 0.9]

    best_score = -1
    best_params = None
    num_iterations = 5

    area_list = [20000, 40000, 50000]
    avg_dice = []
    best_score = -1
    best_params = None

    for _ in tqdm(range(num_iterations), desc="Iteration: Hyper-params tuning", unit="iteration"):

        random.shuffle(points_per_side_range)
        random.shuffle(stability_score_thresh_range)
        random.shuffle(pred_iou_thresh_range)
        random.shuffle(area_list)

        points_per_side = random.choice(points_per_side_range)
        stability_score_thresh = random.choice(stability_score_thresh_range)
        pred_iou_thresh = random.choice(pred_iou_thresh_range)
        crop_nms_thresh = random.choice(crop_nms_thresh_range)
        box_nms_thresh = random.choice(box_nms_thresh_range)
        area = random.choice(area_list)

        print(f"hyper-params: points_per_side: {points_per_side}, stability_score_thresh: {stability_score_thresh},pred_iou_thresh: {pred_iou_thresh} ")
        avg_dice_coeff = 0
        dice_scores =[]
        
        for idx in random_indices:

            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                points_per_batch=256,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                crop_nms_thresh=crop_nms_thresh,
                min_mask_region_area=5000,
                
            )
        
            image, gt, _, _, _ = dataset[idx]
            masks = mask_generator.generate(image)
            masks = [mask for mask in masks if mask["area"] < area] # area filtering

            img_crops = get_crops(image, masks, prompt_mode)
            max_indices, _  = retrieve_relevant_crop(img_crops, prompts, clip_model, preprocess, config)
            bboxes, _ , _ = get_sam_prompts(image, masks,max_indices , img_crops)
            preds = sam_predicton(sam, image, resize_transform, bboxes, config, mode)

            dice_score, _ = dice_coeff(gt, preds)
            print("dice:", dice_score)
            dice_scores.append(dice_score)

   
        avg_dice_coeff = np.mean(dice_scores)

        if avg_dice_coeff > best_score:
            best_score = avg_dice_coeff
            print(f"best dice: {best_score}")

            best_params = {
                'points_per_side': points_per_side,
                'pred_iou_thresh': pred_iou_thresh,
                'stability_score_thresh': stability_score_thresh,
                "box_nms_thresh": box_nms_thresh, 
                "crop_nms_thresh" : crop_nms_thresh,
                "area": area
            }
        
    print("Best Parameters:", best_params)

    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side= best_params["points_per_side"],
    points_per_batch= 128,
    pred_iou_thresh= best_params["pred_iou_thresh"],
    stability_score_thresh= best_params["stability_score_thresh"],
    box_nms_thresh= best_params["box_nms_thresh"],
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    crop_nms_thresh=best_params["crop_nms_thresh"],
    min_mask_region_area=200, 
    )

    return mask_generator, best_params["area"]

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()



def get_crops(image, masks, prompt_mode):
    imgs_bboxes = []
    indices_to_remove = []

    for i, mask in enumerate(masks):
        box = mask["bbox"]
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        
        if x2 > x1 and y2 > y1:  # Check if the bounding box has non-zero dimensions

            if prompt_mode == "crops":
                # crops
                seg_mask = np.array([mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1,2,0)
                cropped_image = np.multiply(image, seg_mask).astype("int")[int(y1):int(y2), int(x1):int(x2)]
                imgs_bboxes.append(cropped_image)
            
            elif prompt_mode == "crop_expand":
                #crops
                seg_mask = np.array([mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1,2,0)
                # Expand bounding box coordinates
                x1_expanded = max(0, x1 - 10)
                y1_expanded = max(0, y1 - 10)
                x2_expanded = min(image.shape[1], x2 + 10)
                y2_expanded = min(image.shape[0], y2 + 10)
                
                if x2_expanded > x1_expanded and y2_expanded > y1_expanded:
                    cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    imgs_bboxes.append(cropped_image)
                
            elif prompt_mode == "bbox":
                # bbox on the image around crop area
                img_bbox = cv2.rectangle(image.copy(), (int(x1), int(y1)), (int(x2), int(y2)), ( 255, 0, 0), 5)
                imgs_bboxes.append(img_bbox)

            elif prompt_mode == "reverse_box_mask":
                # highlight roi and gray out the rest
                res = image.copy()
                box_mask = np.zeros(res.shape, dtype=np.uint8)
                box_mask = cv2.rectangle(box_mask, (x1, y1), (x2, y2),
                                        color=(255, 255, 255), thickness=-1)[:, :, 0]
                overlay = res.copy()
                overlay[box_mask == 0] = np.array((124, 116, 104))
                alpha = 0.5 # Transparency factor.
                res = cv2.addWeighted(overlay, alpha, res, 1 - alpha, 0.0)
                imgs_bboxes.append(res)

            elif prompt_mode == "contour":
                
                #contour around the mask and overlay on the image
                res = image.copy()
                mask = mask["segmentation"]
                contours, hierarchy = cv2.findContours(mask.astype(
                        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                res = cv2.drawContours(res, contours, contourIdx=-1,
                                        color=(255, 0, 0), thickness=3)#, gave 60.04
                imgs_bboxes.append(res)

        else:
            print("Skipping zero-sized bounding box.")
            indices_to_remove.append(i)

            
        for index in sorted(indices_to_remove, reverse=True):
                del masks[index] 

    return imgs_bboxes

def retrieve_relevant_crop(crops, class_names, model, preprocess, config):
    crops_uint8 = [image.astype(np.uint8) for image in crops]

    pil_images = []
    for image in crops_uint8:
        if image.shape[0] > 0 and image.shape[1] > 0:
            pil_image = Image.fromarray(image)
            pil_images.append(pil_image)

    preprocessed_images = [preprocess(image).to("cuda") for image in pil_images]
    stacked_images = torch.stack(preprocessed_images)

    similarity_scores = {class_name: [] for class_name in class_names}

    with torch.no_grad():

        image_features = model.encode_image(stacked_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        for class_name in class_names:
            class_descriptions = class_names[class_name]
            class_text_features = [model.encode_text(clip.tokenize(description).to("cuda")) for description in class_descriptions]

            mean_text_feature = torch.mean(torch.stack(class_text_features), dim=0)
            mean_text_feature /= mean_text_feature.norm(dim=-1, keepdim=True)
            
            similarity_score = 100. * image_features @ mean_text_feature.T
            similarity_scores[class_name] = similarity_score.squeeze().tolist()

        
        if config.dataset == "cxr":
            max_indices = {key: sorted(range(len(similarity_scores[key])), key=lambda i: similarity_scores[key][i], reverse=True)[:2] for key in similarity_scores} # 2 lungs so getting top 2

        else:
            max_indices = {key: similarity_scores[key].index(max(similarity_scores[key])) for key in similarity_scores}

    return max_indices, similarity_scores

def get_sam_prompts(image, masks, max_indices, imgs_bboxes):
        
    # ------  bbox prompts cordinates relevant to ROI for SAM------
        
        bboxes = []
        relevant_crop = []
        img_with_bboxes = []

        for key, indices in max_indices.items():
            for index, value in enumerate(indices):
                bbox = masks[value]["bbox"]
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                if index == 0:
                    img = image.copy()

                img_with_bboxes = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
                bboxes.append(bbox)
                relevant_crop.append(imgs_bboxes[value])

        bboxes = np.array(bboxes)
        return bboxes, relevant_crop, img_with_bboxes

def sam_predicton(sam, image, resize_transform, bboxes, config, mode):
        
        # ------ SAM format ------
    
        batched_input = [{
            'image': prepare_image(image, resize_transform, "cuda").to("cuda"),
            'boxes': resize_transform.apply_boxes_torch(torch.from_numpy( np.array(bboxes)), image.shape[:2]).to("cuda"),
            'original_size': image.shape[:2] 


        }]
        
        preds = sam(batched_input, multimask_output=False)
        binary_masks = torch.sigmoid(preds) > 0.5
        binary_masks = binary_masks.squeeze().cpu().numpy()

        if config.dataset == "cxr" and mode == "sam_clip" or mode == "sam_prompted":
            binary_masks = np.bitwise_or(binary_masks[0], binary_masks[1])

        return binary_masks

def get_eval(dataset, sam, config, suffix, wandb_mode, prompt_mode, mode):

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    wandb_run = wandb.init( project='SAM', entity='sidra-aleem2', name = config['model_name'] + "_" + suffix +"_"+ folder_time, mode = wandb_mode)
  
   # ----- loading the models  ----- 
    
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
    sam_checkpoint = config.sam_ckpt
    
    sam = sam_model_registry[config.model_type](checkpoint=sam_checkpoint)
    sam.to("cuda")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # ----- CLIP Prompts -----
    
    with open(config.clip_prompts, "r") as file:
        prompts = json.load(file)

    dice_scores = []
    mask_generator, area = hyper_params_tuning(sam, dataset, prompt_mode,resize_transform, prompts,clip_model, preprocess, config, mode)

    # ----- Inference -----
    with torch.no_grad(): 
        for idx in tqdm(range(len(dataset)), desc= f"Processing  images", unit= "image"): 

            image, gt, _, bounding_boxes, file_name = dataset[idx]

            if mode == "sam_clip":
                masks = mask_generator.generate(image)
                masks = [mask for mask in masks if mask["area"] < area] # area filtering based on area value from hyper-params tuning

                img_crops = get_crops(image, masks, prompt_mode)
                max_indices, scores = retrieve_relevant_crop(img_crops, prompts, clip_model, preprocess, config)

                # ----- logging crops to wandb  ----- 
                log_images_to_wandb_batch(scores, img_crops, file_name)
                
                 # ------  bbox cordinates relevant to crop ------
                bboxes, relevant_crop, img_with_bboxes = get_sam_prompts(image, masks,max_indices , img_crops)

            elif mode == "sam_prompted":
                # bounding  box prompt from ground truth
                bboxes = bounding_boxes

            preds = sam_predicton(sam, image, resize_transform, bboxes, config, mode)
            dice_score, miou = dice_coeff(gt, preds)

            print("dice:", dice_score, "miou:", miou)
            dice_scores.append((dice_score, miou)) 
            
            # with open(f"{config.dataset}_{mode}_dice_scores_.txt", "a") as file:
            #     file.write(f"File Name: {file_name}, Dice: {dice_score}, MIoU: {miou}\n")
            
            # ----- logging images to wandb ----- 
            if wandb_mode == "online":
                if mode == "sam_clip":
                    log_images_to_wandb(image, gt, masks,preds,  dice_score, file_name, relevant_crop, img_with_bboxes) # sam_clip
                else:
                    log_images_to_wandb(image, gt, masks, preds, dice_score, file_name, None, None ) # prompted/unprompted SAM 
    
        dice_scores = np.array(dice_scores)
        average_dice_score = np.mean(dice_scores[:, 0])
        miou = np.mean(dice_scores[:, 1])
        print("Average Dice Score:", average_dice_score, "mIoU:", miou) 

        return average_dice_score

 


        
        









