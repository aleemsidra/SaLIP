# coding=utf-8
import os
import cv2
import numpy as np
import json
import wandb
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython import embed
import torchvision

import argparse

import json
import torch
from IPython import embed
# import LoRA.loralib as lora

from easydict import EasyDict as edict

def process_config(jsonfile=None):
    try:
        if jsonfile is not None:
            with open(jsonfile, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    print("Config: ", config_args)
    print("\n")

    return config_args


def check_config_dict(config_dict):
    """
    check configuration
    :param config_dict: input config
    :return: 
    """
    if isinstance(config_dict["model_module_name"],str) is False:
        raise TypeError("model_module_name param input err...")
    if isinstance(config_dict["model_net_name"],str) is False:
        raise TypeError("model_net_name param input err...")
    if isinstance(config_dict["gpu_id"],str) is False:

        raise TypeError("gpu_id param input err...")
    if isinstance(config_dict["async_loading"],bool) is False:
        raise TypeError("async_loading param input err...")
    if isinstance(config_dict["is_tensorboard"],bool) is False:
        raise TypeError("is_tensorboard param input err...")
    if isinstance(config_dict["evaluate_before_train"],bool) is False:
        raise TypeError("evaluate_before_train param input err...")
    if isinstance(config_dict["shuffle"],bool) is False:
        raise TypeError("shuffle param input err...")
    if isinstance(config_dict["data_aug"],bool) is False:
        raise TypeError("data_aug param input err...")

    if isinstance(config_dict["num_epochs"],int) is False:
        raise TypeError("num_epochs param input err...")
    if isinstance(config_dict["img_height"],int) is False:
        raise TypeError("img_height param input err...")
    if isinstance(config_dict["img_width"],int) is False:
        raise TypeError("img_width param input err...")
    if isinstance(config_dict["num_channels"],int) is False:
        raise TypeError("num_channels param input err...")
    if isinstance(config_dict["num_classes"],int) is False:
        raise TypeError("num_classes param input err...")
    if isinstance(config_dict["batch_size"],int) is False:
        raise TypeError("batch_size param input err...")
    if isinstance(config_dict["dataloader_workers"],int) is False:
        raise TypeError("dataloader_workers param input err...")
    if isinstance(config_dict["learning_rate"],(int,float)) is False:
        raise TypeError("learning_rate param input err...")
    if isinstance(config_dict["learning_rate_decay"],(int,float)) is False:
        raise TypeError("learning_rate_decay param input err...")
    if isinstance(config_dict["learning_rate_decay_epoch"],int) is False:
        raise TypeError("learning_rate_decay_epoch param input err...")

    if isinstance(config_dict["train_mode"],str) is False:
        raise TypeError("train_mode param input err...")
    if isinstance(config_dict["file_label_separator"],str) is False:
        raise TypeError("file_label_separator param input err...")
    if isinstance(config_dict["pretrained_path"],str) is False:
        raise TypeError("pretrained_path param input err...")
    if isinstance(config_dict["pretrained_file"],str) is False:
        raise TypeError("pretrained_file param input err...")
    if isinstance(config_dict["save_path"],str) is False:
        raise TypeError("save_path param input err...")
    if isinstance(config_dict["save_name"],str) is False:
        raise TypeError("save_name param input err...")

    if not os.path.exists(os.path.join(config_dict["pretrained_path"], config_dict["pretrained_file"])):
        raise ValueError("cannot find pretrained_path or pretrained_file...")
    if not os.path.exists(config_dict["save_path"]):
        raise ValueError("cannot find save_path...")

    if isinstance(config_dict["train_data_root_dir"],str) is False:
        raise TypeError("train_data_root_dir param input err...")
    if isinstance(config_dict["val_data_root_dir"],str) is False:
        raise TypeError("val_data_root_dir param input err...")
    if isinstance(config_dict["train_data_file"],str) is False:
        raise TypeError("train_data_file param input err...")
    if isinstance(config_dict["val_data_file"],str) is False:
        raise TypeError("val_data_file param input err...")

    if not os.path.exists(config_dict["train_data_root_dir"]):
        raise ValueError("cannot find train_data_root_dir...")
    if not os.path.exists(config_dict["val_data_root_dir"]):
        raise ValueError("cannot find val_data_root_dir...")
    if not os.path.exists(config_dict["train_data_file"]):
        raise ValueError("cannot find train_data_file...")
    if not os.path.exists(config_dict["val_data_file"]):
        raise ValueError("cannot find val_data_file...")



#global_config = process_config('configs/config.json')

if __name__ == '__main__':
    config = global_config
    print(config['experiment_dir'])
    print('done')



def log_images(input, preds, gt, epoch, stage, img_id = ""):

    grid_img = vutils.make_grid(input, 
                                normalize=False, scale_each=False)
    
    wandb.log({f"{stage}_Input images_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

    grid_img = vutils.make_grid(preds,
                                normalize=False,
                                scale_each=False)
    

    wandb.log({f"{stage}_predictions_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

    grid_img = vutils.make_grid(gt,
                                normalize=False,
                                scale_each=False)
    
    wandb.log({f"{stage}_Ground truth_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)



def save_model( model, config, suffix, folder_time, step = False):
        """
        implement the logic of saving model
        """
        print("Saving model...")
        save_path = config.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        try:
            if not os.path.exists(os.path.join (save_path, config.model_name)):
                save_dir = os.path.join(save_path, config.model_name + "_"+ suffix +"_" + folder_time)
                os.makedirs(save_dir)
              
        except:
             pass

        
        save_name = os.path.join(save_dir, config.save_name)
        print("save full model")
        torch.save(model.state_dict(), save_name)
        
        if step:
            print(f"saving lora model, step: {step}")
            split_string = save_name.split("/")
            split_string[-1] = "lora_only.pth" 
            save_name = "/".join(split_string) 
            torch.save(lora.lora_state_dict(model, bias='lora_only'),  save_name )

# def log_images(input, preds, gt, epoch, stage, img_id = ""):

#     grid_img = vutils.make_grid(torch.from_numpy(input), 
#                                 normalize=False, scale_each=False)
    
#     wandb.log({f"{stage}_Input images_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

#     grid_img = vutils.make_grid(preds,
#                                 normalize=False,
#                                 scale_each=False)
    

#     wandb.log({f"{stage}_predictions_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

#     # grid_img = vutils.make_grid(torch.from_numpy(gt),
#     #                             normalize=False,
#     #                             scale_each=False)
#     grid_img = vutils.make_grid(gt.numpy(),
#                                 normalize=False,
#                                 scale_each=False)
    
#     wandb.log({f"{stage}_Ground truth_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)
            

def log_images(input, preds, gt, epoch, stage, img_id = ""):

    grid_img = vutils.make_grid(torch.from_numpy(input), 
                                normalize=False, scale_each=False)
    
    wandb.log({f"{stage}_Input images_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

    grid_img = vutils.make_grid(preds,
                                normalize=False,
                                scale_each=False)
    

    wandb.log({f"{stage}_predictions_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)

    grid_img = vutils.make_grid(gt,
                                normalize=False,
                                scale_each=False)
    
    wandb.log({f"{stage}_Ground truth_Epoch: {epoch}_{img_id}": wandb.Image(grid_img)}, step=wandb.run.step)



# def log_images_to_wandb(image, gt, overlay_image, bbox1, bbox2, dice_score, idx):
# # def log_images_to_wandb(image, gt, overlay_image, bbox1,  dice_score, idx):
#         fig, axs = plt.subplots(1, 5, figsize=(20, 10))  # Increase the figsize to make the images larger
#         # fig, axs = plt.subplots(1, 4, figsize=(20, 10))  # Increase the figsize to make the images larger
#         fig.suptitle(f"image id : {idx}")  # Set the general title for the plot
#         axs[0].imshow(image)
#         axs[0].set_title("input")
#         axs[0].axis('off')

#         axs[1].imshow(gt, cmap="gray")
#         axs[1].set_title("ground truth")
#         axs[1].axis('off')

#         axs[2].imshow(bbox1)
#         axs[2].set_title("lung")
#         axs[2].axis('off')

#         axs[3].imshow(bbox2)
#         # axs[3].set_title("right bbox")
#         axs[3].set_title("lung")
#         axs[3].axis('off')

#         axs[4].imshow(overlay_image, cmap="gray")
#         axs[4].set_title("pred dice: " + str(round(dice_score, 3)))
#         axs[4].axis('off')

#         wandb.log({f"img_id: {idx}": wandb.Image(fig)})
#         plt.close(fig)

# def log_images_to_wandb_batch(imgs_bboxes, class_name, idx):
def log_images_to_wandb_batch(scores, imgs_bboxes, idx):

    num_images = len(imgs_bboxes)
    num_cols = 5  # Number of columns in each row
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed
    # print("in log")
    # embed()
    for class_name, class_scores in scores.items():  # Use items() instead of keys() to iterate over both keys and values
        
        sorted_indices = sorted(range(len(class_scores)), key=lambda i: class_scores[i], reverse=True)

        print(f"image id: {idx}, class_name: {class_name}") #, sorted scores: {class_scores[sorted_indices]}')  # Print the sorted scores
        sorted_imgs_bboxes = [imgs_bboxes[i] for i in sorted_indices]
        sorted_values = [class_scores[i] for i in sorted_indices]  # Get the values based on sorted indices
        print(f"sorted scroes: {sorted_values}")

        num_images = len(sorted_imgs_bboxes)
        num_cols = min(num_images, num_cols)  # Adjust the number of columns based on the number of images
        num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))  # Increase the figsize to make the images larger
        fig.suptitle(f"{class_name} id: {idx}")  # Set the general title for the plot

        # embed()
        for i, img_bbox in enumerate(sorted_imgs_bboxes):
            row = i // num_cols  # Calculate the row index
            col = i % num_cols  # Calculate the column index

            axs[row, col].imshow(img_bbox)
            axs[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, num_rows * num_cols):
            row = i // num_cols  
            col = i % num_cols  
            axs[row, col].axis('off')

        wandb.log({f"bboxes_img_id {idx}": wandb.Image(fig)})
        plt.close(fig)



# def log_images_to_wandb(image, gt, pred, crops, img_with_bboxes, dice_score, file_name):
#     num_cols = 6  # Number of columns in each row
#     num_rows = (len(crops) + num_cols - 1) // num_cols  # Calculate the number of rows needed

#     # fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 10))  # Increase the figsize to make the images larger
#     fig, axs = plt.subplots(len(crops) + 2, 1, figsize=(10, 10))  # Increase the figsize to make the images larger

#     axs[0].imshow(image)
#     axs[0].set_title(f"input : {file_name}")
#     axs[0].axis('off')

#     axs[1].imshow(gt)
#     axs[1].set_title("ground truth")
#     axs[1].axis('off')

#     # Plot the crops
#     for i, crop in enumerate(crops):
#         ax = axs[i + 2]
#         ax.imshow(crop)
#         ax.set_title(f"crop {i+1}")
#         ax.axis('off')

#     # img_with_bboxes = img_with_bboxes.astype(float) / 255.0
#     axs[5].imshow(img_with_bboxes)
#     axs[5].set_title("bbox cordinates")
#     axs[5].axis('off')


    
#     last_crop_index = len(crops) + 3
#     axs[last_crop_index].imshow(pred)
#     axs[last_crop_index].set_title(f"pred, dice: {dice_score}")
#     axs[last_crop_index].axis('off')

#     wandb.log({f"img_id: {file_name}": wandb.Image(fig)})
#     plt.close(fig)


# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)

def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    if ax is None:
        ax = plt.gca()  # Fallback to current axes if none specified
    ax.set_autoscale_on(False)

    # Assuming the first annotation's segmentation shape can represent the whole image size
    img_shape = anns[0]['segmentation'].shape
    img = np.ones((img_shape[0], img_shape[1], 4))
    img[:, :, 3] = 0  # Set alpha channel to 0 for transparency
    for ann in anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Random color with alpha value
        img[m] = color_mask

    ax.imshow(img)



# def log_images_to_wandb(image, gt, sam_masks, pred, dice_score, file_name, crops= None, img_with_bboxes= None):
#     # num_cols = 6  # Number of columns in each row
    
#     if crops is not None:
#         num_cols = 7
#         num_rows = (len(crops) + num_cols - 1) // num_cols  # Calculate the number of rows needed
#     else:
#         num_cols = 3
#         num_rows = 1
  
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 10))  # Increase the figsize to make the images larger
#     # fig, axs = plt.subplots(len(crops) + 2, 1, figsize=(10, 10))  # Increase the figsize to make the images larger


#     axs[0].imshow(image)
#     axs[0].set_title(f"input : {file_name}")
#     axs[0].axis('off')

#     axs[1].imshow(gt)
#     axs[1].set_title("ground truth")
#     axs[1].axis('off')


#     # Plot the crops
#     if crops is not None:
        
#         for i, crop in enumerate(crops):
#             if crop.dtype != np.uint8:
#                 crop = crop.astype(np.uint8)
#             ax = axs[i + 2]
#             ax.imshow(crop)
#             main_image_height, main_image_width = image.shape[:2]
#             resized_crop = cv2.resize(crop, (main_image_height, main_image_width))


#             ax.imshow(resized_crop)
#             ax.set_title(f"crop {i+1}")
#             ax.axis('off')
#             # ax.set_aspect('equal')  # Set the aspect ratio to equal
      
      
#     if img_with_bboxes is not None:
#         axs[4].imshow(img_with_bboxes)
#         axs[4].set_title("bbox cordinates")
#         axs[4].axis('off')


    
#     if crops is not None:
#         last_index = len(crops) + 3
#     else:
#         last_index = 2

   
#     pred =  np.sum(pred, axis=0) 
#     axs[last_index].imshow(pred)
#     axs[last_index].set_title(f"pred, dice: {round(dice_score, 4)}")
#     axs[last_index].axis('off')

    

#     axs[6].imshow(image)
#     show_anns(sam_masks)
#     axs[6].set_title("region proposals")
#     axs[6].axis('off')


#     wandb.log({f"img_id: {file_name}": wandb.Image(fig)})
#     plt.close(fig)





# def log_images_to_wandb(image, gt, sam_masks, pred, dice_score, file_name, crops=None, img_with_bboxes=None):
#     if crops is not None:
#         num_cols = 7  # Adjusted number of columns for crops
#         num_rows = (len(crops) + num_cols - 1) // num_cols  # Calculate the number of rows needed
#     else:
#         num_cols = 4
#         num_rows = 1
  
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 10))

#     axs[0].imshow(image)
#     axs[0].set_title(f"input : {file_name}")
#     axs[0].axis('off')

#     axs[1].imshow(gt)
#     axs[1].set_title("ground truth")
#     axs[1].axis('off')

#     # Moved region proposals to axs[2]
#     axs[2].imshow(image)
#     show_anns(sam_masks, ax=axs[2])  # Ensure show_anns can receive an ax argument
#     axs[2].set_title("region proposals")
#     axs[2].axis('off')

#     # Adjusted the starting index for crops to 3
#     if crops is not None:
#         for i, crop in enumerate(crops):
#             if crop.dtype != np.uint8:
#                 crop = crop.astype(np.uint8)
#             ax = axs[i + 3]  # Start from index 3 instead of 2
#             ax.imshow(crop)
#             main_image_height, main_image_width = image.shape[:2]
#             resized_crop = cv2.resize(crop, (main_image_height, main_image_width))

#             ax.imshow(resized_crop)
#             ax.set_title(f"crop {i+1}")
#             ax.axis('off')

#     # Adjust placement for img_with_bboxes and pred based on whether crops are present
#     if crops is not None:
#         bbox_index = len(crops) + 3
#         pred_index = len(crops) + 4
#     else:
#         # bbox_index = 3
#         pred_index = 4

#     if img_with_bboxes is not None:
#         axs[bbox_index].imshow(img_with_bboxes)
#         axs[bbox_index].set_title("bbox coordinates")
#         axs[bbox_index].axis('off')

#     pred = np.sum(pred, axis=0)
#     axs[pred_index].imshow(pred)
#     axs[pred_index].set_title(f"pred, dice: {round(dice_score, 4)}")
#     axs[pred_index].axis('off')

#     wandb.log({f"img_id: {file_name}": wandb.Image(fig)})
#     plt.close(fig)


def log_images_to_wandb(image, gt, sam_masks, pred, dice_score, file_name, crops=None, img_with_bboxes=None):
    if crops is not None:
        num_cols = 7  # Adjusted number of columns for crops
        num_rows = (len(crops) + num_cols - 1) // num_cols  # Calculate the number of rows needed
    else:
        num_cols = 4  # Ensure this matches the actual number of images you have in the non-crops case
        num_rows = 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 10))

    axs[0].imshow(image)
    axs[0].set_title(f"input : {file_name}")
    axs[0].axis('off')

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("ground truth")
    axs[1].axis('off')

    # Region proposals at axs[2]
    axs[2].imshow(image)
    show_anns(sam_masks, ax=axs[2])
    axs[2].set_title("region proposals")
    axs[2].axis('off')

    if crops is not None:
        for i, crop in enumerate(crops):
            if crop.dtype != np.uint8:
                crop = crop.astype(np.uint8)
            ax = axs[i + 3]
            ax.imshow(crop)
            main_image_height, main_image_width = image.shape[:2]
            resized_crop = cv2.resize(crop, (main_image_height, main_image_width))
            ax.imshow(resized_crop)
            ax.set_title(f"crop {i+1}")
            ax.axis('off')

    # Correct placement for pred based on whether crops and img_with_bboxes are present
    if crops is not None:
        pred_index = len(crops) + 3
    else:
        pred_index = 3  # Corrected index for pred when crops and img_with_bboxes are None

    if img_with_bboxes is not None:
        axs[pred_index].imshow(img_with_bboxes)
        axs[pred_index].set_title("bbox coordinates")
        axs[pred_index].axis('off')
        pred_index += 1

    if pred_index < num_cols:  # This check ensures that we do not exceed the subplot bounds
        axs[pred_index].imshow(pred, cmap='gray')
        axs[pred_index].set_title(f"pred, dice: {round(dice_score, 4)}")
        axs[pred_index].axis('off')
    else:
        print(f"Error: pred_index {pred_index} is out of bounds for the number of columns {num_cols}.")

    wandb.log({f"img_id: {file_name}": wandb.Image(fig)})

    plt.close(fig)


def dice_coeff(truth, prediction ):
    

    # if len(prediction > 1):
    #     # for lungs
    #     prediction = np.bitwise_or(prediction[0], prediction[1])
    # dice
    intersection = np.sum(truth * prediction)
    union = np.sum(truth) + np.sum(prediction)
    dice = 2.0 * intersection / union
    
    # mIoU
    iou = np.mean(intersection/(union-intersection))

    # precision
    total_pixel_pred = np.sum(prediction)
    precision = np.mean(intersection/total_pixel_pred)

    # recall
    total_pixel_truth = np.sum(truth)
    recall = np.mean(intersection/total_pixel_truth)

    return  dice, iou
