import random
import os 
import numpy as np

import torch
from torch.utils.data import DataLoader
import time

from data.cc359 import cc359_3d_volume
from data.cxr import CXR
from data.hc18 import hc18_2d

from sam_clip import get_eval
from sam_clip_3d import get_eval_3d


from segment_anything import sam_model_registry, SamPredictor
from utils.utils import process_config

import argparse

def main(args, suffix, wandb_mode, prompt_mode):

    #  ========== add the seed to make sure the results are reproducible ==========

    np.random.seed(args.seed)     # set random seed for numpy
    random.seed(args.seed)        # set random seed for python for image transformations
    torch.manual_seed(args.seed)  # set random seed for CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # set random seed for  GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    config = process_config(os.path.join(os.path.dirname(__file__), args.config))

    #  ========== model and data preparation ==========

    # register sam model
    model = sam_model_registry[config.model_type](checkpoint=config.sam_ckpt)
    # load data  
    if args.data == "cc359":
        test_data =   cc359_3d_volume(config, args.site, train = False)
        
    elif args.data == "cxr":
        test_data = CXR(config)
    
    else:
        test_data = hc18_2d(config, train= False)

    #  ========== SAM: 3d or 2d data ==========
        
    if args.mode == "sam_clip" or args.mode == "sam_prompted":
        final_avg_dice  = get_eval(test_data,  model, config, suffix, wandb_mode, prompt_mode, args.mode)
        print(f"Final average dice score: {final_avg_dice}")#, Total loss: {loss}")

    elif args.mode == "sam_clip_3d":
        final_avg_dice = get_eval_3d(test_data,  model, config, suffix, wandb_mode)
        print(f"Final average dice score: {final_avg_dice}")#, Total loss: {loss}")

  
if __name__ == '__main__':

    # ========== parameters setting ==========
    parser = argparse.ArgumentParser(description='SAM')
    # define arguments
    parser.add_argument('--mode', type = str, required = True, help = "sam_clip/sam_clip_3d ")
    parser.add_argument('--vit_name', type=str, default='vit_h', help = 'select the vit model for the image encoder of sam')
    parser.add_argument('--prompt_mode', type = str, required = True,  default= "crops", help = "CLIP visual prompt type e.g. bbox, blur, etc.")
    parser.add_argument('--config', type=str, required=True, help = 'path to config file. This file has all the parameters specificed in ./config folder')
    parser.add_argument('--data', type = str, required = True, help = "specify the name of dataset")
    parser.add_argument('--site', type = int, required = False, help = "specify the site of dataset (optional), only for cc359 dataset")
    parser.add_argument('--seed', type = int, required = True, help = "random seed")
    parser.add_argument('--suffix', type=str, required = False, help = "checkpoint suffix")
    parser.add_argument('--wandb_mode', type=str, required = False, help='wandb mode')
    

    args = parser.parse_args()
    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    print('----------------------------------------------------------------------')
    print('Time: ' + now)
    print('----------------------------------------------------------------------')
    print('                    Now start ...')
    print('----------------------------------------------------------------------')

    main(args, args.suffix, args.wandb_mode, args.prompt_mode)


    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')


  