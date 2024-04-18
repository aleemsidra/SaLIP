## Test-Time Adaptation with SaLIP: A Cascade of SAM and CLIP for Zero shot Medical Image Segmentation

This project is an implementation of the paper ["Test-Time Adaptation with SaLIP: A Cascade of SAM and CLIP for Zero shot Medical Image Segmentation"](https://arxiv.org/pdf/2404.06362.pdf),  accepted at [CVPRW 2024].

## SaLIP
<p align="center"><img width="60%" src="/imgs/framework.png" /></p>

## Setup
To set up the project environment using conda, follow these steps:

1. Clone the repository: ```git@github.com:aleemsidra/SaLIP.git```
2. Navigate to the project directory: ```cd SaLIP```
3. Create a conda environment: ```conda create --name sam python=3.8```
5. Download SAM's checkpoint from [here ](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)




## Arguments
The ```main.py``` script is the main entry point of the project. It performs the following steps:

```mode:``` Specifies the baseline e.g. sam_clip, sam_clip_3d

```prompt_mode:``` Specifies the visual prompt mode for CLIP (default: "crops")

```vit_name: ``` Specifies SAM's checkpoint version (default: "vit_h" )

```data:``` Specifies the dataset 

```config:``` Specifies the path to config file

```seed:``` Specifies seed value for reproducibility



## Example command

```
python main.py --config ./config/lung.json --mode "sam_clip"  --prompt_mode "crops" --dataset "cxr" --seed 1234 
```

## Contact
Feel free to raise an issue or contact me at sidra.aleem2@mail.dcu.ie for queries and discussions.
