# Lung nodule segmentation 
* Lung nodule segmentation implemented in Pytorch.
* Read and implement 2 papers: 
    + [ResUNet++: An Advanced Architecture for Medical Image Segmentation](https://arxiv.org/abs/1911.07067)
    + [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)
## Requirements
* Python >= 3.5 (3.6 recommended).
    ```
    pip install -r requirements.txt
    ```

## Dataset
* [Luna16 part 1](https://zenodo.org/record/3723295)
* [Luna16 part 2](https://zenodo.org/record/4121926)

## Folder Structure
```
.
├── backbone
│   ├── __init__.py 
│   ├── modules.py
│   ├── res_unet_plus.py
│   ├── res_unet.py
│   └── unet.py
├── checkpoints
│   └── default
│       └── exp8_512.pt - pretrained model with image size 512 x 512
├── config
│   └── default.yaml - holds configuration for training
├── dataset
│   └── dataloader.py - anything about data loading goes here
├── inference.py
├── luna_mask_extraction.py - extracts raw data to image, mask
├── lung_segment_png.py
├── preprocess.py
├── README.md
├── requirements.txt
├── train.py
└── utils
    ├── augmentation.py
    ├── hparams.py
    ├── __init__.py
    ├── logger.py
    └── metrics.py

6 directories, 20 files
```
## Usage 
* Training
  + From scratch:
    ```
    python train.py --name "default" --config "config/default.yaml"
    ```
  + Load from: 
    ```
    python train.py --name "default" --config "config/default.yaml" \ 
    --load_from "checkpoints/default/exp8_512.pt"
    ```
* Inference: 
```
python inference.py
```

## Results
![Alt text](image.png)