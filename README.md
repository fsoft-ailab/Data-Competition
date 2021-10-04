#### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model & Metrics](#run)
4. [How to Run](#quickstart)
   - [Quickstart](#quickstart)
   - [Install](#install)
   - [Train](#training)
   - [Evaluate](#evaluation)
   - [Detect](#detection)


<p align="center">
  <h1 align="center", id="introduction">DATA COMPETITION</h1></p>


## Dataset<a name="dataset"></a>

## Model & Metrics <a name="model"></a>
* The challenge is defined as object detection challenge. In the competition,
We use [YOLOv5s](https://github.com/ultralytics/yolov5/releases) and also use a pre-trained model
trained with easy mask dataset to greatly reduce training time.
* We fix all [hyperparameters](config/hyps/hyp_finetune.yaml) of the model
and **do not use any augmentation tips** in the source code.
Therefore, each participant need to build the best possible dataset by relabeling
incorrect labels, splitting train/val, augmentation tips, adding new dataset, etc.

* In training process, Early Stopping method with patience setten to 100 
is used to keep track of validation set's wAP@0.5. Detail about wAP@0.5 metric:
<p align="center">
wAP@0.5 = weighted_AP@0.5 = 0.2 * AP50_w + 0.3 * AP50_nw + 0.5 * AP50_wi

Where, \
AP50_w: \
AP50_nw:  
AP50_wi:
</p>

* The wAP@0.5 metric is also used as the main metric
to evaluate participant's submission on private testing set.


## How to Run<a name="run"></a>
### QuickStart <a name="quickstart"></a>
### Install requirements <a name="install"></a>

* All  requirements are included in [requirements.txt](https://github.com/fsoft-ailab/Data-Competition/blob/main/requirements.txt)


* Run the script below to clone and install all requirements


```angular2html
git clone https://github.com/fsoft-ailab/Data-Competition
cd Data-Competition
pip3 install -r requirements.txt
```


###Training <a name="training"></a>


* Put your dataset into the Data-Competition folder.
The structure of dataset folder is followed as below folder structure:
```bash
folder-name
├── images
│   ├── train
│   │   ├── train_img1.jpg
│   │   ├── train_img2.jpg
│   │   └── ...
│   │   
│   └── val
│       ├── val_img1.jpg
│       ├── val_img2.jpg
│       └── ...
│   
└── labels
    ├── train
    │   ├── train_img1.txt
    │   ├── train_img2.txt
    │   └── ...
    │   
    └── val
        ├── val_img1.txt
        ├── val_img2.txt
        └── ...

```
* Change relative paths to train and val images folder in `config/data_cfg.yaml` [file](config/data_cfg.yaml)

* [train_cfg.yaml](config/train_cfg.yaml) where we set up the model during training. 
You should not change such parameters because it will result in incorrect results. The training results are saved
in the `results/train/<name_version>`.
* Run the script below to train the model. Specify particular name to identify your experiment:
```angular2html
python train.py --batch-size 64 --device 0 --name <name_version> 
```
`Note`: If you get out of memory error, you can decrease batch-size to multiple of 2 as 32, 16.

### Evaluation <a name="evaluation"></a>

* You can evaluate the weights obtained after the training process. Please specify which dataset is evaluated at --task . Try the script below 


* Results are saved at results/evaluation/< task>/<name_folder>.


```angular2html
python val.py --weights <path_to_weight> --task train --name <name_folder>
                                                  val
                                                  test
```

### Detection <a name="detection"></a>

* You can use this script to make inferences on images

* Results are saved at < dir>.
```angular2html
python detect.py --weights <path_to_weight> --source <path_to_image/folder> --dir <save_dir>
```

* You can find more arguments at [detect.py](https://github.com/fsoft-ailab/Data-Competition/blob/main/train.py)

## References
Our source code is based on Ultralytics's implementation: https://github.com/ultralytics/yolov5