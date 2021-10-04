#### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model & Metrics](#model)
4. [QuickStart](#quickstart)
   - [Install](#install)
    - [Train](#training)
    - [Evaluate](#evaluation)
    - [Detect](#detection)


<p align="center">
  <h1 align="center", id="introduction">DATA COMPETITION</h1></p>


## Dataset<a name="dataset"></a>

## Model & Metrics <a name="model"></a>
* The challenge is defined as object detection challenge. Therefore, We use [YOLOv5s](https://github.com/ultralytics/yolov5/releases)
model in the competition. We fix all hyperparameters of the model and do not use any augmentation tips in the source code.
And so, each participant need to build the best possible dataset.
* There are some fixed hyperparameters:
  

## QuickStart<a name="quickstart"></a>
### Install requirements <a name="install"></a>

* All  requirements are included in [requirements.txt](https://github.com/fsoft-ailab/Data-Competition/blob/main/requirements.txt)


* Run the script below to clone and install all requirements


```angular2html
git clone https://github.com/fsoft-ailab/Data-Competition
cd Data-Competition
pip3 install -r requirements.txt
```


###Training <a name="training"></a>


* You must put the dataset into the Data-Competition folder. The directory containing the dataset is named "dataset".
All configurations of the dataset, you can see at [data_cfg.yaml](https://github.com/fsoft-ailab/Data-Competition/blob/main/config/data_cfg.yaml). 


* [train_cfg.yaml](https://github.com/fsoft-ailab/Data-Competition/blob/main/config/train_cfg.yaml) where we set up the model during training. 
You should not change such parameters because it will result in incorrect results. The training results are saved in the results/train/<name_version>.
Run the script below to train the model:
```angular2html
python train.py --batch-size 64 --device 0 --version <name_version> 
```


### Evaluation <a name="evaluation"></a>

* You can evaluate the weights obtained after the training process. Please specify which dataset is evaluated at --task . Try the script below 


* Results are saved at results/evaluation/< task>/<name_folder>.


```angular2html
python val.py --weights <path_to_weight> --task train --name <name_folder>
                                                  val
                                                  test
```

### Detection <a name="detection"</a>

* You can use this script to make inferences on images

* Results are saved at < dir>.
```angular2html
python detect.py --weights <path_to_weight> --source <path_to_image/folder> --dir <save_dir>
```

* You can find more arguments at [detect.py](https://github.com/fsoft-ailab/Data-Competition/blob/main/train.py)

## References
Our source code is based on Ultralytics's implementation: https://github.com/ultralytics/yolov5