#### Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
2. [QuickStart](#quickstart)
   - [Install](#install)
	- [Train](#train)
	- [Evaluate](#evaluate)
    - [Detect](#detect)
    
 <p align="center">
  <h1 align="center", id="introduction">DATA COMPETITION</h1>
</p>

## Dataset<a name="dataset"></a>
## QuickStart<a name="quickstart"></a>
<details open>
<summary>Install</summary><a name="install"></a>


* All  requirements are included in [requirements.txt](https://github.com/fsoft-ailab/Data-Competition/blob/main/requirements.txt)


* Run the script below to clone and install all requirements


```bash
$ git clone https://github.com/fsoft-ailab/Data-Competition
$ cd Data-Competition
$ pip install -r requirements.txt
```
</details>

<details open>
<summary>Train</summary> <a name="train"></a>


* You must put the dataset into the Data-Competition folder. The directory containing the dataset is named "dataset".
All configurations of the dataset, you can see at [data_cfg.yaml](https://github.com/fsoft-ailab/Data-Competition/blob/main/config/data_cfg.yaml). 


* [train_cfg.yaml](https://github.com/fsoft-ailab/Data-Competition/blob/main/config/train_cfg.yaml) where we set up the model during training. 
You should not change such parameters because it will result in incorrect results. The training results are saved in the results/train/<name_version>.
Run the script below to train the model:
```bash
$ python train.py --batch-size 64 --device 0 --version <name_version> 
```

</details>

<details open >
<summary>Evaluate</summary> <a name="evaluate"></a>


* You can evaluate the weights obtained after the training process. Please specify which dataset is evaluated at --task . Try the script below 


* Results are saved at results/evaluation/< task>/<name_folder>.


```bash
$ python val.py --weights <path_to_weight> --task train --name <name_folder>
                                                  val
                                                  test
```
</details>

<details open>
<summary>Detect</summary>  <a name="detect"></a>


* You can use this script to make inferences on images

* Results are saved at < dir>.
```bash
$ python detect.py --weights <path_to_weight> --source <path_to_image/folder> --dir <save_dir>
```

* You can find more arguments at [detect.py](https://github.com/fsoft-ailab/Data-Competition/blob/main/train.py)

</details>
