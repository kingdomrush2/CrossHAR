# CrossHAR
This is the implementation of the IMWUT 2024 paper "CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining". While CrossHAR is evaluated with IMU sensor data for human activity recognition, it also has the potential to be applied to other applications where sequence-based sensing data is involved, e.g., EMG, EEG, and ECG. 

![image](https://github.com/kingdomrush2/CrossHAR/assets/58900218/d2c53a72-ca86-4044-b902-d4e887642691)


# Abstract
The increasing availability of low-cost wearable devices and smartphones has significantly advanced the field of sensor-based
human activity recognition (HAR), attracting considerable research interest. One of the major challenges in HAR is the
domain shift problem in cross-dataset activity recognition, which occurs due to variations in users, device types, and sensor
placements between the source dataset and the target dataset. Although domain adaptation methods have shown promise,
they typically require access to the target dataset during the training process, which might not be practical in some scenarios.
To address these issues, we introduce CrossHAR, a new HAR model designed to improve model performance on unseen target
datasets. CrossHAR involves three main steps: (i) CrossHAR explores the sensor data generation principle to diversify the data
distribution and augment the raw sensor data. (ii) CrossHAR then employs a hierarchical self-supervised pretraining approach
with the augmented data to develop a generalizable representation. (iii) Finally, CrossHAR fine-tunes the pretrained model with
a small set of labeled data in the source dataset, enhancing its performance in cross-dataset HAR. Our extensive experiments
across multiple real-world HAR datasets demonstrate that CrossHAR outperforms current state-of-the-art methods by 10.83%
in accuracy, demonstrating its effectiveness in generalizing to unseen target datasets.

## Requirement
Please check requirements.txt for required packages (Python 3.8.19).
```shell
pip install -r requirements.txt
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

## Dataset
We use UCI, MotionSense, Shoaib, and HHAR for evaluation. You can replace them with any dataset you prefer. 

## How to run
Three important files are pretrain.py, embedding.py, classifier.py, and cross_dataset.py, which share the same usage pattern.
```shell
usage: pretrain.py [-h] [-mv MODEL_VERSION] [-d {hhar,motion,uci,shoaib}] [-td {hhar,motion,uci,shoaib}] [-dv {20_120}] [-g GPU] [-f MODEL_FILE] [-t TRAIN_CFG] [-a MASK_CFG][-l LABEL_INDEX] [-s SAVE_MODEL] [-lr LABEL_RATE] [-am AUGUMENT_METHOD]

optional arguments:
  -h, --help            show this help message and exit
  -mv MODEL_VERSION, --model_version MODEL_VERSION
                        Model config
  -d {hhar,motion,uci,shoaib}, --dataset {hhar,motion,uci,shoaib}
                        Dataset name
  -td {hhar,motion,uci,shoaib}, --target_dataset {hhar,motion,uci,shoaib}
                        Dataset name
  -dv {20_120}, --dataset_version {20_120}
                        Dataset version
  -g GPU, --gpu GPU     Set specific GPU
  -f MODEL_FILE, --model_file MODEL_FILE
                        Pretrain model file
  -t TRAIN_CFG, --train_cfg TRAIN_CFG
                        Training config json file path
  -a MASK_CFG, --mask_cfg MASK_CFG
                        Mask strategy json file path
  -l LABEL_INDEX, --label_index LABEL_INDEX
                        Label Index
  -s SAVE_MODEL, --save_model SAVE_MODEL
                        The saved model name
  -lr LABEL_RATE, --label_rate LABEL_RATE
                        use finetune data ratio
  -am AUGUMENT_METHOD, --augument_method AUGUMENT_METHOD
```
Our experiment includes three stages, (1) self-supervised pretraining, (2) finetuning, and (3) cross-dataset evaluation.

### (1) Self-supervised pretraining
Example:
```shell
python pretrain.py -d uci
```
For this command, we will pretrain our model with the UCI dataset "data_20_120.npy" and "label_20_120.npy". The pretrained model will be saved as "model_LIMUBert_6_1.pt" and "model_TC_6_1.pt" in the saved/pretrain_base_uci_20_120 folder.
### (2) Finetuning
Example:
```shell
python embedding.py -d uci
```
For this command, we will load the pretrained model file "model_LIMUBert_6_1.pt" and "model_TC_6_1.pt" in the saved/pretrain_base_uci_20_120 folder. And embedding.py will save the learned representations as "embed_uci_20_120.npy" in the embed folder, and the label will be saved in the embed folder too.
```shell
python classifier.py -d uci -lr 0.1
```
For this command, we will load the embeddings or representations from "embed_uci_20_120.npy" and train the activity classifier. The trained classifier will be saved as "model_transformer.pt" in the saved/classifier_base_transformer_uci_20_120 folder.

### (3) Cross-dataset evaluation
```shell
python cross_dataset_test.py -d uci -td hhar
```
For this command, we will test the model trained by uci dataset using the HHAR dataset by loading the trained model and the HHAR dataset "data_20_120.npy" and "label_20_120.npy". 

## Reference

```shell
@article{CrossHAR,
author = {Hong, Zhiqing and Li, Zelong and Zhong, Shuxin and Lyu, Wenjun and Wang, Haotian and Ding, Yi and He, Tian and Zhang, Desheng},
title = {CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining},
year = {2024},
issue_date = {May 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {2},
url = {https://doi.org/10.1145/3659597},
doi = {10.1145/3659597},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {may},
articleno = {64},
numpages = {26},
}
```
