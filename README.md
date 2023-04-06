## [Self-distillation object segmentation via pyramid knowledge representation and transfer)
by Yunfei Zheng

## Introduction
We proposed a Self-Distillation framework via pyramid knowledge representation and transfer (PSPSD) for object segmentation task. Firstly, we constructed a simple and yet effective inference network by fusing four level features of the resnet backbone. Secondly, we proposed a SD method via pyramid knowledge representation and transfer, including a auxiliary SD network and a distillation loss. Extensive experiments are conducted on five object segmentation datasets (COD, CAMP, DUTO, SOC and THUR) to demonstrate the effectiveness of our SD framework. With the proposed SD method, our inference network with fewer parameters performs better typical object segmentation networks. The proposed PSPSD method outperforms recent SD methods (TF, FR, BYOT, DHM, SA and DKS).

This code is the implementation of our SD method. We also transferred other 6 SD methods which were used to image recognition task for the object segmentation task. 

## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.1.0](http://pytorch.org/)
- [OpenCV 3.4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)
- [Apex](https://github.com/NVIDIA/apex)

## Setting pretraining model path
- `ResNet-50` is used as the backbone 
|-- /data/PreModels
    |-- resnet50-19c8e357.pth


## Setting dataset path (taking 'DUTO' as a example) 
|-- data
    |-- train
        |-- DUTO
        	|-- Image
        	|-- Mask
		|-- train.txt
    |-- test
	|-- DUTO
        	|-- Image
        	|-- Mask
		|-- test.txt

## Training and Testing
This file includes training and testing codes of our baseline network, TF, FR, BYOT, DHM, SA, DKS, and our PSPSD network. related files are as follow:
our baseline network: 'train_baseline.py' and 'test_baseline.py'.
our PSPSD network: 'train_pspskd.py' and 'test_pspskd.py'.
TF: 'train_tfkd.py' and 'test_tfkd.py'.
FR: 'train_frskd.py' and 'test_frskd.py'.
BYOT: 'train_byot.py' and 'test_byot.py'.
DHM: 'train_dhm.py' and 'test_dhm.py'.
SA: 'train_satd.py' and 'test_satd.py'.
DKS: 'train_dks.py' and 'test_dks.py'.

The training and testing parameters can be set in related files:
   'cfg = Dataset.Config(datapath='/data/ExpData/Train/THUR/', savepath='./out/', model_version='resnet50', pretrain='/data/PreModels/resnet50-19c8e357.pth',
                      mode='train', batch=10, lr=0.05, momen=0.9, decay=5e-4, epoch=20)'

```shell
    cd src/
    python3 train.py
```
- `ResNet-50` is used as the backbone of our network 
- `batch=10`, `lr=0.05`, `momen=0.9`, `decay=5e-4`, `epoch=20`
- Warm-up and linear decay strategies are used to change the learning rate `lr`
- After training, the result models will be saved in `out` folder


## Citation
TF:Revisting knowledge distillation via label smooth regularization. CVPR2020;
BYOT:Be your own teacher:improve the performance of convolutional neural networks via self distillation.ICCV2019;
DHM:Dynamic hierarchical mimicking towards consistent optimization objectives.CVPR2020;
SA:Learning lightweight lane detection cnns by self attention distillation.ICCV2019;
DKS:Deeply-supervised knowledge synery.CVPR2020;
FR:Refine myself by teaching myself:feature refinement via self-knowledge distillation.





