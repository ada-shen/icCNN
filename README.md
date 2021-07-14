# icCNN
This repository is a pytorch implementation of interpretable compositional convolutional neural networks.


Document Structure

### utils 
--utils.py [the utility modules used for networks]

--train_all.py [the top module of networks]

--celeb.py [the dataset driver for Large-scale CelebFaces Attributes (CelebA) dataset]

--cub_voc.py [the dataset driver for CUB200-2011 dataset/PASCAL-Part dataset/Helen Facial Feature dataset]

--vgg_iccnn_multi_train.py [the compositional CNN network used for VGGs of Multi-category classification]

--vgg_iccnn_train.py [the compositional CNN network used for VGGs of Single-category classification]

--vgg_ori_train.py [the traditional CNN network used for VGGs]

--resnet_iccnn_multi_train.py [the compositional CNN network used for Resnets of Multi-category classification]

--resnet_iccnn_train.py [the compositional CNN network used for Resnets of Single-category classification]

--resnet_ori_train.py [the traditional CNN network used for Resnets]

--densenet_iccnn_multi_train.py [the compositional CNN network used for Densenets of Multi-category classification]

--densenet_iccnn_train.py [the compositional CNN network used for Densenets of Single-category classification]

--densenet_ori_train.py [the traditional CNN network used for Densenets]

--load_utils.py [the driver of utility modules]

--newPad2d.py [the rewrite of deplicated padding]

--SpectralClustering.py [the module of Spectral Clustering]

--Similar_Mask_Generate.py [the module to generate similar masks]

### Train/Test

```
python3 train_all.py -type [ori/iccnn] -is_multi [single/multi 0/1] -model [vgg/resnet/densenet]
```
