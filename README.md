# icCNN
This repository is a pytorch implementation of interpretable compositional convolutional neural networks ([arXiv](https://arxiv.org/abs/2107.04474)), which has been published at IJCAI 2021.



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

### Models corresponding to results in the paper

We have uploaded the model files used in the paper to the following link https://b2y4v0n4v2.feishu.cn/drive/folder/fldcnwJeKnPEuQVydwhqNH8op5d
We have uploaded the model files for multi-category classification to the following link https://pan.baidu.com/s/1RlCfjiJCdGxI8fGCpeClKQ?pwd=87ps

### Datasets for train and visualization

We have uploaded datasets for train icCNNs and those images for visualization to the following link https://pan.baidu.com/s/1RlCfjiJCdGxI8fGCpeClKQ?pwd=87ps

## Citation

If you use this project in your research, please cite it.

```
@inproceedings{shen2021interpretable,
 title={Interpretable Compositional Convolutional Neural Networks},
 author={Shen, Wen and Wei, Zhihua and Huang, Shikun and Zhang, Binbin and Fan, Jiaqi and Zhao, Ping and Zhang, Quanshi},
 booktitle={Proceedings of the International Joint Conference on Artificial Intelligence},
 year={2021}
}
```
