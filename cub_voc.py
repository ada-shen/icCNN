import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import numpy as np
# object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
#                      'bottle', 'bus', 'car', 'cat', 'chair',
#                      'cow', 'diningtable', 'dog', 'horse',
#                      'motorbike', 'person', 'pottedplant',
#                      'sheep', 'sofa', 'train', 'tvmonitor']
object_categories = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']

class CUB_VOC(Dataset):

    def __init__(self, root, dataname, mytype, train=True, transform=None, loader=default_loader, is_frac=None, sample_num=-1):
        self.root = os.path.expanduser(root)
        self.dataname = dataname
        self.mytype = mytype
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.is_frac = is_frac
        self.sample_num = sample_num
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        data_txt = None
        if self.dataname in object_categories:
            data_txt = '%s_info.txt' % self.dataname
        elif self.dataname == 'cub':
            if self.mytype == 'ori':
                data_txt = 'image_info.txt'
            else:
                data_txt = 'cubsample_info.txt'
        elif self.dataname == 'helen':
            data_txt = 'helen_info.txt'
        elif self.dataname == 'voc_multi':
            data_txt = 'animal_info.txt'

        self.data = pd.read_csv(os.path.join(self.root, data_txt),
                             names=['img_id','file_path','target','is_training_img'])
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        if self.is_frac is not None:
            self.data = self.data[self.data.target == self.is_frac]

        if self.sample_num != -1:
            self.data = self.data[0:self.sample_num]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, row.file_path)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.file_path)
        target = sample.target  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
