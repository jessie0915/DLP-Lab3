import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import collections


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')

        print('train:')
        print(collections.Counter(np.squeeze(label.values)))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')

        print('test:')
        print(collections.Counter(np.squeeze(label.values)))
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode, transform):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)
            transform : any transform, e.g., random flipping, rotation, cropping, and normalization.

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """retuen a pair of (image, label) which image is changed from self.tranform """

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img_path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(img_path).convert('RGB')
        label = int(self.label[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, label
