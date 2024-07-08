# encoding: utf-8
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#############################################################
##### Dataset with memory bank and contrastive samples. #####
#############################################################


class ISIC_InstanceSample(Dataset):

    def __init__(self, root_dir, csv_file, CCD_mode, transform=None, p=10, k=4096,
                 mode='exact', is_sample=True, percent=1.0):
        super(ISIC_InstanceSample, self).__init__()

        self.p = p
        self.k = k
        self.mode = mode
        self.CCD_mode = CCD_mode
        self.is_sample = is_sample

        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['id_code'].values  # image name
        self.labels = file['diagnosis'].values.astype(int)  # scalar label
        n_classes = len(np.unique(self.labels))
        # one hot. [num_images, num_classes]
        self.labels = np.eye(n_classes)[self.labels.reshape(-1)]
        self.transform = transform

        print('Total # images:{}, labels:{}'.format(
            len(self.images), len(self.labels)))

        num_samples = len(self.images)
        label = np.argmax(self.labels, axis=1)

        self.cls_positive = [[] for i in range(n_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(n_classes)]
        for i in range(n_classes):
            for j in range(n_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i])
                             for i in range(n_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i])
                             for i in range(n_classes)]

        self.class_index = self.cls_positive

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(n_classes)]

        self.cls_positive = np.asarray(self.cls_positive, dtype=object)
        self.cls_negative = np.asarray(self.cls_negative, dtype=object)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.images[index]+'.png')
        img = Image.open(image_name).convert('RGB')
        target = np.argmax(self.labels, axis=1)[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_sample:
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
            elif self.mode == 'multi_pos':
                pos_idx = np.random.choice(
                    self.cls_positive[target], self.p, replace=False)
            else:
                raise NotImplementedError(self.mode)

            if self.CCD_mode == "sup":
                replace = True if self.k > len(
                    self.cls_negative[target]) else False
                neg_idx = np.random.choice(
                    self.cls_negative[target], self.k, replace=replace)
            elif self.CCD_mode == "unsup":
                pos_others = np.setdiff1d(self.cls_positive[target], ([index]))
                all_negative = np.hstack(
                    (pos_others, self.cls_negative[target]))
                neg_idx = np.random.choice(all_negative, self.k, replace=True)

            if self.mode == 'exact' or self.mode == 'relax':
                sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            elif self.mode == 'multi_pos':
                sample_idx = np.hstack((pos_idx, neg_idx))

            return img, label, index, sample_idx

    def __len__(self):
        return len(self.images)


def load_dataset(args, p=10, k=4096, mode='exact', is_sample=True, percent=1.0):
    csv_file_train = args.csv_file_path + \
        args.dataset + '/' + args.split + '_train.csv'
    csv_file_test = args.csv_file_path + args.dataset + '/' + args.split + '_test.csv'

    train_transform = TransformTwice(transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = ISIC_InstanceSample(root_dir=args.root_path,
                                    csv_file=csv_file_train,
                                    CCD_mode=args.CCD_mode,
                                    transform=train_transform,
                                    p=p,
                                    k=k,
                                    mode=mode,
                                    is_sample=is_sample,
                                    percent=percent)

    test_set = ISIC_Dataset(root_dir=args.root_path,
                            csv_file=csv_file_test, transform=test_transform)

    return train_set, test_set


# Dataset without memory bank
class ISIC_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ISIC_Dataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['id_code'].values  # image name
        self.labels = file['diagnosis'].values.astype(int)
        self.n_classes = len(np.unique(self.labels))
        self.labels = np.eye(self.n_classes)[
            self.labels.reshape(-1)]  # one_hot labels
        self.transform = transform

        print('Total # images:{}, labels:{}'.format(
            len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = os.path.join(self.root_dir, self.images[index]+'.png')
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
