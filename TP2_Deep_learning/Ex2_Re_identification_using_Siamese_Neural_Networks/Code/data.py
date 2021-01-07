from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from PIL import Image
from opt import opt
import os
import re
import numpy as np


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Market1501(train_transform, 'train', opt.data_path)
        self.testset = Market1501(test_transform, 'test', opt.data_path)
        self.queryset = Market1501(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)#8

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)#8
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)#8

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if int(path.split('/')[-1].split('_')[0])  != -1]

        num_pids, num_imgs, num_cams, = self.get_imagedata_info(self.imgs)
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  ", dtype ," |", str(num_pids) , "| ", str(num_imgs) ," | ", str(num_cams) )
        print("  ----------------------------------------")

        self.ids, self.cameras = self.get_id_cam()
        self.unique_ids = sorted(set(self.ids))
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)} # Build a dictionary where the key is the id and the element is the label
        # print(self._id2label)

    def __getitem__(self, index):
        # please write the __getitem__ method
        img = Image.open(self.imgs[index-1]) #3*64*128
        target = self._id2label[self.get_id(self.imgs[index-1])]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        # please write the __len__ method
        return len(self.imgs)

    def get_id_cam(self):
        """"
        :return: person id list and camera list corresponding to dataset image paths
        """
        return [int(file_path.split('/')[-1].split('_')[0]) for file_path in self.imgs], [int(file_path.split('/')[-1].split('_')[1][1]) for file_path in self.imgs]

    def get_id(self,file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    def list_pictures(self, directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset does not exists!{}'.format(directory)
        return sorted([os.path.join(root, f).replace('\\','/')
                       for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    def get_imagedata_info(self, data):
        pids, cams = [], []

        for file_path in data:
            camid = int(file_path.split('/')[-1].split('_')[1][1])
            pid=int(file_path.split('/')[-1].split('_')[0])
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
