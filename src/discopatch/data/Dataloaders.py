from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import data_raw_dir, data_dir
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

class ImageNetPatchDataset(Dataset):
    def __init__(self, transform_fn, train = False, patches = 16):

        self.dataset = load_dataset('benjamin-paine/imagenet-1k-256x256')
        self.transform_fn = transform_fn
        self.train = train
        self.dataset.set_transform(self.transform_fn)
        self.dataset = self.dataset['train' if train else 'validation']
        self.patches = patches

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # get self.patches random patches from the image
        image = self.dataset[idx]['pixel_values']
        patches = []
        for i in range(self.patches):
            x = np.random.randint(0, image.shape[1] - 64)
            y = np.random.randint(0, image.shape[2] - 64)
            patches.append(image[:, x:x+64, y:y+64])
        patches = torch.stack(patches)
        return patches, self.dataset[idx]['label']

def imagenetpatch_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0, patches = 16):

        if normalize:
            transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

        else:
            transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ])

        # Define a function to apply transformations to each dataset sample
        def transform_fn(examples):
            examples['pixel_values'] = [transform(image) for image in examples['image']]
            return examples

        dataset = ImageNetPatchDataset(transform_fn, train = True, patches=patches)

        training_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers = num_workers)
        
        return training_loader
        
        
def imagenetpatch_val_loader(batch_size, normalize = False, input_shape = None, patches = 16, num_workers = 0):

        if normalize:
            transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

        else:
            transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ])

        # Define a function to apply transformations to each dataset sample
        def transform_fn(examples):
            examples['pixel_values'] = [transform(image) for image in examples['image']]
            return examples
        
        dataset = ImageNetPatchDataset(transform_fn, train = False, patches=patches)

        validation_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers = num_workers)
        
        return validation_loader
        
class CustomDataset(Dataset):
    def __init__(self, root, split, transform=None, patches=16):
        '''Custom dataset for loading images from a list of paths
        Args:
            root (str): root directory of the images
            split (str): path to a text file containing the list of image paths
            transform (torchvision.transforms): image transformations
        '''
        self.root = root
        self.transform = transform
        with open(f'{split}', 'r') as f:
            self.data = f.readlines()
        #remove newline characters
        self.data = [x.strip() for x in self.data]
        self.data = [os.path.join(root, x) for x in self.data]
        self.patches = patches
        
    
    def __len__(self):
        '''Returns the number of images in the dataset'''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''Returns the image at the given index
        Args:
            idx (int): index of the image
        Returns:
            image (PIL.Image): the image at the given index
            label (int): the label of the image
        '''
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 32 random 32x32 patches
        patches = []
        for i in range(self.patches):
            x = np.random.randint(0, 192)
            y = np.random.randint(0, 192)
            patches.append(image[:, x:x+64, y:y+64])
        return torch.stack(patches), 0
    
def inaturalist_dataloader(batch_size, img_size, num_workers=0, patches=16):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(data_raw_dir, os.path.join(data_dir,'splits','test_inaturalist.txt'), transform, patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def ninco_dataloader(batch_size, img_size, num_workers=0, patches=16):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(data_raw_dir, os.path.join(data_dir,'splits','test_ninco.txt'), transform, patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def ssbhard_dataloader(batch_size, img_size, num_workers=0, patches=16):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(data_raw_dir, os.path.join(data_dir,'splits','test_ssb_hard.txt'), transform, patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def openimageo_dataloader(batch_size, img_size, num_workers=0, patches=16):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(data_raw_dir, os.path.join(data_dir,'splits','test_openimage_o.txt'), transform, patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def dtd_dataloader(batch_size, img_size, num_workers=0, patches=16):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(data_raw_dir, os.path.join(data_dir,'splits','test_dtd_3.txt'), transform, patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

class ImageNetCPatch(Dataset):
    def __init__(self, root, corruption, intensity, transform=None, patches=16):
        self.root = root
        self.transform = transform
        self.data = glob(os.path.join(root, 'ImageNet-C', f'{corruption}', f'{intensity}','*', '*.JPEG'))
        self.patches = patches
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        patches = []
        for i in range(self.patches):
            x = np.random.randint(0, 192)
            y = np.random.randint(0, 192)
            patch = image[:, x:x+64, y:y+64]
            patches.append(patch)
        return torch.stack(patches), 0
    
def imagenetc_dataloader(batch_size, corruption, intensity, num_workers=0, patches=16):
    
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        dataset = ImageNetCPatch(data_raw_dir, corruption, intensity, transform, patches)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

def pick_dataset(name='cifar10', train=True, batch_size=64, img_size=256, num_workers=0, patches=16):
    '''Returns a dataloader for the given dataset
    Args:
        name (str): name of the dataset
        train (bool): whether to load the training or test set
        batch_size (int): batch size
        img_size (int): size of the images
        Returns:
        dataloader (torch.utils.data.DataLoader): dataloader for the given dataset
    '''
    if name == 'imagenet':
        if train:
            return imagenetpatch_train_loader(batch_size, input_shape=img_size, patches=patches, num_workers=num_workers)
        else:
            return imagenetpatch_val_loader(batch_size, input_shape=img_size, patches=patches, num_workers=num_workers)
    elif name == 'dtd':
        return dtd_dataloader(batch_size, img_size, patches=patches, num_workers=num_workers)
    elif name == 'inaturalist':
        return inaturalist_dataloader(batch_size, img_size, patches=patches, num_workers=num_workers)
    elif name == 'ninco':
        return ninco_dataloader(batch_size, img_size, patches=patches, num_workers=num_workers)
    elif name == 'ssbhard' or name == 'ssb-hard':
        return ssbhard_dataloader(batch_size, img_size, patches=patches, num_workers=num_workers)
    elif name == 'openimageo':
        return openimageo_dataloader(batch_size, img_size, patches=patches, num_workers=num_workers)
    else:
        raise ValueError(f'Unknown dataset: {name}')