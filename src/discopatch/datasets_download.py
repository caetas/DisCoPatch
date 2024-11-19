import os
from config import data_raw_dir
import gdown
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import time
import argparse

def parse_args():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--imagenet', action='store_true', help='Download ImageNet-C dataset')
    return parser.parse_args()

def reporthook(count, block_size, total_size):
    '''
    Callback function to show download progress
    '''
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

class DatasetDownloader:
    def __init__(self, dataset_name, file_name, url, origin, type):
        '''
        Initialize the DatasetDownloader class
        :param dataset_name: Name of the dataset
        :param file_name: Name of the file
        :param url: URL of the dataset
        :param origin: Origin of the dataset (zenodo or google_drive)
        :param type: Type of the file (tar or zip)
        '''
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.dataset_path = os.path.join(data_raw_dir, file_name.split('.')[0])
        self.dataset_url = url
        self.origin = origin
        self.type = type

    def download(self):
        '''
        Download the dataset
        '''
        if os.path.exists(self.dataset_path):
            print(f'{self.dataset_name} already exists')
            return
        
        if os.path.exists(os.path.join(data_raw_dir,self.file_name)):
            print(f'{self.file_name} already downloaded')
        
        else:
            if self.origin == 'zenodo':
                # Download from url
                print(f'Downloading {self.dataset_name} from {self.dataset_url}')
                urlretrieve(self.dataset_url, os.path.join(data_raw_dir,self.file_name), reporthook)
                
            elif self.origin == 'google_drive':
                # Download from Google Drive
                print(f'Downloading {self.dataset_name} from Google Drive')
                gdown.download(id=self.dataset_url, output=os.path.join(data_raw_dir,self.file_name), quiet=False)
            
        # Extract the file
        if self.type == 'tar':
            print(f'\nExtracting {self.file_name}')
            with tarfile.open(os.path.join(data_raw_dir,self.file_name)) as tar:
                tar.extractall(path=data_raw_dir)
            # Remove the tar file
            os.remove(os.path.join(data_raw_dir,self.file_name))
        elif self.type == 'zip':
            print(f'Extracting {self.file_name}')
            with zipfile.ZipFile(os.path.join(data_raw_dir,self.file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(data_raw_dir, self.file_name.split('.')[0]))
            # Remove the zip file
            os.remove(os.path.join(data_raw_dir,self.file_name))

args = parse_args()

dataset_names = ['dtd', 'ssbhard', 'ninco', 'openimageo', 'inaturalist']

file_names = ['texture.zip', 'ssb_hard.zip', 'ninco.zip', 'openimage_o.zip', 'inaturalist.zip']

dataset_urls = ['1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
                '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
                '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
                '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
                '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj']

origins = ['google_drive', 'google_drive', 'google_drive', 'google_drive', 'google_drive']

types = ['zip', 'zip', 'zip', 'zip', 'zip']

for dataset_name, file_name, dataset_url, origin, type in zip(dataset_names, file_names, dataset_urls, origins, types):
    dataset_downloader = DatasetDownloader(dataset_name, file_name, dataset_url, origin, type)
    dataset_downloader.download()

if args.imagenet:
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C')):
        print('ImageNet-C already exists')
    else:
        os.makedirs(os.path.join(data_raw_dir, 'ImageNet-C'), exist_ok=True)

    # check if defocus_blur, glass_blur, motion_blur and zoom_blur already exist
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'defocus_blur')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'glass_blur')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'motion_blur')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'zoom_blur')):
        print('ImageNet-C blur already exists')
    else: 
        if not os.path.exists(os.path.join(data_raw_dir, 'Imagenet-C' 'blur.tar')):
            print('Downloading ImageNet-C blur from https://zenodo.org/record/2239423/files/blur.tar?download=1')
            urlretrieve('https://zenodo.org/records/2235448/files/blur.tar?download=1', os.path.join(data_raw_dir, 'ImageNet-C', 'blur.tar'), reporthook)
        print('\nExtracting blur.tar')
        with tarfile.open(os.path.join(data_raw_dir, 'ImageNet-C', 'blur.tar')) as tar:
            tar.extractall(path=os.path.join(data_raw_dir, 'ImageNet-C'))
        # Remove the tar file
        os.remove(os.path.join(data_raw_dir, 'ImageNet-C', 'blur.tar'))

    # check contrat, elastic_transform, pixelate and jpeg_compression already exist
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'contrast')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'elastic_transform')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'pixelate')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'jpeg_compression')):
        print('ImageNet-C digital already exists')
    else:
        if not os.path.exists(os.path.join(data_raw_dir, 'Imagenet-C' 'digital.tar')):
            print('Downloading ImageNet-C digital from https://zenodo.org/records/2235448/files/digital.tar?download=1')
            urlretrieve('https://zenodo.org/records/2235448/files/digital.tar?download=1', os.path.join(data_raw_dir, 'ImageNet-C', 'digital.tar'), reporthook)
        print('\nExtracting digital.tar')
        with tarfile.open(os.path.join(data_raw_dir, 'ImageNet-C', 'digital.tar')) as tar:
            tar.extractall(path=os.path.join(data_raw_dir, 'ImageNet-C'))
        # Remove the tar file
        os.remove(os.path.join(data_raw_dir, 'ImageNet-C', 'digital.tar'))

    # check if frost, snow, fog, and brightness already exist
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'frost')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'snow')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'fog')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'brightness')):
        print('ImageNet-C weather already exists')
    else:
        if not os.path.exists(os.path.join(data_raw_dir, 'Imagenet-C' 'weather.tar')):
            print('Downloading ImageNet-C weather from https://zenodo.org/records/2235448/files/weather.tar?download=1')
            urlretrieve('https://zenodo.org/records/2235448/files/weather.tar?download=1', os.path.join(data_raw_dir, 'ImageNet-C', 'weather.tar'), reporthook)
        print('\nExtracting weather.tar')
        with tarfile.open(os.path.join(data_raw_dir, 'ImageNet-C', 'weather.tar')) as tar:
            tar.extractall(path=os.path.join(data_raw_dir, 'ImageNet-C'))
        # Remove the tar file
        os.remove(os.path.join(data_raw_dir, 'ImageNet-C', 'weather.tar'))

    # check if speckle_noise, spatter, gaussian_blur, and saturate already exist. if not download extra
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'speckle_noise')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'spatter')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'gaussian_blur')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'saturate')):
        print('ImageNet-C extra already exists')
    else:
        if not os.path.exists(os.path.join(data_raw_dir, 'Imagenet-C' 'extra.tar')):
            print('Downloading ImageNet-C extra from https://zenodo.org/records/2235448/files/extra.tar?download=1')
            urlretrieve('https://zenodo.org/records/2235448/files/extra.tar?download=1', os.path.join(data_raw_dir, 'ImageNet-C', 'extra.tar'), reporthook)
        print('\nExtracting extra.tar')
        with tarfile.open(os.path.join(data_raw_dir, 'ImageNet-C', 'extra.tar')) as tar:
            tar.extractall(path=os.path.join(data_raw_dir, 'ImageNet-C'))
        # Remove the tar file
        os.remove(os.path.join(data_raw_dir, 'ImageNet-C', 'extra.tar'))

    # check if gaussian_noise, shot_noise, and impulse_noise already exist. if not download noise
    if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'gaussian_noise')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'shot_noise')) and os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C', 'impulse_noise')):
        print('ImageNet-C noise already exists')
    else:
        if not os.path.exists(os.path.join(data_raw_dir, 'Imagenet-C' 'noise.tar')):
            print('Downloading ImageNet-C noise from https://zenodo.org/records/2235448/files/noise.tar?download=1')
            urlretrieve('https://zenodo.org/records/2235448/files/noise.tar?download=1', os.path.join(data_raw_dir, 'ImageNet-C', 'noise.tar'), reporthook)
        print('\nExtracting noise.tar')
        with tarfile.open(os.path.join(data_raw_dir, 'ImageNet-C', 'noise.tar')) as tar:
            tar.extractall(path=os.path.join(data_raw_dir, 'ImageNet-C'))
        # Remove the tar file
        os.remove(os.path.join(data_raw_dir, 'ImageNet-C', 'noise.tar'))