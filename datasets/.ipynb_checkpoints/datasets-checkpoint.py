from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
# from .flickr.flickr import FLICKR
from .coco.coco import COCO, get_loader
# from .cityscapes256.cityscapes256 import CITYSCAPES256
# from .ade20k.ade20k import ADE20K
import torch
import os


def get_dataset(config):
    batch_size = config.training.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    # if config.data.dataset == 'flickr':
    #     dataset_train = FLICKR(train=True)
    #     data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    #     dataset_eval = FLICKR(train=False)
    #     data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if config.data.dataset == 'coco':
        data_transform = transforms.Compose([ 
                            transforms.Resize(256),                          # smaller edge of image resized to 256
                            transforms.RandomCrop(224),                      # get 224x224 crop from random location
                            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                            transforms.ToTensor(),                           # convert the PIL Image to a tensor
                            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                 (0.229, 0.224, 0.225))])
        
        # Build data loader.
        data_loader_train = get_loader(transform=data_transform,
                                            mode='train',
                                            batch_size=batch_size, # batch size
                                            vocab_threshold=4,     # minimum word count threshold
                                            vocab_from_file=False)  # if True, load existing vocab file
        
        data_loader_eval = get_loader(transform=data_transform,
                                           mode='test',
                                           batch_size=batch_size, # batch size
                                           vocab_threshold=4,     # minimum word count threshold
                                           vocab_from_file=False)  # if True, load existing vocab file
        
#     if config.data.dataset == 'cityscapes256':
#         dataset_train = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='train', mode='fine',
#                                       crop=config.data.crop_to_square)
#         data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
#         dataset_eval = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='test', mode='fine',
#                                      crop=config.data.crop_to_square)
#         data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
        
#     if config.data.dataset == 'ade20k':
#         dataset_train = ADE20K('/export/data/tkuechle/datasets/ade20k', train=True, crop=True)
#         data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
#         dataset_eval = ADE20K('/export/data/tkuechle/datasets/ade20k', train=False, crop=True)
#         data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return data_loader_train, data_loader_eval
    

def get_img2txt_sample_data(config):
    dataset_dir = config.sampling.sample_data_dir
    
    if config.data.dataset == 'coco':
        data_transform = transforms.Compose([ 
                            transforms.Resize(256),                          # smaller edge of image resized to 256
                            transforms.RandomCrop(224),                      # get 224x224 crop from random location
                            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                            transforms.ToTensor(),                           # convert the PIL Image to a tensor
                            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                 (0.229, 0.224, 0.225))])
        
        # data_transform = transforms.Compose([ 
        #             transforms.Resize(256),                          # smaller edge of image resized to 256
        #             transforms.RandomCrop(224),                      # get 224x224 crop from random location
        #             transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        #             transforms.Resize(800),
        #             transforms.ToTensor(),                           # convert the PIL Image to a tensor
        #             transforms.Normalize((102.9801, 115.9465, 122.7717),      # normalize image for pre-trained model
        #                                  (1, 1, 1))])
        
        # Build data loader.
        data_loader = get_loader(transform=data_transform,
                                 mode='test',
                                 batch_size=1, # batch size
                                 vocab_threshold=4,     # minimum word count threshold
                                 vocab_from_file=True)  # if True, load existing vocab file
        
#     if config.data.dataset == 'flickr':
#         dataset = FLICKR(train=False, sample=True, scale_and_crop=False, root=dataset_dir)
#         data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
#     if config.data.dataset == 'cityscapes256':
#         dataset = CITYSCAPES256(root=dataset_dir, split='val', mode='fine', crop=False)
#         data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
#     if config.data.dataset == 'ade20k':
#         dataset = ADE20K(dataset_dir, train=False, crop=True)
#         data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    return data_loader