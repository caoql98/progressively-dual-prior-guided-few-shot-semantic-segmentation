from __future__ import print_function
from __future__ import absolute_import

from config import settings
from data.transforms import transforms
from data.transforms import transform1
from torch.utils.data import DataLoader
from data.voc_train import voc_train
from data.voc_val import voc_val
from data.coco_train import coco_train
from data.coco_val import coco_val


def data_loader(args,k_shot=1):

    batch = args.batch_size
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals
    size = settings.size

    tsfm_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_vals, std_vals)
                                      ])
    # value_scale = 255
    # mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    # std = [0.229, 0.224, 0.225]
    # std = [item * value_scale for item in std]
    # tsfm_train = transform1.Compose([transform1.RandScale([0.9, 1.1]),
    #                                 transform1.RandRotate([-10, 10], padding=mean, ignore_label=255),
    #                                 transform1.RandomGaussianBlur(),
    #                                 transform1.RandomHorizontalFlip(),
    #                                 transform1.Crop([size, size], crop_type='rand', padding=mean, ignore_label=255),
    #                                 transform1.ToTensor(),
    #                                 transform1.Normalize(mean=mean, std=std)
    #                                 ])
    
    if args.dataset == 'coco':
        img_train = coco_train(args, transform=tsfm_train,k_shot=k_shot)
    if args.dataset == 'voc':
        img_train = voc_train(args, transform=tsfm_train,k_shot=k_shot)

    train_loader = DataLoader(img_train, batch_size=batch, shuffle=True, num_workers=1)

    return train_loader

def val_loader(args, k_shot=1):
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals

    tsfm_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(321,321)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals)
                                    ])
    
    # value_scale = 255
    # mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    # std = [0.229, 0.224, 0.225]
    # std = [item * value_scale for item in std]
    # tsfm_val = transform1.Compose([
    #             transform1.Resize(size=321),
    #             transform1.ToTensor(),
    #             transform1.Normalize(mean=mean, std=std)
    #             ]) 
    
    if args.dataset == 'coco':
        img_val = coco_val(args, transform=tsfm_val, k_shot=k_shot)
    if args.dataset == 'voc':
        img_val = voc_val(args, transform=tsfm_val, k_shot=k_shot)


    val_loader = DataLoader(img_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return val_loader
