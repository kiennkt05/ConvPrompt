# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *
from continual_datasets.video_datasets import UCF101, ActivityNet

import utils

def _unwrap_to_base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset



class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def collate_video(batch):
    """
    Returns (B, S, C, H, W) videos and (B,) labels — no flattening.
    Segment averaging happens inside evaluate() after the forward pass.
    """
    videos, labels = zip(*batch)
    return torch.stack(videos, dim=0), torch.as_tensor(labels, dtype=torch.long)

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val, dataset_feat_train = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val, dataset_feat_train = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
                dataset_feat_train.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        base_train = _unwrap_to_base_dataset(dataset_train)
        video_collate = (
            collate_video if (isinstance(base_train, ActivityNet) or isinstance(base_train, UCF101)) else None
        )

        # Flattening (B, S, C, H, W) -> (B*S, C, H, W) multiplies per-step activations by S; keep the
        # effective frame batch near args.batch_size by loading fewer videos per step.
        train_bs = args.batch_size
        val_bs = args.batch_size
        if video_collate is not None:
            n_seg = getattr(base_train, "num_segments", 1) or 1
            train_bs = max(1, args.batch_size // n_seg)
            val_bs = max(1, args.batch_size // n_seg)
            if utils.is_main_process() and train_bs != args.batch_size:
                print(
                    f"DataLoader batch_size {args.batch_size} -> {train_bs} videos/step "
                    f"(num_segments={n_seg}) so flattened inputs are ~{train_bs * n_seg} frames/step."
                )
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=train_bs,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=video_collate,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=val_bs,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=video_collate,
        )

        data_loader_feat_train = torch.utils.data.DataLoader(
            dataset_feat_train, sampler=sampler_train,
            batch_size=val_bs,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=video_collate,
        )
        # print("Dataloader len: ", len(data_loader_val))
        # for batch_number, batch in enumerate(data_loader_val):
        #     # Print the shape of the first element in the batch
        #     first_batch_item = batch[0]
        #     print("Shape of the first element in the batch:", first_batch_item.shape)
        #     print("Batch: ",batch)
        #     break
        # exit(0)

        dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'feat_train': data_loader_feat_train})

    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)
        dataset_feat_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
        dataset_feat_train = CUB200(args.data_path, train=True, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
        dataset_feat_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_val).data

    elif dataset == 'UCF101':
        dataset_train = UCF101(args.data_path, train=True, num_tasks = args.num_tasks, transform=transform_train)
        dataset_val   = UCF101(args.data_path, train=False, num_tasks = args.num_tasks, transform=transform_val)

    elif dataset == 'ActivityNet':
        dataset_train = ActivityNet(args.data_path, train=True, num_tasks = args.num_tasks, transform=transform_train)
        dataset_val   = ActivityNet(args.data_path, train=False, num_tasks = args.num_tasks, transform=transform_val)
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val, dataset_feat_train

def split_single_dataset(dataset_train, dataset_val, args, dataset_feat_train=None,):
    if (
        hasattr(dataset_train, "task_ids")
        and hasattr(dataset_val, "task_ids")
        and hasattr(dataset_train, "class_mask")
        and len(dataset_train.task_ids) == len(dataset_train.targets)
        and len(dataset_val.task_ids) == len(dataset_val.targets)
    ):
        split_datasets = []
        mask = [list(task_scope) for task_scope in dataset_train.class_mask]

        for task_id in range(args.num_tasks):
            train_split_indices = [i for i, t_id in enumerate(dataset_train.task_ids) if t_id == task_id]
            test_split_indices = [i for i, t_id in enumerate(dataset_val.task_ids) if t_id == task_id]
            subset_train = Subset(dataset_train, train_split_indices)
            subset_val = Subset(dataset_val, test_split_indices)
            split_datasets.append([subset_train, subset_val])

        return split_datasets, mask

    nb_classes = len(dataset_val.classes)
    
    # Handle uneven splits (e.g. UCF-101: 101 classes with 10 or 20 tasks).
    # Remainder classes are absorbed into the FIRST task.
    # Even splits (CIFAR-100, ImageNet-R, ActivityNet) are unaffected.
    remainder        = nb_classes % args.num_tasks
    base_per_task    = nb_classes // args.num_tasks
    first_task_size  = base_per_task + remainder   # e.g. 11 for 10-task, 6 for 20-task

    labels = list(range(nb_classes))

    split_datasets = []
    mask = []

    if args.shuffle:
        random.shuffle(labels)

    for task_id in range(args.num_tasks):
        test_split_indices  = []

        # First task gets the extra remainder classes; all others get base_per_task
        chunk = first_task_size if task_id == 0 else base_per_task
        scope  = labels[:chunk]
        labels = labels[chunk:]

        mask.append(scope)

        scope_set = set(scope)

        for k, target in enumerate(dataset_train.targets):
            if int(target) in scope_set:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train = Subset(dataset_train, train_split_indices)
        subset_val = Subset(dataset_val, test_split_indices)
        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask

            # T.ColorJitter(brightness=.5, hue=.3),
            # T.RandomPerspective(distortion_scale=0.6, p=1.0),
            # T.RandomRotation(degrees=(0, 180)),
            # T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            # T.RandomInvert(),
            # T.RandomPosterize(bits=2),
            # T.RandomSolarize(threshold=192.0),
            # T.RandomAdjustSharpness(sharpness_factor=2),
            # T.RandomAutocontrast(),
            # T.RandomEqualize()

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        if args.use_transform:
            if "CUB" in args.dataset:
                transform = transforms.Compose([
                    transforms.Resize((300, 300), interpolation=3),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            elif "CIFAR" in args.dataset:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=63/255),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                    transforms.RandomInvert(),
                    transforms.RandomPosterize(bits=2),
                    transforms.RandomSolarize(threshold=192.0),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.RandomAutocontrast(),
                    transforms.RandomEqualize(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())

    if args.use_transform:
        if "CUB" in args.dataset:
            t = [transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        elif "CIFAR" in args.dataset:
            t = [transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),]
        else:
            t = [transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
    
    return transforms.Compose(t)