# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import logging
import yaml

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils
from attribute_matching import RainbowAttributeMatcher


def parse_rainbow_args():
    parser = argparse.ArgumentParser(description='RainbowPrompt configuration')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_tasks', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--print_freq', type=int, default=None)
    parser.add_argument('--data_path', '--data-path', dest='data_path', type=str, default=None)
    
    # Pixel prompt arguments (defaults match LGSP/train.py)
    parser.add_argument('--pixel_prompt', type=str, default='NO')
    parser.add_argument('--pool_size', type=int, default=24)  # LGSP/train.py default=24
    parser.add_argument('--prompt_hid_dim', type=int, default=3)  # LGSP/train.py default=3
    parser.add_argument('--first_kernel_size', type=int, default=3)  # LGSP/train.py default=3
    parser.add_argument('--second_kernel_size', type=int, default=5)  # LGSP/train.py default=5
    parser.add_argument('--Dropout_Prompt', type=float, default=0.1)  # LGSP/train.py default=0.1
    parser.add_argument('--lr_local', type=float, default=2e-4)  # LGSP/train.py default=2e-4
    
    # Frequency mask arguments (defaults match LGSP/train.py)
    parser.add_argument('--Frequency_mask', type=bool, default=False)
    parser.add_argument('--num_r', type=int, default=100)  # LGSP/train.py default=100
    parser.add_argument('--temperature', type=float, default=0.1)  # LGSP/train.py default=0.1
    parser.add_argument('--lr_Frequency_mask', type=float, default=0.03)  # LGSP/train.py default=0.03
    
    # Adaptive weighting arguments (defaults match LGSP/train.py)
    parser.add_argument('--adaptive_weighting', type=bool, default=False)

    known_args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Unrecognized arguments ignored: {unknown}")

    with open(known_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    overrides = {k: v for k, v in vars(known_args).items() if k != 'config' and v is not None}
    config_dict.update(overrides)

    config_dict.setdefault('device', 'cuda')
    config_dict.setdefault('pin_mem', True)
    config_dict.setdefault('world_size', 1)
    config_dict.setdefault('dist_url', 'env://')
    config_dict.setdefault('method', 'rainbow')
    config_dict.setdefault('freeze', ['blocks', 'patch_embed', 'cls_token', 'pos_embed'])

    if 'rainbow' not in config_dict:
        config_dict['rainbow'] = {}

    fallback_defaults = {
        'use_transform': False,
        'use_clip_grad': True,
        'SLCA': False,
        'shuffle': False,
        'prompt_pool': False,
        'prompt_key': False,
        'prompt_key_init': 'uniform',
        'top_k': 1,
        'kernel_size': 17,
        'num_prompts_per_task': 5,
        'variable_num_prompts': True,
        'use_prompt_mask': True,
        'mask_first_epoch': False,
        'batchwise_prompt': False,
        'embedding_key': 'cls',
        'same_key_value': False,
        'use_g_prompt': False,
        'use_e_prompt': False,
        'use_prefix_tune_for_g_prompt': True,
        'use_prefix_tune_for_e_prompt': True,
        'g_prompt_length': 5,
        'g_prompt_layer_idx': [],
    }
    for key, value in fallback_defaults.items():
        config_dict.setdefault(key, value)

    if isinstance(config_dict.get('g_prompt_layer_idx'), tuple):
        config_dict['g_prompt_layer_idx'] = list(config_dict['g_prompt_layer_idx'])

    namespace = argparse.Namespace(**config_dict)
    namespace.config_path = known_args.config
    return namespace


def main_rainbow(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    if class_mask:
        args.nb_classes = sum(len(mask) for mask in class_mask)
        args.num_tasks = len(class_mask)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        num_tasks=args.num_tasks,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        prompt_length=args.prompt_length,
        use_rainbow_prompt=True,
        rainbow_config=args.rainbow,
        use_g_prompt=False,
        use_e_prompt=False,
        prompt_pool=False,
        args=args,
    )
    model.to(device)

    matcher = RainbowAttributeMatcher(
        num_tasks=args.num_tasks,
        embed_dim=model.embed_dim,
        hidden_dim=args.rainbow.get('align_hidden_dim', model.embed_dim),
    )
    matcher.to(device)

    args.lambda_sparse = args.rainbow.get('lambda_sparse', 0.0)
    args.lambda_match = args.rainbow.get('lambda_match', 0.0)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.eval:
        if not args.resume:
            raise ValueError('--resume checkpoint is required for evaluation mode')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'matcher' in checkpoint:
            matcher.load_state_dict(checkpoint['matcher'])
        model.to(device)
        matcher.to(device)

        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        trained_task = checkpoint.get('task_id', args.num_tasks - 1)
        max_task = min(trained_task, args.num_tasks - 1)
        evaluate_rainbow_till_now(
            model=model,
            matcher=matcher,
            data_loader=data_loader,
            device=device,
            task_id=max_task,
            class_mask=class_mask,
            acc_matrix=acc_matrix,
            args=args,
        )
        return

    start_time = time.time()
    train_and_evaluate_rainbow(
        model=model,
        matcher=matcher,
        criterion=criterion,
        data_loader=data_loader,
        device=device,
        class_mask=class_mask,
        args=args,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


def parse_legacy_args():
    print("Started main")
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    print("Parser created: ", parser)
    
    print("Getting config")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_convprompt':
        from configs.cifar100_convprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_convprompt', help='Split-CIFAR100 configs for ConvPrompt')
    elif config == 'imr_convprompt':
        from configs.imr_convprompt import get_args_parser
        config_parser = subparser.add_parser('imr_convprompt', help='Split-ImageNet-R configs for ConvPrompt')
    elif config == 'cub_convprompt':
        from configs.cub_convprompt import get_args_parser
        config_parser = subparser.add_parser('cub_convprompt', help='Split-CUB configs for ConvPrompt')
    elif config == 'cifar100_slca':
        from configs.cifar100_slca import get_args_parser
        config_parser = subparser.add_parser('cifar100_slca', help='Split-CIFAR100 SLCA configs')
    elif config == 'imr_slca':
        from configs.imr_slca import get_args_parser
        config_parser = subparser.add_parser('imr_slca', help='Split-ImageNet-R SLCA configs')
    elif config == 'cub_slca':
        from configs.cub_slca import get_args_parser
        config_parser = subparser.add_parser('cub_slca', help='Split-CUB SLCA configs')
    elif config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
    elif config == 'cub_dualprompt':
        from configs.cub_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cub_dualprompt', help='Split-CUB DualPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    print("Reached here")
    args = parser.parse_args()
    return args

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')



def main(args):
    # Print args
    print(args)
    if args.method == 'rainbow':
        main_rainbow(args)
        return

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    print("NB CLasses: ", args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        num_tasks=args.num_tasks,
        kernel_size=args.kernel_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        prompts_per_task=args.num_prompts_per_task,
        args=args
    )
    model.to(device)  

    if args.freeze:
        
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                if n.find('norm1')>=0 or n.find('norm2')>=0:
                    # print(n)
                    pass
                else:
                    p.requires_grad = False
            #         print(n)

        # exit(0)
        
    
    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    criterion = torch.nn.CrossEntropyLoss().to(device)

    milestones = [18] if "CIFAR" in args.dataset else [40]
    lrate_decay = 0.1
    param_list = list(model.parameters())
 

    network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    
    if not args.SLCA:
        optimizer = create_optimizer(args, model)
        if args.sched != 'constant':
            # lr_scheduler, _ = create_scheduler(args, optimizer)
            # Create cosine lr scheduler
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.sched == 'constant':
            lr_scheduler = None
    else:
        optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model,
                    criterion, data_loader, lr_scheduler, optimizer,
                    device, class_mask, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    if '--config' in sys.argv:
        args = parse_rainbow_args()
    else:
        args = parse_legacy_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
    sys.exit(0)
