# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn import MSELoss

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import logging

# for attribute matching of tasks
from attribute_matching import num_new_prompts
from timm.scheduler import create_scheduler


def _task_logit_slice(class_mask, task_id):
    """Column range in concatenated multi-head logits for this task."""
    start = sum(len(class_mask[j]) for j in range(task_id))
    end = start + len(class_mask[task_id])
    return start, end


def _global_to_task_local(target, class_mask, task_id, nb_classes, device):
    """Map dataset global class ids to 0..K-1 for CrossEntropy on the task head."""
    g2l = torch.full((nb_classes,), -100, dtype=torch.long, device=device)
    for local_i, g in enumerate(class_mask[task_id]):
        g2l[g] = local_i
    return g2l[target]


def train_one_epoch(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    optimizer=None,
                    old_prompt_matcher = None,
                    old_prompt = None,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    old_num_k=5,):

    model.train(set_training_mode)

    s = old_num_k

    # Freezing previous tasks' filters
    for name, param in model.named_parameters():
        if name.find('e_prompt.v_conv_vals') >=0  or name.find('e_prompt.k_conv_vals') >=0:
            for i in range(s):
                if name.find('.{}.weight'.format(i)) >=0 or name.find('.{}.bias'.format(i)) >=0:
                    param.requires_grad = False


    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # metric_logger.add_meter('Lr_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if args.SLCA:
        metric_logger.add_meter('Lr_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Lr_rps', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    for step, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Detect multi-segment inputs and flatten for the ViT backbone
        is_multiseg = (input.ndim == 5)
        if is_multiseg:
            B, S, C, H, W = input.shape
            flat_input = input.reshape(B * S, C, H, W)
        else:
            flat_input = input
            B = input.shape[0]

        output = model(flat_input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # TSN consensus: average logits over segments before loss
        if is_multiseg:
            logits = logits.reshape(B, S, -1).mean(dim=1)

        # Multi-head outputs are task-local columns; targets are global class ids from the dataset.
        start, end = _task_logit_slice(class_mask, task_id)
        task_logits = logits[:, start:end]
        cur_targets = _global_to_task_local(
            target, class_mask, task_id, args.nb_classes, device
        )
        loss = criterion(task_logits, cur_targets)

        if args.use_e_prompt or args.use_g_prompt:
            if task_id > 0:
                l1_loss = 0.0
                for old_wt, new_wt in zip(old_prompt_matcher.parameters(), model.e_prompt.prompt_embed_matcher.parameters()):
                    l1_loss += torch.norm(old_wt.detach() - new_wt, p=1)
                loss = loss + 0.01 * l1_loss
                prompt_loss = torch.norm(old_prompt.detach() - model.e_prompt.prompt, p=1)
                loss = loss + 0.01 * prompt_loss


        acc1, acc5 = accuracy(task_logits, cur_targets, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        loss = loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        if args.SLCA:
            metric_logger.update(Lr_cls=optimizer.param_groups[0]["lr"])
            metric_logger.update(Lr_rps=optimizer.param_groups[1]["lr"]) 
        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
        metric_logger.meters['Acc@5'].update(acc5.item(), n=B)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Batchwise Eval time
    start_eval_time = time.time()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # -----------------------------------------------------------
            # 1. Detect multi-segment inputs and flatten for the ViT
            # -----------------------------------------------------------
            is_multiseg = (input.ndim == 5)
            if is_multiseg:
                B, S, C, H, W = input.shape
                flat_input = input.reshape(B * S, C, H, W)
            else:
                flat_input = input
                B = input.shape[0]
            # -----------------------------------------------------------

            # compute output

            output = model(flat_input, task_id=task_id)
            logits = output['logits']

            # -----------------------------------------------------------
            # 2. Average the logits over the temporal segments (TSN consensus)
            # -----------------------------------------------------------
            if is_multiseg:
                # Reshape from (B*S, nb_classes) -> (B, S, nb_classes) and mean across S
                logits = logits.reshape(B, S, -1).mean(dim=1)
            # -----------------------------------------------------------

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask
                eval_logits = logits
                eval_targets = target
            elif class_mask is not None:
                es, ee = _task_logit_slice(class_mask, task_id)
                eval_logits = logits[:, es:ee]
                eval_targets = _global_to_task_local(
                    target, class_mask, task_id, args.nb_classes, device
                )
            else:
                eval_logits = logits
                eval_targets = target

            loss = criterion(eval_logits, eval_targets)
            # print('Loss')
            predicts = torch.max(eval_logits, dim=1)[1]
            correct += (predicts == eval_targets).sum()
            total += len(target)
            acc1, acc5 = accuracy(eval_logits, eval_targets, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=B)

    end_eval_time = time.time()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    
    print(f"Batchwise eval time for task {task_id+1} = {(end_eval_time - start_eval_time)/len(data_loader)}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total, correct


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id+1):
        logging.info('Evaluating task {}...'.format(i+1))
        test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args,)

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total)*100.0
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, final_acc, avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, lr_scheduler, optimizer, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    old_num_k = 0

    for task_id in range(args.num_tasks):

        if task_id>0:
            model.head.update(len(class_mask[task_id]))
     
        # Create new optimizer for each task to clear optimizer status
        not_n_params = []
        n_params = []
        if args.SLCA:
            milestones = [18] if "CIFAR" in args.dataset else [40]

        lrate_decay = 0.1
        param_list = list(model.parameters())
        if task_id:
            for n, p in model.named_parameters():                
                if n.find('norm1')>=0 or n.find('norm2') >= 0 or n.startswith('norm') or n.find('fc_norm') >= 0:
                    # print(f'Param: {n} Param.requires_grad: {p.requires_grad}')
                    n_params.append(p)
                else:
                    not_n_params.append(p)
            
            network_params = [{'params': not_n_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                                {'params': n_params, 'lr': 0.005*args.lr, 'weight_decay': args.weight_decay}] 
        else:
            network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    
        if not args.SLCA:
            print("Using adam optimizer")
            print("Reinitialising optimizer")
            optimizer = optim.Adam(network_params, weight_decay=args.weight_decay)
            if args.sched != 'constant':
                # lr_scheduler, _ = create_scheduler(args, optimizer)
                # Create cosine lr scheduler
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
            elif args.sched == 'constant':
                lr_scheduler = None

        else:
            optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        

        if args.use_e_prompt or args.use_g_prompt:
            old_prompt = copy.deepcopy(model.e_prompt.prompt.clone().detach())
            old_prompt_matcher = copy.deepcopy(model.e_prompt.prompt_embed_matcher)
            
            curr_num_k = num_new_prompts(class_mask, task_id, args)  # Returns number of prompts to be added for this task

            model.e_prompt.process_new_task(old_num_k, old_num_k + curr_num_k)
        else:
            curr_num_k = 0
            old_prompt_matcher = None
            old_prompt = None
        
        print("Task number: ", task_id)
        
        for epoch in range(args.epochs):     
            logging.info('Training for task {} epoch {}/{}'.format(task_id, epoch, args.epochs))     
            train_stats = train_one_epoch(model=model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], 
                                        optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad,
                                        old_prompt_matcher=old_prompt_matcher,
                                        old_prompt=old_prompt,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                        old_num_k=old_num_k,)
            
            if lr_scheduler:
                lr_scheduler.step()


        if args.use_e_prompt or args.use_g_prompt:
            old_num_k += curr_num_k
        
        eval_start_time = time.time()
        test_stats = evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        eval_end_time = time.time()
        print(f"Eval time for task {task_id+1} = {eval_end_time - eval_start_time}")

        
        if args.output_dir and utils.is_main_process():
            path = args.output_dir+'_'+args.dataset
            Path(os.path.join(path, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            
            checkpoint_path = os.path.join(path, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
