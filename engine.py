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

    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # metric_logger.add_meter('Lr_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if args.SLCA:
        metric_logger.add_meter('Lr_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Lr_rps', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # Masking and computing loss
        known_classes = task_id*len(class_mask[0])
        cur_targets = torch.where(target-known_classes>=0,target-known_classes,-100)
        loss = criterion(logits[:, known_classes:], cur_targets) # base criterion (CrossEntropyLoss)


        if args.use_e_prompt or args.use_g_prompt:
            if task_id > 0:
                l1_loss = 0.0
                for old_wt, new_wt in zip(old_prompt_matcher.parameters(), model.e_prompt.prompt_embed_matcher.parameters()):
                    l1_loss += torch.norm(old_wt.detach() - new_wt, p=1)
                loss = loss + 0.01 * l1_loss
                prompt_loss = torch.norm(old_prompt.detach() - model.e_prompt.prompt, p=1)
                loss = loss + 0.01 * prompt_loss


        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        if args.use_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        if args.SLCA:
            metric_logger.update(Lr_cls=optimizer.param_groups[0]["lr"])
            metric_logger.update(Lr_rps=optimizer.param_groups[1]["lr"]) 
        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
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

            # compute output

            output = model(input, task_id=task_id)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            # print('Loss')
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts == target).sum()
            total += len(target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

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
        test_stats['Forgetting'] = forgetting
        test_stats['Backward'] = backward
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


def _class_offset(class_mask, task_id):
    if class_mask is None:
        return 0
    return sum(len(mask) for mask in class_mask[:task_id])


def _task_num_classes(class_mask, task_id):
    if class_mask is None:
        return 0
    return len(class_mask[task_id])


def build_rainbow_optimizer(args, model, matcher):
    optimizer_params = []
    
    # Pixel prompt parameters (if enabled)
    # Check args.rainbow first (for Rainbow configs), then fall back to top-level (for legacy configs)
    if hasattr(args, 'rainbow') and isinstance(args.rainbow, dict):
        pixel_prompt = args.rainbow.get('pixel_prompt', getattr(args, 'pixel_prompt', 'NO'))
    else:
        pixel_prompt = getattr(args, 'pixel_prompt', 'NO')
    if pixel_prompt == "YES":
        prompt_branch_params = []
        for prompt_net in model.prompt_generators:
            prompt_branch_params.extend(list(prompt_net.parameters()))

        params_Mask = [
            p
            for p in prompt_branch_params
            if p.requires_grad
        ]
        if params_Mask:
            optimizer_params.append({'params': params_Mask, 'lr': getattr(args, 'lr_local', 2e-4)})  # LGSP default: 2e-4

    # Frequency mask parameters (if enabled)
    if hasattr(args, 'rainbow') and isinstance(args.rainbow, dict):
        frequency_mask_enabled = args.rainbow.get('Frequency_mask', getattr(args, 'Frequency_mask', False))
    else:
        frequency_mask_enabled = getattr(args, 'Frequency_mask', False)
    if frequency_mask_enabled:
        if not hasattr(model, 'weights'):
            raise AttributeError("Frequency_mask is enabled but model.weights was not created during initialization. "
                               "This may indicate a mismatch between config and initialization code.")
        params_Frequency_mask = [model.weights]
        optimizer_params.append({'params': params_Frequency_mask, 'lr': getattr(args, 'lr_Frequency_mask', 0.03)})

    # Adaptive weighting parameters (if enabled)
    if hasattr(args, 'rainbow') and isinstance(args.rainbow, dict):
        adaptive_weighting = args.rainbow.get('adaptive_weighting', getattr(args, 'adaptive_weighting', False))
    else:
        adaptive_weighting = getattr(args, 'adaptive_weighting', False)
    if adaptive_weighting:
        if not (hasattr(model, 'alpha') and hasattr(model, 'beta')):
            raise AttributeError("adaptive_weighting is enabled but model.alpha/beta were not created during initialization. "
                               "This may indicate a mismatch between config and initialization code.")
        params_adaptive = [model.alpha, model.beta]
        optimizer_params.append({'params': params_adaptive, 'lr': 0.1})
    
    # Rainbow prompt parameters (base prompts for current session)
    base_lr = args.optimizer.get('lr', 1e-3)  # Use consistent learning rate for joint training
    for layer_idx in range(len(model.blocks)):
        for prompt in model.rainbow_prompt.base_prompts[layer_idx]:
            if prompt.requires_grad:
                optimizer_params.append({'params': [prompt], 'lr': base_lr})
    
    # Rainbow evolution parameters (W_evolution)
    optimizer_params.append({'params': model.rainbow_prompt.evolutions.parameters(), 'lr': base_lr})
    
    # Rainbow gate parameters (G_t / δ_tl) - if adaptive gating is used
    if model.rainbow_prompt.current_gate is not None:
        optimizer_params.append({'params': model.rainbow_prompt.current_gate.parameters(), 'lr': base_lr})

    # Matcher parameters (e_t - task embedding)
    optimizer_params.append({'params': [p for p in matcher.parameters() if p.requires_grad], 'lr': base_lr})

    # Classifier head (ϕ) - only train current task's head
    # In continual learning, we explicitly train only the last (current) head
    # to avoid any potential interference with frozen old task heads
    head_lr_multiplier = args.rainbow.get('head_lr_multiplier', 2.0)
    head_lr = base_lr * head_lr_multiplier

    if hasattr(model.head, 'heads') and len(model.head.heads) > 0:
        # Multi-head setup: explicitly use only the last (current task) head
        current_head = model.head.heads[-1]
        head_params = [p for p in current_head.parameters() if p.requires_grad]
        if head_params:
            optimizer_params.append({'params': head_params, 'lr': head_lr})
    else:
        # Fallback for single-head or other head types
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        if head_params:
            optimizer_params.append({'params': head_params, 'lr': head_lr})

    opt_cfg = args.optimizer
    opt_name = opt_cfg.get('name', '').lower()
    weight_decay = opt_cfg.get('weight_decay', 0.0)

    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(optimizer_params, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = opt_cfg.get('momentum', 0.9)
        optimizer = torch.optim.SGD(optimizer_params, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    scheduler = None
    sched_cfg = getattr(args, 'scheduler', None)
    if sched_cfg:
        sched_name = sched_cfg.get('name', '').lower()
        if sched_name == 'cosine':
            eta_min = sched_cfg.get('min_lr', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
        elif sched_name == 'constant':
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

    return optimizer, scheduler

def train_one_epoch_rainbow(
    model: torch.nn.Module,
    matcher,
    criterion,
    data_loader: Iterable,
    optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    task_id: int,
    class_mask,
    args,
):
    model.train(True)
    model.rainbow_prompt.set_training(True)
    model.rainbow_set_epoch(epoch, max_epochs)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('CE', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Match', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Sparsity', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # Meters for time and data loading time
    metric_logger.add_meter('time', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('data', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))

    offset = _class_offset(class_mask, task_id)
    task_classes = _task_num_classes(class_mask, task_id)

    # Gradient accumulation parameters
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad() 
    
    accumulation_step = 0
    end = time.time()
    
    # Iterate without log_every to suppress batch-wise output
    for samples, targets in data_loader:
        data_time = time.time() - end
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        task_embedding = matcher.get_task_embedding(task_id, device)
        model.rainbow_set_task_embedding(task_embedding)

        output = model(samples, task_id=task_id, train=True)
        logits = output['logits']
        logits_current = logits[:, offset: offset + task_classes]

        adjusted_targets = targets - offset
        ce_loss = criterion(logits_current, adjusted_targets)

        total_loss = ce_loss
        sparsity_loss = torch.tensor(0.0, device=device)
        match_loss = torch.tensor(0.0, device=device)

        aux = output.get('rainbow_aux', {})
        if args.lambda_sparse > 0 and aux:
            sparsity_loss = sum(aux.values())
            total_loss = total_loss + args.lambda_sparse * sparsity_loss

        if args.lambda_match > 0:
            match_loss = matcher.match_loss(output['pre_logits'], task_embedding)
            total_loss = total_loss + args.lambda_match * match_loss

        total_loss.backward()

        accumulation_step += 1

        if accumulation_step % gradient_accumulation_steps == 0:
            if getattr(args, 'clip_grad', None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = logits_current.argmax(dim=1)
            acc = (preds == adjusted_targets).float().mean() * 100.0

        batch_time = time.time() - end
        
        metric_logger.update(
            Loss=total_loss.item(),
            CE=ce_loss.item(),
            Match=match_loss.item(),
            Sparsity=sparsity_loss.item(),
            Acc=acc.item() if torch.is_tensor(acc) else acc,
            time=batch_time,
            data=data_time
        )
        end = time.time()
    
    # Handle remaining gradients
    if accumulation_step % gradient_accumulation_steps != 0:
        if getattr(args, 'clip_grad', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize and print summary
    metric_logger.synchronize_between_processes()
    
    print(f"Rainbow Train Task[{task_id + 1}/{args.num_tasks}] Epoch[{epoch + 1}/{max_epochs}] "
          f"Loss: {metric_logger.Loss.global_avg:.4f} "
          f"CE: {metric_logger.CE.global_avg:.4f} "
          f"Match: {metric_logger.Match.global_avg:.4f} "
          f"Sparsity: {metric_logger.Sparsity.global_avg:.4f} "
          f"Acc: {metric_logger.Acc.global_avg:.2f} "
          f"time: {metric_logger.time.global_avg:.4f} "
          f"data: {metric_logger.data.global_avg:.4f} "
          f"max mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_rainbow(
    model: torch.nn.Module,
    matcher,
    data_loader: Iterable,
    device: torch.device,
    task_id: int,
    class_mask,
    args,
    current_training_step: int = None,
):
    model.eval()
    model.rainbow_prompt.set_training(False)

    if current_training_step is None:
        current_training_step = task_id
    total_seen_classes = _class_offset(class_mask, current_training_step + 1) if class_mask else None

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Acc@1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('Acc@5', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    criterion = torch.nn.CrossEntropyLoss()

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    # Iterate without log_every to suppress batch-wise output
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        task_embedding = matcher.get_task_embedding(task_id, device)
        model.rainbow_set_task_embedding(task_embedding)
        
        output = model(samples, task_id=task_id, train=False)
        logits = output['logits']
        
        if total_seen_classes is not None:
            logits_eval = logits[:, :total_seen_classes]
        else:
            logits_eval = logits
        
        loss = criterion(logits_eval, targets)

        acc1, acc5 = accuracy(logits_eval, targets, topk=(1, 5))
        batch_size = targets.size(0)
        total += batch_size
        correct_top1 += acc1.item() / 100.0 * batch_size
        correct_top5 += acc5.item() / 100.0 * batch_size

        metric_logger.meters['Loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['Acc@1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['Acc@5'].update(acc5.item(), n=batch_size)

    avg_acc1 = metric_logger.meters['Acc@1'].global_avg
    avg_acc5 = metric_logger.meters['Acc@5'].global_avg

    metric_logger.synchronize_between_processes()
    
    # Concise single-line output per eval task
    print(f"Rainbow Eval Task[{task_id + 1}/{args.num_tasks}] "
          f"Acc@1 {avg_acc1:.3f} Acc@5 {avg_acc5:.3f} Loss {metric_logger.meters['Loss'].global_avg:.3f}")

    return {
        'loss': metric_logger.meters['Loss'].global_avg,
        'acc1': avg_acc1,
        'acc5': avg_acc5,
        'total': total,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5,
    }


def evaluate_rainbow_till_now(
    model: torch.nn.Module,
    matcher,
    data_loader,
    device: torch.device,
    task_id: int,
    class_mask,
    acc_matrix,
    args,
):
    num_tasks = args.num_tasks
    stat_matrix = np.zeros((3, num_tasks))

    total_samples = 0
    total_correct_top1 = 0
    total_correct_top5 = 0

    for eval_task in range(task_id + 1):
        stats = evaluate_rainbow(
            model,
            matcher,
            data_loader[eval_task]['val'],
            device,
            eval_task,
            class_mask,
            args,
            current_training_step=task_id, 
        )
        acc_matrix[eval_task, task_id] = stats['acc1']
        stat_matrix[0, eval_task] = stats['acc1']
        stat_matrix[1, eval_task] = stats['acc5']
        stat_matrix[2, eval_task] = stats['loss']

        total_samples += stats['total']
        total_correct_top1 += stats['correct_top1']
        total_correct_top5 += stats['correct_top5']

    avg_stat = np.divide(np.sum(stat_matrix[:, : task_id + 1], axis=1), task_id + 1)

    final_acc1 = 100.0 * total_correct_top1 / max(total_samples, 1)
    final_acc5 = 100.0 * total_correct_top5 / max(total_samples, 1)

    forgetting = 0.0
    backward = 0.0
    if task_id > 0:
        previous_max = np.max(acc_matrix[:, :task_id], axis=1)
        current_acc = acc_matrix[:, task_id]
        forgetting = float(np.mean((previous_max - current_acc)[:task_id]))
        diagonal = np.diag(acc_matrix)
        backward = float(np.mean((current_acc - diagonal)[:task_id]))

    summary_str = (
        f"[Average accuracy till task{task_id + 1}]\tAcc@1: {final_acc1:.4f}\tAcc@5: {avg_stat[1]:.4f}"
        f"\tLoss: {avg_stat[2]:.4f}"
    )
    if task_id > 0:
        summary_str += f"\tForgetting: {forgetting:.4f}\tBackward: {backward:.4f}"

    print(summary_str)
    logging.info(summary_str)

    return {
        'avg_acc1': final_acc1,
        'forgetting': forgetting,
    }


def train_and_evaluate_rainbow(
    model: torch.nn.Module,
    matcher,
    criterion,
    data_loader,
    device: torch.device,
    class_mask,
    args,
):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    acc, forgetting = [], []
    for task_id in range(args.num_tasks):
        if task_id > 0 and hasattr(model.head, 'update'):
            model.head.update(len(class_mask[task_id]))

        model.rainbow_start_task(task_id)

        optimizer, scheduler = build_rainbow_optimizer(args, model, matcher)

        for epoch in range(args.epochs):
            train_stats = train_one_epoch_rainbow(
                model=model,
                matcher=matcher,
                criterion=criterion,
                data_loader=data_loader[task_id]['train'],
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_epochs=args.epochs,
                task_id=task_id,
                class_mask=class_mask,
                args=args,
            )

            if scheduler:
                scheduler.step()

            logging.info('Task %d Epoch %d stats: %s', task_id + 1, epoch + 1, train_stats)

        model.rainbow_finalize_task(task_id)

        summary_stats = evaluate_rainbow_till_now(
            model=model,
            matcher=matcher,
            data_loader=data_loader,
            device=device,
            task_id=task_id,
            class_mask=class_mask,
            acc_matrix=acc_matrix,
            args=args,
        )
        acc.append(summary_stats['avg_acc1'])
        forgetting.append(summary_stats['forgetting'])

        if getattr(args, 'output_dir', None) and utils.is_main_process():
            ckpt_dir = Path(args.output_dir) / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'matcher': matcher.state_dict(),
                'task_id': task_id,
            }
            torch.save(checkpoint, ckpt_dir / f'task_{task_id + 1}.pth')

    print("\n\n")
    print("="*20 + "Final Results" + "="*20)
    print("Average Accuracy:\n", acc)
    print("Average Forgetting:\n", forgetting)
    print("="*50)
    print("\n\n")
    
    return acc_matrix