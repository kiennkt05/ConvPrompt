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
    params = [p for p in model.parameters() if p.requires_grad]
    params += [p for p in matcher.parameters() if p.requires_grad]

    opt_cfg = args.optimizer
    opt_name = opt_cfg.get('name', 'adamw').lower()
    lr = opt_cfg.get('lr', 1e-4)
    weight_decay = opt_cfg.get('weight_decay', 0.0)

    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = opt_cfg.get('momentum', 0.9)
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
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

    header = f'Rainbow Train Task[{task_id + 1}/{args.num_tasks}] Epoch[{epoch + 1}/{max_epochs}]'

    offset = _class_offset(class_mask, task_id)
    task_classes = _task_num_classes(class_mask, task_id)

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

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

        if getattr(args, 'clip_grad', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        with torch.no_grad():
            preds = logits_current.argmax(dim=1)
            acc = (preds == adjusted_targets).float().mean() * 100.0

        metric_logger.update(
            Loss=total_loss.item(),
            CE=ce_loss.item(),
            Match=match_loss.item(),
            Sparsity=sparsity_loss.item(),
            Acc=acc.item() if torch.is_tensor(acc) else acc,
        )

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
):
    model.eval()
    model.rainbow_prompt.set_training(False)
    model.rainbow_load_task(task_id, device)

    task_embedding = matcher.get_task_embedding(task_id, device)
    model.rainbow_set_task_embedding(task_embedding)

    offset = _class_offset(class_mask, task_id)
    task_classes = _task_num_classes(class_mask, task_id)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    criterion = torch.nn.CrossEntropyLoss()

    total = 0
    correct = 0

    header = f'Rainbow Eval Task[{task_id + 1}/{args.num_tasks}]'

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(samples, task_id=task_id, train=False)
        logits = output['logits']
        logits_current = logits[:, offset: offset + task_classes]
        adjusted_targets = targets - offset

        loss = criterion(logits_current, adjusted_targets)
        preds = logits_current.argmax(dim=1)

        total += targets.size(0)
        correct += (preds == adjusted_targets).sum().item()
        acc = (preds == adjusted_targets).float().mean() * 100.0

        metric_logger.update(Loss=loss.item(), Acc=acc.item())

    return {
        'loss': metric_logger.meters['Loss'].global_avg,
        'acc1': metric_logger.meters['Acc'].global_avg,
        'total': total,
        'correct': correct,
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
    task_accuracies = []
    total_samples = 0
    total_correct = 0

    for eval_task in range(task_id + 1):
        stats = evaluate_rainbow(model, matcher, data_loader[eval_task]['val'], device, eval_task, class_mask, args)
        acc_matrix[eval_task, task_id] = stats['acc1']
        task_accuracies.append(stats['acc1'])
        total_samples += stats['total']
        total_correct += stats['correct']

    avg_accuracy = float(np.mean(task_accuracies)) if task_accuracies else 0.0

    forgetting = 0.0
    if task_id > 0:
        previous_max = np.max(acc_matrix[:, :task_id], axis=1)
        current_acc = acc_matrix[:, task_id]
        forgetting = float(np.mean((previous_max - current_acc)[:task_id]))

    overall_acc = 100.0 * total_correct / max(total_samples, 1)

    result = {
        'average_accuracy': avg_accuracy,
        'overall_accuracy': overall_acc,
        'forgetting': forgetting,
    }

    logging.info(
        "[Rainbow avg till task %d]\tAcc@1: %.4f\tOverall: %.4f\tForgetting: %.4f",
        task_id + 1,
        avg_accuracy,
        overall_acc,
        forgetting,
    )

    print(
        "[Rainbow avg till task{}]\tAcc@1: {:.4f}\tOverall: {:.4f}\tForgetting: {:.4f}".format(
            task_id + 1, avg_accuracy, overall_acc, forgetting
        )
    )

    return result


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

    for task_id in range(args.num_tasks):
        if task_id > 0 and hasattr(model.head, 'update'):
            model.head.update(len(class_mask[task_id]))

        model.rainbow_start_task(task_id)

        optimizer, scheduler = build_rainbow_optimizer(args, model, matcher)

        for epoch in range(args.epochs):
            logging.info('Rainbow training task %d epoch %d/%d', task_id + 1, epoch + 1, args.epochs)
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

        eval_stats = evaluate_rainbow_till_now(
            model=model,
            matcher=matcher,
            data_loader=data_loader,
            device=device,
            task_id=task_id,
            class_mask=class_mask,
            acc_matrix=acc_matrix,
            args=args,
        )

        logging.info(
            'After task %d: Avg Acc %.4f | Overall Acc %.4f | Forgetting %.4f',
            task_id + 1,
            eval_stats['average_accuracy'],
            eval_stats['overall_accuracy'],
            eval_stats['forgetting'],
        )

        if getattr(args, 'output_dir', None) and utils.is_main_process():
            ckpt_dir = Path(args.output_dir) / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'matcher': matcher.state_dict(),
                'task_id': task_id,
            }
            torch.save(checkpoint, ckpt_dir / f'task_{task_id + 1}.pth')

    return acc_matrix