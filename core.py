import os
import time
import sys

import numpy as np
import torch
import argparse

from DeepLearning import util
from DeepLearning.base import Base
from DeepLearning.datasets import Dataset
from DeepLearning.losses import Loss
from DeepLearning.optimizers import Optimizer
from DeepLearning.models import Model
from DeepLearning.schedulers import Scheduler


def init_args(parser):
    parser.add_argument('--load', type=str,help='path to checkpoint')
    parser.add_argument("--dont_load_args", default=False, const=True, nargs='?',
                        help="If loading from checkpoint, don't load args instead take from command line and defaults")
    parser.add_argument("--cuda", nargs='?', default=False, const=True, type=bool)

def core_args(parser):
    
    parser.add_argument("--model", type=str, choices=Base.options(Model).keys(), required=True)
    parser.add_argument("--dataset", type=str, choices=Base.options(Dataset).keys(), required=True)
    parser.add_argument("--dataset_wrappers", type=str, choices=Base.options(Dataset.__wrapper__).keys(), nargs='*', default=None)
    parser.add_argument("--loss", type=str, choices=Base.options(Loss).keys(), required=True)
    parser.add_argument("--optimizer", type=str, choices=Base.options(Optimizer).keys(), required=True)
    parser.add_argument("--scheduler", type=str, choices=Base.options(Scheduler).keys(), default=None)
    
    parser.add_argument("--print_freq", type=int, default=10,
                        help="number of batches between two logs (default: 10)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="number of epochs between two validations and model saves (default: 10, 0 means no saving)")
    parser.add_argument('--best_loss', type=float, default=np.inf,
                        help="manual best value to start to compare for saving the best model")

    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs  (default: 100)")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="manual start epoch to start at for training")    



def init(args, device):
    training_tracker = {'train': [], 'val': []}
    optimizer_state_dict = None
    scheduler_state_dict = None
    model_state_dict = None

    if args.load:
        print("=> loading checkpoint '{}'".format(args.load))

        checkpoint = torch.load(args.load, map_location=device)

        if 'args' in checkpoint and not args.dont_load_args:
            for arg in checkpoint['args']:
                if f'--{arg}' not in sys.argv:
                    value = checkpoint['args'][arg]
                    if value:
                        sys.argv.append(f'--{arg}')
                        if not isinstance(value, list):
                            sys.argv.append(f'{value}')
                        else:
                            for _value in value:
                                sys.argv.append(f'{_value}')

                        print(f"===> {arg} : {value}")

        model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if 'optimizer' in checkpoint:
            optimizer_state_dict = checkpoint['optimizer']
        if 'scheduler' in checkpoint:
            optimizer_state_dict = checkpoint['scheduler']
        if 'tracker' in checkpoint:
            training_tracker = checkpoint['tracker']

        print("=> loaded checkpoint '{}'"
                .format(args.load))
    return model_state_dict, optimizer_state_dict, scheduler_state_dict, training_tracker

def get_dataloaders(dataset, args):

    train_loader = None
    val_loader = None

    if dataset.dataset_type == Dataset.DatasetType.train:
            train_loader = dataset.dataloader()
            val_loader = Base.get_instance(args.dataset, parent=Dataset, wrappers=args.dataset_wrappers, dataset_type=Dataset.DatasetType.validate)[0].dataloader()
    else:
        val_loader = dataset.dataloader()

    return train_loader, val_loader


def main():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    init_args(parser)

    args, _ = parser.parse_known_args()

    device = torch.device("cpu")
    if args.cuda:
        if torch.cuda.device_count() == 0:
            print("=> no cuda devices available")
        elif not torch.cuda.is_available():
            print("=> cuda is not available")
        else: 
            device = torch.device("cuda")
    
    model_state_dict, optimizer_state_dict, scheduler_state_dict, training_tracker = init(args, device)

    parser = argparse.ArgumentParser(allow_abbrev=False)

    core_args(parser)

    args, _ = parser.parse_known_args()
    kwargs = vars(args)

    dataset, _kwargs = Base.get_instance(args.dataset, parent=Dataset, wrappers=args.dataset_wrappers)
    kwargs.update(_kwargs)

    train_loader, val_loader = get_dataloaders(dataset, args)

    model, _kwargs = Base.get_instance(args.model, parent=Model, dataset=dataset)
    model = model.to(device)
    kwargs.update(_kwargs)

    if model_state_dict:
        model.load_state_dict(model_state_dict)

    optimizer, _kwargs = Base.get_instance(args.optimizer, parent=Optimizer, params=model.parameters())
    kwargs.update(_kwargs)

    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = None

    if args.scheduler:
        scheduler, _kwargs = Base.get_instance(args.scheduler, parent=Scheduler, optimizer=optimizer)
        kwargs.update(_kwargs)

        if scheduler_state_dict:
            scheduler.load_state_dict(scheduler_state_dict)

    criterion, _kwargs = Base.get_instance(args.loss, parent=Loss, dataset=dataset)
    criterion = criterion.to(device)
    kwargs.update(_kwargs)

    if dataset.dataset_type is Dataset.DatasetType.validate:
        print("=> validation mode")
        _, results = validate(val_loader, model, criterion,
                              device, args.print_freq, save_results=True)
    elif dataset.dataset_type is Dataset.DatasetType.predict:
        print("=> prediction mode")
        results = predict(model, val_loader, device)
    else:
        print("=> training mode")
        train(model, criterion, train_loader, val_loader, optimizer, scheduler,
              range(args.start_epoch, args.epochs), device,
              args.print_freq, args.save_freq, args.best_loss, training_tracker, kwargs)

def train_epoch(model, criterion, train_loader, optimizer, epoch, device, print_freq):
    model.train()

    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')

    progress = util.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (data, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return losses.avg

def train(model, criterion, train_loader, val_loader, optimizer, scheduler, epoch_range, device, print_freq, save_freq, best_loss, training_tracker, args):

    for epoch in epoch_range:

        if scheduler:
            scheduler.step()

        train_loss = train_epoch(model, criterion, train_loader,
                                 optimizer, epoch, device, print_freq)

        training_tracker['train'].append((epoch, train_loss))

        if (epoch + 1) % save_freq == 0:
            val_loss, _ = validate(
                val_loader, model, criterion, device, print_freq)

            training_tracker['val'].append((epoch, val_loss))

            util.save_training_tracker(
                './training.png', training_tracker['train'], training_tracker['val'])

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            args['best_loss'] = best_loss
            args['start_epoch'] = epoch + 1

            util.save_checkpoint({
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'tracker': training_tracker,
                'args': args
            }, is_best)

            print("=> model saved")

def validate(val_loader, model, criterion, device, print_freq, save_results=False):
    model.eval()

    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')

    progress =  util.ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Validation: ')

    results = []
    acc = []

    with torch.no_grad():
        end = time.time()
        for i, (data, targets) in enumerate(val_loader):
            data, targets = data.to(device), targets.to(device)

            output = model(data)
            acc.append((targets.cpu().numpy() == output.cpu().numpy().argmax(axis=1)).sum() / len(targets))
            loss = criterion(output, targets)
            losses.update(loss.item(), data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
            if save_results:
                results.extend([(i * len(data) + ii, output[ii].cpu().numpy()) for ii in range(len(data))])
    print(sum(acc) / len(acc))

    return losses.avg, results

def predict(model, loader, device):
    model.eval()

    results = []

    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            output = model(data)

            results.extend([(i * len(data) + ii, output[ii].cpu().numpy()) for ii in range(len(data))])

    return results
