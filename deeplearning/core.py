import os
import time
import sys

import numpy as np
import torch
import argparse

from deeplearning import util
from deeplearning.base import Base
from deeplearning.datasets import Dataset
from deeplearning.losses import Loss
from deeplearning.optimizers import Optimizer
from deeplearning.models import Model
from deeplearning.models.wrappers import DataParallel
from deeplearning.schedulers import Scheduler
from deeplearning.validators import Validator


def init_args(parser):
    parser.add_argument('--load', type=str,help='path to checkpoint')
    parser.add_argument("--dont_load_args", default=False, const=True, nargs='?',
                        help="If loading from checkpoint, don't load args instead take from command line and defaults")
    parser.add_argument("--cuda", nargs='?', default=False, const=True, type=bool)

def core_args(parser):
    
    parser.add_argument("--model", type=str, choices=Base.options(Model).keys(), required=True)
    parser.add_argument("--model_wrappers", type=str, choices=Base.options(Model.__wrapper__).keys(), nargs='*', default=None)
    parser.add_argument("--dataset", type=str, choices=Base.options(Dataset).keys(), required=True)
    parser.add_argument("--dataset_wrappers", type=str, choices=Base.options(Dataset.__wrapper__).keys(), nargs='*', default=None)
    parser.add_argument("--loss", type=str, choices=Base.options(Loss).keys(), required=True)
    parser.add_argument("--optimizer", type=str, choices=Base.options(Optimizer).keys(), required=True)
    parser.add_argument("--scheduler", type=str, choices=Base.options(Scheduler).keys(), default=None)
    parser.add_argument("--validators", type=str, choices=Base.options(Validator).keys(), nargs='*', default=None)

    
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
        
        model_state_dict, checkpoint = util.load(args.load, device, dont_load_args=args.dont_load_args)

        if 'optimizer' in checkpoint:
            optimizer_state_dict = checkpoint['optimizer']
        if 'scheduler' in checkpoint:
            scheduler_state_dict = checkpoint['scheduler']
        if 'tracker' in checkpoint:
            training_tracker = checkpoint['tracker']

    return model_state_dict, optimizer_state_dict, scheduler_state_dict, training_tracker

def get_dataloaders(dataset, args):

    train_loader = None
    val_loader = None

    if dataset.dataset_type == Dataset.DatasetType.train:
            train_loader = dataset.dataloader()
            val_loader = Base.get_instance(args.dataset, parent=Dataset, wrappers=args.dataset_wrappers, dataset_type=Dataset.DatasetType.validate, device=dataset.device)[0].dataloader()
    else:
        val_loader = dataset.dataloader()

    return train_loader, val_loader



def main():
    util.import_submodules(__package__)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    init_args(parser)

    _args, _ = parser.parse_known_args()

    device = util.get_device('cuda' if _args.cuda else 'cpu')
    
    model_state_dict, optimizer_state_dict, scheduler_state_dict, training_tracker = init(_args, device)

    parser = argparse.ArgumentParser(allow_abbrev=False)

    core_args(parser)

    args, _ = parser.parse_known_args()
    kwargs = vars(args)

    dataset, _kwargs = Base.get_instance(args.dataset, parent=Dataset, wrappers=args.dataset_wrappers, device=device)
    kwargs.update(_kwargs)

    train_loader, val_loader = get_dataloaders(dataset, args)

    model, _kwargs = Base.get_instance(args.model, parent=Model, wrappers=args.model_wrappers)
    kwargs.update(_kwargs)

    if model_state_dict:
        if isinstance(model, DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

    model = model.to(device)

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

    criterion, _kwargs = Base.get_instance(args.loss, parent=Loss)
    criterion = criterion.to(device)

    kwargs.update(_kwargs)

    validators = []

    if args.validators:

        validators, _kwargss = list(zip(*[Base.get_instance(validator, parent=Validator) for validator in args.validators]))

        [kwargs.update(_kwargs) for _kwargs in _kwargss]

    if dataset.dataset_type is Dataset.DatasetType.validate:
        print("=> validation mode")
        _, results = validate(val_loader, model, criterion,
                              device, args.print_freq, validators, save_results=True)
    elif dataset.dataset_type is Dataset.DatasetType.predict: 
        print("=> prediction mode")
        results = predict(model, val_loader, device)
    else:
        print("=> training mode")
        train(model, criterion, train_loader, val_loader, optimizer, scheduler,
              range(args.start_epoch, args.epochs), device,
              args.print_freq, args.save_freq, args.best_loss, training_tracker, validators, kwargs)

def train_epoch(model, criterion, loader, optimizer, epoch, device, print_freq):
    model.train()

    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')

    progress = util.ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (data, targets) in enumerate(loader):
        
        data_time.update(time.time() - end)
        
        if not isinstance(model, DataParallel):
            data = util.to_device(data, device)
        targets = util.to_device(targets, device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, targets)
        losses.update(loss.item(), loader.batch_size)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return losses.avg

def train(model, criterion, train_loader, val_loader, optimizer, scheduler, epoch_range, device, print_freq, save_freq, best_loss, training_tracker, validators,  args):

    for epoch in epoch_range:

        train_loss = train_epoch(model, criterion, train_loader,
                                 optimizer, epoch, device, print_freq)

        if scheduler:
            scheduler.step()

        training_tracker['train'].append((epoch, train_loss))

        if (epoch + 1) % save_freq == 0:
            val_loss, _ = validate(
                val_loader, model, criterion, device, print_freq, validators)

            training_tracker['val'].append((epoch, val_loss))

            util.save_training_tracker(
                './training.png', training_tracker['train'], training_tracker['val'])

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            args['best_loss'] = best_loss
            args['start_epoch'] = epoch + 1

            util.save_checkpoint({
                'state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'tracker': training_tracker,
                'args': args
            }, is_best)

            print("=> model saved")

def validate(loader, model, criterion, device, print_freq, validators, save_results=False):
    model.eval()

    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')

    validator_meters = [util.AverageMeter(validator.name(), validator.format()) for validator in validators]

    progress =  util.ProgressMeter(
        len(loader),
        validator_meters + [batch_time, losses],
        prefix='Validation: ')

    results = []

    with torch.no_grad():
        end = time.time()
        for i, (data, targets) in enumerate(loader):
            
            if not isinstance(model, DataParallel):
                data = util.to_device(data, device)
            targets = util.to_device(targets, device)

            output = model(data)
            loss = criterion(output, targets)
            losses.update(loss.item(), loader.batch_size)

            for vi, meter in enumerate(validator_meters):
                meter.update(validators[vi](output, targets))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
            if save_results:
                results.extend([(i * len(data) + ii, output[ii].cpu().numpy()) for ii in range(len(data))])

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
