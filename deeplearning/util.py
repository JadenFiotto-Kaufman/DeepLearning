import matplotlib.pyplot as plt
import torch
import shutil
import sys
import importlib
import argparse

def import_class(cls):

    levels = cls.split('.')

    module = importlib.import_module('.'.join(levels[:-1]))
    cls = getattr(module, levels[-1])

    return cls


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def save_training_tracker(path, train_loss, val_loss, lr):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    sc1 = ax1.scatter(*zip(*train_loss), label='Train', c='red')
    sc2 = ax1.scatter(*zip(*val_loss), label='Val', c='blue')
    st1 = ax2.step(*zip(*lr), label='lr', c='lightpink', alpha=.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale('log')

    fig.legend(loc="upper right")
    
    plt.savefig(path,bbox_inches='tight',dpi=100)
    plt.clf()

def get_device(device):

    if 'cuda' in str(device):
        if not torch.cuda.is_available():
            print("=> cuda is not available")
        elif torch.cuda.device_count() == 0:
            print("=> no cuda devices available")
        else: 
            return torch.device(device)

    return torch.device("cpu")

def to_device(data, device):

    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, tuple):
        return tuple([to_device(_data, device) for _data in data])

    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = to_device(data[i], device)

    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)

    return data

def load(path, device, dont_load_args=False):

    print("=> loading checkpoint '{}'".format(path))

    checkpoint = torch.load(path, map_location=device)

    if 'args' in checkpoint and not dont_load_args:
        print("==> loading args")
        for arg in checkpoint['args']:
            if f'--{arg}' not in sys.argv:
                value = checkpoint['args'][arg]
                if value is not None:
                    sys.argv.append(f'--{arg}')
                    if not isinstance(value, list):
                        sys.argv.append(f'{value}')
                    else:
                        for _value in value:
                            sys.argv.append(f'{_value}')

                    print(f"===> {arg} : {value}")

    model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    print("=> loaded checkpoint '{}'"
            .format(path))

    return model_state_dict, checkpoint

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
