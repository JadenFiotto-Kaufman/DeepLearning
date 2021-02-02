import pkgutil
import importlib
import matplotlib.pyplot as plt
import torch
import shutil
import sys


def subclasses(cls):

    _subclasses = {}
    
    for scls in cls.__subclasses__():
        if not scls.__name__.startswith('_'):
            key = '.'.join((f'{scls.__module__}.{scls.__name__}'.split('.')[2:]))
            _subclasses[key] = scls
        _subclasses.update(subclasses(scls))

    return _subclasses


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        if len(full_name.split('.')) > 2:
            results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results




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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_training_tracker(path, train_loss, val_loss):

    plt.scatter(*zip(*train_loss), label='Train', c='red')
    plt.scatter(*zip(*val_loss), label='Val', c='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(path)
    plt.clf()

def get_device(device):

    if 'cuda' in str(device):
        if torch.cuda.device_count() == 0:
            print("=> no cuda devices available")
        elif not torch.cuda.is_available():
            print("=> cuda is not available")
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

    print("=> loaded checkpoint '{}'"
            .format(path))

    return model_state_dict, checkpoint