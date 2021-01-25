from .core import main
from .util import import_submodules

if __name__ == '__main__':
    import_submodules(__package__)
    main()