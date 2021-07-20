import argparse
from deeplearning.util import import_class


class Base():

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def kwargs(cls):

        parser = argparse.ArgumentParser(allow_abbrev=False)
        cls.args(parser)
        return vars(parser.parse_known_args()[0])

    @staticmethod
    def args(parser):
        pass

    @staticmethod
    def get_instance(cls, parent=None, wrappers=None, **kwargs):

        if isinstance(cls, str):
            cls = import_class(cls)

        _kwargs = Base.kwargs(cls)

        instance = cls(**{**_kwargs, **kwargs})

        if wrappers:
            for wrapper in wrappers:
                if isinstance(wrapper, str):
                    wrapper = import_class(wrapper)
                wkwargs = Base.kwargs(wrapper)
                instance = wrapper(obj=instance, **{**wkwargs, **kwargs})

                _kwargs.update(wkwargs)

        return instance, _kwargs


class _Wrapper(Base):
    def __init__(self, obj, **kwargs):
        self.__dict__['_obj'] = obj
        self.__dict__.update(obj.__dict__)

        return kwargs

    def __len__(self):
        return len(self._obj)

    @staticmethod
    def args(parser):
        pass


Base.__wrapper__ = _Wrapper
