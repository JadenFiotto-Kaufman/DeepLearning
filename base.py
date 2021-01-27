import argparse
from .util import subclasses




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
    def options(cls):
        return dict(subclasses(cls))

    @staticmethod
    def get_instance(cls, parent=None, wrappers=None, **kwargs):

        if isinstance(cls, str):
            cls = Base.options(parent)[cls]
            
        _kwargs = Base.kwargs(cls)
        
        instance = cls(**{**_kwargs, **kwargs})

        if wrappers:
            for wrapper in wrappers:
                if isinstance(wrapper, str):
                    wrapper = Base.options(cls.__wrapper__)[wrapper]
                wkwargs = Base.kwargs(wrapper)
                instance = wrapper(obj=instance, **wkwargs)

                _kwargs.update(wkwargs)
        
        return instance, _kwargs


class _Wrapper(Base):
        def __init__(self, obj, **kwargs):
            self._obj = obj

        def __getattr__(self, name):
            return getattr(self._obj, name)

        @staticmethod
        def args(parser):
            pass


Base.__wrapper__ = _Wrapper