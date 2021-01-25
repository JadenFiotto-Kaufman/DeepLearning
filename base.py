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
    def get_instance(cls, parent, **kwargs):

        if isinstance(cls, str):
            cls = Base.options(parent)[cls]
            
        kwargs = {**Base.kwargs(cls), **kwargs}
        
        return cls(**kwargs)




    
