from importlib import import_module

from . import cosmology


def import_norm(path):
    mod, func = path.rsplit('.', 1)
    mod = import_module('.' + mod, __name__)
    func = getattr(mod, func)
    return func
