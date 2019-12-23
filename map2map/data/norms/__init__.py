from importlib import import_module

from . import cosmology


def import_norm(path):
    mod, fun = path.rsplit('.', 1)
    mod = import_module('.' + mod, __name__)
    fun = getattr(mod, fun)
    return fun
