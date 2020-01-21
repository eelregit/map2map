from importlib import import_module

from . import cosmology


def import_norm(norm):
    if callable(norm):
        return norm

    mod, fun = norm.rsplit('.', 1)
    mod = import_module('.' + mod, __name__)
    fun = getattr(mod, fun)
    return fun
