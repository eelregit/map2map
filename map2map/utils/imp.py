import os
import importlib


def import_attr(name, pkg, callback_at=None):
    """Import attribute. Try package first and then callback directory.

    To use a callback, `name` must contain a module, formatted as 'mod.attr'.

    Examples
    --------
    >>> import_attr('attr', 'pkg1.pkg2')

    tries to import attr from pkg1.pkg2.

    >>> import_attr('mod.attr', 'pkg1.pkg2', 'path/to/cb_dir')

    first tries to import attr from pkg1.pkg2.mod, then from
    'path/to/cb_dir/mod.py'.
    """
    if name.count('.') == 0:
        attr = name

        return getattr(importlib.import_module(pkg), attr)
    else:
        mod, attr = name.rsplit('.', 1)

        try:
            return getattr(importlib.import_module(pkg + '.' + mod), attr)
        except (ModuleNotFoundError, AttributeError):
            if callback_at is None:
                raise

        callback_at = os.path.join(callback_at, mod + '.py')
        assert os.path.isfile(callback_at), 'callback file not found'

        spec = importlib.util.spec_from_file_location(mod, callback_at)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, attr)
