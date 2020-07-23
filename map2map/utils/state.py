import sys
import warnings
from pprint import pformat


def load_model_state_dict(module, state_dict, strict=True):
    bad_keys = module.load_state_dict(state_dict, strict)

    if len(bad_keys.missing_keys) > 0:
        warnings.warn('Missing keys in state_dict:\n{}'.format(
            pformat(bad_keys.missing_keys)))
    if len(bad_keys.unexpected_keys) > 0:
        warnings.warn('Unexpected keys in state_dict:\n{}'.format(
            pformat(bad_keys.unexpected_keys)))
    sys.stderr.flush()
