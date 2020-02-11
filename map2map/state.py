import warnings
from pprint import pformat


def load_model_state_dict(model, state_dict, strict=True):
    bad_keys = model.load_state_dict(state_dict, strict)

    if bad_keys.missing_keys or bad_keys.unexpected_keys:
        warnings.warn(pformat(bad_keys))
